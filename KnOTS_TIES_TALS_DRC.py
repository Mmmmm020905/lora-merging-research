import os
import gc
import traceback
import argparse
from collections import OrderedDict
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import yaml

from transformers import AutoTokenizer, AutoProcessor, AutoModelForSeq2SeqLM
from eval_model import eval_iteris_model
from KnOTS import (
    get_loras_path,
    set_seed,
    ensure_csv_header,
    append_csv_row,
    construct_base_model,
    load_adapter_config,
    construct_fresh_peft_model,
    ordered_ft_state_dict,
    check_state_dict_keys_match,
    get_lora_scaling_from_adapter_cfg,
    load_adapter_state_dict_from_safetensors,
    lora_state_dict_to_delta_matrices,
    get_task_directions,
    add_direction_to_base_model,
)


def directions_to_reps(directions):
    if isinstance(directions, list):
        return [directions_to_reps(direction) for direction in directions]
    sorted_direction = OrderedDict(sorted(directions.items()))
    return torch.nn.utils.parameters_to_vector(
        [value.reshape(-1) for value in sorted_direction.values()]
    )


def rep_to_state_dict(vector, state_dict):
    if isinstance(vector, list):
        return [rep_to_state_dict(v, state_dict) for v in vector]
    if hasattr(vector, "dim") and vector.dim() == 2:
        return [rep_to_state_dict(v, state_dict) for v in vector]

    reference_dict = OrderedDict(
        (key, value.clone()) for key, value in OrderedDict(sorted(state_dict.items())).items()
    )
    torch.nn.utils.vector_to_parameters(vector, reference_dict.values())
    return reference_dict


def apply_knots_svd(ftms_task_dirs, concat_across_output=True, svd_tol=1e-5, verbose_svd=False):
    num_tasks = len(ftms_task_dirs)
    if num_tasks < 2:
        raise ValueError("[KnOTS-TIES] 至少需要两个任务方向用于 KnOTS-TIES。")

    layer_names = list(ftms_task_dirs[0].keys())
    U_dict = OrderedDict()
    task_sV_dicts = [OrderedDict() for _ in range(num_tasks)]

    for layer_name in layer_names:
        layer_mats = [task_dir[layer_name].to(torch.float32) for task_dir in ftms_task_dirs]
        if concat_across_output:
            concat_matrix = torch.cat(layer_mats, dim=1)
        else:
            concat_matrix = torch.cat([mat.t() for mat in layer_mats], dim=1)

        concat_matrix = concat_matrix.to(torch.float64)
        U, s, Vh = torch.linalg.svd(concat_matrix, full_matrices=False)
        keep = s > svd_tol
        if verbose_svd:
            kept_rank = int(keep.sum().item())
            total_rank = int(s.numel())
            max_sv = float(s.max().item()) if total_rank > 0 else 0.0
            min_sv = float(s.min().item()) if total_rank > 0 else 0.0
            min_kept_sv = float(s[keep].min().item()) if kept_rank > 0 else 0.0
            print(
                f"[KnOTS-SVD] layer={layer_name} shape={tuple(concat_matrix.shape)} "
                f"rank_kept={kept_rank}/{total_rank} svd_tol={svd_tol} "
                f"max_sv={max_sv:.6e} min_sv={min_sv:.6e} min_kept_sv={min_kept_sv:.6e}"
            )

        if keep.sum().item() == 0:
            rank_dim = 1
            rows = layer_mats[0].shape[0] if concat_across_output else layer_mats[0].shape[1]
            cols = layer_mats[0].shape[1] if concat_across_output else layer_mats[0].shape[0]
            U_keep = torch.zeros((rows, rank_dim), dtype=torch.float32)
            split_width = cols
            V_chunks = [torch.zeros((rank_dim, split_width), dtype=torch.float32) for _ in range(num_tasks)]
        else:
            U_keep = U[:, keep].to(torch.float32)
            s_keep = s[keep].to(torch.float32)
            Vh_keep = Vh[keep].to(torch.float32)
            split_width = Vh_keep.shape[1] // num_tasks
            V_chunks = [
                torch.diag(s_keep) @ chunk
                for chunk in torch.split(Vh_keep, split_width, dim=1)
            ]

        U_dict[layer_name] = U_keep.cpu()
        for idx, chunk in enumerate(V_chunks):
            task_sV_dicts[idx][layer_name] = chunk.cpu()

    return U_dict, task_sV_dicts


def reconstruct_merged_directions(U_dict, merged_sV_sd, concat_across_output=True):
    merged_direction_sd = OrderedDict()
    for key, U in U_dict.items():
        merged_matrix = (U @ merged_sV_sd[key]).to(torch.float32)
        if not concat_across_output:
            merged_matrix = merged_matrix.t()
        merged_direction_sd[key] = merged_matrix.cpu()
    return merged_direction_sd


def topk_values_mask(M, K=0.7):
    if K > 1:
        K /= 100

    if K >= 1:
        return M, torch.ones_like(M), torch.ones_like(M).mean(dim=-1)

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    _, d = M.shape
    k = int(d * K)
    k = d - k
    if M.flatten().shape[-1] == 1:
        kth_values = M.abs()
    else:
        kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    mask = M.abs() >= kth_values

    if original_shape == M.squeeze().shape:
        final_mask = mask.squeeze()
        M = M.squeeze()
    else:
        final_mask = mask

    return M * final_mask, final_mask, final_mask.float().mean(dim=-1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())
    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(tensor, mode="sum_of_values"):
    if mode == "sum_of_signs":
        sign_to_mult = torch.sign(torch.sum(torch.sign(tensor), dim=0))
    elif mode == "sum_of_values":
        sign_to_mult = torch.sign(tensor.sum(dim=0))
        sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    else:
        raise ValueError(
            f"[KnOTS-TIES] Unknown sign_resolve_mode: {mode}. "
            f"Expected sum_of_values or sum_of_signs."
        )
    return sign_to_mult


def ties_masking(vectors, topK=100, sign_resolve_mode="sum_of_values"):
    stacked_vectors = torch.vstack(vectors).clone()
    pruned_vectors, prune_mask, _ = topk_values_mask(stacked_vectors, K=topK)
    vector_signs = resolve_sign(pruned_vectors, mode=sign_resolve_mode)
    sign_mask = torch.where(
        vector_signs.unsqueeze(0) > 0,
        pruned_vectors > 0,
        pruned_vectors < 0,
    )
    ties_mask = sign_mask * prune_mask
    return ties_mask


def chunked_disjoint_mean(vectors, chunk_size=10000):
    num_chunks = vectors.size(0) // chunk_size + (1 if vectors.size(0) % chunk_size != 0 else 0)
    total_sum = torch.zeros_like(vectors[0])
    non_zero_counts = torch.zeros_like(vectors[0])

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, vectors.size(0))
        chunk = vectors[start_idx:end_idx]
        total_sum += torch.sum(chunk, dim=0)
        non_zero_counts += (chunk != 0).sum(dim=0)

    disjoint_aggs = total_sum / torch.clamp(non_zero_counts.float(), min=1)
    disjoint_aggs[non_zero_counts == 0] = 0
    return disjoint_aggs


def chunked_sum(tensor, chunk_size=10000):
    num_chunks = tensor.size(0) // chunk_size + (1 if tensor.size(0) % chunk_size != 0 else 0)
    total_sum = torch.zeros_like(tensor[0])
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, tensor.size(0))
        chunk = tensor[start_idx:end_idx]
        total_sum += torch.sum(chunk, dim=0)
    return total_sum


def masked_merge(vectors, merge_func, weights=None):
    vectors = vectors.clone()
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=vectors.dtype, device=vectors.device)
        if weights.dim() == 0:
            weights = weights.unsqueeze(0)
        if weights.numel() == 1:
            weights = weights.repeat(vectors.shape[0])
        if weights.numel() != vectors.shape[0]:
            raise ValueError(
                f"[KnOTS-TIES] weights length mismatch: got {weights.numel()}, "
                f"expected {vectors.shape[0]}"
            )
        vectors = vectors * weights.view(-1, 1)

    if merge_func == "mean":
        disjoint_aggs = chunked_disjoint_mean(vectors, chunk_size=10000)
    elif merge_func == "sum":
        disjoint_aggs = chunked_sum(vectors)
    elif merge_func == "max":
        disjoint_aggs = vectors.abs().max(dim=0)[0]
    elif merge_func == "unmerged":
        disjoint_aggs = vectors
    else:
        raise ValueError(f"[KnOTS-TIES] Merge method {merge_func} is not defined.")

    return disjoint_aggs


def merge_knots_ties_param(
    lora_path,
    model_name,
    task_targets,
    seed,
    topK=20,
    sign_resolve_mode="sum_of_values",
    scaling_coeffs=0.5,
    merging_type="mean",
    concat_across_output=True,
    svd_tol=1e-5,
    verbose_svd=False,
):
    assert len(lora_path) == len(task_targets), "lora_path 数量必须和 task_targets 一致"
    if len(task_targets) != 2:
        raise ValueError("[KnOTS-TIES] 当前实现只支持 GLUE pairwise 两任务融合。")

    print(f"[KnOTS-TIES] task_targets = {task_targets}")
    print(f"[KnOTS-TIES] lora_path = {lora_path}")
    print(
        f"[KnOTS-TIES] topK = {topK}, sign_resolve_mode = {sign_resolve_mode}, "
        f"scaling_coeffs = {scaling_coeffs}, merging_type = {merging_type}, "
        f"concat_across_output = {concat_across_output}, svd_tol = {svd_tol}, "
        f"verbose_svd = {verbose_svd}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    start_time = datetime.now()

    adapter_cfg = load_adapter_config(lora_path[0])
    fresh_peft_model = construct_fresh_peft_model(
        model_name=model_name,
        adapter_cfg=adapter_cfg,
        seed=seed,
        device="cpu",
    )
    ptm_reference_params = lora_state_dict_to_delta_matrices(
        ordered_ft_state_dict(fresh_peft_model),
        device="cpu",
        adapter_cfg=adapter_cfg,
    )

    ftms_relevant_params = []
    for path in lora_path:
        adapter_state_dict = load_adapter_state_dict_from_safetensors(path, device="cpu")
        ft_sd = lora_state_dict_to_delta_matrices(
            adapter_state_dict,
            device="cpu",
            adapter_cfg=adapter_cfg,
        )
        ftms_relevant_params.append(ft_sd)

    check_state_dict_keys_match([ptm_reference_params] + ftms_relevant_params)
    ftms_task_dirs = get_task_directions(ptm_reference_params, ftms_relevant_params)

    U_dict, task_sV_dicts = apply_knots_svd(
        ftms_task_dirs,
        concat_across_output=concat_across_output,
        svd_tol=svd_tol,
        verbose_svd=verbose_svd,
    )

    ftms_reps = directions_to_reps(task_sV_dicts)
    masks = ties_masking(ftms_reps, topK=topK, sign_resolve_mode=sign_resolve_mode)

    ftms_reps = torch.vstack(ftms_reps).clone()
    masks = masks.to(ftms_reps.device).to(ftms_reps.dtype)
    masked_sVs = ftms_reps * masks

    pre_merge_sVs_dict = rep_to_state_dict(masked_sVs, task_sV_dicts[0])
    rescaled_reps = torch.stack(directions_to_reps(pre_merge_sVs_dict), dim=0)
    merged_sV = masked_merge(
        vectors=rescaled_reps,
        merge_func=merging_type,
        weights=[scaling_coeffs],
    )
    merged_sV_sd = rep_to_state_dict(merged_sV.cpu(), task_sV_dicts[0])
    merged_direction_sd = reconstruct_merged_directions(
        U_dict,
        merged_sV_sd,
        concat_across_output=concat_across_output,
    )

    model = construct_base_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    number_update = add_direction_to_base_model(model, merged_direction_sd, scaling_coeff=1.0)

    if number_update != len(merged_direction_sd):
        print(
            f"[KnOTS-TIES][Warn] Updated {number_update}/{len(merged_direction_sd)} modules. "
            f"请检查 LoRA 层名与 base model 参数名是否一致。"
        )

    fusion_time = round((datetime.now() - start_time).total_seconds(), 4)
    fusion_peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
        if torch.cuda.is_available()
        else 0.0
    )

    fusion_stats = {
        "fusion_iter_time_avg_sec": fusion_time,
        "fusion_iter_time_max_sec": fusion_time,
        "fusion_peak_vram_avg_mb": fusion_peak_vram_mb,
        "fusion_peak_vram_max_mb": fusion_peak_vram_mb,
    }

    print(f"[KnOTS-TIES] Fusion time: {fusion_time} sec")
    print(f"[KnOTS-TIES] Fusion peak VRAM: {fusion_peak_vram_mb} MB")

    return model, fusion_stats, adapter_cfg




def load_required_coarse_model_from_dir(model_dir, model_label="coarse"):
    """
    Load a previously selected best coarse merged model.

    This is intentionally strict: if the expected directory does not exist,
    we raise an error instead of silently rebuilding the coarse model with
    fixed/default hyperparameters.
    """
    if model_dir is None or str(model_dir).strip() == "":
        raise ValueError(f"[KnOTS_TIES_TALS_DRC] Empty {model_label} model directory.")
    model_dir = str(model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"[KnOTS_TIES_TALS_DRC] Required {model_label} model dir not found: {model_dir}\n"
            f"Please run the corresponding KnOTS-TIES baseline first and save the best coarse model."
        )
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(
            f"[KnOTS_TIES_TALS_DRC] {model_label} model dir exists but config.json is missing: {model_dir}\n"
            f"This does not look like a valid HuggingFace saved model directory."
        )
    print(f"[KnOTS_TIES_TALS_DRC] Loading required {model_label} coarse model from: {model_dir}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return model


def make_loaded_fusion_stats():
    return {
        "fusion_iter_time_avg_sec": 0.0,
        "fusion_iter_time_max_sec": 0.0,
        "fusion_peak_vram_avg_mb": 0.0,
        "fusion_peak_vram_max_mb": 0.0,
    }


def parse_scalar_or_candidates(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return [float(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if "/" in s:
            return [float(x.strip()) for x in s.split("/") if x.strip()]
        if "," in s:
            return [float(x.strip()) for x in s.split(",") if x.strip()]
        return float(s)
    raise ValueError(f"Unsupported value type: {type(value)} for value={value}")


def get_normalized_metric(eval_result, task_name):
    if task_name == "cola":
        return (float(eval_result.get("eval_MCC", 0.0)) + 1.0) / 2.0
    return float(eval_result.get("eval_accuracy", 0.0))


def eval_knots_ties_pair_average(model, tokenizer, model_name, task_targets, max_length, per_device_eval_batch_size):
    normalized_metrics = []
    per_task_results = {}
    for task_name in task_targets:
        eval_result = eval_iteris_model(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            task_name=task_name,
            max_length=max_length,
            per_device_eval_batch_size=per_device_eval_batch_size,
        )
        normalized_metric = get_normalized_metric(eval_result, task_name)
        normalized_metrics.append(normalized_metric)
        per_task_results[task_name] = eval_result
    avg_score = float(np.mean(normalized_metrics))
    return avg_score, per_task_results


def search_best_knots_ties_config(
    lora_path,
    model_name,
    task_targets,
    seed,
    tokenizer,
    max_length,
    per_device_eval_batch_size,
    concat_across_output,
    svd_tol,
    default_params,
    search_config,
    order_of_processing_params,
    sign_resolve_mode,
    merging_type,
    search_strategy="linear",
    early_stop=False,
    verbose_svd=False,
):
    """
    Search KnOTS-TIES hyperparameters.

    search_strategy:
      - "linear": sequentially search scaling_coeffs then topK, using the best value found
                  for previous parameters. This is close to the linear-search protocol used
                  in many model-merging repos, but early_stop is disabled by default here.
      - "grid": full Cartesian product over scaling_coeffs and topK. Slower, but safer
                for diagnosing unstable GLUE/LoRA behavior.
    """
    search_strategy = str(search_strategy).lower().strip()
    print(f"[KnOTS-TIES-SEARCH] default_params = {default_params}")
    print(f"[KnOTS-TIES-SEARCH] order_of_processing_params = {order_of_processing_params}")
    print(f"[KnOTS-TIES-SEARCH] search_config = {search_config}")
    print(f"[KnOTS-TIES-SEARCH] search_strategy = {search_strategy}, early_stop = {early_stop}")

    def evaluate_params(instance_params):
        print(f"[KnOTS-TIES-SEARCH] Try params = {instance_params}")
        model, _, _ = merge_knots_ties_param(
            lora_path=lora_path,
            model_name=model_name,
            task_targets=task_targets,
            seed=seed,
            topK=int(instance_params["topK"]),
            sign_resolve_mode=sign_resolve_mode,
            scaling_coeffs=float(instance_params["scaling_coeffs"]),
            merging_type=merging_type,
            concat_across_output=concat_across_output,
            svd_tol=svd_tol,
            verbose_svd=verbose_svd,
        )
        avg_score, _ = eval_knots_ties_pair_average(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            task_targets=task_targets,
            max_length=max_length,
            per_device_eval_batch_size=per_device_eval_batch_size,
        )
        print(f"[KnOTS-TIES-SEARCH] avg_normalized_metric = {avg_score:.6f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return avg_score

    if search_strategy == "grid":
        best = deepcopy(default_params)
        best["avg_normalized_metric"] = -1e9
        for scaling_value in search_config.get("scaling_coeffs", [default_params["scaling_coeffs"]]):
            for topk_value in search_config.get("topK", [default_params["topK"]]):
                params = deepcopy(default_params)
                params["scaling_coeffs"] = float(scaling_value)
                params["topK"] = int(float(topk_value))
                score = evaluate_params(params)
                if score >= best.get("avg_normalized_metric", -1e9):
                    best = deepcopy(params)
                    best["avg_normalized_metric"] = score
        print(f"[KnOTS-TIES-SEARCH] Best config = {best}")
        return best

    if search_strategy != "linear":
        raise ValueError("knots_ties_search_strategy must be 'linear' or 'grid'.")

    best_val_results = {"avg_normalized_metric": -1e9}
    running_defaults = deepcopy(default_params)

    for param in order_of_processing_params:
        best_for_param = deepcopy(best_val_results)
        found_improvement_once = False
        for value in search_config[param]:
            instance_params = deepcopy(running_defaults)
            instance_params[param] = value
            score = evaluate_params(instance_params)
            if score >= best_for_param.get("avg_normalized_metric", -1e9):
                best_for_param = deepcopy(instance_params)
                best_for_param["avg_normalized_metric"] = score
                found_improvement_once = True
            elif early_stop and found_improvement_once:
                print(
                    f"[KnOTS-TIES-SEARCH] Early stop on param={param}, value={value}, "
                    f"score={score:.6f}, best={best_for_param.get('avg_normalized_metric'):.6f}"
                )
                break

        running_defaults[param] = best_for_param[param]
        best_val_results = deepcopy(best_for_param)

    print(f"[KnOTS-TIES-SEARCH] Best config = {best_val_results}")
    return best_val_results



# ======================================================================================
# TALS-LER extension on top of KnOTS-TIES coarse merge
# ======================================================================================
# This script intentionally reuses the already validated helpers in Linear_TALS_DRC.py.
# The key difference from Linear_TALS_DRC is the definition of the coarse parameter delta:
#   Linear:       ΔW_c = sum_j λ_j ΔW_j
#   KnOTS-TIES:   ΔW_c = W_KnOTS-TIES - W_base
# Then missing update is:
#   R_t^W = ΔW_t - ΔW_c
# and TALS-LER constructs:
#   d_t = omega * Normalize(V_k G_k V_k^T r_act)

from Linear_TALS_DRC import (
    normalize_target_modules,
    select_drc_targets,
    get_task_samples,
    collect_features_by_position,
    load_single_lora_dense_model,
    build_all_task_lora_deltas,
    apply_tals_filter_to_activation_residual,
    apply_ler_layerwise_reweight,
    stable_int_hash,
    parse_float_list,
    parse_optional_float,
    normalize_direction,
    register_drc_hooks_by_position,
    remove_hooks,
    get_primary_metric,
    append_alpha_search_row,
)


def short_hash(text, length=12):
    import hashlib
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()[:length]


def make_knots_ties_tals_cache_path(
    cache_dir,
    method_name,
    pair_name,
    inject_position,
    target_layers,
    target_modules,
    topK,
    scaling_coeffs,
    sign_resolve_mode,
    merging_type,
    concat_across_output,
    svd_tol,
    knots_ties_do_search,
    knots_ties_search_strategy,
    tals_subspace_source,
    tals_rank,
    tals_gamma,
    tals_weight_norm,
    tals_svd_center,
    tals_use_layer_weight,
    tals_layer_weight_score,
    tals_layer_weight_norm,
):
    os.makedirs(cache_dir, exist_ok=True)

    layer_tag = (
        "l" + "_".join(map(str, target_layers))
        if target_layers is not None
        else "lall"
    )
    if inject_position == "lora_input":
        module_tag = "m" + "_".join(normalize_target_modules(target_modules))
    else:
        module_tag = "mblock"

    cfg_text = (
        f"method={method_name}|pair={pair_name}|inject={inject_position}|"
        f"layers={target_layers}|modules={normalize_target_modules(target_modules)}|"
        f"topK={topK}|scaling_coeffs={scaling_coeffs}|"
        f"sign_resolve_mode={sign_resolve_mode}|merging_type={merging_type}|"
        f"concat_across_output={concat_across_output}|svd_tol={svd_tol}|"
        f"knots_ties_do_search={knots_ties_do_search}|"
        f"knots_ties_search_strategy={knots_ties_search_strategy}|"
        f"tals_subspace_source={tals_subspace_source}|tals_rank={tals_rank}|"
        f"tals_gamma={tals_gamma}|tals_weight_norm={tals_weight_norm}|"
        f"tals_svd_center={tals_svd_center}|"
        f"tals_use_layer_weight={tals_use_layer_weight}|"
        f"tals_layer_weight_score={tals_layer_weight_score}|"
        f"tals_layer_weight_norm={tals_layer_weight_norm}"
    )
    h = short_hash(cfg_text, length=12)

    safe_method = str(method_name).replace("/", "_").replace(" ", "_")[:48]
    gamma_tag = str(tals_gamma).replace(".", "p")
    scale_tag = str(scaling_coeffs).replace(".", "p")
    tol_tag = str(svd_tol).replace(".", "p").replace("-", "m")
    filename = (
        f"{safe_method}_{pair_name}_{layer_tag}_{module_tag}_"
        f"kt{topK}_c{scale_tag}_svd{tol_tag}_"
        f"src{tals_subspace_source}_r{tals_rank}_g{gamma_tag}_"
        f"lw{int(bool(tals_use_layer_weight))}_{h}.pt"
    )
    return os.path.join(cache_dir, filename)


def module_weight_delta_from_base(base_model, coarse_model, target_key):
    base_modules = dict(base_model.named_modules())
    coarse_modules = dict(coarse_model.named_modules())

    if target_key not in base_modules:
        return None, f"missing_base_module:{target_key}"
    if target_key not in coarse_modules:
        return None, f"missing_coarse_module:{target_key}"

    base_module = base_modules[target_key]
    coarse_module = coarse_modules[target_key]

    if not hasattr(base_module, "weight"):
        return None, f"base_module_has_no_weight:{target_key}"
    if not hasattr(coarse_module, "weight"):
        return None, f"coarse_module_has_no_weight:{target_key}"

    base_w = base_module.weight.detach().float().cpu()
    coarse_w = coarse_module.weight.detach().float().cpu()

    if base_w.shape != coarse_w.shape:
        return None, f"shape_mismatch:{target_key}:{tuple(base_w.shape)}!={tuple(coarse_w.shape)}"

    return coarse_w - base_w, ""


def build_dense_coarse_delta_w_dict(model_name, coarse_model, target_keys):
    """
    Build ΔW_c = W_coarse - W_base for arbitrary coarse merged model.

    For KnOTS-TIES, the coarse update is not a simple linear combination of source LoRA
    deltas, so TALS must use the dense difference between the merged model and the
    original base model at each LoRA target module.
    """
    print("[KnOTS_TIES_TALS_DRC] Build dense coarse delta W = W_KnOTS-TIES - W_base ...")
    base_model = construct_base_model(model_name)
    coarse_delta = {}
    missing = []

    for key in target_keys:
        delta, reason = module_weight_delta_from_base(base_model, coarse_model, key)
        if delta is None:
            missing.append((key, reason))
            continue
        coarse_delta[key] = delta

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[KnOTS_TIES_TALS_DRC] Built dense coarse delta for "
        f"{len(coarse_delta)}/{len(target_keys)} targets."
    )
    if missing:
        print(f"[KnOTS_TIES_TALS_DRC][Warn] Missing dense coarse delta for {len(missing)} targets.")
        for item in missing[:10]:
            print(f"    {item}")

    return coarse_delta


def build_task_specific_knots_ties_tals_directions(
    model_name,
    tokenizer,
    task_targets,
    lora_path_dict,
    coarse_model,
    rank,
    max_length,
    seed,
    samples_per_task=50,
    select_long=False,
    shuffle=True,
    if_balance=True,
    target_part="encoder",
    target_modules=None,
    target_layers=None,
    normalize=True,
    inject_position="lora_input",
    lora_alpha_list=None,
    tals_rank=8,
    tals_gamma=0.5,
    tals_eps=1e-6,
    tals_weight_norm="mean",
    tals_svd_center=False,
    tals_subspace_source="missing",
    tals_fallback_to_base=False,
    tals_use_layer_weight=False,
    tals_layer_weight_score="ler_act",
    tals_layer_weight_norm="mean_one",
    tals_layer_weight_clip_min=None,
    tals_layer_weight_clip_max=None,
):
    selected_target_keys = select_drc_targets(
        inject_position=inject_position,
        lora_path_dict=lora_path_dict,
        task_targets=task_targets,
        linear_model=coarse_model,
        target_part=target_part,
        target_modules=target_modules,
        target_layers=target_layers,
    )

    if inject_position != "lora_input":
        raise ValueError(
            "KnOTS_TIES_TALS_DRC currently supports only drc_inject_position='lora_input', "
            "because TALS uses input-side singular vectors of LoRA target-module ΔW."
        )

    all_lora_deltas = build_all_task_lora_deltas(
        task_targets=task_targets,
        lora_path_dict=lora_path_dict,
        target_keys=selected_target_keys,
        rank=rank,
        lora_alpha_list=lora_alpha_list,
    )

    coarse_delta_w_dict = build_dense_coarse_delta_w_dict(
        model_name=model_name,
        coarse_model=coarse_model,
        target_keys=selected_target_keys,
    )

    all_task_directions = {}
    direction_stats = {}

    for task_name in task_targets:
        print(f"\n[KnOTS_TIES_TALS_DRC] Building task-specific TALS-LER direction for task={task_name}")

        input_ids, attention_mask = get_task_samples(
            model_name=model_name,
            tokenizer=tokenizer,
            max_length=max_length,
            task_name=task_name,
            samples_num=samples_per_task,
            select_long=select_long,
            seed=seed,
            shuffle=shuffle,
            if_balance=if_balance,
        )

        # Single-task LoRA dense model features.
        print(f"[KnOTS_TIES_TALS_DRC] Collect single-LoRA features for {task_name}...")
        single_model = load_single_lora_dense_model(
            model_name=model_name,
            lora_path=lora_path_dict[task_name],
            rank=rank,
        ).to("cuda")
        single_features = collect_features_by_position(
            model=single_model,
            inject_position=inject_position,
            target_keys=selected_target_keys,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        del single_model
        torch.cuda.empty_cache()
        gc.collect()

        # Coarse KnOTS-TIES merged model features.
        print(f"[KnOTS_TIES_TALS_DRC] Collect KnOTS-TIES coarse features on {task_name} samples...")
        coarse_features = collect_features_by_position(
            model=coarse_model,
            inject_position=inject_position,
            target_keys=selected_target_keys,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        torch.cuda.empty_cache()
        gc.collect()

        task_direction = {}
        task_stats = {}

        missing_report = {
            "missing_single": [],
            "missing_coarse": [],
            "missing_lora_delta": [],
            "missing_coarse_delta_w": [],
            "tals_filter_failed": [],
            "zero_or_tiny_norm": [],
        }

        for key in selected_target_keys:
            if key not in single_features:
                missing_report["missing_single"].append(key)
                continue
            if key not in coarse_features:
                missing_report["missing_coarse"].append(key)
                continue
            if key not in all_lora_deltas.get(task_name, {}):
                missing_report["missing_lora_delta"].append((task_name, key))
                continue
            if key not in coarse_delta_w_dict:
                missing_report["missing_coarse_delta_w"].append(key)
                continue

            # activation residual: h_single - h_KnOTS-TIES
            activation_residual = single_features[key] - coarse_features[key]

            task_delta_w = all_lora_deltas[task_name][key].float()
            coarse_delta_w = coarse_delta_w_dict[key].float()

            source = str(tals_subspace_source).lower().strip()
            if source == "missing":
                source_delta_w = task_delta_w - coarse_delta_w
            elif source == "single":
                source_delta_w = task_delta_w
            elif source == "random":
                source_delta_w = None
            else:
                raise ValueError(
                    f"Unsupported tals_subspace_source={tals_subspace_source}. "
                    "Please use 'missing', 'single', or 'random'."
                )

            random_seed = seed + stable_int_hash(f"knots_ties|{task_name}|{key}|{tals_rank}")

            tals_delta, tals_stats = apply_tals_filter_to_activation_residual(
                activation_residual=activation_residual,
                source_delta_w=source_delta_w,
                tals_rank=tals_rank,
                tals_gamma=tals_gamma,
                tals_eps=tals_eps,
                tals_weight_norm=tals_weight_norm,
                tals_svd_center=tals_svd_center,
                tals_subspace_source=source,
                random_seed=random_seed,
            )

            if tals_delta is None:
                missing_report["tals_filter_failed"].append((key, tals_stats))
                if tals_fallback_to_base:
                    print(
                        f"[KnOTS_TIES_TALS_DRC][Warn] TALS filter failed at {key}; "
                        "fallback to base DRC direction."
                    )
                    tals_delta = activation_residual.float().cpu()
                    tals_stats = {"fallback_to_base": True, **tals_stats}
                else:
                    continue

            raw_norm = float(torch.norm(tals_delta.float()))

            if normalize:
                delta_normed, _ = normalize_direction(tals_delta, eps=tals_eps)
                if delta_normed is None:
                    missing_report["zero_or_tiny_norm"].append((key, raw_norm))
                    continue

                task_direction[key] = delta_normed.cpu()
                task_stats[key] = {
                    "raw_norm": raw_norm,
                    "used_norm": 1.0,
                    **tals_stats,
                }
            else:
                if raw_norm < tals_eps:
                    missing_report["zero_or_tiny_norm"].append((key, raw_norm))
                    continue

                task_direction[key] = tals_delta.float().cpu()
                task_stats[key] = {
                    "raw_norm": raw_norm,
                    "used_norm": raw_norm,
                    **tals_stats,
                }

        print(f"[KnOTS_TIES_TALS_DRC][Debug] Missing/skip report for task={task_name}:")
        for reason, items in missing_report.items():
            print(f"  - {reason}: {len(items)}")
            for item in items[:10]:
                print(f"      {item}")

        print(
            f"[KnOTS_TIES_TALS_DRC] Task {task_name}: built directions for "
            f"{len(task_direction)}/{len(selected_target_keys)} targets."
        )

        if tals_use_layer_weight and len(task_direction) > 0:
            task_direction, task_stats, lw_summary = apply_ler_layerwise_reweight(
                task_direction=task_direction,
                task_stats=task_stats,
                score_type=tals_layer_weight_score,
                norm_type=tals_layer_weight_norm,
                clip_min=tals_layer_weight_clip_min,
                clip_max=tals_layer_weight_clip_max,
                eps=tals_eps,
            )
            print(f"[KnOTS_TIES_TALS_DRC][LayerWeight] task={task_name}, summary={lw_summary}")
            direction_stats[f"{task_name}__layer_weight_summary"] = lw_summary

        all_task_directions[task_name] = task_direction
        direction_stats[task_name] = task_stats

    return all_task_directions, direction_stats, selected_target_keys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="GLUE_t5")
    parser.add_argument("--config", type=str, default="config/methods-config/iteris-config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    seed = config_data.get("seed", 42)
    set_seed(seed)

    if args.task_type not in config_data:
        raise ValueError(f"Cannot find task_type={args.task_type} in config.")

    task_cfg = config_data[args.task_type]
    model_name = task_cfg["model_name"]
    task_targets = task_cfg["task_targets"]
    rank = int(task_cfg.get("rank", 8))

    tokenizer = (
        AutoTokenizer.from_pretrained(model_name)
        if "blip" not in model_name
        else AutoProcessor.from_pretrained(model_name)
    )

    lora_path_dict = get_loras_path(args.task_type, model_name)
    lora_path = [lora_path_dict[task] for task in task_targets]

    pair_name = "_".join(task_targets)

    knots_ties_tals_load_coarse_from_dir = bool(task_cfg.get("knots_ties_tals_load_coarse_from_dir", False))
    knots_ties_tals_coarse_model_dir = task_cfg.get(
        "knots_ties_tals_coarse_model_dir",
        f"merged_model/KnOTS-TIES_{pair_name}",
    )

    if knots_ties_tals_load_coarse_from_dir:
        print("[KnOTS_TIES_TALS_DRC] knots_ties_tals_load_coarse_from_dir = True")
        print(f"[KnOTS_TIES_TALS_DRC] Expected best coarse model dir = {knots_ties_tals_coarse_model_dir}")

    method_name = task_cfg.get(
        "knots_ties_tals_method_name",
        task_cfg.get("tals_method_name", task_cfg.get("drc_method_name", "KnOTS_TIES_TALS_DRC")),
    )
    if knots_ties_tals_load_coarse_from_dir:
        method_name = f"{method_name}_loaded_{os.path.basename(str(knots_ties_tals_coarse_model_dir)).replace('/', '_')}"

    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{method_name}_{pair_name}"

    # ------------------------------
    # KnOTS-TIES coarse merge config
    # ------------------------------
    raw_topK = task_cfg.get("knots_ties_topK", task_cfg.get("topK", 20))
    sign_resolve_mode = task_cfg.get(
        "knots_ties_sign_resolve_mode",
        task_cfg.get("sign_resolve_mode", "sum_of_values"),
    )
    raw_scaling_coeffs = task_cfg.get("knots_ties_scaling_coeffs", task_cfg.get("scaling_coeffs", 0.5))
    merging_type = task_cfg.get("knots_ties_merging_type", task_cfg.get("merging_type", "mean"))
    concat_across_output = bool(task_cfg.get("knots_concat_across_output", True))
    svd_tol = float(task_cfg.get("knots_svd_tol", 1e-5))
    knots_ties_do_search = bool(task_cfg.get("knots_ties_do_search", False))
    knots_ties_search_strategy = str(task_cfg.get("knots_ties_search_strategy", "linear")).lower().strip()
    knots_ties_search_early_stop = bool(task_cfg.get("knots_ties_search_early_stop", False))
    knots_verbose_svd = bool(task_cfg.get("knots_verbose_svd", False))

    topK = None
    scaling_coeffs = None
    adapter_cfg = {}

    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, "pair_merge_results.csv")
    registry_csv = os.path.join(results_dir, "experiment_registry.csv")
    alpha_search_csv = os.path.join(results_dir, "drc_alpha_search_results.csv")

    results_header = [
        "experiment_id", "method", "pair_name", "task_a", "task_b",
        "evaluated_task", "primary_metric_name", "primary_metric_value",
        "normalized_metric", "eval_accuracy", "eval_mcc", "eval_f1",
        "eval_loss", "eval_runtime", "eval_samples_per_second",
        "eval_steps_per_second", "eval_peak_vram_mb",
        "split", "merged_model_dir", "log_file", "error"
    ]

    registry_header = [
        "experiment_id", "experiment_type", "method", "model_name", "pair_name",
        "task_targets", "linear_weights", "rank", "lora_alpha",
        "fusion_total_time_sec", "fusion_iter_time_avg_sec",
        "fusion_iter_time_max_sec", "fusion_peak_vram_avg_mb",
        "fusion_peak_vram_max_mb", "pair_avg_normalized_metric",
        "status", "error"
    ]

    ensure_csv_header(results_csv, results_header)
    ensure_csv_header(registry_csv, registry_header)

    log_file = os.environ.get("LOG_FILE", "")

    try:
        if knots_ties_tals_load_coarse_from_dir:
            topK = "loaded"
            scaling_coeffs = "loaded"
            merged_model_dir = str(knots_ties_tals_coarse_model_dir)
            coarse_model = load_required_coarse_model_from_dir(
                merged_model_dir,
                model_label="KnOTS-TIES best",
            ).to("cuda")
            fusion_stats = make_loaded_fusion_stats()
            adapter_cfg = {
                "r": rank,
                "lora_alpha": task_cfg.get("lora_alpha", ""),
            }
            print("[KnOTS_TIES_TALS_DRC] Loaded saved best coarse model. No KnOTS-TIES merge/search is run in this script.")
        else:
            parsed_topK = parse_scalar_or_candidates(raw_topK)
            parsed_scaling_coeffs = parse_scalar_or_candidates(raw_scaling_coeffs)
            explicit_topK_candidates = task_cfg.get(
                "knots_ties_topK_candidates",
                [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            )
            explicit_scaling_candidates = task_cfg.get(
                "knots_ties_scaling_coeffs_candidates",
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )

            if knots_ties_do_search:
                topK_candidates = (
                    parsed_topK if isinstance(parsed_topK, list)
                    else [float(x) for x in explicit_topK_candidates]
                )
                scaling_candidates = (
                    parsed_scaling_coeffs if isinstance(parsed_scaling_coeffs, list)
                    else [float(x) for x in explicit_scaling_candidates]
                )
                search_result = search_best_knots_ties_config(
                    lora_path=lora_path,
                    model_name=model_name,
                    task_targets=task_targets,
                    seed=seed,
                    tokenizer=tokenizer,
                    max_length=task_cfg["max_length"],
                    per_device_eval_batch_size=task_cfg.get("per_device_eval_batch_size", 8),
                    concat_across_output=concat_across_output,
                    svd_tol=svd_tol,
                    default_params={
                        "scaling_coeffs": float(
                            task_cfg.get("knots_ties_scaling_coeffs", task_cfg.get("scaling_coeffs", 0.5))
                        ),
                        "topK": float(task_cfg.get("knots_ties_topK", task_cfg.get("topK", 20))),
                    },
                    search_config={
                        "scaling_coeffs": [float(x) for x in scaling_candidates],
                        "topK": [int(float(x)) for x in topK_candidates],
                    },
                    order_of_processing_params=["scaling_coeffs", "topK"],
                    sign_resolve_mode=sign_resolve_mode,
                    merging_type=merging_type,
                    search_strategy=knots_ties_search_strategy,
                    early_stop=knots_ties_search_early_stop,
                    verbose_svd=knots_verbose_svd,
                )
                scaling_coeffs = float(search_result["scaling_coeffs"])
                topK = int(search_result["topK"])
            else:
                if isinstance(parsed_topK, list):
                    raise ValueError(
                        "knots_ties_topK 当前是多个候选值。"
                        "若要搜索，请设置 knots_ties_do_search: true；"
                        "若不搜索，请把 knots_ties_topK 改成单个数值。"
                    )
                if isinstance(parsed_scaling_coeffs, list):
                    raise ValueError(
                        "knots_ties_scaling_coeffs 当前是多个候选值。"
                        "若要搜索，请设置 knots_ties_do_search: true；"
                        "若不搜索，请把 knots_ties_scaling_coeffs 改成单个数值。"
                    )
                topK = int(parsed_topK)
                scaling_coeffs = float(parsed_scaling_coeffs)

            # Build KnOTS-TIES coarse merged model.
            coarse_model, fusion_stats, adapter_cfg = merge_knots_ties_param(
                lora_path=lora_path,
                model_name=model_name,
                task_targets=task_targets,
                seed=seed,
                topK=topK,
                sign_resolve_mode=sign_resolve_mode,
                scaling_coeffs=scaling_coeffs,
                merging_type=merging_type,
                concat_across_output=concat_across_output,
                svd_tol=svd_tol,
                verbose_svd=knots_verbose_svd,
            )
            coarse_model = coarse_model.to("cuda")

            merged_model_dir = task_cfg.get(
                "knots_ties_tals_merged_model_dir",
                f"merged_model/{method_name}_{pair_name}",
            )
            if task_cfg.get("save", 0):
                os.makedirs(merged_model_dir, exist_ok=True)
                coarse_model.save_pretrained(merged_model_dir)
                if hasattr(tokenizer, "save_pretrained"):
                    tokenizer.save_pretrained(merged_model_dir)
                print(f"[KnOTS_TIES_TALS_DRC] KnOTS-TIES coarse merged model saved to {merged_model_dir}")

        # ------------------------------
        # TALS-LER config
        # ------------------------------
        drc_inject_position = task_cfg.get("drc_inject_position", "lora_input")
        drc_alpha = float(task_cfg.get("drc_alpha", 0.03))
        drc_alpha_search = bool(task_cfg.get("drc_alpha_search", False))
        drc_alpha_candidates = parse_float_list(
            task_cfg.get("drc_alpha_candidates", None),
            default=[0.0, 0.03, 0.1, 0.2, 0.3, 0.5],
        )
        drc_samples_per_task = int(task_cfg.get("drc_samples_per_task", task_cfg.get("samples_num", 50)))
        drc_select_long = task_cfg.get("drc_select_long", task_cfg.get("select_long", False))
        drc_shuffle = task_cfg.get("drc_shuffle", task_cfg.get("shuffle", True))
        drc_if_balance = task_cfg.get("drc_if_balance", task_cfg.get("if_balance", True))

        drc_target_part = task_cfg.get("drc_target_part", "encoder")
        drc_target_modules = task_cfg.get("drc_target_modules", ["q", "v"])
        drc_target_layers = task_cfg.get("drc_target_layers", None)
        drc_normalize_direction = bool(task_cfg.get("drc_normalize_direction", True))
        drc_use_hidden_norm_scale = bool(task_cfg.get("drc_use_hidden_norm_scale", False))
        drc_rebuild_cache = bool(task_cfg.get("drc_rebuild_cache", True))

        tals_rank = int(task_cfg.get("tals_rank", task_cfg.get("tals_subspace_rank", 8)))
        tals_gamma = float(task_cfg.get("tals_gamma", 0.5))
        tals_eps = float(task_cfg.get("tals_eps", 1e-6))
        tals_weight_norm = task_cfg.get("tals_weight_norm", "mean")
        tals_svd_center = bool(task_cfg.get("tals_svd_center", False))
        tals_subspace_source = str(task_cfg.get("tals_subspace_source", "missing")).lower().strip()
        if tals_subspace_source not in ["missing", "single", "random"]:
            raise ValueError(
                f"Unsupported tals_subspace_source={tals_subspace_source}. "
                "Please use 'missing', 'single', or 'random'."
            )
        tals_fallback_to_base = bool(task_cfg.get("tals_fallback_to_base", False))

        tals_use_layer_weight = bool(task_cfg.get("tals_use_layer_weight", True))
        tals_layer_weight_score = task_cfg.get("tals_layer_weight_score", "ler_act")
        tals_layer_weight_norm = task_cfg.get("tals_layer_weight_norm", "mean_one")
        tals_layer_weight_clip_min = parse_optional_float(
            task_cfg.get("tals_layer_weight_clip_min", None),
            default=None,
        )
        tals_layer_weight_clip_max = parse_optional_float(
            task_cfg.get("tals_layer_weight_clip_max", None),
            default=None,
        )

        cache_dir = task_cfg.get("drc_cache_dir", "direction_cache_knots_ties_tals")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = make_knots_ties_tals_cache_path(
            cache_dir=cache_dir,
            method_name=method_name,
            pair_name=pair_name,
            inject_position=drc_inject_position,
            target_layers=drc_target_layers,
            target_modules=drc_target_modules,
            topK=topK,
            scaling_coeffs=scaling_coeffs,
            sign_resolve_mode=sign_resolve_mode,
            merging_type=merging_type,
            concat_across_output=concat_across_output,
            svd_tol=svd_tol,
            knots_ties_do_search=knots_ties_do_search,
            knots_ties_search_strategy=knots_ties_search_strategy,
            tals_subspace_source=tals_subspace_source,
            tals_rank=tals_rank,
            tals_gamma=tals_gamma,
            tals_weight_norm=tals_weight_norm,
            tals_svd_center=tals_svd_center,
            tals_use_layer_weight=tals_use_layer_weight,
            tals_layer_weight_score=tals_layer_weight_score,
            tals_layer_weight_norm=tals_layer_weight_norm,
        )

        print(f"[KnOTS_TIES_TALS_DRC] drc_inject_position = {drc_inject_position}")
        print(f"[KnOTS_TIES_TALS_DRC] drc_alpha = {drc_alpha}")
        print(f"[KnOTS_TIES_TALS_DRC] drc_alpha_search = {drc_alpha_search}")
        print(f"[KnOTS_TIES_TALS_DRC] drc_alpha_candidates = {drc_alpha_candidates}")
        print(f"[KnOTS_TIES_TALS_DRC] drc_samples_per_task = {drc_samples_per_task}")
        print(f"[KnOTS_TIES_TALS_DRC] drc_target_part = {drc_target_part}")
        print(f"[KnOTS_TIES_TALS_DRC] drc_target_modules = {drc_target_modules}")
        print(f"[KnOTS_TIES_TALS_DRC] drc_target_layers = {drc_target_layers}")
        print(f"[KnOTS_TIES_TALS_DRC] tals_subspace_source = {tals_subspace_source}")
        print(f"[KnOTS_TIES_TALS_DRC] tals_rank = {tals_rank}")
        print(f"[KnOTS_TIES_TALS_DRC] tals_gamma = {tals_gamma}")
        print(f"[KnOTS_TIES_TALS_DRC] tals_use_layer_weight = {tals_use_layer_weight}")
        print(f"[KnOTS_TIES_TALS_DRC] cache_path = {cache_path}")

        start_time = datetime.now()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if (not drc_rebuild_cache) and os.path.exists(cache_path):
            print(f"[KnOTS_TIES_TALS_DRC] Load TALS direction cache from: {cache_path}")
            cache = torch.load(cache_path, map_location="cpu")
            drc_directions = cache["directions"]
            direction_stats = cache.get("direction_stats", {})
            selected_target_keys = cache.get("selected_target_keys", [])
        else:
            drc_directions, direction_stats, selected_target_keys = build_task_specific_knots_ties_tals_directions(
                model_name=model_name,
                tokenizer=tokenizer,
                task_targets=task_targets,
                lora_path_dict=lora_path_dict,
                coarse_model=coarse_model,
                rank=rank,
                max_length=task_cfg["max_length"],
                seed=seed,
                samples_per_task=drc_samples_per_task,
                select_long=drc_select_long,
                shuffle=drc_shuffle,
                if_balance=drc_if_balance,
                target_part=drc_target_part,
                target_modules=drc_target_modules,
                target_layers=drc_target_layers,
                normalize=drc_normalize_direction,
                inject_position=drc_inject_position,
                lora_alpha_list=task_cfg.get("lora_alpha", []),
                tals_rank=tals_rank,
                tals_gamma=tals_gamma,
                tals_eps=tals_eps,
                tals_weight_norm=tals_weight_norm,
                tals_svd_center=tals_svd_center,
                tals_subspace_source=tals_subspace_source,
                tals_fallback_to_base=tals_fallback_to_base,
                tals_use_layer_weight=tals_use_layer_weight,
                tals_layer_weight_score=tals_layer_weight_score,
                tals_layer_weight_norm=tals_layer_weight_norm,
                tals_layer_weight_clip_min=tals_layer_weight_clip_min,
                tals_layer_weight_clip_max=tals_layer_weight_clip_max,
            )

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(
                {
                    "pair_name": pair_name,
                    "method": method_name,
                    "base_method": "KnOTS-TIES",
                    "directions": drc_directions,
                    "direction_stats": direction_stats,
                    "selected_target_keys": selected_target_keys,
                    "drc_inject_position": drc_inject_position,
                    "drc_alpha": drc_alpha,
                    "drc_samples_per_task": drc_samples_per_task,
                    "drc_target_part": drc_target_part,
                    "drc_target_modules": normalize_target_modules(drc_target_modules),
                    "drc_target_layers": drc_target_layers,
                    "drc_normalize_direction": drc_normalize_direction,
                    "knots_ties_topK": topK,
                    "knots_ties_scaling_coeffs": scaling_coeffs,
                    "knots_ties_sign_resolve_mode": sign_resolve_mode,
                    "knots_ties_merging_type": merging_type,
                    "knots_concat_across_output": concat_across_output,
                    "knots_svd_tol": svd_tol,
                    "tals_subspace_source": tals_subspace_source,
                    "tals_rank": tals_rank,
                    "tals_gamma": tals_gamma,
                    "tals_weight_norm": tals_weight_norm,
                    "tals_svd_center": tals_svd_center,
                    "tals_use_layer_weight": tals_use_layer_weight,
                    "tals_layer_weight_score": tals_layer_weight_score,
                    "tals_layer_weight_norm": tals_layer_weight_norm,
                },
                cache_path,
            )
            print(f"[KnOTS_TIES_TALS_DRC] Saved TALS direction cache to: {cache_path}")

        tals_build_time = round((datetime.now() - start_time).total_seconds(), 4)
        tals_peak_vram_mb = (
            round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
            if torch.cuda.is_available()
            else 0.0
        )

        # Include TALS direction-building time in registry as the total post-merge cost.
        fusion_stats = {
            "fusion_iter_time_avg_sec": float(fusion_stats.get("fusion_iter_time_avg_sec", 0.0)) + tals_build_time,
            "fusion_iter_time_max_sec": float(fusion_stats.get("fusion_iter_time_max_sec", 0.0)) + tals_build_time,
            "fusion_peak_vram_avg_mb": max(float(fusion_stats.get("fusion_peak_vram_avg_mb", 0.0)), tals_peak_vram_mb),
            "fusion_peak_vram_max_mb": max(float(fusion_stats.get("fusion_peak_vram_max_mb", 0.0)), tals_peak_vram_mb),
        }

        normalized_metrics = []
        selected_alpha_by_task = {}

        for task_name in task_targets:
            print(f"\n[Eval] Evaluating KnOTS-TIES + TALS-LER on {task_name}...")

            if task_name not in drc_directions:
                raise ValueError(f"Cannot find TALS direction for task: {task_name}")

            candidate_alphas = drc_alpha_candidates if drc_alpha_search else [drc_alpha]
            best_record = None

            for alpha in candidate_alphas:
                print(f"[KnOTS_TIES_TALS_DRC][AlphaSearch] task={task_name}, alpha={alpha}")

                handles = register_drc_hooks_by_position(
                    model=coarse_model,
                    directions=drc_directions[task_name],
                    inject_position=drc_inject_position,
                    alpha=float(alpha),
                    use_hidden_norm_scale=drc_use_hidden_norm_scale,
                )

                eval_result = eval_iteris_model(
                    model=coarse_model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    task_name=task_name,
                    max_length=task_cfg["max_length"],
                    per_device_eval_batch_size=task_cfg.get("per_device_eval_batch_size", 8),
                )

                remove_hooks(handles)
                torch.cuda.empty_cache()
                gc.collect()

                eval_accuracy = eval_result.get("eval_accuracy", "")
                eval_mcc = eval_result.get("eval_MCC", "")
                eval_f1 = eval_result.get("eval_f1-score", "")
                eval_loss = eval_result.get("eval_loss", "")
                eval_runtime = eval_result.get("eval_runtime", "")
                eval_sps = eval_result.get("eval_samples_per_second", "")
                eval_stepsps = eval_result.get("eval_steps_per_second", "")
                eval_peak_vram_mb = eval_result.get("eval_peak_vram_mb", "")

                primary_metric_name, primary_metric_value, normalized_metric = get_primary_metric(
                    task_name=task_name,
                    eval_result=eval_result,
                )

                append_alpha_search_row(
                    alpha_search_csv,
                    [
                        experiment_id, method_name, pair_name, task_name,
                        float(alpha), primary_metric_name, primary_metric_value,
                        normalized_metric, eval_accuracy, eval_mcc, eval_f1,
                        eval_loss, eval_runtime, eval_peak_vram_mb,
                    ],
                )

                current_record = {
                    "alpha": float(alpha),
                    "primary_metric_name": primary_metric_name,
                    "primary_metric_value": primary_metric_value,
                    "normalized_metric": normalized_metric,
                    "eval_accuracy": eval_accuracy,
                    "eval_mcc": eval_mcc,
                    "eval_f1": eval_f1,
                    "eval_loss": eval_loss,
                    "eval_runtime": eval_runtime,
                    "eval_sps": eval_sps,
                    "eval_stepsps": eval_stepsps,
                    "eval_peak_vram_mb": eval_peak_vram_mb,
                }

                if best_record is None or normalized_metric > best_record["normalized_metric"]:
                    best_record = current_record

            selected_alpha_by_task[task_name] = best_record["alpha"]
            normalized_metrics.append(best_record["normalized_metric"])

            print(
                f"[KnOTS_TIES_TALS_DRC][AlphaSearch][Best] task={task_name}, "
                f"alpha={best_record['alpha']}, normalized_metric={best_record['normalized_metric']}"
            )

            append_csv_row(
                results_csv,
                [
                    experiment_id, method_name, pair_name, task_targets[0], task_targets[1],
                    task_name,
                    best_record["primary_metric_name"],
                    best_record["primary_metric_value"],
                    best_record["normalized_metric"],
                    best_record["eval_accuracy"],
                    best_record["eval_mcc"],
                    best_record["eval_f1"],
                    best_record["eval_loss"],
                    best_record["eval_runtime"],
                    best_record["eval_sps"],
                    best_record["eval_stepsps"],
                    best_record["eval_peak_vram_mb"],
                    "validation",
                    merged_model_dir,
                    log_file,
                    "",
                ],
            )

        pair_avg_normalized_metric = float(np.mean(normalized_metrics))
        selected_alpha_str = "|".join(
            [f"{task}:{selected_alpha_by_task.get(task, drc_alpha)}" for task in task_targets]
        )

        cfg_str = (
            f"base=KnOTS-TIES|"
            f"topK={topK}|sign_resolve_mode={sign_resolve_mode}|"
            f"scaling_coeffs={scaling_coeffs}|merging_type={merging_type}|"
            f"concat_across_output={concat_across_output}|svd_tol={svd_tol}|"
            f"knots_search={knots_ties_do_search}|knots_search_strategy={knots_ties_search_strategy}|"
            f"knots_early_stop={knots_ties_search_early_stop}|"
            f"inject_position={drc_inject_position}|"
            f"alpha_search={drc_alpha_search}|"
            f"alpha_candidates={drc_alpha_candidates}|"
            f"selected_alpha_by_task={selected_alpha_str}|"
            f"samples_per_task={drc_samples_per_task}|"
            f"target_part={drc_target_part}|"
            f"target_modules={normalize_target_modules(drc_target_modules)}|"
            f"target_layers={drc_target_layers}|"
            f"normalize={drc_normalize_direction}|"
            f"use_hidden_norm_scale={drc_use_hidden_norm_scale}|"
            f"source={tals_subspace_source}|"
            f"tals_rank={tals_rank}|"
            f"tals_gamma={tals_gamma}|"
            f"tals_weight_norm={tals_weight_norm}|"
            f"tals_svd_center={tals_svd_center}|"
            f"tals_use_layer_weight={tals_use_layer_weight}|"
            f"tals_layer_weight_score={tals_layer_weight_score}|"
            f"tals_layer_weight_norm={tals_layer_weight_norm}"
        )

        append_csv_row(
            registry_csv,
            [
                experiment_id, "GLUE_pairwise", method_name, model_name, pair_name,
                "|".join(task_targets), cfg_str,
                adapter_cfg.get("r", task_cfg.get("rank", "")),
                str(adapter_cfg.get("lora_alpha", task_cfg.get("lora_alpha", ""))),
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_max_sec"],
                fusion_stats["fusion_peak_vram_avg_mb"],
                fusion_stats["fusion_peak_vram_max_mb"],
                pair_avg_normalized_metric,
                "success", "",
            ],
        )

        print(f"\n[Done] KnOTS_TIES_TALS_DRC finished for pair: {pair_name}")
        print(f"[Done] selected_alpha_by_task = {selected_alpha_by_task}")
        print(f"[Done] pair_avg_normalized_metric = {pair_avg_normalized_metric}")

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)

        cfg_str = (
            f"base=KnOTS-TIES|"
            f"topK={topK}|sign_resolve_mode={sign_resolve_mode}|"
            f"scaling_coeffs={scaling_coeffs}|merging_type={merging_type}|"
            f"concat_across_output={concat_across_output}|svd_tol={svd_tol}|"
            f"knots_search={knots_ties_do_search}|knots_search_strategy={knots_ties_search_strategy}|"
            f"knots_early_stop={knots_ties_search_early_stop}"
        )

        append_csv_row(
            registry_csv,
            [
                experiment_id, "GLUE_pairwise", method_name, model_name, pair_name,
                "|".join(task_targets), cfg_str,
                task_cfg.get("rank", ""),
                str(task_cfg.get("lora_alpha", "")),
                "", "", "", "", "", "", "failed", error_msg,
            ],
        )
        raise e


if __name__ == "__main__":
    main()
