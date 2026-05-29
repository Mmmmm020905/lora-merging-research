import os
import gc
import csv
import yaml
import time
import torch
import random
import argparse
import traceback
import numpy as np
import fcntl
from copy import deepcopy
from datetime import datetime
from collections import OrderedDict

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    BlipForConditionalGeneration,
)

from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)
from peft.utils import get_peft_model_state_dict
from safetensors import safe_open

from eval_model import eval_iteris_model


GLUE_task_name = [
    "mnli", "rte", "cola", "sst2", "qqp", "qnli", "mrpc", "wnli",
]



def is_blip_model(model_name):
    return "blip" in str(model_name).lower()


def get_loras_path(task_type, model_name, lora_root=None):
    """
    Return LoRA adapter directories.

    This function is intentionally local to DARE.py so VLM support does not
    depend on the current state of KnOTS.py. It preserves GLUE/T5 behavior and
    adds TASKS_blip_base support.
    """
    lora_path_dict = {}

    if lora_root is None:
        if task_type == "TASKS_blip_base":
            lora_root = "loras/SENTICAP-lora-blip"
        else:
            lora_root = "best_LoRA"

    model_name_l = str(model_name).lower()

    if "t5" in model_name_l and task_type == "GLUE_t5":
        lora_path_dict["cola"] = f"{lora_root}/T5-COLA-LoRA"
        lora_path_dict["sst2"] = f"{lora_root}/T5-SST2-LoRA"
        lora_path_dict["rte"]  = f"{lora_root}/T5-RTE-LoRA"
        lora_path_dict["qnli"] = f"{lora_root}/T5-QNLI-LoRA"
        lora_path_dict["qqp"]  = f"{lora_root}/T5-QQP-LoRA"
        lora_path_dict["mrpc"] = f"{lora_root}/T5-MRPC-LoRA"
        lora_path_dict["mnli"] = f"{lora_root}/T5-MNLI-LoRA"
        lora_path_dict["wnli"] = f"{lora_root}/T5-WNLI-LoRA"

    if task_type == "TASKS_blip_base":
        lora_path_dict["positive"] = f"{lora_root}/positive"
        lora_path_dict["negative"] = f"{lora_root}/negative"
        # Keep these optional mappings for future FlickrStyle10k experiments.
        lora_path_dict["roman"] = "loras/FlickrStyle10k-lora-blip/roman"
        lora_path_dict["humor"] = "loras/FlickrStyle10k-lora-blip/humor"

    return lora_path_dict


def construct_base_model(model_name):
    model_name_l = str(model_name).lower()
    if "t5" in model_name_l:
        return T5ForConditionalGeneration.from_pretrained(model_name)
    if "bart" in model_name_l:
        return BartForConditionalGeneration.from_pretrained(model_name)
    if "blip" in model_name_l:
        return BlipForConditionalGeneration.from_pretrained(model_name)
    raise ValueError(f"[DARE] Unsupported model_name: {model_name}")


def load_adapter_config(lora_dir):
    config_path = os.path.join(lora_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[DARE] adapter_config.json not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_lora_config_from_adapter_cfg(adapter_cfg):
    task_type_value = adapter_cfg.get("task_type", "SEQ_2_SEQ_LM")
    if isinstance(task_type_value, str):
        task_type = getattr(TaskType, task_type_value)
    else:
        task_type = TaskType.SEQ_2_SEQ_LM

    kwargs = dict(
        task_type=task_type,
        r=adapter_cfg["r"],
        lora_alpha=adapter_cfg["lora_alpha"],
        lora_dropout=adapter_cfg.get("lora_dropout", 0.0),
        target_modules=adapter_cfg.get("target_modules", None),
        bias=adapter_cfg.get("bias", "none"),
        inference_mode=False,
    )

    optional_fields = [
        "fan_in_fan_out",
        "modules_to_save",
        "layers_to_transform",
        "layers_pattern",
        "rank_pattern",
        "alpha_pattern",
        "init_lora_weights",
        "use_rslora",
        "use_dora",
    ]
    for field in optional_fields:
        if field in adapter_cfg and adapter_cfg[field] is not None:
            kwargs[field] = adapter_cfg[field]

    return LoraConfig(**kwargs)


def construct_fresh_peft_model(model_name, adapter_cfg, seed, device="cpu"):
    set_seed(seed)
    base_model = construct_base_model(model_name)
    lora_config = build_lora_config_from_adapter_cfg(adapter_cfg)
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.to(device)
    return peft_model


def ordered_ft_state_dict(peft_model):
    return OrderedDict(sorted(get_peft_model_state_dict(peft_model).items()))


def load_adapter_state_dict_from_safetensors(lora_dir, device="cpu"):
    adapter_file = os.path.join(lora_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"[DARE] adapter_model.safetensors not found: {adapter_file}")
    tensor_dict = safe_open(adapter_file, framework="pt")
    state_dict = OrderedDict()
    for key in sorted(tensor_dict.keys()):
        state_dict[key] = tensor_dict.get_tensor(key).to(device)
    return state_dict


def check_state_dict_keys_match(state_dicts):
    ref_keys = list(state_dicts[0].keys())
    for i, sd in enumerate(state_dicts[1:], start=1):
        cur_keys = list(sd.keys())
        if cur_keys != ref_keys:
            diff = set(ref_keys).symmetric_difference(set(cur_keys))
            raise ValueError(
                f"[DARE] state_dict keys mismatch between model 0 and model {i}. "
                f"Different keys: {sorted(diff)}"
            )


def normalize_lora_layer_name(base_name):
    prefixes = [
        "base_model.model.",
        "base_model.",
        "model.",
    ]
    for prefix in prefixes:
        if base_name.startswith(prefix):
            return base_name[len(prefix):]
    return base_name


def extract_lora_base_name(full_key):
    suffixes = [
        ".lora_A.default.weight",
        ".lora_B.default.weight",
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_A.default",
        ".lora_B.default",
        ".lora_A",
        ".lora_B",
    ]
    for suffix in suffixes:
        if full_key.endswith(suffix):
            return normalize_lora_layer_name(full_key[: -len(suffix)])
    return None


def get_lora_scale_for_layer(layer_name, adapter_cfg):
    """
    PEFT effective LoRA update is:
        ΔW = scaling * (B @ A), scaling = lora_alpha / r

    Some adapters may include alpha_pattern / rank_pattern. If present, follow
    PEFT-style per-layer override; otherwise use the global r/lora_alpha.
    """
    if adapter_cfg is None:
        return 1.0

    r = adapter_cfg.get("r", adapter_cfg.get("rank", 1))
    alpha = adapter_cfg.get("lora_alpha", adapter_cfg.get("alpha", r))

    rank_pattern = adapter_cfg.get("rank_pattern", None) or {}
    alpha_pattern = adapter_cfg.get("alpha_pattern", None) or {}

    # PEFT patterns can be stored with suffix-like keys. Use the longest
    # matching pattern to be robust.
    best_rank_key = None
    for k in rank_pattern.keys():
        if layer_name.endswith(k) or k in layer_name:
            if best_rank_key is None or len(k) > len(best_rank_key):
                best_rank_key = k
    if best_rank_key is not None:
        r = rank_pattern[best_rank_key]

    best_alpha_key = None
    for k in alpha_pattern.keys():
        if layer_name.endswith(k) or k in layer_name:
            if best_alpha_key is None or len(k) > len(best_alpha_key):
                best_alpha_key = k
    if best_alpha_key is not None:
        alpha = alpha_pattern[best_alpha_key]

    return float(alpha) / float(r)


def lora_state_dict_to_delta_matrices(state_dict, device="cpu", adapter_cfg=None):
    """
    Convert PEFT LoRA state dict to equivalent dense ΔW matrices.

    Important fix:
    The previous version used raw B@A. That is not the actual LoRA update.
    The effective PEFT update is (lora_alpha / r) * B@A, optionally with
    per-layer alpha/rank patterns. This function now matches the dense update
    used by Linear.py / get_lora_matrix().
    """
    layer2lora_parameters = {}
    for key, val in state_dict.items():
        if ".lora_A" in key:
            base_name = extract_lora_base_name(key)
            if base_name is None:
                continue
            layer2lora_parameters.setdefault(base_name, {})["A"] = val.to(device)
        elif ".lora_B" in key:
            base_name = extract_lora_base_name(key)
            if base_name is None:
                continue
            layer2lora_parameters.setdefault(base_name, {})["B"] = val.to(device)

    task_parameters = OrderedDict()
    fan_in_fan_out = bool(adapter_cfg.get("fan_in_fan_out", False)) if adapter_cfg is not None else False

    for name, key2val in sorted(layer2lora_parameters.items()):
        if "A" not in key2val or "B" not in key2val:
            raise ValueError(f"[DARE] Incomplete LoRA pair for layer: {name}")

        scale = get_lora_scale_for_layer(name, adapter_cfg)
        delta = (key2val["B"] @ key2val["A"]) * scale

        # For Conv1D-style layers PEFT may store fan_in_fan_out=True. Linear BLIP/T5
        # normally does not need this, but keeping it here makes the implementation
        # faithful to PEFT.
        if fan_in_fan_out:
            delta = delta.T

        task_parameters[name] = delta.to(torch.float32)

    return task_parameters


def get_task_directions(ptm_params, ftms_params):
    finetuned_directions = []
    for ftm_params in ftms_params:
        direction_sd = OrderedDict()
        for key, finetuned_val in ftm_params.items():
            if key not in ptm_params:
                ptm_val = torch.zeros_like(finetuned_val)
            else:
                ptm_val = ptm_params[key]
            direction_sd[key] = finetuned_val - ptm_val
        finetuned_directions.append(OrderedDict(sorted(direction_sd.items())))
    return finetuned_directions


def add_direction_to_base_model(model, direction_sd, scaling_coeff=1.0):
    updated = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.endswith(".weight"):
                continue
            key = name[:-7]
            if key in direction_sd:
                delta = direction_sd[key].to(param.device).to(param.dtype)
                param.copy_(param + scaling_coeff * delta)
                updated += 1
    return updated


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_csv_header(csv_path, header):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            file_is_empty = (f.tell() == 0)
            if file_is_empty:
                writer = csv.writer(f)
                writer.writerow(header)
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_csv_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_dare_search_row(csv_path, row):
    """
    记录 DARE 内部超参数搜索明细。
    注意：该文件只用于分析，不参与 pair_merge_results 的正式汇总。
    """
    if csv_path is None:
        return

    header = [
        "search_id", "pair_name", "merge_method", "searched_param", "candidate_value",
        "scaling_coeffs", "dare_pruning_coeffs", "topK", "dare_seed",
        "use_rescale", "avg_normalized_metric",
    ]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            need_header = f.tell() == 0
            writer = csv.writer(f)
            if need_header:
                writer.writerow(header)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def ordered_dict_values_as_vector(state_dict):
    ordered = OrderedDict(sorted(state_dict.items()))
    return torch.nn.utils.parameters_to_vector([value.reshape(-1) for value in ordered.values()])


def directions_to_reps(directions):
    if isinstance(directions, list):
        return [directions_to_reps(direction) for direction in directions]
    return ordered_dict_values_as_vector(directions)


def rep_to_state_dict(vector, state_dict):
    if isinstance(vector, list) or (hasattr(vector, "shape") and len(vector.shape) == 2):
        return [rep_to_state_dict(v, state_dict) for v in vector]
    reference_dict = OrderedDict((k, v.clone()) for k, v in OrderedDict(sorted(state_dict.items())).items())
    torch.nn.utils.vector_to_parameters(vector, reference_dict.values())
    return reference_dict


def randbin(shape, p, generator=None, device="cpu", dtype=torch.float32):
    if p < 0.0 or p >= 1.0:
        raise ValueError(f"[DARE] drop_rate p must satisfy 0 <= p < 1, got {p}")
    keep_prob = 1.0 - p
    return torch.bernoulli(
        torch.full(shape, keep_prob, device=device, dtype=dtype),
        generator=generator,
    )


def apply_dare_to_directions(ftms_params, p, dare_seed=0, use_rescale=True):
    """
    对 task delta 参数执行 DARE:
        delta_hat = mask * delta / (1 - p)
    其中 mask ~ Bernoulli(1 - p)。

    这里使用 seed + task_idx 保证不同任务 mask 不完全相同，同时可复现。
    """
    print("DARE seed:", dare_seed)
    if p < 0.0 or p >= 1.0:
        raise ValueError(f"[DARE] drop_rate p must satisfy 0 <= p < 1, got {p}")

    finetuned_directions = []
    for idx, ftm_params in enumerate(ftms_params):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(dare_seed) + idx)
        direction_sd = OrderedDict()
        for key, finetuned_val in ftm_params.items():
            mask = randbin(
                finetuned_val.shape,
                p,
                generator=generator,
                device=finetuned_val.device,
                dtype=finetuned_val.dtype,
            )
            masked = finetuned_val * mask
            if use_rescale:
                masked = masked / (1.0 - p)
            direction_sd[key] = masked
        finetuned_directions.append(OrderedDict(sorted(direction_sd.items())))
    return finetuned_directions


def topk_values_mask(M, K=0.7):
    if K > 1:
        K /= 100.0

    if K >= 1:
        return M, torch.ones_like(M, dtype=torch.bool), torch.ones_like(M).float().mean(dim=-1)

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
    if majority_sign == 0:
        majority_sign = torch.ones_like(majority_sign)
    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def resolve_sign(tensor, mode="sum_of_values"):
    if mode == "sum_of_signs":
        sign_to_mult = torch.sign(torch.sum(torch.sign(tensor), dim=0))
        sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    elif mode == "sum_of_values":
        sign_to_mult = torch.sign(tensor.sum(dim=0))
        sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    else:
        raise ValueError(
            f"[DARE-TIES] Unknown sign_resolve_mode: {mode}. "
            f"Expected sum_of_values or sum_of_signs."
        )
    return sign_to_mult


def ties_masking(vectors, topK=20, sign_resolve_mode="sum_of_values"):
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
                f"[DARE-TIES] weights length mismatch: got {weights.numel()}, "
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
        raise ValueError(f"[DARE-TIES] Merge method {merge_func} is not defined.")

    return disjoint_aggs


def tv_merging(vectors, weights=None, merging_type="sum"):
    vectors_ = torch.vstack(vectors).clone() if isinstance(vectors, list) else vectors.clone()
    if weights is not None:
        weights = torch.as_tensor(weights, dtype=vectors_.dtype, device=vectors_.device)
        if weights.dim() == 0:
            weights = weights.unsqueeze(0)
        if weights.numel() == 1:
            weights = weights.repeat(vectors_.shape[0])
        if weights.numel() != vectors_.shape[0]:
            raise ValueError(
                f"[DARE] weights length mismatch: got {weights.numel()}, expected {vectors_.shape[0]}"
            )
        vectors_ = vectors_ * weights.view(-1, 1)
    if merging_type == "mean":
        return torch.mean(vectors_, dim=0), None, None
    return torch.sum(vectors_, dim=0), None, None


def merge_dare_param(
    lora_path,
    model_name,
    task_targets,
    drop_rate,
    scaling_coeffs,
    use_rescale=True,
    dare_seed=42,
    merge_method="dare-ties",
    topK=20,
    sign_resolve_mode="sum_of_values",
    merging_type="mean",
):
    assert len(lora_path) == len(task_targets)
    print(f"[DARE] task_targets = {task_targets}")
    print(f"[DARE] lora_path = {lora_path}")
    print(f"[DARE] merge_method = {merge_method}")
    print(f"[DARE] drop_rate = {drop_rate}, use_rescale = {use_rescale}, dare_seed = {dare_seed}")
    print(f"[DARE] scaling_coeffs = {scaling_coeffs}")
    if merge_method == "dare-ties":
        print(f"[DARE] topK = {topK}, sign_resolve_mode = {sign_resolve_mode}, merging_type = {merging_type}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    start_time = time.time()

    adapter_cfg = load_adapter_config(lora_path[0])

    ft_state_dicts = [load_adapter_state_dict_from_safetensors(path, device="cpu") for path in lora_path]
    check_state_dict_keys_match(ft_state_dicts)
    ftms_relevant_params = [
        lora_state_dict_to_delta_matrices(sd, device="cpu", adapter_cfg=adapter_cfg)
        for sd in ft_state_dicts
    ]

    # Important fix:
    # A LoRA adapter's dense ΔW is already a task vector relative to the frozen
    # base model. There is no meaningful "pretrained LoRA delta" to subtract.
    # The previous version constructed a fresh PEFT LoRA and subtracted its
    # dense delta. That is unnecessary and can corrupt the direction if PEFT
    # initialization is not exactly zero. Use the scaled dense LoRA deltas
    # directly as DARE task directions.
    finetuned_directions = [OrderedDict(sorted(sd.items())) for sd in ftms_relevant_params]

    # Debug norm summary for sanity checking against Linear:
    for task_name, direction_sd in zip(task_targets, finetuned_directions):
        total_sq = 0.0
        max_norm = 0.0
        max_key = ""
        for k, v in direction_sd.items():
            n = float(torch.norm(v.float()))
            total_sq += n * n
            if n > max_norm:
                max_norm = n
                max_key = k
        print(
            f"[DARE][DeltaNorm] task={task_name}, "
            f"global_norm={total_sq ** 0.5:.6f}, max_layer_norm={max_norm:.6f}, max_key={max_key}"
        )

    finetuned_directions = apply_dare_to_directions(
        finetuned_directions,
        p=float(drop_rate),
        dare_seed=dare_seed,
        use_rescale=use_rescale,
    )

    representations = directions_to_reps(finetuned_directions)
    scaling_tensor = torch.tensor(
        [scaling_coeffs] * len(representations) if isinstance(scaling_coeffs, (int, float)) else scaling_coeffs,
        dtype=torch.float32,
    )

    if merge_method == "dare-ties":
        masks = ties_masking(
            vectors=representations,
            topK=topK,
            sign_resolve_mode=sign_resolve_mode,
        )
        ftms_reps = torch.vstack(representations).clone()
        masked_reps = ftms_reps * masks
        merged_vector = masked_merge(
            vectors=masked_reps,
            merge_func=merging_type,
            weights=scaling_tensor,
        )
    elif merge_method == "dare":
        merged_vector, _, _ = tv_merging(
            vectors=representations,
            weights=scaling_tensor,
            merging_type="sum",
        )
    else:
        raise ValueError(f"[DARE] Unsupported merge_method: {merge_method}")

    merged_delta_dict = rep_to_state_dict(merged_vector, finetuned_directions[0])
    model = construct_base_model(model_name).to("cuda")
    number_update = add_direction_to_base_model(model, merged_delta_dict, scaling_coeff=1.0)

    if number_update == len(merged_delta_dict):
        print("[DARE] All target modules updated successfully.")
    else:
        print(f"[DARE][Warn] Updated {number_update}/{len(merged_delta_dict)} modules.")

    fusion_time = round(time.time() - start_time, 4)
    fusion_peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2) if torch.cuda.is_available() else 0.0

    fusion_stats = {
        "fusion_iter_time_avg_sec": fusion_time,
        "fusion_iter_time_max_sec": fusion_time,
        "fusion_peak_vram_avg_mb": fusion_peak_vram_mb,
        "fusion_peak_vram_max_mb": fusion_peak_vram_mb,
    }

    print(f"[DARE] Fusion time: {fusion_time} sec")
    print(f"[DARE] Fusion peak VRAM: {fusion_peak_vram_mb} MB")

    return model, fusion_stats


def get_normalized_metric(eval_result, task_name, task_type=None):
    if task_type == "TASKS_blip_base":
        return float(eval_result.get("acc", eval_result.get("style_acc", eval_result.get("eval_accuracy", 0.0))))
    if task_name == "cola":
        return (float(eval_result.get("eval_MCC", 0.0)) + 1.0) / 2.0
    return float(eval_result.get("eval_accuracy", 0.0))


def get_primary_metric_any(eval_result, task_name, task_type=None):
    if task_type == "TASKS_blip_base":
        style_acc = float(eval_result.get("acc", eval_result.get("style_acc", 0.0)))
        return "style_accuracy", style_acc, style_acc

    if task_name == "cola":
        mcc = float(eval_result.get("eval_MCC", 0.0))
        return "matthews_correlation", mcc, (mcc + 1.0) / 2.0

    acc = float(eval_result.get("eval_accuracy", 0.0))
    return "accuracy", acc, acc


def get_vlm_bleu(eval_result, idx):
    bleu = eval_result.get("bleu", ["", "", "", ""])
    try:
        return bleu[idx]
    except Exception:
        return ""


def append_vlm_caption_row(
    csv_path,
    experiment_id,
    method_name,
    pair_name,
    task_targets,
    task_name,
    eval_result,
    merged_model_dir,
    log_file,
):
    append_csv_row(
        csv_path,
        [
            experiment_id,
            method_name,
            pair_name,
            task_targets[0],
            task_targets[1],
            task_name,
            eval_result.get("acc", eval_result.get("style_acc", "")),
            eval_result.get("cider", eval_result.get("CIDEr", "")),
            get_vlm_bleu(eval_result, 0),
            get_vlm_bleu(eval_result, 1),
            get_vlm_bleu(eval_result, 2),
            get_vlm_bleu(eval_result, 3),
            eval_result.get("rougeL", ""),
            eval_result.get("div_1", ""),
            eval_result.get("div_2", ""),
            eval_result.get("vocab_size", ""),
            "validation",
            merged_model_dir,
            log_file,
        ],
    )


def eval_pair_average(model, tokenizer, model_name, task_targets, max_length, per_device_eval_batch_size, task_type=None):
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
        normalized_metric = get_normalized_metric(eval_result, task_name, task_type=task_type)
        normalized_metrics.append(normalized_metric)
        per_task_results[task_name] = eval_result
    return float(np.mean(normalized_metrics)), per_task_results


def search_best_dare_config(
    lora_path,
    model_name,
    task_targets,
    tokenizer,
    max_length,
    per_device_eval_batch_size,
    task_type,
    default_params,
    search_config,
    order_of_processing_params,
    merge_method,
    use_rescale,
    dare_seed,
    sign_resolve_mode,
    merging_type,
    search_early_stop=False,
    search_csv=None,
    search_id="",
    pair_name="",
):
    print(f"[DARE-SEARCH] default_params = {default_params}")
    print(f"[DARE-SEARCH] order_of_processing_params = {order_of_processing_params}")
    print(f"[DARE-SEARCH] search_config = {search_config}")
    print(f"[DARE-SEARCH] search_early_stop = {search_early_stop}")

    best_val_results = {**deepcopy(default_params), "avg_normalized_metric": -1e9}
    running_defaults = deepcopy(default_params)

    for param in order_of_processing_params:
        best_for_param = None
        best_score_for_param = -1e9

        for value in search_config[param]:
            instance_params = deepcopy(running_defaults)
            instance_params[param] = value
            print(f"[DARE-SEARCH] Try params = {instance_params}")

            model = None
            avg_score = None
            try:
                model, _ = merge_dare_param(
                    lora_path=lora_path,
                    model_name=model_name,
                    task_targets=task_targets,
                    drop_rate=float(instance_params["dare_pruning_coeffs"]),
                    scaling_coeffs=float(instance_params["scaling_coeffs"]),
                    use_rescale=use_rescale,
                    dare_seed=dare_seed,
                    merge_method=merge_method,
                    topK=int(instance_params.get("topK", 20)),
                    sign_resolve_mode=sign_resolve_mode,
                    merging_type=merging_type,
                )

                avg_score, _ = eval_pair_average(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    task_targets=task_targets,
                    max_length=max_length,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                    task_type=task_type,
                )
                print(f"[DARE-SEARCH] avg_normalized_metric = {avg_score:.6f}")

                append_dare_search_row(
                    search_csv,
                    [
                        search_id,
                        pair_name,
                        merge_method,
                        param,
                        value,
                        float(instance_params["scaling_coeffs"]),
                        float(instance_params["dare_pruning_coeffs"]),
                        int(instance_params.get("topK", 20)),
                        dare_seed,
                        use_rescale,
                        avg_score,
                    ],
                )

                if avg_score >= best_score_for_param:
                    best_for_param = deepcopy(instance_params)
                    best_for_param["avg_normalized_metric"] = avg_score
                    best_score_for_param = avg_score
                elif search_early_stop:
                    print(f"[DARE-SEARCH] Early stop on param={param}, value={value}")
                    break

            finally:
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        if best_for_param is None:
            raise RuntimeError(f"[DARE-SEARCH] No valid candidate found for param={param}")

        running_defaults[param] = best_for_param[param]
        best_val_results = deepcopy(best_for_param)
        print(f"[DARE-SEARCH] Best after {param}: {best_val_results}")

    print(f"[DARE-SEARCH] Best config = {best_val_results}")
    return best_val_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="GLUE_t5")
    parser.add_argument("--config", type=str, default="config/methods-config/iteris-config.yaml")
    args = parser.parse_args()

    task_type = args.task_type

    with open(args.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    set_seed(config_data.get("seed", 42))

    if task_type not in config_data:
        raise ValueError(f"Cannot find task_type={task_type} in config.")

    task_cfg = config_data[task_type]

    model_name = task_cfg["model_name"]
    task_targets = task_cfg["task_targets"]
    rank = task_cfg.get("rank", 8)

    drop_rate = float(task_cfg.get("dare_drop_rate", 0.5))
    use_rescale = bool(task_cfg.get("dare_use_rescale", True))
    dare_seed = int(task_cfg.get("dare_seed", config_data.get("seed", 42)))
    merge_method = task_cfg.get("dare_merge_method", "dare-ties")
    topK = int(task_cfg.get("dare_topK", task_cfg.get("ties_topK", task_cfg.get("topK", 20))))
    sign_resolve_mode = task_cfg.get(
        "dare_sign_resolve_mode",
        task_cfg.get(
            "ties_sign_resolve_mode",
            task_cfg.get("sign_resolve_mode", "sum_of_values"),
        ),
    )
    merging_type = task_cfg.get(
        "dare_merging_type",
        task_cfg.get(
            "ties_merging_type",
            task_cfg.get("merging_type", "mean"),
        ),
    )
    scaling_coeffs = float(task_cfg.get("dare_scaling_coeffs", task_cfg.get("scaling_coeffs", 1.0)))
    dare_do_search = bool(task_cfg.get("dare_do_search", False))
    dare_search_early_stop = bool(task_cfg.get("dare_search_early_stop", False))

    scaling_candidates = task_cfg.get(
        "dare_scaling_coeffs_candidates",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    drop_candidates = task_cfg.get(
        "dare_pruning_coeffs_candidates",
        task_cfg.get("dare_drop_rates", [drop_rate]),
    )
    topk_candidates = task_cfg.get("dare_topK_candidates", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    tokenizer = AutoTokenizer.from_pretrained(model_name) if "blip" not in model_name else AutoProcessor.from_pretrained(model_name)

    if task_type == "TASKS_blip_base":
        lora_root = task_cfg.get("lora_root", "loras/SENTICAP-lora-blip")
    else:
        lora_source = str(task_cfg.get("lora_source", "default")).lower().strip()
        if lora_source in ["default", "best", "normal", "gaussian"]:
            lora_root = task_cfg.get("lora_root", "best_LoRA")
        elif lora_source == "osrm":
            lora_root = task_cfg.get("lora_root", "OSRM_LoRA")
        else:
            lora_root = task_cfg.get("lora_root", lora_source)

    print(f"[DARE] lora_root = {lora_root}")
    lora_path_dict = get_loras_path(task_type, model_name, lora_root=lora_root)
    missing_lora = [task for task in task_targets if task not in lora_path_dict]
    if missing_lora:
        raise ValueError(f"[DARE] Missing LoRA path for tasks: {missing_lora}")
    lora_path = [lora_path_dict[task] for task in task_targets]

    pair_name = "_".join(task_targets)

    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, "pair_merge_results.csv")
    registry_csv = os.path.join(results_dir, "experiment_registry.csv")
    dare_search_csv = os.path.join(results_dir, "dare_search_results.csv")
    vlm_results_csv = os.path.join(results_dir, "vlm_caption_results.csv")
    log_file = os.environ.get("LOG_FILE", "")

    search_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_DARE_SEARCH_{pair_name}"

    if dare_do_search:
        order_of_processing_params = ["scaling_coeffs", "dare_pruning_coeffs"]
        search_config = {
            "scaling_coeffs": [float(x) for x in scaling_candidates],
            "dare_pruning_coeffs": [float(x) for x in drop_candidates],
        }
        if merge_method == "dare-ties":
            order_of_processing_params.append("topK")
            search_config["topK"] = [int(x) for x in topk_candidates]
        best_cfg = search_best_dare_config(
            lora_path=lora_path,
            model_name=model_name,
            task_targets=task_targets,
            tokenizer=tokenizer,
            max_length=task_cfg["max_length"],
            per_device_eval_batch_size=task_cfg.get("per_device_eval_batch_size", 8),
            task_type=task_type,
            default_params={
                "scaling_coeffs": scaling_coeffs,
                "dare_pruning_coeffs": drop_rate,
                "topK": topK,
            },
            search_config=search_config,
            order_of_processing_params=order_of_processing_params,
            merge_method=merge_method,
            use_rescale=use_rescale,
            dare_seed=dare_seed,
            sign_resolve_mode=sign_resolve_mode,
            merging_type=merging_type,
            search_early_stop=dare_search_early_stop,
            search_csv=dare_search_csv,
            search_id=search_id,
            pair_name=pair_name,
        )
        scaling_coeffs = float(best_cfg["scaling_coeffs"])
        drop_rate = float(best_cfg["dare_pruning_coeffs"])
        if merge_method == "dare-ties":
            topK = int(best_cfg["topK"])

    method_tag = "DARE_TIES" if merge_method == "dare-ties" else "DARE"
    method_name = f"{method_tag}_p{str(drop_rate).replace('.', 'p')}_s{dare_seed}_k{topK}_c{str(scaling_coeffs).replace('.', 'p')}"
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{method_name}_{pair_name}"

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
        "dare_drop_rate", "dare_use_rescale", "dare_seed",
        "fusion_total_time_sec", "fusion_iter_time_avg_sec",
        "fusion_iter_time_max_sec", "fusion_peak_vram_avg_mb",
        "fusion_peak_vram_max_mb", "pair_avg_normalized_metric",
        "status", "error"
    ]

    ensure_csv_header(results_csv, results_header)
    ensure_csv_header(registry_csv, registry_header)

    if task_type == "TASKS_blip_base":
        ensure_csv_header(
            vlm_results_csv,
            [
                "experiment_id",
                "method",
                "pair_name",
                "task_a",
                "task_b",
                "evaluated_task",
                "style_acc",
                "cider",
                "bleu_1",
                "bleu_2",
                "bleu_3",
                "bleu_4",
                "rougeL",
                "div_1",
                "div_2",
                "vocab_size",
                "split",
                "merged_model_dir",
                "log_file",
            ],
        )

    try:
        model, fusion_stats = merge_dare_param(
            lora_path=lora_path,
            model_name=model_name,
            task_targets=task_targets,
            drop_rate=drop_rate,
            scaling_coeffs=scaling_coeffs,
            use_rescale=use_rescale,
            dare_seed=dare_seed,
            merge_method=merge_method,
            topK=topK,
            sign_resolve_mode=sign_resolve_mode,
            merging_type=merging_type,
        )

        # Stable save path for the selected/best DARE coarse model.
        # This is important for later DARE_TALS_DRC: it should load exactly the
        # coarse model selected by DARE search, not re-run DARE with fixed params.
        merged_model_dir = task_cfg.get(
            "dare_merged_model_dir",
            f"merged_model/{method_tag}_best_{pair_name}",
        )

        save_best_model = bool(task_cfg.get("save_best_model", task_cfg.get("save", 0)))
        if save_best_model:
            os.makedirs(merged_model_dir, exist_ok=True)
            model.save_pretrained(merged_model_dir)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(merged_model_dir)

            selected_cfg = {
                "method_name": method_name,
                "method_tag": method_tag,
                "pair_name": pair_name,
                "task_targets": task_targets,
                "model_name": model_name,
                "merge_method": merge_method,
                "dare_drop_rate": float(drop_rate),
                "dare_use_rescale": bool(use_rescale),
                "dare_seed": int(dare_seed),
                "dare_scaling_coeffs": float(scaling_coeffs),
                "dare_topK": int(topK),
                "dare_sign_resolve_mode": sign_resolve_mode,
                "dare_merging_type": merging_type,
                "dare_do_search": bool(dare_do_search),
                "dare_search_early_stop": bool(dare_search_early_stop),
                "search_id": search_id if dare_do_search else "",
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(os.path.join(merged_model_dir, "dare_selected_config.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(selected_cfg, f, allow_unicode=True, sort_keys=False)

            print(f"[DARE] Selected/best coarse merged model saved to {merged_model_dir}")
            print(f"[DARE] Selected config saved to {os.path.join(merged_model_dir, 'dare_selected_config.yaml')}")

        normalized_metrics = []

        for task_name in task_targets:
            print(f"[Eval] Evaluating merged model on {task_name}...")

            eval_result = eval_iteris_model(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                task_name=task_name,
                max_length=task_cfg["max_length"],
                per_device_eval_batch_size=task_cfg.get("per_device_eval_batch_size", 8),
            )

            eval_accuracy = eval_result.get("eval_accuracy", "")
            eval_mcc = eval_result.get("eval_MCC", "")
            eval_f1 = eval_result.get("eval_f1-score", "")
            eval_loss = eval_result.get("eval_loss", "")
            eval_runtime = eval_result.get("eval_runtime", eval_result.get("eval_wall_time_sec", ""))
            eval_sps = eval_result.get("eval_samples_per_second", "")
            eval_stepsps = eval_result.get("eval_steps_per_second", "")
            eval_peak_vram_mb = eval_result.get("eval_peak_vram_mb", "")

            primary_metric_name, primary_metric_value, normalized_metric = get_primary_metric_any(
                eval_result=eval_result,
                task_name=task_name,
                task_type=task_type,
            )

            if task_type == "TASKS_blip_base":
                eval_accuracy = primary_metric_value
                append_vlm_caption_row(
                    csv_path=vlm_results_csv,
                    experiment_id=experiment_id,
                    method_name=method_name,
                    pair_name=pair_name,
                    task_targets=task_targets,
                    task_name=task_name,
                    eval_result=eval_result,
                    merged_model_dir=merged_model_dir,
                    log_file=log_file,
                )

            normalized_metrics.append(normalized_metric)

            append_csv_row(
                results_csv,
                [
                    experiment_id, method_name, pair_name, task_targets[0], task_targets[1],
                    task_name, primary_metric_name, primary_metric_value,
                    normalized_metric, eval_accuracy, eval_mcc, eval_f1,
                    eval_loss, eval_runtime, eval_sps, eval_stepsps,
                    eval_peak_vram_mb, "validation", merged_model_dir, log_file, ""
                ]
            )

        pair_avg_normalized_metric = float(np.mean(normalized_metrics))

        dare_cfg_str = (
            f"merge_method={merge_method}|topK={topK}|sign_resolve_mode={sign_resolve_mode}|"
            f"merging_type={merging_type}|scaling_coeffs={scaling_coeffs}|"
            f"drop_rate={drop_rate}|use_rescale={use_rescale}|dare_seed={dare_seed}|"
            f"search={dare_do_search}|search_early_stop={dare_search_early_stop}"
        )

        append_csv_row(
            registry_csv,
            [
                experiment_id, "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise", method_name, model_name, pair_name,
                "|".join(task_targets), dare_cfg_str,
                rank, "|".join(map(str, task_cfg.get("lora_alpha", [32 for _ in task_targets]))),
                drop_rate, use_rescale, dare_seed,
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_max_sec"],
                fusion_stats["fusion_peak_vram_avg_mb"],
                fusion_stats["fusion_peak_vram_max_mb"],
                pair_avg_normalized_metric,
                "success", ""
            ]
        )

        print(f"[Done] {method_tag} finished for pair: {pair_name}")
        print(f"[Done] pair_avg_normalized_metric = {pair_avg_normalized_metric}")

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)

        dare_cfg_str = (
            f"merge_method={merge_method}|topK={topK}|sign_resolve_mode={sign_resolve_mode}|"
            f"merging_type={merging_type}|scaling_coeffs={scaling_coeffs}|"
            f"drop_rate={drop_rate}|use_rescale={use_rescale}|dare_seed={dare_seed}|"
            f"search={dare_do_search}|search_early_stop={dare_search_early_stop}"
        )

        append_csv_row(
            registry_csv,
            [
                experiment_id, "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise", method_name, model_name, pair_name,
                "|".join(task_targets), dare_cfg_str,
                rank, "|".join(map(str, task_cfg.get("lora_alpha", [32 for _ in task_targets]))),
                drop_rate, use_rescale, dare_seed,
                "", "", "", "", "", "", "failed", error_msg
            ]
        )

        raise e


if __name__ == "__main__":
    main()
