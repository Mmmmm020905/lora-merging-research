#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KnOTS_TIES.py

KnOTS-TIES LoRA merging script.

Purpose:
  - Verify KnOTS-TIES (KnOTS SVD alignment + TIES topK/sign mask merge)
    on NLP/T5 and VLM/BLIP.
  - For TASKS_blip_base, this script uses the corrected LoRA delta convention:
      ΔW = (lora_alpha / r) * B @ A
    and does NOT subtract a fresh LoRA reference.
  - For GLUE/T5, default behavior preserves the previous raw-delta +
    fresh-reference task-vector style unless overridden in config.

Notes:
  - On BLIP-SentiCap, TIES-style sign masking may collapse generation; this file
    is intended to run/record that baseline with correct implementation and
    VLM metrics, not to force it to be stable.
"""

import os
import gc
import csv
import yaml
import time
import fcntl
import torch
import random
import argparse
import traceback
import numpy as np
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

from peft import LoraConfig, get_peft_model, TaskType
from peft.utils import get_peft_model_state_dict
from safetensors import safe_open

from eval_model import eval_iteris_model


GLUE_task_name = [
    "mnli", "rte", "cola", "sst2", "qqp", "qnli", "mrpc", "wnli",
]


# ============================================================
# Generic helpers
# ============================================================

def is_blip_model(model_name):
    return "blip" in str(model_name).lower()


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
    lock_path = csv_path + ".lock"
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
            if need_header:
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    f.flush()
                    os.fsync(f.fileno())
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def append_csv_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    lock_path = csv_path + ".lock"
    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


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


def short_float_tag(x):
    return str(x).replace(".", "p").replace("-", "m")


# ============================================================
# Model / LoRA path helpers
# ============================================================

def get_loras_path(task_type, model_name, lora_root=None):
    """
    Return LoRA adapter directories.

    For TASKS_blip_base:
      positive -> loras/SENTICAP-lora-blip/positive
      negative -> loras/SENTICAP-lora-blip/negative

    For GLUE_t5:
      default root is best_LoRA unless overridden.
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
        # Optional future mappings
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
    raise ValueError(f"[KnOTS] Unsupported model_name: {model_name}")


# ============================================================
# PEFT / LoRA extraction helpers
# ============================================================

def load_adapter_config(lora_dir):
    config_path = os.path.join(lora_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[KnOTS] adapter_config.json not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_lora_scaling_from_adapter_cfg(adapter_cfg):
    r = adapter_cfg.get("r", None)
    alpha = adapter_cfg.get("lora_alpha", None)
    if r is None or alpha is None:
        return 1.0
    return float(alpha) / float(r)


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
        raise FileNotFoundError(f"[KnOTS] adapter_model.safetensors not found: {adapter_file}")
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
                f"[KnOTS] state_dict keys mismatch between model 0 and model {i}. "
                f"Different keys: {sorted(diff)[:20]} ..."
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


def lora_state_dict_to_delta_matrices(
    state_dict,
    device="cpu",
    adapter_cfg=None,
    use_effective_delta=False,
):
    """
    Convert PEFT LoRA state dict to equivalent dense delta matrices.

    If use_effective_delta=True:
        ΔW = (lora_alpha / r) * B @ A
    If False:
        ΔW = B @ A
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

    scale = get_lora_scaling_from_adapter_cfg(adapter_cfg) if use_effective_delta else 1.0

    task_parameters = OrderedDict()
    for name, key2val in sorted(layer2lora_parameters.items()):
        if "A" not in key2val or "B" not in key2val:
            raise ValueError(f"[KnOTS] Incomplete LoRA pair for layer: {name}")
        task_parameters[name] = (float(scale) * (key2val["B"] @ key2val["A"])).to(torch.float32)

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
                param.copy_(param + float(scaling_coeff) * delta)
                updated += 1
    return updated


def print_delta_norms(task_dirs, prefix="[KnOTS]"):
    for idx, sd in enumerate(task_dirs):
        if not sd:
            print(f"{prefix}[DeltaNorm] task_idx={idx}: empty")
            continue
        total_sq = 0.0
        max_norm = -1.0
        max_key = None
        for key, val in sd.items():
            n = float(torch.norm(val.float()))
            total_sq += n * n
            if n > max_norm:
                max_norm = n
                max_key = key
        print(
            f"{prefix}[DeltaNorm] task_idx={idx}, "
            f"global_norm={total_sq ** 0.5:.6f}, "
            f"max_layer_norm={max_norm:.6f}, max_key={max_key}"
        )



# ============================================================
# KnOTS-TIES core
# ============================================================

def apply_knots_svd(ftms_task_dirs, concat_across_output=True, svd_tol=1e-5, verbose_svd=False):
    num_tasks = len(ftms_task_dirs)
    if num_tasks < 2:
        raise ValueError("[KnOTS-TIES] At least two task directions are required.")

    layer_names = list(ftms_task_dirs[0].keys())
    U_dict = OrderedDict()
    task_sV_dicts = [OrderedDict() for _ in range(num_tasks)]

    for layer_name in layer_names:
        layer_mats = [task_dir[layer_name].to(torch.float32) for task_dir in ftms_task_dirs]

        if concat_across_output:
            concat_matrix = torch.cat(layer_mats, dim=1)
        else:
            concat_matrix = torch.cat([mat.t() for mat in layer_mats], dim=1)

        concat_matrix64 = concat_matrix.to(torch.float64)
        U, s, Vh = torch.linalg.svd(concat_matrix64, full_matrices=False)
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
            V_chunks = [torch.zeros((rank_dim, cols), dtype=torch.float32) for _ in range(num_tasks)]
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


def topk_values_mask(M, K=0.7):
    """
    TIES topK mask. If K > 1, treat it as percentage, e.g. 20 -> 20%.
    K=100 keeps all nonzero positions before sign masking.
    """
    if K > 1:
        K /= 100.0

    if K >= 1:
        return M, torch.ones_like(M), torch.ones_like(M).float().mean(dim=-1)

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    _, d = M.shape
    k = int(d * K)
    k = d - k
    k = max(1, min(k, d))
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
        sign_to_mult = sign_to_mult.clone()
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult = sign_to_mult.clone()
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
        return chunked_disjoint_mean(vectors, chunk_size=10000)
    if merge_func == "sum":
        return chunked_sum(vectors)
    if merge_func == "max":
        return vectors.abs().max(dim=0)[0]
    if merge_func == "unmerged":
        return vectors
    raise ValueError(f"[KnOTS-TIES] Merge method {merge_func} is not defined.")


def merge_knots_ties_param(
    lora_path,
    model_name,
    task_targets,
    seed,
    topK=100,
    sign_resolve_mode="sum_of_values",
    scaling_coeffs=0.05,
    merging_type="mean",
    concat_across_output=True,
    svd_tol=1e-5,
    verbose_svd=False,
    use_effective_delta=False,
    subtract_fresh_reference=True,
):
    assert len(lora_path) == len(task_targets), "lora_path must match task_targets."

    print(f"[KnOTS-TIES] task_targets = {task_targets}")
    print(f"[KnOTS-TIES] lora_path = {lora_path}")
    print(
        f"[KnOTS-TIES] topK = {topK}, sign_resolve_mode = {sign_resolve_mode}, "
        f"scaling_coeffs = {scaling_coeffs}, merging_type = {merging_type}, "
        f"concat_across_output = {concat_across_output}, svd_tol = {svd_tol}, "
        f"verbose_svd = {verbose_svd}, use_effective_delta={use_effective_delta}, "
        f"subtract_fresh_reference={subtract_fresh_reference}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    start_time = time.time()

    adapter_cfg = load_adapter_config(lora_path[0])

    ft_state_dicts = [load_adapter_state_dict_from_safetensors(path, device="cpu") for path in lora_path]
    check_state_dict_keys_match(ft_state_dicts)

    ftms_relevant_params = [
        lora_state_dict_to_delta_matrices(
            sd,
            device="cpu",
            adapter_cfg=adapter_cfg,
            use_effective_delta=use_effective_delta,
        )
        for sd in ft_state_dicts
    ]

    if subtract_fresh_reference:
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
            use_effective_delta=use_effective_delta,
        )
        del fresh_peft_model
        gc.collect()

        check_state_dict_keys_match([ptm_reference_params] + ftms_relevant_params)
        ftms_task_dirs = get_task_directions(ptm_reference_params, ftms_relevant_params)
    else:
        check_state_dict_keys_match(ftms_relevant_params)
        ftms_task_dirs = [OrderedDict(sorted(sd.items())) for sd in ftms_relevant_params]

    print_delta_norms(ftms_task_dirs, prefix="[KnOTS-TIES]")

    U_dict, task_sV_dicts = apply_knots_svd(
        ftms_task_dirs,
        concat_across_output=concat_across_output,
        svd_tol=svd_tol,
        verbose_svd=verbose_svd,
    )

    ftms_reps = directions_to_reps(task_sV_dicts)
    masks = ties_masking(ftms_reps, topK=topK, sign_resolve_mode=sign_resolve_mode)

    stacked_sVs = torch.vstack(ftms_reps).clone()
    masks = masks.to(stacked_sVs.device).to(stacked_sVs.dtype)
    masked_sVs = stacked_sVs * masks

    pre_merge_sVs_dict = rep_to_state_dict(masked_sVs, task_sV_dicts[0])
    rescaled_reps = torch.stack(directions_to_reps(pre_merge_sVs_dict), dim=0)

    merged_sV = masked_merge(
        vectors=rescaled_reps,
        merge_func=merging_type,
        weights=[float(scaling_coeffs)],
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

    if number_update == len(merged_direction_sd):
        print("[KnOTS-TIES] All target modules updated successfully.")
    else:
        print(
            f"[KnOTS-TIES][Warn] Updated {number_update}/{len(merged_direction_sd)} modules. "
            f"Please check LoRA layer names vs base model parameter names."
        )

    fusion_time = round(time.time() - start_time, 4)
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


def append_knots_ties_search_row(csv_path, row):
    header = [
        "search_id", "pair_name", "scaling_coeffs", "topK",
        "avg_normalized_metric",
    ]
    ensure_csv_header(csv_path, header)
    append_csv_row(csv_path, row)


def search_best_knots_ties_config(
    lora_path,
    model_name,
    task_targets,
    seed,
    tokenizer,
    max_length,
    per_device_eval_batch_size,
    task_type,
    concat_across_output,
    svd_tol,
    default_params,
    scaling_candidates,
    topK_candidates,
    sign_resolve_mode,
    merging_type,
    search_strategy="linear",
    early_stop=False,
    verbose_svd=False,
    use_effective_delta=False,
    subtract_fresh_reference=True,
    search_csv=None,
    search_id="",
    pair_name="",
):
    search_strategy = str(search_strategy).lower().strip()
    print(f"[KnOTS-TIES-SEARCH] default_params = {default_params}")
    print(f"[KnOTS-TIES-SEARCH] scaling_candidates = {scaling_candidates}")
    print(f"[KnOTS-TIES-SEARCH] topK_candidates = {topK_candidates}")
    print(f"[KnOTS-TIES-SEARCH] search_strategy = {search_strategy}, early_stop = {early_stop}")

    def evaluate_params(params):
        model = None
        try:
            print(f"[KnOTS-TIES-SEARCH] Try params = {params}")
            model, _, _ = merge_knots_ties_param(
                lora_path=lora_path,
                model_name=model_name,
                task_targets=task_targets,
                seed=seed,
                topK=int(params["topK"]),
                sign_resolve_mode=sign_resolve_mode,
                scaling_coeffs=float(params["scaling_coeffs"]),
                merging_type=merging_type,
                concat_across_output=concat_across_output,
                svd_tol=svd_tol,
                verbose_svd=verbose_svd,
                use_effective_delta=use_effective_delta,
                subtract_fresh_reference=subtract_fresh_reference,
            )

            avg_score, _ = eval_knots_ties_pair_average(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                task_targets=task_targets,
                max_length=max_length,
                per_device_eval_batch_size=per_device_eval_batch_size,
                task_type=task_type,
            )
            print(f"[KnOTS-TIES-SEARCH] avg_normalized_metric = {avg_score:.6f}")

            if search_csv is not None:
                append_knots_ties_search_row(
                    search_csv,
                    [
                        search_id,
                        pair_name,
                        float(params["scaling_coeffs"]),
                        int(params["topK"]),
                        avg_score,
                    ],
                )

            return avg_score
        finally:
            if model is not None:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    if search_strategy == "grid":
        best = deepcopy(default_params)
        best["avg_normalized_metric"] = -1e9
        for scaling in scaling_candidates:
            for topK in topK_candidates:
                params = deepcopy(default_params)
                params["scaling_coeffs"] = float(scaling)
                params["topK"] = int(float(topK))
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

    for param, candidates in [("scaling_coeffs", scaling_candidates), ("topK", topK_candidates)]:
        best_for_param = deepcopy(best_val_results)
        found_improvement_once = False

        for value in candidates:
            params = deepcopy(running_defaults)
            params[param] = float(value) if param == "scaling_coeffs" else int(float(value))
            score = evaluate_params(params)
            if score >= best_for_param.get("avg_normalized_metric", -1e9):
                best_for_param = deepcopy(params)
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

# ============================================================
# Evaluation helpers
# ============================================================

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


def eval_knots_ties_pair_average(
    model,
    tokenizer,
    model_name,
    task_targets,
    max_length,
    per_device_eval_batch_size,
    task_type=None,
):
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
    avg_score = float(np.mean(normalized_metrics))
    return avg_score, per_task_results




# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_type", type=str, default="GLUE_t5")
    parser.add_argument("--config", type=str, default="config/methods-config/iteris-config.yaml")
    args = parser.parse_args()

    task_type = args.task_type

    with open(args.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    seed = config_data.get("seed", 42)
    set_seed(seed)

    if task_type not in config_data:
        raise ValueError(f"Cannot find task_type={task_type} in config.")

    task_cfg = config_data[task_type]
    model_name = task_cfg["model_name"]
    task_targets = task_cfg["task_targets"]
    rank = task_cfg.get("rank", 8)

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

    print(f"[KnOTS-TIES] lora_root = {lora_root}")
    lora_path_dict = get_loras_path(task_type, model_name, lora_root=lora_root)
    missing_lora = [task for task in task_targets if task not in lora_path_dict]
    if missing_lora:
        raise ValueError(f"[KnOTS-TIES] Missing LoRA path for tasks: {missing_lora}")
    lora_path = [lora_path_dict[task] for task in task_targets]

    raw_topK = task_cfg.get("knots_ties_topK", task_cfg.get("topK", 100))
    raw_scaling_coeffs = task_cfg.get("knots_ties_scaling_coeffs", task_cfg.get("scaling_coeffs", 0.05))
    sign_resolve_mode = task_cfg.get("knots_ties_sign_resolve_mode", task_cfg.get("sign_resolve_mode", "sum_of_values"))
    merging_type = task_cfg.get("knots_ties_merging_type", task_cfg.get("merging_type", "mean"))
    concat_across_output = bool(task_cfg.get("knots_concat_across_output", True))
    svd_tol = float(task_cfg.get("knots_svd_tol", 1e-5))
    knots_ties_do_search = bool(task_cfg.get("knots_ties_do_search", False))
    knots_ties_search_strategy = str(task_cfg.get("knots_ties_search_strategy", "linear")).lower().strip()
    knots_ties_search_early_stop = bool(task_cfg.get("knots_ties_search_early_stop", False))
    knots_verbose_svd = bool(task_cfg.get("knots_verbose_svd", False))

    # Preserve old NLP default, but use corrected effective delta for BLIP/VLM.
    use_effective_delta = bool(task_cfg.get("knots_ties_use_effective_delta", task_cfg.get("knots_use_effective_delta", is_blip_model(model_name))))
    subtract_fresh_reference = bool(task_cfg.get("knots_ties_subtract_fresh_reference", task_cfg.get("knots_subtract_fresh_reference", not is_blip_model(model_name))))

    method_name = task_cfg.get("knots_ties_method_name", "KnOTS_TIES")
    pair_name = "_".join(task_targets)
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{method_name}_{pair_name}"

    tokenizer = (
        AutoTokenizer.from_pretrained(model_name)
        if not is_blip_model(model_name)
        else AutoProcessor.from_pretrained(model_name)
    )

    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, "pair_merge_results.csv")
    registry_csv = os.path.join(results_dir, "experiment_registry.csv")
    vlm_results_csv = os.path.join(results_dir, "vlm_caption_results.csv")
    knots_ties_search_csv = os.path.join(results_dir, "knots_ties_search_results.csv")
    log_file = os.environ.get("LOG_FILE", "")

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
        "task_targets", "merge_config", "rank", "lora_alpha",
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

    topK = None
    scaling_coeffs = None
    adapter_cfg = {}

    try:
        parsed_topK = parse_scalar_or_candidates(raw_topK)
        parsed_scaling_coeffs = parse_scalar_or_candidates(raw_scaling_coeffs)

        if knots_ties_do_search:
            if isinstance(parsed_topK, list):
                topK_candidates = parsed_topK
            else:
                topK_candidates = [
                    float(x) for x in task_cfg.get(
                        "knots_ties_topK_candidates",
                        [70, 90, 100],
                    )
                ]
            if isinstance(parsed_scaling_coeffs, list):
                scaling_candidates = parsed_scaling_coeffs
            else:
                scaling_candidates = [
                    float(x) for x in task_cfg.get(
                        "knots_ties_scaling_coeffs_candidates",
                        [0.0, 0.0001, 0.0003, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05],
                    )
                ]

            default_scaling = float(task_cfg.get("knots_ties_default_scaling_coeffs", task_cfg.get("knots_ties_scaling_coeffs", 0.05)))
            default_topK = int(float(task_cfg.get("knots_ties_default_topK", task_cfg.get("knots_ties_topK", 100))))

            search_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_KNOTS_TIES_SEARCH_{pair_name}"
            search_result = search_best_knots_ties_config(
                lora_path=lora_path,
                model_name=model_name,
                task_targets=task_targets,
                seed=seed,
                tokenizer=tokenizer,
                max_length=task_cfg["max_length"],
                per_device_eval_batch_size=task_cfg.get("per_device_eval_batch_size", 8),
                task_type=task_type,
                concat_across_output=concat_across_output,
                svd_tol=svd_tol,
                default_params={
                    "scaling_coeffs": default_scaling,
                    "topK": default_topK,
                },
                scaling_candidates=scaling_candidates,
                topK_candidates=topK_candidates,
                sign_resolve_mode=sign_resolve_mode,
                merging_type=merging_type,
                search_strategy=knots_ties_search_strategy,
                early_stop=knots_ties_search_early_stop,
                verbose_svd=knots_verbose_svd,
                use_effective_delta=use_effective_delta,
                subtract_fresh_reference=subtract_fresh_reference,
                search_csv=knots_ties_search_csv,
                search_id=search_id,
                pair_name=pair_name,
            )
            scaling_coeffs = float(search_result["scaling_coeffs"])
            topK = int(search_result["topK"])
        else:
            if isinstance(parsed_topK, list):
                raise ValueError(
                    "knots_ties_topK is a candidate list. "
                    "Set knots_ties_do_search: true, or set knots_ties_topK to a single value."
                )
            if isinstance(parsed_scaling_coeffs, list):
                raise ValueError(
                    "knots_ties_scaling_coeffs is a candidate list. "
                    "Set knots_ties_do_search: true, or set knots_ties_scaling_coeffs to a single value."
                )
            topK = int(parsed_topK)
            scaling_coeffs = float(parsed_scaling_coeffs)

        method_name = f"{method_name}_k{topK}_c{short_float_tag(scaling_coeffs)}"

        model, fusion_stats, adapter_cfg = merge_knots_ties_param(
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
            use_effective_delta=use_effective_delta,
            subtract_fresh_reference=subtract_fresh_reference,
        )

        merged_model_dir = task_cfg.get(
            "knots_ties_merged_model_dir",
            f"merged_model/KnOTS_TIES_{pair_name}",
        )

        save_best_model = bool(task_cfg.get("save_best_model", task_cfg.get("save", 0)))
        if save_best_model:
            os.makedirs(merged_model_dir, exist_ok=True)
            model.save_pretrained(merged_model_dir)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(merged_model_dir)

            selected_cfg = {
                "method_name": method_name,
                "pair_name": pair_name,
                "task_targets": task_targets,
                "model_name": model_name,
                "knots_ties_topK": int(topK),
                "knots_ties_scaling_coeffs": float(scaling_coeffs),
                "knots_ties_sign_resolve_mode": sign_resolve_mode,
                "knots_ties_merging_type": merging_type,
                "knots_concat_across_output": bool(concat_across_output),
                "knots_svd_tol": float(svd_tol),
                "knots_ties_do_search": bool(knots_ties_do_search),
                "knots_ties_search_strategy": knots_ties_search_strategy,
                "knots_ties_search_early_stop": bool(knots_ties_search_early_stop),
                "knots_verbose_svd": bool(knots_verbose_svd),
                "knots_ties_use_effective_delta": bool(use_effective_delta),
                "knots_ties_subtract_fresh_reference": bool(subtract_fresh_reference),
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(os.path.join(merged_model_dir, "knots_ties_selected_config.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(selected_cfg, f, allow_unicode=True, sort_keys=False)

            print(f"[KnOTS-TIES] Selected/best coarse merged model saved to {merged_model_dir}")
            print(f"[KnOTS-TIES] Selected config saved to {os.path.join(merged_model_dir, 'knots_ties_selected_config.yaml')}")

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
        knots_ties_cfg_str = (
            f"topK={topK}|sign_resolve_mode={sign_resolve_mode}|"
            f"scaling_coeffs={scaling_coeffs}|merging_type={merging_type}|"
            f"concat_across_output={concat_across_output}|svd_tol={svd_tol}|"
            f"search={knots_ties_do_search}|search_strategy={knots_ties_search_strategy}|"
            f"early_stop={knots_ties_search_early_stop}|verbose_svd={knots_verbose_svd}|"
            f"use_effective_delta={use_effective_delta}|subtract_fresh_reference={subtract_fresh_reference}"
        )

        append_csv_row(
            registry_csv,
            [
                experiment_id,
                "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise",
                method_name,
                model_name,
                pair_name,
                "|".join(task_targets),
                knots_ties_cfg_str,
                adapter_cfg.get("r", rank),
                str(adapter_cfg.get("lora_alpha", task_cfg.get("lora_alpha", ""))),
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_max_sec"],
                fusion_stats["fusion_peak_vram_avg_mb"],
                fusion_stats["fusion_peak_vram_max_mb"],
                pair_avg_normalized_metric,
                "success",
                ""
            ]
        )

        print(f"[Done] KnOTS-TIES finished for pair: {pair_name}")
        print(f"[Done] pair_avg_normalized_metric = {pair_avg_normalized_metric}")

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)

        knots_ties_cfg_str = (
            f"topK={topK}|sign_resolve_mode={sign_resolve_mode}|"
            f"scaling_coeffs={scaling_coeffs}|merging_type={merging_type}|"
            f"concat_across_output={concat_across_output}|svd_tol={svd_tol}|"
            f"search={knots_ties_do_search}|search_strategy={knots_ties_search_strategy}|"
            f"early_stop={knots_ties_search_early_stop}|verbose_svd={knots_verbose_svd}|"
            f"use_effective_delta={use_effective_delta}|subtract_fresh_reference={subtract_fresh_reference}"
        )

        append_csv_row(
            registry_csv,
            [
                experiment_id,
                "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise",
                method_name,
                model_name,
                pair_name,
                "|".join(task_targets),
                knots_ties_cfg_str,
                task_cfg.get("rank", ""),
                str(task_cfg.get("lora_alpha", "")),
                "", "", "", "", "", "",
                "failed",
                error_msg
            ]
        )
        raise e


if __name__ == "__main__":
    main()
