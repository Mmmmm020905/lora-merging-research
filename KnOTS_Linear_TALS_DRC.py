#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KnOTS_LINEAR_TALS_DRC.py

KnOTS+Linear + current full TALS-LER inference-time correction.

Coarse merge:
    KnOTS+Linear or KnOTS+Linear-TIES merged model M_c.

TALS-LER direction:
    r_act = h_single - h_knots_linear
    R_W   = ΔW_t - (W_knots_linear - W_base)
    d     = omega * Normalize(V_k G_k V_k^T r_act)
    s     = LER * ||r_act||
    omega = N * s / sum(s)

Important:
    This script reuses low-level TALS/DRC helper functions from the latest
    Linear_TALS_DRC.py. Please keep the verified current full-version
    Linear_TALS_DRC.py in the same project root.
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

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForSeq2SeqLM,
    BlipForConditionalGeneration,
)

from eval_model import eval_iteris_model
from safetensors import safe_open
from get_midfeatures import get_samples as get_iteris_samples
from get_midfeatures import merge_peft



# Reuse verified TALS-LER utilities.
from Linear_TALS_DRC import (
    construct_base_model,
    load_single_lora_dense_model,
    normalize_target_modules,
    select_drc_targets,
    collect_features_by_position,
    get_task_samples,
    normalize_direction,
    build_all_task_lora_deltas,
    apply_tals_filter_to_activation_residual,
    apply_ler_layerwise_reweight,
    register_drc_hooks_by_position,
    remove_hooks,
    stable_int_hash,
    parse_optional_float,
)


GLUE_task_name = [
    "mnli", "rte", "cola", "sst2", "qqp", "qnli", "mrpc", "wnli",
]


def get_loras_path(task_type, model_name, lora_root=None):
    """
    Local LoRA-path resolver used by KnOTS+Linear_TALS_DRC.

    VLM:
      positive -> <lora_root>/positive
      negative -> <lora_root>/negative

    GLUE/T5:
      task -> <lora_root>/T5-<TASK>-LoRA
    """
    if lora_root is None:
        lora_root = "loras/SENTICAP-lora-blip" if task_type == "TASKS_blip_base" else "best_LoRA"

    path_dict = {}
    if task_type == "TASKS_blip_base":
        path_dict["positive"] = f"{lora_root}/positive"
        path_dict["negative"] = f"{lora_root}/negative"
        return path_dict

    if task_type == "GLUE_t5" or "t5" in str(model_name).lower():
        path_dict["cola"] = f"{lora_root}/T5-COLA-LoRA"
        path_dict["sst2"] = f"{lora_root}/T5-SST2-LoRA"
        path_dict["rte"] = f"{lora_root}/T5-RTE-LoRA"
        path_dict["qnli"] = f"{lora_root}/T5-QNLI-LoRA"
        path_dict["qqp"] = f"{lora_root}/T5-QQP-LoRA"
        path_dict["mrpc"] = f"{lora_root}/T5-MRPC-LoRA"
        path_dict["mnli"] = f"{lora_root}/T5-MNLI-LoRA"
        path_dict["wnli"] = f"{lora_root}/T5-WNLI-LoRA"
        return path_dict

    return path_dict


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


def is_blip_model(model_name):
    return "blip" in str(model_name).lower()


def tensor_to_cuda(x):
    if torch.is_tensor(x):
        return x.detach().clone().to("cuda")
    return torch.as_tensor(x).to("cuda")


def construct_base_model_any(model_name):
    if is_blip_model(model_name):
        return BlipForConditionalGeneration.from_pretrained(model_name)
    return construct_base_model(model_name)


def load_required_coarse_model_from_dir_any(model_dir, model_name, model_label="coarse"):
    if model_dir is None or str(model_dir).strip() == "":
        raise ValueError(f"[KnOTS_LINEAR_TALS_DRC] Empty {model_label} model directory.")
    model_dir = str(model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"[KnOTS_LINEAR_TALS_DRC] Required {model_label} model dir not found: {model_dir}\n"
            f"Please run the corresponding KnOTS+Linear baseline first and save the selected coarse model."
        )
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(
            f"[KnOTS_LINEAR_TALS_DRC] {model_label} model dir exists but config.json is missing: {model_dir}\n"
            f"This does not look like a valid HuggingFace saved model directory."
        )

    print(f"[KnOTS_LINEAR_TALS_DRC] Loading required {model_label} coarse model from: {model_dir}")
    if is_blip_model(model_name):
        return BlipForConditionalGeneration.from_pretrained(model_dir)
    return AutoModelForSeq2SeqLM.from_pretrained(model_dir)


def read_lora_scale_from_adapter_config(lora_dir, fallback_alpha=None, fallback_rank=None):
    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    alpha = fallback_alpha
    rank = fallback_rank
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        alpha = cfg.get("lora_alpha", cfg.get("alpha", alpha))
        rank = cfg.get("r", cfg.get("rank", rank))
    if alpha is None:
        alpha = 1.0
    if rank is None or float(rank) == 0:
        rank = 1.0
    return float(alpha) / float(rank)


def normalize_lora_layer_name_local(base_name):
    prefixes = ["base_model.model.", "base_model.", "model."]
    for prefix in prefixes:
        if base_name.startswith(prefix):
            return base_name[len(prefix):]
    return base_name


def extract_lora_base_name_local(full_key):
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
            return normalize_lora_layer_name_local(full_key[: -len(suffix)])
    return None


def load_scaled_lora_delta_dict_local(lora_dir, device="cpu", fallback_alpha=None, fallback_rank=None):
    adapter_file = os.path.join(lora_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"[KnOTS_LINEAR_TALS_DRC] adapter_model.safetensors not found: {adapter_file}")

    scale = read_lora_scale_from_adapter_config(
        lora_dir,
        fallback_alpha=fallback_alpha,
        fallback_rank=fallback_rank,
    )

    layer2ab = {}
    with safe_open(adapter_file, framework="pt") as f:
        for key in sorted(f.keys()):
            tensor = f.get_tensor(key).to(device)
            base_name = extract_lora_base_name_local(key)
            if base_name is None:
                continue
            if ".lora_A" in key:
                layer2ab.setdefault(base_name, {})["A"] = tensor
            elif ".lora_B" in key:
                layer2ab.setdefault(base_name, {})["B"] = tensor

    delta_dict = {}
    for name, ab in sorted(layer2ab.items()):
        if "A" not in ab or "B" not in ab:
            continue
        delta_dict[name] = (scale * (ab["B"] @ ab["A"])).float().cpu()

    print(
        f"[KnOTS_LINEAR_TALS_DRC][LoRAΔW] {lora_dir}: loaded {len(delta_dict)} scaled deltas, "
        f"scale={scale:.6g}"
    )
    return delta_dict


def select_drc_targets_any(
    model_name,
    inject_position,
    lora_path_dict,
    task_targets,
    linear_model,
    target_part="encoder",
    target_modules=None,
    target_layers=None,
):
    if not is_blip_model(model_name):
        return select_drc_targets(
            inject_position=inject_position,
            lora_path_dict=lora_path_dict,
            task_targets=task_targets,
            linear_model=linear_model,
            target_part=target_part,
            target_modules=target_modules,
            target_layers=target_layers,
        )

    if inject_position != "lora_input":
        raise ValueError("[KnOTS_LINEAR_TALS_DRC][VLM] only drc_inject_position='lora_input' is supported.")

    modules = normalize_target_modules(target_modules or ["query", "value"])
    layers = target_layers if target_layers is not None else list(range(12))
    named_modules = dict(linear_model.named_modules())

    selected = []
    for layer in layers:
        for module in modules:
            key = f"text_decoder.bert.encoder.layer.{int(layer)}.attention.self.{module}"
            if key in named_modules:
                selected.append(key)
            else:
                print(f"[KnOTS_LINEAR_TALS_DRC][VLM][Warn] target module not found: {key}")

    print(
        f"[KnOTS_LINEAR_TALS_DRC][VLM] Selected {len(selected)} BLIP DRC targets. "
        f"target_modules={modules}, target_layers={layers}"
    )
    return selected


def build_all_task_lora_deltas_any(task_targets, lora_path_dict, target_keys, rank, lora_alpha_list=None, model_name=None):
    if not is_blip_model(model_name):
        return build_all_task_lora_deltas(
            task_targets=task_targets,
            lora_path_dict=lora_path_dict,
            target_keys=target_keys,
            rank=rank,
            lora_alpha_list=lora_alpha_list,
        )

    all_deltas = {}
    for idx, task_name in enumerate(task_targets):
        fallback_alpha = None
        if isinstance(lora_alpha_list, (list, tuple)) and idx < len(lora_alpha_list):
            fallback_alpha = lora_alpha_list[idx]
        elif lora_alpha_list is not None:
            fallback_alpha = lora_alpha_list
        full_delta = load_scaled_lora_delta_dict_local(
            lora_path_dict[task_name],
            device="cpu",
            fallback_alpha=fallback_alpha,
            fallback_rank=rank,
        )
        all_deltas[task_name] = {}
        missing = []
        for key in target_keys:
            if key in full_delta:
                all_deltas[task_name][key] = full_delta[key]
            else:
                missing.append(key)
        if missing:
            print(f"[KnOTS_LINEAR_TALS_DRC][VLM][Warn] task={task_name}: missing {len(missing)} LoRA deltas.")
            for item in missing[:10]:
                print(f"    missing delta: {item}")
        print(f"[KnOTS_LINEAR_TALS_DRC][VLM] task={task_name}: kept {len(all_deltas[task_name])}/{len(target_keys)} target deltas.")
    return all_deltas


def get_task_samples_any(
    model_name,
    tokenizer,
    max_length,
    task_name,
    samples_num,
    select_long,
    seed,
    shuffle,
    if_balance,
):
    if is_blip_model(model_name):
        batch = get_iteris_samples(
            model_name=model_name,
            tokenizer=tokenizer,
            max_length=max_length,
            task_name=task_name,
            samples_num=samples_num,
            select_long=select_long,
            seed=seed,
            shuffle=shuffle,
            if_balance=if_balance,
        )
        return {
            "pixel_values": tensor_to_cuda(batch["pixel_values"]),
            "input_ids": tensor_to_cuda(batch["input_ids"]),
            "attention_mask": tensor_to_cuda(batch["attention_mask"]),
        }

    return get_task_samples(
        model_name=model_name,
        tokenizer=tokenizer,
        max_length=max_length,
        task_name=task_name,
        samples_num=samples_num,
        select_long=select_long,
        seed=seed,
        shuffle=shuffle,
        if_balance=if_balance,
    )


def collect_features_by_position_any(
    model,
    model_name,
    inject_position,
    target_keys,
    sample_batch=None,
    input_ids=None,
    attention_mask=None,
    max_length=None,
):
    if not is_blip_model(model_name):
        return collect_features_by_position(
            model=model,
            inject_position=inject_position,
            target_keys=target_keys,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    if inject_position != "lora_input":
        raise ValueError("[KnOTS_LINEAR_TALS_DRC][VLM] only lora_input feature collection is supported.")

    modules = dict(model.named_modules())
    accum = {}
    counts = {}
    handles = []

    def make_pre_hook(name):
        def hook_fn(module, inputs):
            if inputs is None or len(inputs) == 0:
                return
            x = inputs[0].detach().float().cpu()
            # BLIP text-decoder self-attention projections usually receive
            # [batch, seq_len, hidden_dim]. TALS expects a vector with the
            # same dimension as ΔW input, so average over batch/tokens/calls.
            if x.dim() >= 2:
                x_vec = x.reshape(-1, x.shape[-1]).mean(dim=0)
            else:
                x_vec = x.reshape(-1)
            if name not in accum:
                accum[name] = x_vec
                counts[name] = 1
            else:
                accum[name] += x_vec
                counts[name] += 1
        return hook_fn

    for key in target_keys:
        if key not in modules:
            print(f"[KnOTS_LINEAR_TALS_DRC][VLM][Warn] cannot hook missing module: {key}")
            continue
        handles.append(modules[key].register_forward_pre_hook(make_pre_hook(key)))

    with torch.no_grad():
        model.generate(
            pixel_values=sample_batch["pixel_values"],
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            max_length=max_length,
        )

    for h in handles:
        h.remove()

    features = {}
    for key, val in accum.items():
        features[key] = (val / max(counts.get(key, 1), 1)).float().cpu()

    for key in list(features.keys())[:3]:
        print(f"[KnOTS_LINEAR_TALS_DRC][VLM][FeatureShape] {key}: {tuple(features[key].shape)}")
    return features


def load_single_lora_dense_model_any(model_name, lora_path, rank):
    if is_blip_model(model_name):
        model = construct_base_model_any(model_name)
        model = merge_peft(model, model_name, lora_path, rank)
        return model
    return load_single_lora_dense_model(
        model_name=model_name,
        lora_path=lora_path,
        rank=rank,
    )


def get_primary_metric_any(task_name, eval_result, task_type=None):
    if task_type == "TASKS_blip_base":
        style_acc = float(eval_result.get("acc", eval_result.get("style_acc", 0.0)))
        return "style_accuracy", style_acc, style_acc

    eval_accuracy = eval_result.get("eval_accuracy", "")
    eval_mcc = eval_result.get("eval_MCC", "")

    if task_name == "cola":
        primary_metric_name = "matthews_correlation"
        primary_metric_value = float(eval_mcc)
        normalized_metric = (primary_metric_value + 1.0) / 2.0
    else:
        primary_metric_name = "accuracy"
        primary_metric_value = float(eval_accuracy)
        normalized_metric = primary_metric_value

    return primary_metric_name, primary_metric_value, normalized_metric


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
    alpha,
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
            float(alpha),
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





def load_required_coarse_model_from_dir(model_dir, model_label="coarse"):
    """
    Load a previously selected best coarse merged model.

    This is intentionally strict: if the expected directory does not exist,
    we raise an error instead of silently rebuilding the coarse model with
    fixed/default hyperparameters.
    """
    if model_dir is None or str(model_dir).strip() == "":
        raise ValueError(f"[KnOTS_LINEAR_TALS_DRC] Empty {model_label} model directory.")
    model_dir = str(model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"[KnOTS_LINEAR_TALS_DRC] Required {model_label} model dir not found: {model_dir}\n"
            f"Please run the corresponding KnOTS+Linear baseline first and save the best coarse model."
        )
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(
            f"[KnOTS_LINEAR_TALS_DRC] {model_label} model dir exists but config.json is missing: {model_dir}\n"
            f"This does not look like a valid HuggingFace saved model directory."
        )
    print(f"[KnOTS_LINEAR_TALS_DRC] Loading required {model_label} coarse model from: {model_dir}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    return model


def make_loaded_fusion_stats():
    return {
        "fusion_iter_time_avg_sec": 0.0,
        "fusion_iter_time_max_sec": 0.0,
        "fusion_peak_vram_avg_mb": 0.0,
        "fusion_peak_vram_max_mb": 0.0,
    }


def parse_float_list(value, default=None):
    if default is None:
        default = [0.0, 0.03, 0.1, 0.2, 0.3, 0.5]

    if value is None:
        return default

    if isinstance(value, list):
        return [float(x) for x in value]

    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]

    raise ValueError(f"Unsupported float list type: {type(value)}")


def append_alpha_search_row(csv_path, row):
    header = [
        "experiment_id", "method", "pair_name", "evaluated_task",
        "alpha", "primary_metric_name", "primary_metric_value",
        "normalized_metric", "eval_accuracy", "eval_mcc", "eval_f1",
        "eval_loss", "eval_runtime", "eval_peak_vram_mb",
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    lock_path = csv_path + ".lock"

    with open(lock_path, "w", encoding="utf-8") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        try:
            need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if need_header:
                    writer.writerow(header)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(lockf, fcntl.LOCK_UN)


def get_dense_coarse_delta_w_dict(model_name, coarse_model, target_keys):
    """
    For non-linear / stochastic coarse merge methods such as KnOTS+Linear, the coarse
    update is extracted from dense weights:

        ΔW_c = W_KnOTS_LINEAR - W_base

    This function extracts ΔW_c for every LoRA target module key.
    """
    print("[KnOTS_LINEAR_TALS_DRC] Building coarse dense ΔW_c = W_KnOTS_LINEAR - W_base ...")
    base_model = construct_base_model_any(model_name)

    base_params = dict(base_model.named_parameters())
    coarse_params = dict(coarse_model.named_parameters())

    delta_dict = {}
    missing_report = []

    with torch.no_grad():
        for key in target_keys:
            param_name = key + ".weight"
            if param_name not in base_params or param_name not in coarse_params:
                missing_report.append(param_name)
                continue

            base_w = base_params[param_name].detach().float().cpu()
            coarse_w = coarse_params[param_name].detach().float().cpu()
            delta_dict[key] = coarse_w - base_w

    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    if missing_report:
        print(f"[KnOTS_LINEAR_TALS_DRC][Warn] Missing dense params for {len(missing_report)} targets.")
        for item in missing_report[:10]:
            print(f"    missing: {item}")

    print(f"[KnOTS_LINEAR_TALS_DRC] Built dense coarse delta for {len(delta_dict)}/{len(target_keys)} targets.")
    return delta_dict


def build_task_specific_tals_directions_for_knots_linear(
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
    tals_use_layer_weight=True,
    tals_layer_weight_score="ler_act",
    tals_layer_weight_norm="mean_one",
    tals_layer_weight_clip_min=None,
    tals_layer_weight_clip_max=None,
):
    """
    Build TALS-LER directions for KnOTS+Linear coarse merged model.

    Main difference from Linear_TALS_DRC:
        Linear_TALS_DRC uses ΔW_c = sum_j λ_j ΔW_j.
        Here we use ΔW_c = W_KnOTS_LINEAR - W_base.
    """
    selected_target_keys = select_drc_targets_any(
        model_name=model_name,
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
            "KnOTS_LINEAR_TALS_DRC currently supports only drc_inject_position='lora_input', "
            "because TALS uses input-side singular vectors of LoRA target-module ΔW."
        )

    all_lora_deltas = build_all_task_lora_deltas_any(
        task_targets=task_targets,
        lora_path_dict=lora_path_dict,
        target_keys=selected_target_keys,
        rank=rank,
        lora_alpha_list=lora_alpha_list,
        model_name=model_name,
    )

    coarse_delta_w_dict = get_dense_coarse_delta_w_dict(
        model_name=model_name,
        coarse_model=coarse_model,
        target_keys=selected_target_keys,
    )

    all_task_directions = {}
    direction_stats = {}

    source = str(tals_subspace_source).lower().strip()
    if source not in ["missing", "single", "random"]:
        raise ValueError(
            f"Unsupported tals_subspace_source={tals_subspace_source}. "
            "Use 'missing', 'single', or 'random'."
        )

    for task_name in task_targets:
        print(f"\n[KnOTS_LINEAR_TALS_DRC] Building task-specific TALS-LER direction for task = {task_name}")

        sample_batch = get_task_samples_any(
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

        print(f"[KnOTS_LINEAR_TALS_DRC] Collect base features on {task_name} samples...")
        base_model = construct_base_model_any(model_name).to("cuda")
        if is_blip_model(model_name):
            base_features = collect_features_by_position_any(
                model=base_model,
                model_name=model_name,
                inject_position=inject_position,
                target_keys=selected_target_keys,
                sample_batch=sample_batch,
                max_length=max_length,
            )
        else:
            input_ids, attention_mask = sample_batch
            base_features = collect_features_by_position_any(
                model=base_model,
                model_name=model_name,
                inject_position=inject_position,
                target_keys=selected_target_keys,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        del base_model
        torch.cuda.empty_cache()
        gc.collect()

        print(f"[KnOTS_LINEAR_TALS_DRC] Collect single-LoRA features for {task_name}...")
        single_lora_path = lora_path_dict[task_name]
        single_model = load_single_lora_dense_model_any(
            model_name=model_name,
            lora_path=single_lora_path,
            rank=rank,
        ).to("cuda")
        if is_blip_model(model_name):
            single_features = collect_features_by_position_any(
                model=single_model,
                model_name=model_name,
                inject_position=inject_position,
                target_keys=selected_target_keys,
                sample_batch=sample_batch,
                max_length=max_length,
            )
        else:
            input_ids, attention_mask = sample_batch
            single_features = collect_features_by_position_any(
                model=single_model,
                model_name=model_name,
                inject_position=inject_position,
                target_keys=selected_target_keys,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        del single_model
        torch.cuda.empty_cache()
        gc.collect()

        print(f"[KnOTS_LINEAR_TALS_DRC] Collect KnOTS+Linear merged features on {task_name} samples...")
        if is_blip_model(model_name):
            knots_linear_features = collect_features_by_position_any(
                model=coarse_model,
                model_name=model_name,
                inject_position=inject_position,
                target_keys=selected_target_keys,
                sample_batch=sample_batch,
                max_length=max_length,
            )
        else:
            input_ids, attention_mask = sample_batch
            knots_linear_features = collect_features_by_position_any(
                model=coarse_model,
                model_name=model_name,
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
            "missing_base": [],
            "missing_single": [],
            "missing_knots_linear": [],
            "missing_lora_delta": [],
            "missing_coarse_delta": [],
            "tals_filter_failed": [],
            "zero_or_tiny_norm": [],
        }

        for key in selected_target_keys:
            if key not in base_features:
                missing_report["missing_base"].append(key)
                continue
            if key not in single_features:
                missing_report["missing_single"].append(key)
                continue
            if key not in knots_linear_features:
                missing_report["missing_knots_linear"].append(key)
                continue

            single_shift = single_features[key] - base_features[key]
            knots_linear_shift = knots_linear_features[key] - base_features[key]
            delta = single_shift - knots_linear_shift  # h_single - h_knots_linear

            task_delta_dict = all_lora_deltas.get(task_name, {})
            if key not in task_delta_dict and source != "random":
                missing_report["missing_lora_delta"].append((task_name, key))
                continue

            source_delta_w = None
            if source == "missing":
                if key not in coarse_delta_w_dict:
                    missing_report["missing_coarse_delta"].append(key)
                    continue
                task_delta_w = task_delta_dict[key].float()
                coarse_delta_w = coarse_delta_w_dict[key].float()
                if task_delta_w.shape != coarse_delta_w.shape:
                    missing_report["missing_coarse_delta"].append(
                        (key, f"shape mismatch task_delta={tuple(task_delta_w.shape)}, coarse_delta={tuple(coarse_delta_w.shape)}")
                    )
                    continue
                source_delta_w = task_delta_w - coarse_delta_w
            elif source == "single":
                source_delta_w = task_delta_dict[key].float()
            elif source == "random":
                source_delta_w = None

            random_seed = int(seed) + stable_int_hash(f"{task_name}|{key}|{tals_rank}|KnOTS_LINEAR_TALS_DRC")

            tals_delta, tals_stats = apply_tals_filter_to_activation_residual(
                activation_residual=delta,
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
                        f"[KnOTS_LINEAR_TALS_DRC][Warn] TALS filter failed at {key}; "
                        "fallback to base DRC direction."
                    )
                    tals_delta = delta.float().cpu()
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

        if tals_use_layer_weight:
            task_direction, task_stats, lw_summary = apply_ler_layerwise_reweight(
                task_direction=task_direction,
                task_stats=task_stats,
                score_type=tals_layer_weight_score,
                norm_type=tals_layer_weight_norm,
                clip_min=tals_layer_weight_clip_min,
                clip_max=tals_layer_weight_clip_max,
                eps=tals_eps,
            )
            print(f"[KnOTS_LINEAR_TALS_DRC][LayerWeight] task={task_name}, summary={lw_summary}")
            task_stats["__layer_weight_summary__"] = lw_summary

        print(f"[KnOTS_LINEAR_TALS_DRC][Debug] Missing/skip report for task={task_name}:")
        for reason, items in missing_report.items():
            print(f"  - {reason}: {len(items)}")
            for item in items[:10]:
                print(f"      {item}")

        print(
            f"[KnOTS_LINEAR_TALS_DRC] Task {task_name}: built directions for "
            f"{len(task_direction)}/{len(selected_target_keys)} targets."
        )

        all_task_directions[task_name] = task_direction
        direction_stats[task_name] = task_stats

    return all_task_directions, direction_stats, selected_target_keys


def get_primary_metric(task_name, eval_result):
    # Backward-compatible wrapper for old GLUE call sites.
    return get_primary_metric_any(task_name=task_name, eval_result=eval_result, task_type=None)


def short_hash(text, length=12):
    import hashlib
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()[:length]

def make_knots_linear_tals_cache_path(
    cache_dir,
    method_name,
    pair_name,
    inject_position,
    target_layers,
    target_modules,
    merge_method,
    drop_rate,
    scaling_coeffs,
    use_rescale,
    knots_linear_seed,
    topK,
    sign_resolve_mode,
    merging_type,
    tals_subspace_source,
    tals_rank,
    tals_gamma,
    tals_weight_norm,
    tals_svd_center,
    tals_use_layer_weight,
    tals_layer_weight_score,
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
        f"method={method_name}|"
        f"pair={pair_name}|"
        f"inject={inject_position}|"
        f"layers={target_layers}|"
        f"modules={normalize_target_modules(target_modules)}|"
        f"knots_linear_merge_method={merge_method}|"
        f"knots_linear_drop_rate={drop_rate}|"
        f"knots_linear_scaling_coeffs={scaling_coeffs}|"
        f"knots_linear_use_rescale={use_rescale}|"
        f"knots_linear_seed={knots_linear_seed}|"
        f"knots_linear_topK={topK}|"
        f"knots_linear_sign_resolve_mode={sign_resolve_mode}|"
        f"knots_linear_merging_type={merging_type}|"
        f"tals_subspace_source={tals_subspace_source}|"
        f"tals_rank={tals_rank}|"
        f"tals_gamma={tals_gamma}|"
        f"tals_weight_norm={tals_weight_norm}|"
        f"tals_svd_center={tals_svd_center}|"
        f"tals_use_layer_weight={tals_use_layer_weight}|"
        f"tals_layer_weight_score={tals_layer_weight_score}"
    )

    h = short_hash(cfg_text, length=12)

    safe_method = str(method_name).replace("/", "_")[:40]
    filename = (
        f"{safe_method}_{pair_name}_{layer_tag}_{module_tag}_"
        f"knots_linearp{str(drop_rate).replace('.', 'p')}_"
        f"s{knots_linear_seed}_k{topK}_"
        f"tals{tals_subspace_source}_r{tals_rank}_g{str(tals_gamma).replace('.', 'p')}_"
        f"lw{int(bool(tals_use_layer_weight))}_{h}.pt"
    )

    return os.path.join(cache_dir, filename)


def build_tals_cfg_string(
    merge_method,
    drop_rate,
    scaling_coeffs,
    use_rescale,
    knots_linear_seed,
    topK,
    sign_resolve_mode,
    merging_type,
    knots_linear_do_search,
    drc_inject_position,
    drc_alpha,
    drc_alpha_search,
    drc_alpha_candidates,
    drc_samples_per_task,
    drc_target_part,
    drc_target_modules,
    drc_target_layers,
    drc_normalize_direction,
    drc_use_hidden_norm_scale,
    drc_rebuild_cache,
    tals_subspace_source,
    tals_rank,
    tals_gamma,
    tals_weight_norm,
    tals_svd_center,
    tals_fallback_to_base,
    tals_use_layer_weight,
    tals_layer_weight_score,
    tals_layer_weight_norm,
    tals_layer_weight_clip_min,
    tals_layer_weight_clip_max,
    cache_path,
):
    return (
        f"KnOTS+Linear: merge_method={merge_method}|drop_rate={drop_rate}|scaling_coeffs={scaling_coeffs}|"
        f"use_rescale={use_rescale}|knots_linear_seed={knots_linear_seed}|topK={topK}|"
        f"sign_resolve_mode={sign_resolve_mode}|merging_type={merging_type}|search={knots_linear_do_search}; "
        f"TALS-LER: inject_position={drc_inject_position}; "
        f"alpha={drc_alpha}; alpha_search={drc_alpha_search}; "
        f"alpha_candidates={drc_alpha_candidates}; "
        f"samples_per_task={drc_samples_per_task}; "
        f"target_part={drc_target_part}; "
        f"target_modules={normalize_target_modules(drc_target_modules)}; "
        f"target_layers={drc_target_layers}; "
        f"normalize={drc_normalize_direction}; "
        f"use_hidden_norm_scale={drc_use_hidden_norm_scale}; "
        f"rebuild_cache={drc_rebuild_cache}; "
        f"subspace_source={tals_subspace_source}; "
        f"tals_rank={tals_rank}; "
        f"tals_gamma={tals_gamma}; "
        f"tals_weight_norm={tals_weight_norm}; "
        f"tals_svd_center={tals_svd_center}; "
        f"tals_fallback_to_base={tals_fallback_to_base}; "
        f"use_layer_weight={tals_use_layer_weight}; "
        f"layer_weight_score={tals_layer_weight_score}; "
        f"layer_weight_norm={tals_layer_weight_norm}; "
        f"layer_weight_clip_min={tals_layer_weight_clip_min}; "
        f"layer_weight_clip_max={tals_layer_weight_clip_max}; "
        f"cache_path={cache_path}"
    )





def main():
    parser = argparse.ArgumentParser(description="KnOTS+Linear coarse model + TALS-DRC for NLP/VLM")
    parser.add_argument("--task_type", type=str, default="GLUE_t5")
    parser.add_argument("--config", type=str, default="config/methods-config/iteris-config.yaml")
    args = parser.parse_args()

    task_type = args.task_type

    with open(args.config, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    seed = int(config_data.get("seed", 42))
    set_seed(seed)

    if task_type not in config_data:
        raise ValueError(f"Cannot find task_type={task_type} in config.")

    task_cfg = config_data[task_type]

    model_name = task_cfg["model_name"]
    task_targets = task_cfg["task_targets"]
    rank = int(task_cfg.get("rank", 8))
    lora_alpha = task_cfg.get("lora_alpha", [32 for _ in task_targets])
    max_length = int(task_cfg.get("max_length", 512))
    per_device_eval_batch_size = int(task_cfg.get("per_device_eval_batch_size", 8))

    tokenizer = (
        AutoTokenizer.from_pretrained(model_name)
        if not is_blip_model(model_name)
        else AutoProcessor.from_pretrained(model_name)
    )

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

    print(f"[KnOTS_LINEAR_TALS_DRC] lora_root = {lora_root}")
    lora_path_dict = get_loras_path(task_type=task_type, model_name=model_name, lora_root=lora_root)
    missing_lora = [task for task in task_targets if task not in lora_path_dict]
    if missing_lora:
        raise ValueError(f"[KnOTS_LINEAR_TALS_DRC] Missing LoRA path for tasks: {missing_lora}")

    for task in task_targets:
        adapter_path = os.path.join(lora_path_dict[task], "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"[KnOTS_LINEAR_TALS_DRC] Cannot find adapter_model.safetensors for task={task}: {adapter_path}"
            )

    pair_name = "_".join(task_targets)

    # Load existing KnOTS+Linear coarse model. Do not rebuild it here.
    load_coarse = bool(task_cfg.get("knots_linear_tals_load_coarse_from_dir", True))
    coarse_model_dir = task_cfg.get(
        "knots_linear_tals_coarse_model_dir",
        task_cfg.get(
            "knots_linear_merged_model_dir",
            f"merged_model/KnOTS_LINEAR_{pair_name}",
        ),
    )

    if not load_coarse:
        raise ValueError(
            "[KnOTS_LINEAR_TALS_DRC] knots_linear_tals_load_coarse_from_dir must be True. "
            "Please run KnOTS_Linear.py first and load its saved coarse model here."
        )

    print("[KnOTS_LINEAR_TALS_DRC] knots_linear_tals_load_coarse_from_dir = True")
    print(f"[KnOTS_LINEAR_TALS_DRC] Expected coarse model dir = {coarse_model_dir}")

    coarse_model = load_required_coarse_model_from_dir_any(
        model_name=model_name,
        model_dir=coarse_model_dir,
        model_label="KnOTS+Linear",
    ).to("cuda")

    print("[KnOTS_LINEAR_TALS_DRC] Loaded saved KnOTS+Linear coarse model. No KnOTS+Linear merge is run in this script.")

    # Method / experiment naming.
    base_method_name = task_cfg.get(
        "knots_linear_tals_method_name",
        task_cfg.get("tals_method_name", "KnOTS_LINEAR_TALS_DRC_LER"),
    )
    method_name = f"{base_method_name}_loaded_{os.path.basename(str(coarse_model_dir)).replace('/', '_')}"
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{method_name}_{pair_name}"

    # Results files.
    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"[KnOTS_LINEAR_TALS_DRC] results_dir = {results_dir}")

    results_csv = os.path.join(results_dir, "pair_merge_results.csv")
    registry_csv = os.path.join(results_dir, "experiment_registry.csv")
    alpha_search_csv = os.path.join(results_dir, "drc_alpha_search_results.csv")
    vlm_results_csv = os.path.join(results_dir, "vlm_caption_results.csv")
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
                "alpha",
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

    # TALS-DRC config.
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

    tals_subspace_source = str(task_cfg.get("tals_subspace_source", "missing")).lower().strip()
    if tals_subspace_source not in ["missing", "single", "random"]:
        raise ValueError(
            f"Unsupported tals_subspace_source={tals_subspace_source}. "
            "Use 'missing', 'single', or 'random'."
        )
    tals_rank = int(task_cfg.get("tals_rank", task_cfg.get("tals_subspace_rank", 8)))
    tals_gamma = float(task_cfg.get("tals_gamma", 0.5))
    tals_eps = float(task_cfg.get("tals_eps", 1e-6))
    tals_weight_norm = task_cfg.get("tals_weight_norm", "mean")
    tals_svd_center = bool(task_cfg.get("tals_svd_center", False))
    tals_fallback_to_base = bool(task_cfg.get("tals_fallback_to_base", False))
    tals_use_layer_weight = bool(task_cfg.get("tals_use_layer_weight", True))
    tals_layer_weight_score = task_cfg.get("tals_layer_weight_score", "ler_act")
    tals_layer_weight_norm = task_cfg.get("tals_layer_weight_norm", "mean_one")
    tals_layer_weight_clip_min = parse_optional_float(task_cfg.get("tals_layer_weight_clip_min", None))
    tals_layer_weight_clip_max = parse_optional_float(task_cfg.get("tals_layer_weight_clip_max", None))

    cache_dir = task_cfg.get(
        "knots_linear_tals_cache_dir",
        task_cfg.get("drc_cache_dir", "direction_cache_knots_linear_tals_ler"),
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Use KnOTS+Linear settings only for cache identity, not for rebuilding.
    knots_linear_beta = float(task_cfg.get("knots_linear_beta", 0.5))
    knots_svd_keep_rank = task_cfg.get("knots_svd_keep_rank", None)
    knots_linear_method = task_cfg.get("knots_linear_method_name", "KnOTS_LINEAR")

    cache_path = make_knots_linear_tals_cache_path(
        cache_dir=cache_dir,
        method_name=method_name,
        pair_name=pair_name,
        inject_position=drc_inject_position,
        target_layers=drc_target_layers,
        target_modules=drc_target_modules,
        merge_method=f"loaded_{knots_linear_method}",
        drop_rate=0.0,
        scaling_coeffs=knots_linear_beta,
        use_rescale=False,
        knots_linear_seed=seed,
        topK=knots_svd_keep_rank if knots_svd_keep_rank is not None else 0,
        sign_resolve_mode="none",
        merging_type="linear_knots_blend",
        tals_subspace_source=tals_subspace_source,
        tals_rank=tals_rank,
        tals_gamma=tals_gamma,
        tals_weight_norm=tals_weight_norm,
        tals_svd_center=tals_svd_center,
        tals_use_layer_weight=tals_use_layer_weight,
        tals_layer_weight_score=tals_layer_weight_score,
    )

    print(f"[KnOTS_LINEAR_TALS_DRC] drc_inject_position = {drc_inject_position}")
    print(f"[KnOTS_LINEAR_TALS_DRC] drc_alpha = {drc_alpha}")
    print(f"[KnOTS_LINEAR_TALS_DRC] drc_alpha_search = {drc_alpha_search}")
    print(f"[KnOTS_LINEAR_TALS_DRC] drc_alpha_candidates = {drc_alpha_candidates}")
    print(f"[KnOTS_LINEAR_TALS_DRC] drc_samples_per_task = {drc_samples_per_task}")
    print(f"[KnOTS_LINEAR_TALS_DRC] drc_target_part = {drc_target_part}")
    print(f"[KnOTS_LINEAR_TALS_DRC] drc_target_modules = {drc_target_modules}")
    print(f"[KnOTS_LINEAR_TALS_DRC] drc_target_layers = {drc_target_layers}")
    print(f"[KnOTS_LINEAR_TALS_DRC] drc_normalize_direction = {drc_normalize_direction}")
    print(f"[KnOTS_LINEAR_TALS_DRC] tals_subspace_source = {tals_subspace_source}")
    print(f"[KnOTS_LINEAR_TALS_DRC] tals_rank = {tals_rank}")
    print(f"[KnOTS_LINEAR_TALS_DRC] tals_gamma = {tals_gamma}")
    print(f"[KnOTS_LINEAR_TALS_DRC] tals_weight_norm = {tals_weight_norm}")
    print(f"[KnOTS_LINEAR_TALS_DRC] tals_svd_center = {tals_svd_center}")
    print(f"[KnOTS_LINEAR_TALS_DRC] tals_use_layer_weight = {tals_use_layer_weight}")
    print(f"[KnOTS_LINEAR_TALS_DRC] tals_layer_weight_score = {tals_layer_weight_score}")
    print(f"[KnOTS_LINEAR_TALS_DRC] tals_layer_weight_norm = {tals_layer_weight_norm}")
    print(f"[KnOTS_LINEAR_TALS_DRC] cache_path = {cache_path}")

    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        if (not drc_rebuild_cache) and os.path.exists(cache_path):
            print(f"[KnOTS_LINEAR_TALS_DRC] Load TALS direction cache from: {cache_path}")
            cache = torch.load(cache_path, map_location="cpu")
            drc_directions = cache["directions"]
            direction_stats = cache.get("direction_stats", {})
            selected_target_keys = cache.get("selected_target_keys", [])
        else:
            drc_directions, direction_stats, selected_target_keys = build_task_specific_tals_directions_for_knots_linear(
                model_name=model_name,
                tokenizer=tokenizer,
                task_targets=task_targets,
                lora_path_dict=lora_path_dict,
                coarse_model=coarse_model,
                rank=rank,
                max_length=max_length,
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
                lora_alpha_list=lora_alpha,
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
                    "base_method": "KnOTS+Linear",
                    "coarse_model_dir": coarse_model_dir,
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
            print(f"[KnOTS_LINEAR_TALS_DRC] Saved TALS direction cache to: {cache_path}")

        tals_build_time = round(time.time() - start_time, 4)
        tals_peak_vram_mb = (
            round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)
            if torch.cuda.is_available()
            else 0.0
        )

        normalized_metrics = []
        selected_alpha_by_task = {}

        for task_name in task_targets:
            print(f"\n[Eval] Evaluating KnOTS+Linear + TALS-LER on {task_name}...")

            if task_name not in drc_directions:
                raise ValueError(f"Cannot find TALS direction for task: {task_name}")

            candidate_alphas = drc_alpha_candidates if drc_alpha_search else [drc_alpha]
            best_record = None
            best_eval_result = None

            for alpha in candidate_alphas:
                print(f"[KnOTS_LINEAR_TALS_DRC][AlphaSearch] task={task_name}, alpha={alpha}")

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
                    max_length=max_length,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                )

                remove_hooks(handles)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                primary_metric_name, primary_metric_value, normalized_metric = get_primary_metric_any(
                    eval_result=eval_result,
                    task_name=task_name,
                    task_type=task_type,
                )

                eval_accuracy = eval_result.get("eval_accuracy", primary_metric_value if task_type == "TASKS_blip_base" else "")
                eval_mcc = eval_result.get("eval_MCC", "")
                eval_f1 = eval_result.get("eval_f1-score", "")
                eval_loss = eval_result.get("eval_loss", "")
                eval_runtime = eval_result.get("eval_runtime", eval_result.get("eval_wall_time_sec", ""))
                eval_sps = eval_result.get("eval_samples_per_second", "")
                eval_stepsps = eval_result.get("eval_steps_per_second", "")
                eval_peak_vram_mb = eval_result.get("eval_peak_vram_mb", "")

                append_alpha_search_row(
                    alpha_search_csv,
                    [
                        experiment_id, method_name, pair_name, task_name,
                        float(alpha), primary_metric_name, primary_metric_value,
                        normalized_metric, eval_accuracy, eval_mcc, eval_f1,
                        eval_loss, eval_runtime, eval_peak_vram_mb,
                    ],
                )

                if task_type == "TASKS_blip_base":
                    append_vlm_caption_row(
                        csv_path=vlm_results_csv,
                        experiment_id=experiment_id,
                        method_name=method_name,
                        pair_name=pair_name,
                        task_targets=task_targets,
                        task_name=task_name,
                        alpha=float(alpha),
                        eval_result=eval_result,
                        merged_model_dir=coarse_model_dir,
                        log_file=log_file,
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
                    best_eval_result = eval_result

            selected_alpha_by_task[task_name] = best_record["alpha"]
            normalized_metrics.append(best_record["normalized_metric"])

            print(
                f"[KnOTS_LINEAR_TALS_DRC][AlphaSearch][Best] task={task_name}, "
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
                    coarse_model_dir,
                    log_file,
                    "",
                ],
            )

        pair_avg_normalized_metric = float(np.mean(normalized_metrics))
        selected_alpha_str = "|".join(
            [f"{task}:{selected_alpha_by_task.get(task, drc_alpha)}" for task in task_targets]
        )

        cfg_str = (
            f"base=KnOTS+Linear|coarse_model_dir={coarse_model_dir}|"
            f"knots_linear_beta={knots_linear_beta}|knots_svd_keep_rank={knots_svd_keep_rank}|"
            f"inject_position={drc_inject_position}|alpha_search={drc_alpha_search}|"
            f"alpha_candidates={drc_alpha_candidates}|selected_alpha_by_task={selected_alpha_str}|"
            f"samples_per_task={drc_samples_per_task}|target_part={drc_target_part}|"
            f"target_modules={normalize_target_modules(drc_target_modules)}|target_layers={drc_target_layers}|"
            f"normalize={drc_normalize_direction}|use_hidden_norm_scale={drc_use_hidden_norm_scale}|"
            f"subspace_source={tals_subspace_source}|tals_rank={tals_rank}|tals_gamma={tals_gamma}|"
            f"tals_weight_norm={tals_weight_norm}|tals_svd_center={tals_svd_center}|"
            f"tals_use_layer_weight={tals_use_layer_weight}|"
            f"tals_layer_weight_score={tals_layer_weight_score}|"
            f"tals_layer_weight_norm={tals_layer_weight_norm}"
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
                cfg_str,
                rank,
                str(lora_alpha),
                tals_build_time,
                tals_build_time,
                tals_build_time,
                tals_peak_vram_mb,
                tals_peak_vram_mb,
                pair_avg_normalized_metric,
                "success",
                "",
            ],
        )

        print(f"\n[Done] KnOTS_LINEAR_TALS_DRC finished for pair: {pair_name}")
        print(f"[Done] selected_alpha_by_task = {selected_alpha_by_task}")
        print(f"[Done] pair_avg_normalized_metric = {pair_avg_normalized_metric}")
        print(f"[Done] pair-task results: {results_csv}")
        if task_type == "TASKS_blip_base":
            print(f"[Done] VLM caption results: {vlm_results_csv}")

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)

        append_csv_row(
            registry_csv,
            [
                experiment_id,
                "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise",
                method_name,
                model_name,
                pair_name,
                "|".join(task_targets),
                f"base=KnOTS+Linear|coarse_model_dir={coarse_model_dir}",
                rank,
                str(lora_alpha),
                "", "", "", "", "",
                "",
                "failed",
                error_msg,
            ],
        )
        raise e


if __name__ == "__main__":
    main()
