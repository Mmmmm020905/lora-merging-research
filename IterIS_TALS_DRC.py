import gc
import os
import pdb
import yaml
import time
import torch
import random
import argparse
import itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from safetensors import safe_open
from peft import PeftModel
from sklearn.metrics import f1_score
from eval_model import eval_iteris_model
from get_midfeatures import T5WithHooks, BartWithHooks, BlipWithHook
from torch.optim.lr_scheduler import StepLR
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, BartForConditionalGeneration, AutoTokenizer, AutoProcessor, BlipForConditionalGeneration
from get_midfeatures import get_all_midfeatures, get_samples, get_pretrain_matrix, get_lora_matrix
from get_midfeatures import get_samples as get_iteris_samples
import csv
from datetime import datetime
import fcntl
import traceback

GLUE_task_name = [
    "mnli", "rte",
    "cola", "sst2", "qqp",
    "qnli", "mrpc", "wnli",
]
EMOTION_task_name = [
    "emoint", "emotion-cause",
    "tec", "isear",
]
SENTICAP_task_name = ['positive', 'negative']
FlickrStyle10k_task_name = ["roman", "humor"]
TASKS_blip_base = ['positive', 'negative', "roman", "humor"]

def get_loras_path(task_type, model_name):
    lora_path_dict = {}

    if 't5' in str(model_name).lower() and task_type == "GLUE_t5":
        lora_path_dict["cola"] = "best_LoRA/T5-COLA-LoRA"
        lora_path_dict["sst2"] = "best_LoRA/T5-SST2-LoRA"
        lora_path_dict["rte"]  = "best_LoRA/T5-RTE-LoRA"
        lora_path_dict["qnli"] = "best_LoRA/T5-QNLI-LoRA"
        lora_path_dict["qqp"]  = "best_LoRA/T5-QQP-LoRA"
        lora_path_dict["mrpc"] = "best_LoRA/T5-MRPC-LoRA"
        lora_path_dict["mnli"] = "best_LoRA/T5-MNLI-LoRA"
        lora_path_dict["wnli"] = "best_LoRA/T5-WNLI-LoRA"

    if task_type == "TASKS_blip_base":
        lora_path_dict["positive"] = "loras/SENTICAP-lora-blip/positive"
        lora_path_dict["negative"] = "loras/SENTICAP-lora-blip/negative"
        # 预留 FlickrStyle10k；没有对应 LoRA 时不会影响 positive/negative。
        lora_path_dict["roman"] = "loras/FlickrStyle10k-lora-blip/roman"
        lora_path_dict["humor"] = "loras/FlickrStyle10k-lora-blip/humor"

    return lora_path_dict

# Set all the seeds the same
def set_seed(seed):
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)  
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


def parse_float_list(value, default=None):
    """
    支持从 YAML 中读取 alpha candidates。

    可接受：
        [0.0, 0.03, 0.1]
        "0.0,0.03,0.1"
    """
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
    """
    记录每个 alpha 候选的评测结果。

    注意：
    - pair_merge_results.csv 只写每个 task 的 best alpha；
    - drc_alpha_search_results.csv 写所有候选 alpha。
    """
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

def reg_math(term, alpha):
    term_list = [term[i] + alpha[i] * torch.eye(term[i].size(0), dtype=term.dtype, device=term.device) for i in range(term.shape[0])]
    return torch.stack(term_list)

def solution_matrix(
    W_list, 
    X_list, 
    X_tilde_list, 
    ceof_list, 
    manual_ceof,
    alpha_1=1e-7, 
    alpha_2=1e-7,
    reg_ceof=5e-4,
):  
    with torch.no_grad():
        N = W_list.shape[0]
        manual_ceof = torch.tensor(manual_ceof).to('cuda')
        X_list, X_tilde_list = X_list.transpose(0,1).flatten(start_dim=1,end_dim=2), X_tilde_list.transpose(0,1).flatten(start_dim=1,end_dim=2)

        X_tilde_list = (1 - reg_ceof) * X_tilde_list + reg_ceof * X_list
        X_X_tilde = torch.matmul(X_list.transpose(-1,-2), X_tilde_list)
        X_X_tilde_norm = torch.norm(X_X_tilde, p='fro', dim=[-2,-1]) * alpha_1
        X_X_tilde = reg_math(X_X_tilde, X_X_tilde_norm)

        X_tilde_X_tilde = torch.matmul(X_tilde_list.transpose(-1,-2), X_tilde_list)
        X_tilde_X_tilde_norm = torch.norm(X_tilde_X_tilde, p='fro', dim=[-2,-1]) * alpha_2
        X_tilde_X_tilde = reg_math(X_tilde_X_tilde, X_tilde_X_tilde_norm)

        term1 = torch.sum(torch.matmul(W_list, X_X_tilde) * (ceof_list*manual_ceof).view(N,1,1), dim=0).double()
        term2 = torch.sum(X_tilde_X_tilde * (ceof_list*manual_ceof).view(N,1,1), dim=0).double()
        results = torch.linalg.solve(term2.t(), term1.t()).double().t()
        
        norm_value = ceof_list * torch.norm(torch.matmul(W_list, X_list.transpose(-1,-2)) - torch.matmul(torch.stack([results, results]).float(), X_list.transpose(-1,-2)), dim=[-2, -1])**2
        # print(norm_value)
        return results.to('cpu')

def update_param(
    seed,
    max_iter,
    lora_path,
    model_name,
    task_targets,
    manual_ceof,
    shuffle,
    with_pretrain_matrix=0,
    max_length=512,
    lora_alpha=[32,32],
    alpha_1=1e-7,
    alpha_2=1e-7,
    reg_ceof=5e-4,
    rank=8,
    select_long=40,
    inner_num=2,
    outer_num=10,
    samples_num=20,
    if_divide=True,
    if_balance=True,
    **generation_kwargs,
):
    input_ids_list, X_dict = get_all_midfeatures(
        rank=rank,
        seed=seed,
        select_long=select_long,
        lora_path=lora_path,
        model_name=model_name,
        max_length=max_length,
        task_targets=task_targets,
        if_divide=if_divide,
        if_balance=if_balance,
        shuffle=shuffle,
        inner_num=inner_num,
        outer_num=outer_num,
        samples_num=samples_num,
        **generation_kwargs,
    )
    
    pretrain_matrix_dict = get_pretrain_matrix(X_dict.keys(), model_name=model_name)
    # Via the names of LoRAs, get the pretrain model matrix

    lora_adapter_path_list = [
        lora_adapter_path + "/adapter_model.safetensors" for lora_adapter_path in lora_path
    ]
    tensors_lora = [safe_open(tensor_lora, framework='pt') for tensor_lora in lora_adapter_path_list]
    torch.cuda.empty_cache()
    X_tilde_dict = {}
    iter_peak_vram_mb = []
    iter_time_sec = []
    for iter_idx in range(max_iter):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        iter_start = time.time()

        tar_lora_list = {}
        print(f"-----------iter: {iter_idx}---------------")
        print("Calculate the opt solution...")
        with torch.no_grad():
            for idx in X_dict.keys():
                # print(idx)
                W_list, X_list = torch.stack(
                    [get_lora_matrix(model_name, tensors_lora[i], idx, lora_alpha[i], rank=rank, no_weight=True) for i in range(len(tensors_lora))]
                ).to('cuda'), X_dict[idx].to('cuda') # Get lora matrix and mid-features
                N = W_list.shape[0]  
                merge_W = W_list + pretrain_matrix_dict[idx].unsqueeze(0).repeat(N, 1, 1).to('cuda')
                ceof_list = torch.norm(merge_W, p='fro', dim=[-2,-1])**2 / \
                            torch.sum(torch.norm(torch.matmul(X_list, merge_W.transpose(1,2)), p='fro', dim=[-2,-1])**2, dim=0)
                # ceof_list = torch.tensor([1.0, 1.0]).to('cuda')
                if with_pretrain_matrix == 0:
                    tar_lora_list[idx] = solution_matrix(W_list, X_list, X_list, ceof_list, manual_ceof, alpha_1, alpha_2, reg_ceof).to('cpu') if iter_idx == 0 else \
                                        solution_matrix(W_list, X_list, X_tilde_dict[idx].to('cuda'), ceof_list, manual_ceof, alpha_1, alpha_2, reg_ceof).to('cpu')
                elif with_pretrain_matrix == 1:
                    tar_lora_list[idx] = solution_matrix(merge_W, X_list, X_list, ceof_list, manual_ceof, alpha_1, alpha_2, reg_ceof).to('cpu') if iter_idx == 0 else \
                                        solution_matrix(merge_W, X_list, X_tilde_dict[idx].to('cuda'), ceof_list, manual_ceof, alpha_1, alpha_2, reg_ceof).to('cpu')
                torch.cuda.empty_cache()
                gc.collect() 
        print("Calculation Done!")
        print("Loading and updating the original model...")
        model = None
        if 't5' in model_name:
            model = T5WithHooks.from_pretrained(model_name, lora_path=lora_path[0] + '/adapter_model.safetensors').to('cuda')
        elif 'bart' in model_name:
            model = BartWithHooks.from_pretrained(model_name, lora_path=lora_path[0] + '/adapter_model.safetensors').to('cuda')
        elif 'blip' in model_name:
            model = BlipWithHook.from_pretrained(model_name).to('cuda')
        # Updating the model
        number_update = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                # name like this: 'decoder.block.1.layer.0.SelfAttention.q.weight'
                if name[:-7] in tar_lora_list.keys(): #delete 'weight'
                    lora_matrix = tar_lora_list[name[:-7]].to('cuda')
                    if with_pretrain_matrix == 0:
                        param.copy_(lora_matrix + param)
                    elif with_pretrain_matrix == 1:
                        param.copy_(lora_matrix)
                    number_update += 1
        if number_update == len(tar_lora_list.keys()):
            print("All the targets which correspond to LoRAs are updated successfully!")
        else:
            print("Something got wrong...")
        torch.cuda.empty_cache()
        max_memory_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        iter_peak_vram_mb.append(round(max_memory_mb, 2))
        iter_time_sec.append(round(time.time() - iter_start, 4))

        print(f"Iter {iter_idx} peak VRAM: {max_memory_mb:.2f} MB", flush=True)

        if iter_idx == max_iter - 1:
            fusion_stats = {
                "iter_peak_vram_mb": iter_peak_vram_mb,
                "iter_time_sec": iter_time_sec,
                "fusion_peak_vram_avg_mb": round(float(np.mean(iter_peak_vram_mb)), 2),
                "fusion_peak_vram_max_mb": round(float(np.max(iter_peak_vram_mb)), 2),
                "fusion_iter_time_avg_sec": round(float(np.mean(iter_time_sec)), 4),
                "fusion_iter_time_max_sec": round(float(np.max(iter_time_sec)), 4),
            }
            return model, fusion_stats
        # Record the mid-features of updated model
        records_list = []
        if if_divide == True:
            assert inner_num * outer_num == len(input_ids_list[0])
            for input_ids in input_ids_list:
                print("Generating lora midfeatures...")
                dict_record_item = {}
                for i in range(outer_num):
                    with torch.no_grad():
                        outputs = model.generate(input_ids[i*inner_num:(i+1)*inner_num, :].to('cuda'))
                    temp_dict = dict(model.inputs_to_track.items())
                    dict_record_item = temp_dict if i == 0 else {key: torch.cat([value, temp_dict[key]], dim=0) for key, value in dict_record_item.items()}
                    model.inputs_to_track.clear()
                    torch.cuda.empty_cache()
                records_list.append(dict_record_item) 
        else:
            for input_ids in input_ids_list:
                model.inputs_to_track.clear()
                torch.cuda.empty_cache()
                print("Generating lora midfeatures...")
                dict_record_item = {}
                with torch.no_grad():
                    if 'blip' in model_name:
                        outputs = model.generate(**input_ids, max_length=max_length)
                    else:
                        outputs = model.generate(input_ids.to('cuda'))
                records_list.append(dict(model.inputs_to_track.items())) 

        for item in records_list[0].keys():
            X_tilde_dict[item] = torch.cat(
                [records[item].unsqueeze(dim=1) for records in records_list], 
                dim=1,
            ).to('cpu')


# TALS-LER helper functions are reused from Linear_TALS_DRC.py.
# Please keep the current verified Linear_TALS_DRC.py in the same project root.
# This script builds IterIS-specific TALS directions, where the coarse update is
# ΔW_c = W_IterIS - W_base, not the Linear weighted LoRA average.
from Linear_TALS_DRC import (
    construct_base_model,
    load_single_lora_dense_model,
    normalize_target_modules,
    select_drc_targets,
    collect_features_by_position,
    get_task_samples,
    build_all_task_lora_deltas,
    apply_tals_filter_to_activation_residual,
    apply_ler_layerwise_reweight,
    register_drc_hooks_by_position,
    remove_hooks,
    make_cache_path,
    parse_optional_float,
    stable_int_hash,
)



# ======================================================================================
# VLM / BLIP helper functions
# ======================================================================================

def is_blip_model(model_name):
    return "blip" in str(model_name).lower()


def tensor_to_cuda(x):
    if torch.is_tensor(x):
        return x.detach().clone().to("cuda")
    return torch.as_tensor(x).to("cuda")


def construct_base_model_any(model_name):
    """Use the original NLP constructor for T5/BART; add BLIP support locally."""
    if is_blip_model(model_name):
        return BlipForConditionalGeneration.from_pretrained(model_name)
    return construct_base_model(model_name)


def load_single_lora_dense_model_any(model_name, lora_path, rank):
    """Load a single-task LoRA and merge it into the dense base model."""
    if is_blip_model(model_name):
        base_model = BlipForConditionalGeneration.from_pretrained(model_name)
        peft_model = PeftModel.from_pretrained(base_model, lora_path)
        return peft_model.merge_and_unload()
    return load_single_lora_dense_model(
        model_name=model_name,
        lora_path=lora_path,
        rank=rank,
    )


def normalize_lora_layer_name_local(base_name):
    prefixes = [
        "base_model.model.",
        "base_model.",
        "model.",
    ]
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


def layer_match_blip_key(key, target_layers):
    if target_layers is None:
        return True
    for layer_idx in target_layers:
        layer_idx = int(layer_idx)
        patterns = [
            f".layer.{layer_idx}.",
            f".layers.{layer_idx}.",
            f".block.{layer_idx}.",
            f".blocks.{layer_idx}.",
        ]
        if any(p in key for p in patterns):
            return True
    return False


def module_match_blip_key(key, target_modules):
    modules = normalize_target_modules(target_modules)
    if not modules:
        return True
    last = key.split(".")[-1].lower()
    key_lower = key.lower()
    for module in modules:
        m = str(module).lower()
        if last == m or key_lower.endswith("." + m):
            return True
    return False


def select_drc_targets_any(
    inject_position,
    lora_path_dict,
    task_targets,
    linear_model,
    target_part,
    target_modules,
    target_layers,
    model_name=None,
):
    """Select DRC target modules. For BLIP, avoid T5-specific layer parsing."""
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

    first_task = task_targets[0]
    adapter_file = os.path.join(lora_path_dict[first_task], "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"[IterIS_TALS_DRC][VLM] adapter_model.safetensors not found: {adapter_file}")

    tensor_file = safe_open(adapter_file, framework="pt")
    keys = sorted(tensor_file.keys())
    target_keys = []
    model_modules = dict(linear_model.named_modules())

    for full_key in keys:
        if ".lora_A" not in full_key:
            continue
        base_name = extract_lora_base_name_local(full_key)
        if base_name is None:
            continue

        if target_part and str(target_part).lower() not in base_name.lower():
            continue
        if not module_match_blip_key(base_name, target_modules):
            continue
        if not layer_match_blip_key(base_name, target_layers):
            continue

        if base_name not in model_modules:
            print(f"[IterIS_TALS_DRC][VLM][Warn] LoRA target not found in dense BLIP modules: {base_name}")
            continue

        target_keys.append(base_name)

    target_keys = sorted(set(target_keys))
    print(f"[IterIS_TALS_DRC][VLM] Selected {len(target_keys)} BLIP DRC targets.")
    for k in target_keys[:20]:
        print(f"    {k}")
    if len(target_keys) == 0:
        raise RuntimeError(
            "[IterIS_TALS_DRC][VLM] No BLIP DRC target selected. "
            "Check drc_target_part / drc_target_modules / drc_target_layers."
        )
    return target_keys


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
    """Return either text-only tensors for NLP or a BLIP processor batch for VLM."""
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

    input_ids, attention_mask = get_task_samples(
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
    return input_ids, attention_mask


def collect_features_by_position_any(
    model,
    model_name,
    inject_position,
    target_keys,
    batch_inputs=None,
    input_ids=None,
    attention_mask=None,
    max_length=None,
):
    """
    Collect target-module input features for NLP or BLIP.

    Important for BLIP/VLM:
    - module input is usually [batch, seq_len, hidden_dim]
    - TALS expects one direction vector with dimension == hidden_dim
    - so we MUST pool over batch/sequence/generation calls to [hidden_dim]

    The previous VLM version stored the full [batch, seq_len, hidden_dim]
    tensor. apply_tals_filter_to_activation_residual then flattened it to
    [batch * seq_len * hidden_dim], causing residual_dim=6412800 while LoRA
    ΔW has input dimension 768. That silently produced zero usable directions.
    """
    if not is_blip_model(model_name):
        return collect_features_by_position(
            model=model,
            inject_position=inject_position,
            target_keys=target_keys,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    if inject_position != "lora_input":
        raise ValueError("[IterIS_TALS_DRC][VLM] Currently only drc_inject_position='lora_input' is supported.")

    model.eval()
    model.to("cuda")

    features_sum = {}
    features_count = {}
    handles = []
    modules = dict(model.named_modules())

    def _pool_blip_feature(x, mask=None):
        """Pool BLIP module input to a single [hidden_dim] vector."""
        if not torch.is_tensor(x):
            return None
        x = x.detach().float()

        # Common case: [batch, seq_len, hidden_dim].
        if x.dim() == 3:
            if mask is not None and torch.is_tensor(mask) and x.size(0) == mask.size(0) and x.size(1) == mask.size(1):
                m = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
                denom = m.sum().clamp_min(1.0)
                pooled = (x * m).sum(dim=(0, 1)) / denom
            else:
                pooled = x.reshape(-1, x.size(-1)).mean(dim=0)
            return pooled.detach().cpu()

        # Generation with cache may call the module on [batch, hidden_dim].
        if x.dim() == 2:
            return x.mean(dim=0).detach().cpu()

        # Fallback: if the last dim is hidden_dim-like, preserve it.
        if x.dim() >= 1:
            last_dim = x.size(-1)
            return x.reshape(-1, last_dim).mean(dim=0).detach().cpu()

        return None

    def make_hook(name):
        def hook_fn(module, inputs, outputs):
            if inputs is None or len(inputs) == 0:
                return
            pooled = _pool_blip_feature(inputs[0], mask=batch_inputs.get("attention_mask", None) if isinstance(batch_inputs, dict) else None)
            if pooled is None:
                return
            if name not in features_sum:
                features_sum[name] = pooled
                features_count[name] = 1
            else:
                features_sum[name] = features_sum[name] + pooled
                features_count[name] += 1
        return hook_fn

    for key in target_keys:
        if key not in modules:
            print(f"[IterIS_TALS_DRC][VLM][Warn] target module not found for hook: {key}")
            continue
        handles.append(modules[key].register_forward_hook(make_hook(key)))

    with torch.no_grad():
        model.generate(
            pixel_values=batch_inputs["pixel_values"],
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs["attention_mask"],
            max_length=max_length,
        )

    for h in handles:
        h.remove()

    features = {}
    for key, val in features_sum.items():
        features[key] = (val / max(features_count.get(key, 1), 1)).float().cpu()

    # Debug one-time shape summary. This should show [768], not [6412800].
    shown = 0
    for key, val in features.items():
        print(f"[IterIS_TALS_DRC][VLM][FeatureShape] {key}: {tuple(val.shape)}")
        shown += 1
        if shown >= 3:
            break

    return features


def find_weight_param_by_target(named_params, target_key):
    """
    target_key:
        encoder.block.1.layer.0.SelfAttention.q

    expected parameter:
        encoder.block.1.layer.0.SelfAttention.q.weight

    T5WithHooks / T5ForConditionalGeneration usually use exactly this name.
    A suffix fallback is kept for robustness.
    """
    pname = target_key + ".weight"
    if pname in named_params:
        return pname, named_params[pname]

    candidates = []
    for name, param in named_params.items():
        if name.endswith(pname):
            candidates.append((name, param))

    if len(candidates) == 1:
        return candidates[0]

    return None, None


def build_iteris_coarse_delta_dict(model_name, coarse_model, target_keys):
    """
    For IterIS coarse merged model, reconstruct the coarse update at each target module:

        ΔW_c = W_IterIS - W_base

    This is the key difference from Linear_TALS_DRC:
    - Linear_TALS_DRC uses ΔW_c = sum_j λ_j ΔW_j.
    - IterIS_TALS_DRC uses the actual dense IterIS merged weight difference.
    """
    print("[IterIS_TALS_DRC] Build IterIS coarse ΔW_c = W_IterIS - W_base ...")

    base_model = construct_base_model_any(model_name)
    base_named_params = dict(base_model.named_parameters())
    coarse_named_params = dict(coarse_model.named_parameters())

    coarse_delta_dict = {}
    missing_report = []

    for key in target_keys:
        base_pname, base_param = find_weight_param_by_target(base_named_params, key)
        coarse_pname, coarse_param = find_weight_param_by_target(coarse_named_params, key)

        if base_param is None or coarse_param is None:
            missing_report.append((key, base_pname, coarse_pname))
            continue

        base_w = base_param.detach().float().cpu()
        coarse_w = coarse_param.detach().float().cpu()

        if base_w.shape != coarse_w.shape:
            missing_report.append((key, tuple(base_w.shape), tuple(coarse_w.shape)))
            continue

        coarse_delta_dict[key] = coarse_w - base_w

    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    print(
        f"[IterIS_TALS_DRC] Built IterIS coarse deltas for "
        f"{len(coarse_delta_dict)}/{len(target_keys)} targets."
    )
    if missing_report:
        print(f"[IterIS_TALS_DRC][Warn] Missing/mismatched coarse weights: {len(missing_report)}")
        for item in missing_report[:10]:
            print(f"    {item}")

    return coarse_delta_dict


def build_task_specific_iteris_tals_ler_directions(
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
    Build task-specific TALS-LER directions for an IterIS coarse merged model.

    For each task t and target module (l,m):
        r_act = (h_single - h_base) - (h_iteris - h_base)
              = h_single - h_iteris

        ΔW_t = LoRA single-task update
        ΔW_c = W_IterIS - W_base
        R_W  = ΔW_t - ΔW_c        when tals_subspace_source == "missing"

        d_TALS = Normalize(V_k G V_k^T r_act)

    If tals_use_layer_weight=True:
        score = LER * ||r_act||
        omega = N * score / sum(score)
        d_final = omega * d_TALS
    """
    selected_target_keys = select_drc_targets_any(
        inject_position=inject_position,
        lora_path_dict=lora_path_dict,
        task_targets=task_targets,
        linear_model=coarse_model,
        target_part=target_part,
        target_modules=target_modules,
        target_layers=target_layers,
        model_name=model_name,
    )

    if inject_position != "lora_input":
        raise ValueError(
            "IterIS_TALS_DRC currently supports only drc_inject_position='lora_input', "
            "because TALS uses input-side singular vectors of LoRA target-module ΔW."
        )

    all_lora_deltas = build_all_task_lora_deltas(
        task_targets=task_targets,
        lora_path_dict=lora_path_dict,
        target_keys=selected_target_keys,
        rank=rank,
        lora_alpha_list=lora_alpha_list,
    )

    coarse_delta_dict = build_iteris_coarse_delta_dict(
        model_name=model_name,
        coarse_model=coarse_model,
        target_keys=selected_target_keys,
    )

    all_task_directions = {}
    direction_stats = {}

    for task_name in task_targets:
        print(f"\n[IterIS_TALS_DRC] Building task-specific TALS-LER direction for task = {task_name}")

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

        # 1. Base model features
        print(f"[IterIS_TALS_DRC] Collect base features on {task_name} samples...")
        base_model = construct_base_model_any(model_name).to("cuda")
        if is_blip_model(model_name):
            base_features = collect_features_by_position_any(
                model=base_model,
                model_name=model_name,
                inject_position=inject_position,
                target_keys=selected_target_keys,
                batch_inputs=sample_batch,
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

        # 2. Single-task LoRA dense model features
        print(f"[IterIS_TALS_DRC] Collect single-LoRA features for {task_name}...")
        single_model = load_single_lora_dense_model_any(
            model_name=model_name,
            lora_path=lora_path_dict[task_name],
            rank=rank,
        ).to("cuda")
        if is_blip_model(model_name):
            single_features = collect_features_by_position_any(
                model=single_model,
                model_name=model_name,
                inject_position=inject_position,
                target_keys=selected_target_keys,
                batch_inputs=sample_batch,
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

        # 3. IterIS coarse merged model features
        print(f"[IterIS_TALS_DRC] Collect IterIS merged features on {task_name} samples...")
        if is_blip_model(model_name):
            coarse_features = collect_features_by_position_any(
                model=coarse_model,
                model_name=model_name,
                inject_position=inject_position,
                target_keys=selected_target_keys,
                batch_inputs=sample_batch,
                max_length=max_length,
            )
        else:
            input_ids, attention_mask = sample_batch
            coarse_features = collect_features_by_position_any(
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
            "missing_iteris": [],
            "missing_lora_delta": [],
            "missing_iteris_delta": [],
            "tals_filter_failed": [],
            "zero_or_tiny_norm": [],
        }

        source = str(tals_subspace_source).lower().strip()
        if source not in ["missing", "single", "random"]:
            raise ValueError(
                f"Unsupported tals_subspace_source={tals_subspace_source}. "
                "Please use 'missing', 'single', or 'random'."
            )

        for key in selected_target_keys:
            if key not in base_features:
                missing_report["missing_base"].append(key)
                continue
            if key not in single_features:
                missing_report["missing_single"].append(key)
                continue
            if key not in coarse_features:
                missing_report["missing_iteris"].append(key)
                continue
            if key not in all_lora_deltas.get(task_name, {}):
                missing_report["missing_lora_delta"].append((task_name, key))
                continue

            single_shift = single_features[key] - base_features[key]
            iteris_shift = coarse_features[key] - base_features[key]
            delta = single_shift - iteris_shift  # h_single - h_iteris

            task_delta_w = all_lora_deltas[task_name][key].float()

            if source == "missing":
                if key not in coarse_delta_dict:
                    missing_report["missing_iteris_delta"].append(key)
                    continue
                source_delta_w = task_delta_w - coarse_delta_dict[key].float()
            elif source == "single":
                source_delta_w = task_delta_w
            elif source == "random":
                source_delta_w = None
            else:
                raise ValueError(f"Unsupported tals_subspace_source={source}")

            random_seed = seed + stable_int_hash(f"iteris|{task_name}|{key}|{tals_rank}")

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
                        f"[IterIS_TALS_DRC][Warn] TALS filter failed at {key}; "
                        "fallback to base DRC direction."
                    )
                    tals_delta = delta.float().cpu()
                    tals_stats = {"fallback_to_base": True, **tals_stats}
                else:
                    continue

            raw_norm = float(torch.norm(tals_delta.float()))

            if normalize:
                norm = torch.norm(tals_delta.float())
                if norm < tals_eps:
                    missing_report["zero_or_tiny_norm"].append((key, raw_norm))
                    continue
                task_direction[key] = (tals_delta.float() / norm).cpu()
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

        print(f"[IterIS_TALS_DRC][Debug] Missing/skip report for task={task_name}:")
        for reason, items in missing_report.items():
            print(f"  - {reason}: {len(items)}")
            for item in items[:10]:
                print(f"      {item}")

        print(
            f"[IterIS_TALS_DRC] Task {task_name}: built directions for "
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
            print(f"[IterIS_TALS_DRC][LayerWeight] task={task_name}, summary={lw_summary}")
            direction_stats[f"{task_name}__layer_weight_summary"] = lw_summary

        all_task_directions[task_name] = task_direction
        direction_stats[task_name] = task_stats

    return all_task_directions, direction_stats, selected_target_keys


def get_primary_metric_for_glue(task_name, eval_results):
    eval_accuracy = eval_results.get("eval_accuracy", "")
    eval_mcc = eval_results.get("eval_MCC", "")

    if task_name == "cola":
        primary_metric_name = "MCC"
        primary_metric_value = eval_mcc
        normalized_metric = (eval_mcc + 1) / 2 if eval_mcc != "" else ""
    else:
        primary_metric_name = "accuracy"
        primary_metric_value = eval_accuracy
        normalized_metric = eval_accuracy

    return primary_metric_name, primary_metric_value, normalized_metric



def get_primary_metric_any(task_name, eval_results, task_type):
    """Return primary metric for GLUE or BLIP/SentiCap."""
    if task_type == "TASKS_blip_base":
        if "acc" in eval_results and eval_results["acc"] not in ["", None]:
            acc = float(eval_results["acc"])
            return "acc", acc, acc
        for key in ["style_acc", "style_accuracy", "eval_style_acc", "eval_accuracy", "accuracy"]:
            if key in eval_results and eval_results[key] not in ["", None]:
                v = float(eval_results[key])
                return key, v, v
        if "cider" in eval_results and eval_results["cider"] not in ["", None]:
            v = float(eval_results["cider"])
            return "cider", v, v
        return "acc", "", ""

    return get_primary_metric_for_glue(task_name, eval_results)


def ensure_vlm_caption_header(vlm_results_csv):
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
            "record_type",
        ],
    )


def append_vlm_caption_row(
    vlm_results_csv,
    experiment_id,
    method_name,
    pair_name,
    task_targets,
    task_name,
    alpha,
    eval_results,
    merged_model_dir,
    log_file,
    record_type="best",
):
    bleu = eval_results.get("bleu", ["", "", "", ""])

    def get_bleu(i):
        try:
            return bleu[i]
        except Exception:
            return ""

    append_csv_row(
        vlm_results_csv,
        [
            experiment_id,
            method_name,
            pair_name,
            task_targets[0],
            task_targets[1],
            task_name,
            alpha,
            eval_results.get("acc", eval_results.get("style_acc", "")),
            eval_results.get("cider", eval_results.get("CIDEr", "")),
            get_bleu(0),
            get_bleu(1),
            get_bleu(2),
            get_bleu(3),
            eval_results.get("rougeL", ""),
            eval_results.get("div_1", ""),
            eval_results.get("div_2", ""),
            eval_results.get("vocab_size", ""),
            "validation",
            merged_model_dir,
            log_file,
            record_type,
        ],
    )


def build_drc_cfg_string(
    drc_inject_position,
    drc_alpha,
    drc_samples_per_task,
    drc_target_part,
    drc_target_modules,
    drc_target_layers,
    drc_normalize_direction,
    drc_use_hidden_norm_scale,
    drc_rebuild_cache,
    cache_path,
):
    return (
        f"DRC: inject_position={drc_inject_position}; "
        f"alpha={drc_alpha}; "
        f"samples_per_task={drc_samples_per_task}; "
        f"target_part={drc_target_part}; "
        f"target_modules={normalize_target_modules(drc_target_modules)}; "
        f"target_layers={drc_target_layers}; "
        f"normalize={drc_normalize_direction}; "
        f"use_hidden_norm_scale={drc_use_hidden_norm_scale}; "
        f"rebuild_cache={drc_rebuild_cache}; "
        f"cache_path={cache_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="IterIS + TALS-LER Inference-time Enhancement Script")
    parser.add_argument(
        '--config',
        type=str,
        default="config/methods-config/iteris-config.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        '--task_type',
        type=str,
        choices=['GLUE_t5', 'EMOTION_t5_large', 'TASKS_blip_base'],
        default='GLUE_t5',
        help="Choose a task type from the list of options.",
    )
    args = parser.parse_args()

    task_type = args.task_type

    with open(args.config, 'r', encoding="utf-8") as file:
        config_data = yaml.safe_load(file)

    set_seed(config_data['seed'])

    if task_type not in config_data:
        raise ValueError(f"Cannot find task_type={task_type} in config.")

    task_cfg = config_data[task_type]

    model_name = task_cfg['model_name']
    task_targets = task_cfg['task_targets']
    pair_name = "_".join(task_targets)

    method_name = task_cfg.get("tals_method_name", task_cfg.get("drc_method_name", "IterIS_TALS_DRC"))
    experiment_id = f"{task_type}_{pair_name}_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Pair-safe output:
    # run_all_glue_pairs.py will set RESULTS_DIR to:
    #   batch_runs/<batch_id>/pair_results/<pair_name>
    # so each pair writes its own CSV files. This avoids multi-process appending
    # to the same global results/*.csv.
    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)

    registry_csv = os.path.join(results_dir, "experiment_registry.csv")
    results_csv = os.path.join(results_dir, "pair_merge_results.csv")
    alpha_search_csv = os.path.join(results_dir, "drc_alpha_search_results.csv")
    vlm_results_csv = os.path.join(results_dir, "vlm_caption_results.csv")
    log_file = os.environ.get("LOG_FILE", f"logs/{method_name}_{pair_name}.log")

    lora_path = [get_loras_path(task_type, model_name)[item] for item in task_targets]
    with_pretrain_matrix = task_cfg['with_pretrain_matrix']
    tokenizer = (
        AutoTokenizer.from_pretrained(model_name)
        if 'blip' not in model_name
        else AutoProcessor.from_pretrained(model_name)
    )
    save = task_cfg['save']

    # 注意：DRC 是推理时 hook，不会被 save_pretrained 保存。
    # 这里保存的是 IterIS 粗融合模型本身，不包含 DRC hook。
    merged_model_dir = task_cfg.get(
        "iteris_drc_merged_model_dir",
        f"merged_model/{method_name}_{pair_name}",
    )

    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ensure_csv_header(
        registry_csv,
        [
            "experiment_id", "stage", "method", "base_model", "pair_name", "task_a", "task_b",
            "lora_a_path", "lora_b_path", "config_path", "seed", "max_iter", "alpha_1", "alpha_2",
            "reg_ceof", "rank", "samples_num", "if_balance", "shuffle", "select_long",
            "with_pretrain_matrix", "save_merged_model", "start_time", "end_time",
            "fusion_total_time_sec", "fusion_iter_time_avg_sec", "fusion_iter_time_max_sec",
            "fusion_peak_vram_avg_mb", "fusion_peak_vram_max_mb", "pair_avg_normalized_metric",
            "merged_model_dir", "log_file", "status", "notes",
        ],
    )

    ensure_csv_header(
        results_csv,
        [
            "experiment_id", "method", "pair_name", "task_a", "task_b", "evaluated_task",
            "primary_metric_name", "primary_metric_value", "normalized_metric",
            "eval_accuracy", "eval_mcc", "eval_f1", "eval_loss", "eval_runtime",
            "eval_samples_per_second", "eval_steps_per_second", "eval_peak_vram_mb",
            "split", "merged_model_dir", "log_file", "notes",
        ],
    )

    if task_type == "TASKS_blip_base":
        ensure_vlm_caption_header(vlm_results_csv)

    # DRC config.
    # 如果 drc_alpha_search=True，则对每个 evaluated task 搜索 drc_alpha_candidates；
    # 否则使用固定 drc_alpha。
    drc_inject_position = task_cfg.get("drc_inject_position", "lora_input")
    drc_alpha = float(task_cfg.get("drc_alpha", 0.3))
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
    drc_normalize_direction = task_cfg.get("drc_normalize_direction", True)
    drc_use_hidden_norm_scale = task_cfg.get("drc_use_hidden_norm_scale", False)
    drc_rebuild_cache = task_cfg.get("drc_rebuild_cache", True)

    # TALS-LER config.
    # Current final version:
    #   d = omega * Normalize(V_k G V_k^T r_act)
    #   omega score = LER * ||r_act||
    #   omega norm  = mean_one
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

    cache_dir = task_cfg.get("drc_cache_dir", "direction_cache")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = make_cache_path(
        cache_dir=cache_dir,
        method_name=method_name,
        pair_name=pair_name,
        inject_position=drc_inject_position,
        target_layers=drc_target_layers,
        target_modules=drc_target_modules,
    )
    cache_path = cache_path.replace(
        ".pt",
        f"_source{tals_subspace_source}"
        f"_talsk{tals_rank}_gamma{str(tals_gamma).replace('.', 'p')}"
        f"_wnorm{tals_weight_norm}_center{int(tals_svd_center)}"
        f"_lw{int(tals_use_layer_weight)}_{tals_layer_weight_score}_{tals_layer_weight_norm}.pt"
    )

    drc_cfg_str = build_drc_cfg_string(
        drc_inject_position=drc_inject_position,
        drc_alpha=drc_alpha,
        drc_samples_per_task=drc_samples_per_task,
        drc_target_part=drc_target_part,
        drc_target_modules=drc_target_modules,
        drc_target_layers=drc_target_layers,
        drc_normalize_direction=drc_normalize_direction,
        drc_use_hidden_norm_scale=drc_use_hidden_norm_scale,
        drc_rebuild_cache=drc_rebuild_cache,
        cache_path=cache_path,
    )
    drc_cfg_str = (
        f"{drc_cfg_str}; "
        f"alpha_search={drc_alpha_search}; "
        f"alpha_candidates={drc_alpha_candidates}; "
        f"tals_rank={tals_rank}; "
        f"tals_gamma={tals_gamma}; "
        f"tals_weight_norm={tals_weight_norm}; "
        f"tals_svd_center={tals_svd_center}; "
        f"tals_subspace_source={tals_subspace_source}; "
        f"tals_fallback_to_base={tals_fallback_to_base}; "
        f"tals_use_layer_weight={tals_use_layer_weight}; "
        f"tals_layer_weight_score={tals_layer_weight_score}; "
        f"tals_layer_weight_norm={tals_layer_weight_norm}; "
        f"tals_layer_weight_clip_min={tals_layer_weight_clip_min}; "
        f"tals_layer_weight_clip_max={tals_layer_weight_clip_max}"
    )

    print(f"[IterIS_TALS_DRC] method_name = {method_name}")
    print(f"[IterIS_TALS_DRC] task_targets = {task_targets}")
    print(f"[IterIS_TALS_DRC] model_name = {model_name}")
    print(f"[IterIS_TALS_DRC] results_dir = {results_dir}")
    print(f"[IterIS_TALS_DRC] log_file = {log_file}")
    print(f"[IterIS_TALS_DRC] drc_alpha = {drc_alpha}")
    print(f"[IterIS_TALS_DRC] drc_alpha_search = {drc_alpha_search}")
    print(f"[IterIS_TALS_DRC] drc_alpha_candidates = {drc_alpha_candidates}")
    print(f"[IterIS_TALS_DRC] tals_subspace_source = {tals_subspace_source}")
    print(f"[IterIS_TALS_DRC] tals_rank = {tals_rank}")
    print(f"[IterIS_TALS_DRC] tals_gamma = {tals_gamma}")
    print(f"[IterIS_TALS_DRC] tals_use_layer_weight = {tals_use_layer_weight}")
    print(f"[IterIS_TALS_DRC] tals_layer_weight_score = {tals_layer_weight_score}")
    print(f"[IterIS_TALS_DRC] tals_layer_weight_norm = {tals_layer_weight_norm}")
    print(f"[IterIS_TALS_DRC] {drc_cfg_str}")

    start_time = time.time()
    status = "done"
    notes = drc_cfg_str
    fusion_stats = {
        "fusion_iter_time_avg_sec": "",
        "fusion_iter_time_max_sec": "",
        "fusion_peak_vram_avg_mb": "",
        "fusion_peak_vram_max_mb": "",
    }
    pair_norm_metrics = []
    selected_alpha_by_task = {}
    model = None
    wrote_result_row = False

    try:
        # 1. Run original IterIS merging to obtain a coarse merged model.
        model, fusion_stats = update_param(
            task_targets=task_targets,
            lora_path=lora_path,
            model_name=model_name,
            with_pretrain_matrix=with_pretrain_matrix,
            max_iter=task_cfg['max_iter'],
            max_length=task_cfg['max_length'],
            lora_alpha=task_cfg['lora_alpha'],
            alpha_1=task_cfg['alpha_1'],
            alpha_2=task_cfg['alpha_2'],
            reg_ceof=task_cfg['reg_ceof'],
            rank=task_cfg['rank'],
            samples_num=task_cfg['samples_num'],
            manual_ceof=task_cfg['manual_ceof'],
            if_divide=task_cfg['if_divide'],
            if_balance=task_cfg['if_balance'],
            inner_num=task_cfg['inner_num'],
            outer_num=task_cfg['outer_num'],
            seed=config_data['seed'],
            select_long=task_cfg['select_long'],
            shuffle=task_cfg['shuffle'],
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()

        # 2. Build or load task-specific TALS-LER directions.
        # For IterIS, the activation residual is h_single - h_iteris,
        # and the missing update is ΔW_t - (W_IterIS - W_base).
        if (not drc_rebuild_cache) and os.path.exists(cache_path):
            print(f"[IterIS_TALS_DRC] Load DRC direction cache from: {cache_path}")
            cache = torch.load(cache_path, map_location="cpu")
            drc_directions = cache["directions"]
            direction_stats = cache.get("direction_stats", {})
            selected_target_keys = cache.get(
                "selected_target_keys",
                cache.get("selected_lora_keys", []),
            )
        else:
            print("[IterIS_TALS_DRC] Build task-specific TALS-LER directions for IterIS merged model.")
            drc_directions, direction_stats, selected_target_keys = build_task_specific_iteris_tals_ler_directions(
                model_name=model_name,
                tokenizer=tokenizer,
                task_targets=task_targets,
                lora_path_dict=get_loras_path(task_type, model_name),
                coarse_model=model,
                rank=task_cfg['rank'],
                max_length=task_cfg['max_length'],
                seed=config_data['seed'],
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

            torch.save(
                {
                    "pair_name": pair_name,
                    "method": method_name,
                    "base_method": "IterIS",
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
                    "tals_rank": tals_rank,
                    "tals_gamma": tals_gamma,
                    "tals_eps": tals_eps,
                    "tals_weight_norm": tals_weight_norm,
                    "tals_svd_center": tals_svd_center,
                    "tals_subspace_source": tals_subspace_source,
                    "tals_fallback_to_base": tals_fallback_to_base,
                    "tals_use_layer_weight": tals_use_layer_weight,
                    "tals_layer_weight_score": tals_layer_weight_score,
                    "tals_layer_weight_norm": tals_layer_weight_norm,
                    "tals_layer_weight_clip_min": tals_layer_weight_clip_min,
                    "tals_layer_weight_clip_max": tals_layer_weight_clip_max,
                },
                cache_path,
            )
            print(f"[IterIS_TALS_DRC] Saved DRC direction cache to: {cache_path}")

        torch.cuda.empty_cache()
        gc.collect()

        # 3. Evaluate IterIS + task-specific TALS-LER.
        # 如果 drc_alpha_search=True:
        #   - 每个 task 会遍历 drc_alpha_candidates；
        #   - drc_alpha_search_results.csv 写入所有候选 alpha 的结果；
        #   - pair_merge_results.csv 只写 best alpha 的最终结果。
        # 如果 drc_alpha_search=False:
        #   - 只使用固定 drc_alpha；
        #   - 不写 drc_alpha_search_results.csv，避免固定 alpha 实验被误认为搜索实验。
        for task_name in task_targets:
            try:
                print(f"\n[Eval] Evaluating IterIS + task-specific TALS-LER on {task_name}...")

                if task_name not in drc_directions:
                    raise ValueError(f"Cannot find DRC direction for task: {task_name}")

                candidate_alphas = drc_alpha_candidates if drc_alpha_search else [drc_alpha]
                best_record = None

                for alpha in candidate_alphas:
                    handles = []
                    try:
                        print(f"[IterIS_TALS_DRC][AlphaSearch] task={task_name}, alpha={alpha}")

                        handles = register_drc_hooks_by_position(
                            model=model,
                            directions=drc_directions[task_name],
                            inject_position=drc_inject_position,
                            alpha=float(alpha),
                            use_hidden_norm_scale=drc_use_hidden_norm_scale,
                        )

                        eval_results = eval_iteris_model(
                            model=model,
                            tokenizer=tokenizer,
                            model_name=model_name,
                            task_name=task_name,
                            max_length=task_cfg['max_length'],
                            per_device_eval_batch_size=task_cfg['per_device_eval_batch_size'],
                        )

                        remove_hooks(handles)
                        handles = []

                        eval_accuracy = eval_results.get("eval_accuracy", "")
                        eval_mcc = eval_results.get("eval_MCC", "")
                        eval_f1 = eval_results.get("eval_f1-score", "")
                        eval_loss = eval_results.get("eval_loss", "")
                        eval_runtime = eval_results.get("eval_runtime", eval_results.get("eval_wall_time_sec", ""))
                        eval_sps = eval_results.get("eval_samples_per_second", "")
                        eval_stepsps = eval_results.get("eval_steps_per_second", "")
                        eval_peak_vram_mb = eval_results.get("eval_peak_vram_mb", "")

                        primary_metric_name, primary_metric_value, normalized_metric = get_primary_metric_any(
                            task_name=task_name,
                            eval_results=eval_results,
                            task_type=task_type,
                        )
                        if task_type == "TASKS_blip_base":
                            # 兼容原 CSV：把 VLM style acc 也放进 eval_accuracy 列，便于统一看主指标。
                            eval_accuracy = primary_metric_value

                        # 只有开启 alpha search 时才记录所有候选 alpha。
                        if drc_alpha_search:
                            append_alpha_search_row(
                                alpha_search_csv,
                                [
                                    experiment_id, method_name, pair_name, task_name,
                                    float(alpha), primary_metric_name, primary_metric_value,
                                    normalized_metric, eval_accuracy, eval_mcc, eval_f1,
                                    eval_loss, eval_runtime, eval_peak_vram_mb,
                                ],
                            )

                        score = float(normalized_metric) if normalized_metric != "" else -float("inf")
                        current_record = {
                            "alpha": float(alpha),
                            "score": score,
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
                            "eval_results": eval_results,
                        }

                        if best_record is None or current_record["score"] > best_record["score"]:
                            best_record = current_record

                    except Exception:
                        try:
                            remove_hooks(handles)
                        except Exception:
                            pass
                        raise
                    finally:
                        torch.cuda.empty_cache()
                        gc.collect()

                if best_record is None:
                    raise RuntimeError(f"No valid evaluation result for task={task_name}")

                selected_alpha_by_task[task_name] = best_record["alpha"]

                if best_record["normalized_metric"] != "":
                    pair_norm_metrics.append(best_record["normalized_metric"])

                print(
                    f"[IterIS_TALS_DRC][AlphaSearch][Best] task={task_name}, "
                    f"alpha={best_record['alpha']}, "
                    f"normalized_metric={best_record['normalized_metric']}"
                )

                if task_type == "TASKS_blip_base":
                    append_vlm_caption_row(
                        vlm_results_csv=vlm_results_csv,
                        experiment_id=experiment_id,
                        method_name=method_name,
                        pair_name=pair_name,
                        task_targets=task_targets,
                        task_name=task_name,
                        alpha=best_record["alpha"],
                        eval_results=best_record.get("eval_results", {}),
                        merged_model_dir=merged_model_dir,
                        log_file=log_file,
                        record_type="best",
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
                        drc_cfg_str,
                    ],
                )
                wrote_result_row = True

            except Exception as e:
                append_csv_row(
                    results_csv,
                    [
                        experiment_id, method_name, pair_name, task_targets[0], task_targets[1],
                        task_name, "", "", "",
                        "", "", "", "", "",
                        "", "", "",
                        "validation", merged_model_dir, log_file,
                        f"EVAL_FAILED: {type(e).__name__}: {str(e)} | {drc_cfg_str}",
                    ],
                )
                wrote_result_row = True
                raise

        # 4. Save the coarse IterIS model if requested.
        # Reminder: DRC is hook-based and is not included in save_pretrained().
        if save == 1:
            os.makedirs(merged_model_dir, exist_ok=True)
            model.save_pretrained(merged_model_dir)
            tokenizer.save_pretrained(merged_model_dir)
            print(f"IterIS coarse merged model saved to: {merged_model_dir}")
            print("[Warn] TALS-LER hooks are not saved in model weights; DRC directions are saved in the cache file.")

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()

    except Exception as e:
        status = "failed"
        notes = traceback.format_exc() + " | " + drc_cfg_str

        if not wrote_result_row:
            append_csv_row(
                results_csv,
                [
                    experiment_id, method_name, pair_name, task_targets[0], task_targets[1],
                    "__merge_failed__", "", "", "",
                    "", "", "", "", "",
                    "", "", "",
                    "validation", merged_model_dir, log_file,
                    f"MERGE_FAILED: {type(e).__name__}: {str(e)} | {drc_cfg_str}",
                ],
            )

    finally:
        end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_time = round(time.time() - start_time, 4)
        pair_avg_normalized_metric = (
            round(float(np.mean(pair_norm_metrics)), 6)
            if len(pair_norm_metrics) > 0
            else ""
        )

        selected_alpha_str = "|".join(
            [f"{task}:{selected_alpha_by_task.get(task, drc_alpha)}" for task in task_targets]
        )
        final_notes = (
            str(notes).replace("\n", " | ")
            + f" | alpha_search={drc_alpha_search}"
            + f" | alpha_candidates={drc_alpha_candidates}"
            + f" | selected_alpha_by_task={selected_alpha_str}"
        )

        append_csv_row(
            registry_csv,
            [
                experiment_id, "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise", method_name, model_name, pair_name,
                task_targets[0], task_targets[1],
                lora_path[0], lora_path[1],
                args.config, config_data["seed"], task_cfg["max_iter"],
                task_cfg["alpha_1"], task_cfg["alpha_2"],
                task_cfg["reg_ceof"], task_cfg["rank"],
                task_cfg["samples_num"], task_cfg["if_balance"],
                task_cfg["shuffle"], task_cfg["select_long"],
                with_pretrain_matrix, save, start_dt, end_dt,
                elapsed_time,
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_max_sec"],
                fusion_stats["fusion_peak_vram_avg_mb"],
                fusion_stats["fusion_peak_vram_max_mb"],
                pair_avg_normalized_metric,
                merged_model_dir, log_file, status, final_notes[:5000],
            ],
        )

        if status == "done":
            print(f"\n[Done] {method_name} finished for pair: {pair_name}")
            print(f"[Done] pair_avg_normalized_metric = {pair_avg_normalized_metric}")
            print(f"[Done] results saved to: {results_csv}")
            print(f"[Done] registry saved to: {registry_csv}")
            print(f"[Done] direction cache = {cache_path}")
        else:
            print(f"\n[Fail] {method_name} failed for pair: {pair_name}")
            print(f"[Fail] status = {status}")
            print("[Fail] full traceback:")
            print(notes)

        if status == "failed":
            raise RuntimeError(f"Experiment failed: {pair_name}. See log: {log_file}")


if __name__ == "__main__":
    main()
