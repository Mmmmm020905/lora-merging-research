import os
import gc
import csv
import yaml
import time
import errno
import torch
import random
import argparse
import traceback
import numpy as np
from collections import OrderedDict
from datetime import datetime
from copy import deepcopy

from transformers import (
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    TaskType,
)
from peft.utils import get_peft_model_state_dict
from safetensors import safe_open

from eval_model import eval_iteris_model


GLUE_task_name = [
    "mnli", "rte",
    "cola", "sst2", "qqp",
    "qnli", "mrpc", "wnli",
]


def get_loras_path(task_type, model_name):
    lora_path_dict = {}
    if "t5" in model_name and task_type == "GLUE_t5":
        lora_path_dict["cola"] = "best_LoRA/T5-COLA-LoRA"
        lora_path_dict["sst2"] = "best_LoRA/T5-SST2-LoRA"
        lora_path_dict["rte"] = "best_LoRA/T5-RTE-LoRA"
        lora_path_dict["qnli"] = "best_LoRA/T5-QNLI-LoRA"
        lora_path_dict["qqp"] = "best_LoRA/T5-QQP-LoRA"
        lora_path_dict["mrpc"] = "best_LoRA/T5-MRPC-LoRA"
        lora_path_dict["mnli"] = "best_LoRA/T5-MNLI-LoRA"
        lora_path_dict["wnli"] = "best_LoRA/T5-WNLI-LoRA"
    return lora_path_dict


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
    with FileLock(lock_path):
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(header)


def append_csv_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    lock_path = csv_path + ".lock"
    with FileLock(lock_path):
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)


class FileLock:
    def __init__(self, lock_path, poll_interval=0.05):
        self.lock_path = lock_path
        self.poll_interval = poll_interval
        self.fd = None

    def __enter__(self):
        while True:
            try:
                self.fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                return self
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                time.sleep(self.poll_interval)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd is not None:
                os.close(self.fd)
        finally:
            try:
                os.unlink(self.lock_path)
            except FileNotFoundError:
                pass


def construct_base_model(model_name):
    if "t5" in model_name:
        return T5ForConditionalGeneration.from_pretrained(model_name)
    if "bart" in model_name:
        return BartForConditionalGeneration.from_pretrained(model_name)
    raise ValueError(f"[KnOTS] Unsupported model_name: {model_name}")


def load_adapter_config(lora_dir):
    config_path = os.path.join(lora_dir, "adapter_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[KnOTS] adapter_config.json not found: {config_path}")
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


def get_lora_scaling_from_adapter_cfg(adapter_cfg):
    r = adapter_cfg.get("r", 1)
    lora_alpha = adapter_cfg.get("lora_alpha", r)
    use_rslora = adapter_cfg.get("use_rslora", False)
    if use_rslora:
        return float(lora_alpha) / np.sqrt(float(r))
    return float(lora_alpha) / float(r)


def lora_state_dict_to_delta_matrices(state_dict, device="cpu", adapter_cfg=None):
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
    for name, key2val in sorted(layer2lora_parameters.items()):
        if "A" not in key2val or "B" not in key2val:
            raise ValueError(f"[KnOTS] Incomplete LoRA pair for layer: {name}")
        # Align with core-space LoRAHandler: task parameters live in raw B@A space.
        task_parameters[name] = (key2val["B"] @ key2val["A"]).to(torch.float32)
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


def get_knots_components(task_matrices, concat_across_output=True, svd_tol=1e-5):
    if len(task_matrices) == 0:
        raise ValueError("[KnOTS] task_matrices 不能为空。")

    mats = [mat.to(torch.float32) for mat in task_matrices]
    if concat_across_output:
        stack = torch.cat(mats, dim=1)
    else:
        stack = torch.cat([mat.t() for mat in mats], dim=1)

    U, s, Vh = torch.linalg.svd(stack.to(torch.float64), full_matrices=False)
    keep = s > svd_tol

    if keep.sum().item() == 0:
        return None, None, None

    U = U[:, keep].to(torch.float32)
    s = s[keep].to(torch.float32)
    Vh = Vh[keep].to(torch.float32)
    Vs = list(torch.split(Vh, Vh.shape[1] // len(task_matrices), dim=1))
    return U, s, Vs


def apply_knots_mean_per_layer(task_matrices, concat_across_output=True, svd_tol=1e-5):
    U, s, Vs = get_knots_components(
        task_matrices,
        concat_across_output=concat_across_output,
        svd_tol=svd_tol,
    )
    if U is None:
        return torch.zeros_like(task_matrices[0], dtype=torch.float32)

    merged_V = torch.stack(Vs, dim=0).mean(dim=0)
    merged_matrix = U @ torch.diag(s) @ merged_V
    if not concat_across_output:
        merged_matrix = merged_matrix.t()
    return merged_matrix.to(torch.float32)


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


def merge_knots_linear_mean_param(
    lora_path,
    model_name,
    task_targets,
    seed,
    scaling_coeff=1.0,
    concat_across_output=True,
    svd_tol=1e-5,
):
    assert len(lora_path) == len(task_targets), "lora_path 数量必须和 task_targets 一致"

    print(f"[KnOTS] task_targets = {task_targets}")
    print(f"[KnOTS] lora_path = {lora_path}")
    print(
        f"[KnOTS] merge = KnOTS mean, "
        f"scaling_coeff = {scaling_coeff}, concat_across_output = {concat_across_output}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()
    start_time = time.time()

    adapter_cfg = load_adapter_config(lora_path[0])

    ftms_relevant_params = []
    for path in lora_path:
        adapter_state_dict = load_adapter_state_dict_from_safetensors(path, device="cpu")
        ft_sd = lora_state_dict_to_delta_matrices(
            adapter_state_dict,
            device="cpu",
            adapter_cfg=adapter_cfg,
        )
        ftms_relevant_params.append(ft_sd)

    merged_direction_sd = OrderedDict()
    check_state_dict_keys_match(ftms_relevant_params)
    for layer_name in ftms_relevant_params[0].keys():
        layer_mats = [ft_params[layer_name] for ft_params in ftms_relevant_params]
        merged_direction_sd[layer_name] = apply_knots_mean_per_layer(
            layer_mats,
            concat_across_output=concat_across_output,
            svd_tol=svd_tol,
        ).cpu()

    model = construct_base_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    number_update = add_direction_to_base_model(
        model,
        merged_direction_sd,
        scaling_coeff=scaling_coeff,
    )

    if number_update == len(merged_direction_sd):
        print("[KnOTS] All target modules updated successfully.")
    else:
        print(
            f"[KnOTS][Warn] Updated {number_update}/{len(merged_direction_sd)} modules. "
            f"请检查 LoRA 层名和 base model 参数名是否完全匹配。"
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

    print(f"[KnOTS] Fusion time: {fusion_time} sec")
    print(f"[KnOTS] Fusion peak VRAM: {fusion_peak_vram_mb} MB")

    return model, fusion_stats, adapter_cfg


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


def eval_knots_pair_average(model, tokenizer, model_name, task_targets, max_length, per_device_eval_batch_size):
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


def search_best_knots_config(
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
):
    EARLY_STOPPING_STEPS = 2
    print(f"[KnOTS-SEARCH] default_params = {default_params}")
    print(f"[KnOTS-SEARCH] order_of_processing_params = {order_of_processing_params}")
    print(f"[KnOTS-SEARCH] search_config = {search_config}")

    best_val_results = {"avg_normalized_metric": -1e9}
    running_defaults = deepcopy(default_params)

    for param in order_of_processing_params:
        best_for_param = deepcopy(best_val_results)
        early_stopping = EARLY_STOPPING_STEPS
        for value in search_config[param]:
            instance_params = deepcopy(running_defaults)
            instance_params[param] = value
            print(f"[KnOTS-SEARCH] Try params = {instance_params}")

            model, _, _ = merge_knots_linear_mean_param(
                lora_path=lora_path,
                model_name=model_name,
                task_targets=task_targets,
                seed=seed,
                scaling_coeff=float(instance_params["scaling_coeffs"]),
                concat_across_output=concat_across_output,
                svd_tol=svd_tol,
            )

            avg_score, _ = eval_knots_pair_average(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                task_targets=task_targets,
                max_length=max_length,
                per_device_eval_batch_size=per_device_eval_batch_size,
            )
            print(f"[KnOTS-SEARCH] avg_normalized_metric = {avg_score:.6f}")

            if avg_score >= best_for_param.get("avg_normalized_metric", -1e9):
                best_for_param = deepcopy(instance_params)
                best_for_param["avg_normalized_metric"] = avg_score
                early_stopping = EARLY_STOPPING_STEPS
            else:
                early_stopping -= 1
                if early_stopping <= 0:
                    print("[KnOTS-SEARCH] Early stopping")
                    break

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        running_defaults[param] = best_for_param[param]
        best_val_results = deepcopy(best_for_param)

    print(f"[KnOTS-SEARCH] Best config = {best_val_results}")
    return best_val_results


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
    raw_knots_scaling_coeff = task_cfg.get("knots_scaling_coeffs", 1.0)
    knots_concat_across_output = bool(task_cfg.get("knots_concat_across_output", True))
    knots_svd_tol = float(task_cfg.get("knots_svd_tol", 1e-5))
    knots_do_search = bool(task_cfg.get("knots_do_search", False))

    tokenizer = (
        AutoTokenizer.from_pretrained(model_name)
        if "blip" not in model_name
        else AutoProcessor.from_pretrained(model_name)
    )

    lora_path_dict = get_loras_path(task_type, model_name)
    lora_path = [lora_path_dict[task] for task in task_targets]

    pair_name = "_".join(task_targets)
    method_name = "KnOTS-Mean"
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{method_name}_{pair_name}"

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    results_csv = os.path.join(results_dir, "pair_merge_results.csv")
    registry_csv = os.path.join(results_dir, "experiment_registry.csv")

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

    log_file = ""

    try:
        parsed_scaling = parse_scalar_or_candidates(raw_knots_scaling_coeff)
        explicit_scaling_candidates = task_cfg.get(
            "knots_scaling_coeffs_candidates",
            [0.1, 0.3, 0.5, 0.7, 1.0],
        )

        if knots_do_search:
            scaling_candidates = (
                parsed_scaling if isinstance(parsed_scaling, list)
                else [float(x) for x in explicit_scaling_candidates]
            )
            search_result = search_best_knots_config(
                lora_path=lora_path,
                model_name=model_name,
                task_targets=task_targets,
                seed=seed,
                tokenizer=tokenizer,
                max_length=task_cfg["max_length"],
                per_device_eval_batch_size=task_cfg.get("per_device_eval_batch_size", 8),
                concat_across_output=knots_concat_across_output,
                svd_tol=knots_svd_tol,
                default_params={"scaling_coeffs": float(task_cfg.get("knots_scaling_coeffs", 0.3))},
                search_config={"scaling_coeffs": [float(x) for x in scaling_candidates]},
                order_of_processing_params=["scaling_coeffs"],
            )
            knots_scaling_coeff = float(search_result["scaling_coeffs"])
        else:
            if isinstance(parsed_scaling, list):
                raise ValueError(
                    "knots_scaling_coeffs 当前是多个候选值。"
                    "若要搜索，请设置 knots_do_search: true；"
                    "若不搜索，请把 knots_scaling_coeffs 改成单个数值。"
                )
            knots_scaling_coeff = float(parsed_scaling)

        model, fusion_stats, adapter_cfg = merge_knots_linear_mean_param(
            lora_path=lora_path,
            model_name=model_name,
            task_targets=task_targets,
            seed=seed,
            scaling_coeff=knots_scaling_coeff,
            concat_across_output=knots_concat_across_output,
            svd_tol=knots_svd_tol,
        )

        merged_model_dir = f"merged_model/{method_name}_{pair_name}"

        if task_cfg.get("save", 0):
            os.makedirs(merged_model_dir, exist_ok=True)
            model.save_pretrained(merged_model_dir)
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(merged_model_dir)
            print(f"[KnOTS] Merged model saved to {merged_model_dir}")

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
            eval_runtime = eval_result.get("eval_runtime", "")
            eval_sps = eval_result.get("eval_samples_per_second", "")
            eval_stepsps = eval_result.get("eval_steps_per_second", "")
            eval_peak_vram_mb = eval_result.get("eval_peak_vram_mb", "")

            if task_name == "cola":
                primary_metric_name = "matthews_correlation"
                primary_metric_value = float(eval_mcc)
                normalized_metric = (primary_metric_value + 1.0) / 2.0
            else:
                primary_metric_name = "accuracy"
                primary_metric_value = float(eval_accuracy)
                normalized_metric = primary_metric_value

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
        knots_cfg_str = (
            f"weights=0.5|0.5|scaling={knots_scaling_coeff}|"
            f"concat_across_output={knots_concat_across_output}|svd_tol={knots_svd_tol}|"
            f"search={knots_do_search}"
        )

        append_csv_row(
            registry_csv,
            [
                experiment_id, "GLUE_pairwise", method_name, model_name, pair_name,
                "|".join(task_targets), knots_cfg_str,
                adapter_cfg.get("r", ""),
                str(adapter_cfg.get("lora_alpha", "")),
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_max_sec"],
                fusion_stats["fusion_peak_vram_avg_mb"],
                fusion_stats["fusion_peak_vram_max_mb"],
                pair_avg_normalized_metric,
                "success", ""
            ]
        )

        print(f"[Done] KnOTS linear-mean baseline finished for pair: {pair_name}")
        print(f"[Done] pair_avg_normalized_metric = {pair_avg_normalized_metric}")

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)

        knots_cfg_str = (
            f"weights=0.5|0.5|scaling={knots_scaling_coeff}|"
            f"concat_across_output={knots_concat_across_output}|svd_tol={knots_svd_tol}|"
            f"search={knots_do_search}"
        )

        append_csv_row(
            registry_csv,
            [
                experiment_id, "GLUE_pairwise", method_name, model_name, pair_name,
                "|".join(task_targets), knots_cfg_str,
                task_cfg.get("rank", ""),
                str(task_cfg.get("lora_alpha", "")),
                "", "", "", "", "", "", "failed", error_msg
            ]
        )

        raise e


if __name__ == "__main__":
    main()
