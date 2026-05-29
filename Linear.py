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
from datetime import datetime
from safetensors import safe_open

from transformers import (
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

from eval_model import eval_iteris_model
from get_midfeatures import get_lora_pos, get_lora_matrix


GLUE_task_name = [
    "mnli", "rte", "cola", "sst2", "qqp", "qnli", "mrpc", "wnli",
]

SENTICAP_task_name = ["positive", "negative"]
FlickrStyle10k_task_name = ["roman", "humor"]
TASKS_blip_base = ["positive", "negative", "roman", "humor"]


def is_blip_model(model_name):
    return "blip" in str(model_name).lower()


def get_loras_path(task_type, model_name, lora_root=None):
    """
    Return task -> LoRA adapter directory.

    Compatibility:
      - GLUE_t5 keeps the original best_LoRA / OSRM_LoRA behavior.
      - TASKS_blip_base uses SentiCap BLIP LoRAs by default.
    """
    lora_path_dict = {}
    model_name_l = str(model_name).lower()

    if "t5" in model_name_l and task_type == "GLUE_t5":
        if lora_root is None:
            lora_root = "best_LoRA"
        lora_path_dict["cola"] = f"{lora_root}/T5-COLA-LoRA"
        lora_path_dict["sst2"] = f"{lora_root}/T5-SST2-LoRA"
        lora_path_dict["rte"]  = f"{lora_root}/T5-RTE-LoRA"
        lora_path_dict["qnli"] = f"{lora_root}/T5-QNLI-LoRA"
        lora_path_dict["qqp"]  = f"{lora_root}/T5-QQP-LoRA"
        lora_path_dict["mrpc"] = f"{lora_root}/T5-MRPC-LoRA"
        lora_path_dict["mnli"] = f"{lora_root}/T5-MNLI-LoRA"
        lora_path_dict["wnli"] = f"{lora_root}/T5-WNLI-LoRA"

    if task_type == "TASKS_blip_base":
        if lora_root is None:
            lora_root = "loras/SENTICAP-lora-blip"
        lora_path_dict["positive"] = f"{lora_root}/positive"
        lora_path_dict["negative"] = f"{lora_root}/negative"

        # Reserved for possible FlickrStyle10k experiments. These keys do not affect
        # positive/negative SentiCap experiments unless selected in task_targets.
        lora_path_dict["roman"] = "loras/FlickrStyle10k-lora-blip/roman"
        lora_path_dict["humor"] = "loras/FlickrStyle10k-lora-blip/humor"

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
    need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    if need_header:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            f.flush()
            os.fsync(f.fileno())


def append_csv_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def parse_float_list(value, default=None):
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, str):
        return [float(x.strip()) for x in value.split(",") if x.strip()]
    raise ValueError(f"Unsupported float list type: {type(value)}")


def normalize_lora_target_name(name):
    prefixes = [
        "base_model.model.",
        "base_model.",
        "model.",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def find_matching_weight_param(named_params, target_key):
    """
    Find target_key + '.weight' in dense model parameters, with a suffix fallback
    for small naming differences between PEFT adapter keys and dense model keys.
    """
    pname = target_key + ".weight"
    if pname in named_params:
        return pname, named_params[pname]

    norm_target = normalize_lora_target_name(target_key)
    pname2 = norm_target + ".weight"
    if pname2 in named_params:
        return pname2, named_params[pname2]

    candidates = []
    for name, param in named_params.items():
        if name.endswith(pname) or name.endswith(pname2):
            candidates.append((name, param))

    if len(candidates) == 1:
        return candidates[0]

    return None, None


def construct_base_model(model_name):
    model_name_l = str(model_name).lower()
    if "t5" in model_name_l:
        return T5ForConditionalGeneration.from_pretrained(model_name)
    if "bart" in model_name_l:
        return BartForConditionalGeneration.from_pretrained(model_name)
    if "blip" in model_name_l:
        return BlipForConditionalGeneration.from_pretrained(model_name)
    raise ValueError(f"[Linear] Unsupported model_name: {model_name}")


def get_primary_metric(task_name, eval_result, task_type):
    if task_type == "TASKS_blip_base":
        for key in ["acc", "style_acc", "style_accuracy", "eval_style_acc", "eval_accuracy", "accuracy"]:
            if key in eval_result and eval_result[key] not in ["", None]:
                v = float(eval_result[key])
                return key, v, v
        if "cider" in eval_result and eval_result["cider"] not in ["", None]:
            v = float(eval_result["cider"])
            return "cider", v, v
        return "acc", "", ""

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


def append_vlm_caption_row(
    vlm_results_csv,
    experiment_id,
    method_name,
    pair_name,
    task_targets,
    task_name,
    eval_result,
    merged_model_dir,
    log_file,
):
    bleu = eval_result.get("bleu", ["", "", "", ""])

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
            eval_result.get("acc", eval_result.get("style_acc", "")),
            eval_result.get("cider", eval_result.get("CIDEr", "")),
            get_bleu(0),
            get_bleu(1),
            get_bleu(2),
            get_bleu(3),
            eval_result.get("rougeL", ""),
            eval_result.get("div_1", ""),
            eval_result.get("div_2", ""),
            eval_result.get("vocab_size", ""),
            "validation",
            merged_model_dir,
            log_file,
        ],
    )


def merge_linear_param(
    lora_path,
    model_name,
    task_targets,
    lora_alpha,
    rank,
    linear_weights=None,
):
    """
    Linear baseline.

    For N LoRAs:
        Delta W_merge = sum_i weight_i * Delta W_i

    get_lora_matrix() returns:
        Delta W_i = alpha / rank * (B @ A)

    Then:
        W_base <- W_base + Delta W_merge
    """
    assert len(lora_path) == len(task_targets)

    num_loras = len(lora_path)

    if linear_weights is None:
        linear_weights = [1.0 / num_loras] * num_loras

    if len(linear_weights) != num_loras:
        raise ValueError(
            f"linear_weights 数量必须等于 LoRA 数量。"
            f"当前 linear_weights={linear_weights}, num_loras={num_loras}"
        )

    linear_weights = [float(w) for w in linear_weights]
    weight_sum = sum(linear_weights)
    if abs(weight_sum - 1.0) > 1e-6:
        print(f"[Warn] linear_weights sum = {weight_sum}, 自动归一化。")
        linear_weights = [w / weight_sum for w in linear_weights]

    if isinstance(lora_alpha, (int, float)):
        lora_alpha = [float(lora_alpha)] * num_loras
    if len(lora_alpha) != num_loras:
        raise ValueError(
            f"lora_alpha 数量必须等于 LoRA 数量。"
            f"当前 lora_alpha={lora_alpha}, num_loras={num_loras}"
        )

    print(f"[Linear] task_targets = {task_targets}")
    print(f"[Linear] lora_path = {lora_path}")
    print(f"[Linear] linear_weights = {linear_weights}")
    print(f"[Linear] lora_alpha = {lora_alpha}, rank = {rank}")
    print(f"[Linear] model_name = {model_name}")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    # 1. Parse LoRA target modules from first adapter.
    first_lora_file = os.path.join(lora_path[0], "adapter_model.safetensors")
    lora_keys = get_lora_pos(first_lora_file)
    lora_keys = sorted(set([normalize_lora_target_name(k) for k in lora_keys]))

    print(f"[Linear] Found {len(lora_keys)} LoRA target modules.")
    for k in lora_keys[:20]:
        print(f"    {k}")

    # 2. Open every adapter.
    lora_files = [
        os.path.join(path, "adapter_model.safetensors")
        for path in lora_path
    ]
    tensors_lora = [safe_open(path, framework="pt") for path in lora_files]

    # 3. Linear merge LoRA deltas.
    merged_delta_dict = {}

    with torch.no_grad():
        for idx in lora_keys:
            delta_list = []

            for i in range(num_loras):
                delta_w = get_lora_matrix(
                    model_name=model_name,
                    load_tensor=tensors_lora[i],
                    idx_str=idx,
                    alpha=float(lora_alpha[i]),
                    rank=rank,
                    no_weight=True,
                )

                if delta_w is None:
                    raise ValueError(
                        f"[Linear] Cannot find LoRA matrix for layer={idx}, "
                        f"lora_file={lora_files[i]}"
                    )

                delta_list.append(delta_w.float())

            merged_delta = sum(
                linear_weights[i] * delta_list[i]
                for i in range(num_loras)
            )

            merged_delta_dict[idx] = merged_delta.cpu()

    print("[Linear] All LoRA delta matrices merged.")

    # 4. Load base model.
    model = construct_base_model(model_name).to("cuda")
    named_params = dict(model.named_parameters())

    # 5. Add merged delta into dense base weights.
    number_update = 0
    missing_targets = []
    shape_mismatch = []

    with torch.no_grad():
        for key, delta_cpu in merged_delta_dict.items():
            pname, param = find_matching_weight_param(named_params, key)

            if param is None:
                missing_targets.append(key)
                continue

            delta = delta_cpu.to(param.device).to(param.dtype)
            if tuple(delta.shape) != tuple(param.shape):
                shape_mismatch.append((key, tuple(delta.shape), tuple(param.shape)))
                continue

            param.copy_(param + delta)
            number_update += 1

    if number_update == len(merged_delta_dict):
        print("[Linear] All target modules updated successfully.")
    else:
        print(
            f"[Linear][Warn] Updated {number_update}/{len(merged_delta_dict)} modules. "
            f"Missing={len(missing_targets)}, shape_mismatch={len(shape_mismatch)}"
        )
        for item in missing_targets[:10]:
            print(f"    [Missing] {item}")
        for item in shape_mismatch[:10]:
            print(f"    [ShapeMismatch] {item}")

    fusion_time = round(time.time() - start_time, 4)
    fusion_peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)

    fusion_stats = {
        "fusion_iter_time_avg_sec": fusion_time,
        "fusion_iter_time_max_sec": fusion_time,
        "fusion_peak_vram_avg_mb": fusion_peak_vram_mb,
        "fusion_peak_vram_max_mb": fusion_peak_vram_mb,
    }

    print(f"[Linear] Fusion time: {fusion_time} sec")
    print(f"[Linear] Fusion peak VRAM: {fusion_peak_vram_mb} MB")

    return model, fusion_stats


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
    lora_alpha = task_cfg.get("lora_alpha", [32 for _ in task_targets])
    rank = task_cfg.get("rank", 8)
    linear_weights = task_cfg.get("linear_weights", None)
    tokenizer = (
        AutoTokenizer.from_pretrained(model_name)
        if not is_blip_model(model_name)
        else AutoProcessor.from_pretrained(model_name)
    )

    # LoRA source.
    # GLUE:
    #   default / best / normal / gaussian -> best_LoRA
    #   osrm -> OSRM_LoRA
    # BLIP:
    #   default -> loras/SENTICAP-lora-blip
    lora_source = str(task_cfg.get("lora_source", "default")).lower()
    if task_type == "TASKS_blip_base":
        lora_root = task_cfg.get("lora_root", "loras/SENTICAP-lora-blip")
    else:
        if lora_source in ["default", "best", "normal", "gaussian"]:
            lora_root = task_cfg.get("lora_root", "best_LoRA")
        elif lora_source == "osrm":
            lora_root = task_cfg.get("lora_root", "OSRM_LoRA")
        else:
            # Allow directly using lora_source as a directory name.
            lora_root = task_cfg.get("lora_root", lora_source)

    print(f"[Linear] lora_source = {lora_source}")
    print(f"[Linear] lora_root = {lora_root}")

    if linear_weights is None:
        linear_weights = [1.0 / len(task_targets)] * len(task_targets)

    lora_path_dict = get_loras_path(
        task_type=task_type,
        model_name=model_name,
        lora_root=lora_root,
    )
    lora_path = [lora_path_dict[task] for task in task_targets]

    for task, path in zip(task_targets, lora_path):
        if not os.path.exists(os.path.join(path, "adapter_model.safetensors")):
            raise FileNotFoundError(
                f"[Linear] Cannot find adapter_model.safetensors for task={task}, path={path}. "
                f"Please check lora_source={lora_source}, lora_root={lora_root}."
            )

    pair_name = "_".join(task_targets)
    if lora_source == "osrm":
        default_method_name = "OSRM_Linear"
    elif task_type == "TASKS_blip_base":
        default_method_name = "Linear_BLIP"
    else:
        default_method_name = "Linear"

    method_name = task_cfg.get("linear_method_name", default_method_name)
    experiment_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{method_name}_{pair_name}"

    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)

    results_csv = os.path.join(results_dir, "pair_merge_results.csv")
    registry_csv = os.path.join(results_dir, "experiment_registry.csv")
    vlm_results_csv = os.path.join(results_dir, "vlm_caption_results.csv")

    results_header = [
        "experiment_id", "method", "pair_name", "task_a", "task_b",
        "evaluated_task", "primary_metric_name", "primary_metric_value",
        "normalized_metric", "eval_accuracy", "eval_mcc", "eval_f1",
        "eval_loss", "eval_runtime", "eval_samples_per_second",
        "eval_steps_per_second", "eval_peak_vram_mb",
        "split", "merged_model_dir", "log_file", "error",
    ]

    registry_header = [
        "experiment_id", "experiment_type", "method", "model_name", "pair_name",
        "task_targets", "linear_weights", "rank", "lora_alpha",
        "lora_source", "lora_root",
        "fusion_total_time_sec", "fusion_iter_time_avg_sec",
        "fusion_iter_time_max_sec", "fusion_peak_vram_avg_mb",
        "fusion_peak_vram_max_mb", "pair_avg_normalized_metric",
        "merged_model_dir", "status", "error",
    ]

    ensure_csv_header(results_csv, results_header)
    ensure_csv_header(registry_csv, registry_header)
    if task_type == "TASKS_blip_base":
        ensure_vlm_caption_header(vlm_results_csv)

    log_file = os.environ.get("LOG_FILE", "")

    if task_cfg.get("linear_merged_model_dir", None) is not None:
        merged_model_dir = task_cfg["linear_merged_model_dir"]
    else:
        merged_model_dir = f"merged_model/{method_name}_{pair_name}"

    try:
        model, fusion_stats = merge_linear_param(
            lora_path=lora_path,
            model_name=model_name,
            task_targets=task_targets,
            lora_alpha=lora_alpha,
            rank=rank,
            linear_weights=linear_weights,
        )

        if task_cfg.get("save", 0):
            os.makedirs(merged_model_dir, exist_ok=True)
            model.save_pretrained(merged_model_dir)
            try:
                tokenizer.save_pretrained(merged_model_dir)
            except Exception as e:
                print(f"[Linear][Warn] tokenizer/processor save failed: {type(e).__name__}: {e}")
            print(f"[Linear] Merged model saved to {merged_model_dir}")

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

            print(f"[Debug][EvalResults][{task_name}] {eval_result}", flush=True)

            eval_accuracy = eval_result.get("eval_accuracy", "")
            eval_mcc = eval_result.get("eval_MCC", "")
            eval_f1 = eval_result.get("eval_f1-score", "")
            eval_loss = eval_result.get("eval_loss", "")
            eval_runtime = eval_result.get("eval_runtime", eval_result.get("eval_wall_time_sec", ""))
            eval_sps = eval_result.get("eval_samples_per_second", "")
            eval_stepsps = eval_result.get("eval_steps_per_second", "")
            eval_peak_vram_mb = eval_result.get("eval_peak_vram_mb", "")

            primary_metric_name, primary_metric_value, normalized_metric = get_primary_metric(
                task_name=task_name,
                eval_result=eval_result,
                task_type=task_type,
            )
            if task_type == "TASKS_blip_base":
                eval_accuracy = primary_metric_value
                append_vlm_caption_row(
                    vlm_results_csv=vlm_results_csv,
                    experiment_id=experiment_id,
                    method_name=method_name,
                    pair_name=pair_name,
                    task_targets=task_targets,
                    task_name=task_name,
                    eval_result=eval_result,
                    merged_model_dir=merged_model_dir,
                    log_file=log_file,
                )

            if normalized_metric != "":
                normalized_metrics.append(float(normalized_metric))

            append_csv_row(
                results_csv,
                [
                    experiment_id, method_name, pair_name, task_targets[0], task_targets[1],
                    task_name, primary_metric_name, primary_metric_value,
                    normalized_metric, eval_accuracy, eval_mcc, eval_f1,
                    eval_loss, eval_runtime, eval_sps, eval_stepsps,
                    eval_peak_vram_mb, "validation", merged_model_dir, log_file, "",
                ],
            )

        pair_avg_normalized_metric = float(np.mean(normalized_metrics)) if len(normalized_metrics) > 0 else ""

        append_csv_row(
            registry_csv,
            [
                experiment_id,
                "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise",
                method_name,
                model_name,
                pair_name,
                "|".join(task_targets),
                "|".join(map(str, linear_weights)),
                rank,
                "|".join(map(str, lora_alpha)),
                lora_source,
                lora_root,
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_max_sec"],
                fusion_stats["fusion_peak_vram_avg_mb"],
                fusion_stats["fusion_peak_vram_max_mb"],
                pair_avg_normalized_metric,
                merged_model_dir,
                "success",
                "",
            ],
        )

        print(f"[Done] {method_name} baseline finished for pair: {pair_name}")
        print(f"[Done] pair_avg_normalized_metric = {pair_avg_normalized_metric}")
        print(f"[Done] results saved to: {results_csv}")
        if task_type == "TASKS_blip_base":
            print(f"[Done] VLM caption results saved to: {vlm_results_csv}")

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)

        try:
            append_csv_row(
                registry_csv,
                [
                    experiment_id,
                    "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise",
                    method_name,
                    model_name,
                    pair_name,
                    "|".join(task_targets),
                    "|".join(map(str, linear_weights)),
                    rank,
                    "|".join(map(str, lora_alpha)),
                    lora_source,
                    lora_root,
                    "", "", "", "", "", "",
                    merged_model_dir,
                    "failed",
                    error_msg,
                ],
            )
        except Exception:
            pass

        raise e


if __name__ == "__main__":
    main()
