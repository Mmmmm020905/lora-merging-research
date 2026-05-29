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
from get_midfeatures import get_lora_matrix, get_all_midfeatures


GLUE_task_name = [
    "mnli", "rte", "cola", "sst2", "qqp", "qnli", "mrpc", "wnli",
]


def is_blip_model(model_name):
    return "blip" in str(model_name).lower()


def get_loras_path(task_type, model_name, lora_root=None):
    """
    Compatible LoRA path resolver.

    GLUE_t5:
        default lora_root = best_LoRA
        e.g. best_LoRA/T5-MNLI-LoRA

    TASKS_blip_base:
        default lora_root = loras/SENTICAP-lora-blip
        e.g. loras/SENTICAP-lora-blip/positive
    """
    model_name_lower = str(model_name).lower()
    lora_path_dict = {}

    if "t5" in model_name_lower and task_type == "GLUE_t5":
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

        # reserved for possible future FlickrStyle10k experiments
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
    if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_csv_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def reduce_non_diag(G, alpha):
    """
    G: [K, d, d]
    return:
        G_tilde = alpha * G + (1 - alpha) * diag(G)
    """
    diag_only = torch.diag_embed(torch.diagonal(G, dim1=-2, dim2=-1))
    return alpha * G + (1.0 - alpha) * diag_only


def regmean_solve(W_list, X_tensor, regmean_alpha=0.1, regmean_eps=1e-6):
    """
    W_list:   [K, out_dim, in_dim]
    X_tensor: feature tensor from get_all_midfeatures().

    Current engineering convention:
        merged_W = (sum_i W_i G_i_tilde) (sum_i G_i_tilde)^(-1)
    """
    with torch.no_grad():
        # same layout convention as IterIS solution_matrix:
        # [num_chunks, K, inner_num, hidden] -> [K, T, hidden]
        X_list = X_tensor.transpose(0, 1).flatten(start_dim=1, end_dim=2).float()

        G_list = torch.matmul(X_list.transpose(-1, -2), X_list)
        G_tilde = reduce_non_diag(G_list, regmean_alpha)

        term1 = torch.sum(torch.matmul(W_list.float(), G_tilde), dim=0).double()
        term2 = torch.sum(G_tilde, dim=0).double()

        eye = torch.eye(term2.size(0), device=term2.device, dtype=term2.dtype)
        term2 = term2 + regmean_eps * eye

        merged_W = torch.linalg.solve(term2.t(), term1.t()).t()
        return merged_W.float().cpu()


def construct_base_model(model_name):
    model_name_lower = str(model_name).lower()

    if "t5" in model_name_lower:
        return T5ForConditionalGeneration.from_pretrained(model_name)
    if "bart" in model_name_lower:
        return BartForConditionalGeneration.from_pretrained(model_name)
    if "blip" in model_name_lower:
        return BlipForConditionalGeneration.from_pretrained(model_name)

    raise ValueError(f"[RegMean] Unsupported model_name: {model_name}")


def merge_regmean_param(
    lora_path,
    model_name,
    task_targets,
    lora_alpha,
    rank,
    seed,
    max_length,
    select_long,
    inner_num,
    outer_num,
    samples_num,
    if_divide,
    if_balance,
    shuffle,
    regmean_alpha,
    regmean_eps,
    max_new_tokens=None,
):
    """
    RegMean baseline for LoRA deltas.

    Steps:
    1) get_all_midfeatures() extracts input features X for each LoRA target module.
       For BLIP/SentiCap, this internally uses image + prompt batches.
    2) For each LoRA target module:
         - read each task LoRA delta W_i
         - compute G_i = X_i^T X_i
         - G_tilde_i = alpha * G_i + (1-alpha) * diag(G_i)
         - merged_delta = (sum_i W_i G_tilde_i)(sum_i G_tilde_i)^(-1)
    3) Add merged_delta to the base model weights.
    """
    assert len(lora_path) == len(task_targets), "lora_path 数量必须和 task_targets 一致"

    print(f"[RegMean] task_targets = {task_targets}")
    print(f"[RegMean] lora_path = {lora_path}")
    print(f"[RegMean] lora_alpha = {lora_alpha}, rank = {rank}")
    print(f"[RegMean] regmean_alpha = {regmean_alpha}, regmean_eps = {regmean_eps}")
    print(
        f"[RegMean] samples_num = {samples_num}, select_long = {select_long}, "
        f"inner_num = {inner_num}, outer_num = {outer_num}, "
        f"if_divide = {if_divide}, if_balance = {if_balance}, shuffle = {shuffle}"
    )

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    generation_kwargs = {}
    if max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = max_new_tokens

    # 1) collect mid-features
    _, X_dict = get_all_midfeatures(
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

    print(f"[RegMean] Collected mid-features for {len(X_dict)} LoRA target modules.")

    # 2) open LoRA safetensors
    lora_files = [os.path.join(path, "adapter_model.safetensors") for path in lora_path]
    tensors_lora = [safe_open(path, framework="pt") for path in lora_files]

    # 3) RegMean solve per LoRA target
    merged_delta_dict = {}

    with torch.no_grad():
        for idx in X_dict.keys():
            W_list = []
            for i in range(len(tensors_lora)):
                delta_w = get_lora_matrix(
                    model_name=model_name,
                    load_tensor=tensors_lora[i],
                    idx_str=idx,
                    alpha=lora_alpha[i],
                    rank=rank,
                    no_weight=True,
                )

                if delta_w is None:
                    raise ValueError(
                        f"[RegMean] Cannot find LoRA matrix for layer={idx}, "
                        f"lora_file={lora_files[i]}"
                    )

                W_list.append(delta_w.float())

            W_list = torch.stack(W_list).to("cuda")
            X_tensor = X_dict[idx].to("cuda")

            merged_delta = regmean_solve(
                W_list=W_list,
                X_tensor=X_tensor,
                regmean_alpha=regmean_alpha,
                regmean_eps=regmean_eps,
            )

            merged_delta_dict[idx] = merged_delta

            del W_list, X_tensor
            torch.cuda.empty_cache()

    print("[RegMean] All LoRA delta matrices merged.")

    # 4) load base model
    model = construct_base_model(model_name).to("cuda")

    # 5) add merged delta to base model target weights
    number_update = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if not name.endswith(".weight"):
                continue

            key = name[:-7]
            if key in merged_delta_dict:
                delta = merged_delta_dict[key].to(param.device).to(param.dtype)
                param.copy_(param + delta)
                number_update += 1

    if number_update == len(merged_delta_dict):
        print("[RegMean] All target modules updated successfully.")
    else:
        print(
            f"[RegMean][Warn] Updated {number_update}/{len(merged_delta_dict)} modules. "
            f"请检查 LoRA 层名和 base model 参数名是否完全匹配。"
        )

    fusion_time = round(time.time() - start_time, 4)
    fusion_peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)

    fusion_stats = {
        "fusion_iter_time_avg_sec": fusion_time,
        "fusion_iter_time_max_sec": fusion_time,
        "fusion_peak_vram_avg_mb": fusion_peak_vram_mb,
        "fusion_peak_vram_max_mb": fusion_peak_vram_mb,
    }

    print(f"[RegMean] Fusion time: {fusion_time} sec")
    print(f"[RegMean] Fusion peak VRAM: {fusion_peak_vram_mb} MB")

    return model, fusion_stats


def get_primary_metric_any(task_name, eval_result, task_type):
    if task_type == "TASKS_blip_base":
        if "acc" in eval_result and eval_result["acc"] not in ["", None]:
            v = float(eval_result["acc"])
            return "acc", v, v

        for key in ["style_acc", "style_accuracy", "eval_style_acc", "eval_accuracy", "accuracy"]:
            if key in eval_result and eval_result[key] not in ["", None]:
                v = float(eval_result[key])
                return key, v, v

        # fallback to CIDEr to avoid empty rows if style score is unavailable
        for key in ["cider", "CIDEr", "eval_cider", "eval_CIDEr"]:
            if key in eval_result and eval_result[key] not in ["", None]:
                v = float(eval_result[key])
                return key, v, v

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
    lora_alpha = task_cfg.get("lora_alpha", [32 for _ in task_targets])
    rank = task_cfg.get("rank", 8)

    # LoRA source remains compatible with Linear.py style configs.
    lora_source = str(task_cfg.get("lora_source", "default")).lower().strip()
    if task_type == "TASKS_blip_base":
        # For BLIP, default path is loras/SENTICAP-lora-blip/{positive,negative}.
        lora_root = task_cfg.get("lora_root", "loras/SENTICAP-lora-blip")
    else:
        if lora_source in ["default", "best", "normal", "gaussian"]:
            lora_root = task_cfg.get("lora_root", "best_LoRA")
        elif lora_source == "osrm":
            lora_root = task_cfg.get("lora_root", "OSRM_LoRA")
        else:
            lora_root = task_cfg.get("lora_root", lora_source)

    print(f"[RegMean] lora_source = {lora_source}")
    print(f"[RegMean] lora_root = {lora_root}")

    # RegMean paper commonly uses alpha=0.9, while earlier T5-base GLUE setup used 0.1.
    default_regmean_alpha = 0.1 if "t5" in str(model_name).lower() else 0.9
    regmean_alpha = float(task_cfg.get("regmean_alpha", default_regmean_alpha))
    regmean_eps = float(task_cfg.get("regmean_eps", 1e-6))

    max_length = task_cfg.get("max_length", 512)
    max_new_tokens = task_cfg.get("max_new_tokens", None)
    select_long = task_cfg.get("select_long", 40)
    inner_num = task_cfg.get("inner_num", 2)
    outer_num = task_cfg.get("outer_num", 10)
    samples_num = task_cfg.get("samples_num", 20)
    if_divide = task_cfg.get("if_divide", True)
    if_balance = task_cfg.get("if_balance", True)
    shuffle = task_cfg.get("shuffle", False)

    tokenizer = (
        AutoTokenizer.from_pretrained(model_name)
        if not is_blip_model(model_name)
        else AutoProcessor.from_pretrained(model_name)
    )

    lora_path_dict = get_loras_path(task_type, model_name, lora_root=lora_root)
    lora_path = [lora_path_dict[task] for task in task_targets]

    for task, path in zip(task_targets, lora_path):
        adapter_path = os.path.join(path, "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"[RegMean] Cannot find adapter_model.safetensors for task={task}: {adapter_path}"
            )

    pair_name = "_".join(task_targets)
    method_name = task_cfg.get("regmean_method_name", "RegMean")
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{method_name}_{pair_name}"

    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)

    results_csv = os.path.join(results_dir, "pair_merge_results.csv")
    registry_csv = os.path.join(results_dir, "experiment_registry.csv")
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
        "task_targets", "linear_weights", "rank", "lora_alpha",
        "fusion_total_time_sec", "fusion_iter_time_avg_sec",
        "fusion_iter_time_max_sec", "fusion_peak_vram_avg_mb",
        "fusion_peak_vram_max_mb", "pair_avg_normalized_metric",
        "status", "error"
    ]

    ensure_csv_header(results_csv, results_header)
    ensure_csv_header(registry_csv, registry_header)
    if task_type == "TASKS_blip_base":
        ensure_vlm_caption_header(vlm_results_csv)

    status = "success"
    error_msg = ""

    try:
        model, fusion_stats = merge_regmean_param(
            lora_path=lora_path,
            model_name=model_name,
            task_targets=task_targets,
            lora_alpha=lora_alpha,
            rank=rank,
            seed=seed,
            max_length=max_length,
            select_long=select_long,
            inner_num=inner_num,
            outer_num=outer_num,
            samples_num=samples_num,
            if_divide=if_divide,
            if_balance=if_balance,
            shuffle=shuffle,
            regmean_alpha=regmean_alpha,
            regmean_eps=regmean_eps,
            max_new_tokens=max_new_tokens,
        )

        merged_model_dir = task_cfg.get(
            "regmean_merged_model_dir",
            f"merged_model/{method_name}_{pair_name}",
        )

        if task_cfg.get("save", 0):
            os.makedirs(merged_model_dir, exist_ok=True)
            model.save_pretrained(merged_model_dir)
            tokenizer.save_pretrained(merged_model_dir)
            print(f"[RegMean] Merged model saved to {merged_model_dir}")

        normalized_metrics = []

        for task_name in task_targets:
            print(f"[Eval] Evaluating merged model on {task_name}...")

            eval_result = eval_iteris_model(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                task_name=task_name,
                max_length=max_length,
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
                task_name=task_name,
                eval_result=eval_result,
                task_type=task_type,
            )
            if task_type == "TASKS_blip_base":
                eval_accuracy = primary_metric_value

            if normalized_metric != "":
                normalized_metrics.append(float(normalized_metric))

            if task_type == "TASKS_blip_base":
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

        pair_avg_normalized_metric = float(np.mean(normalized_metrics)) if normalized_metrics else ""

        regmean_cfg_str = f"alpha={regmean_alpha}|eps={regmean_eps}|samples={samples_num}"

        append_csv_row(
            registry_csv,
            [
                experiment_id,
                "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise",
                method_name,
                model_name,
                pair_name,
                "|".join(task_targets),
                regmean_cfg_str,
                rank,
                "|".join(map(str, lora_alpha)),
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_max_sec"],
                fusion_stats["fusion_peak_vram_avg_mb"],
                fusion_stats["fusion_peak_vram_max_mb"],
                pair_avg_normalized_metric,
                status,
                error_msg,
            ]
        )

        print(f"[Done] RegMean baseline finished for pair: {pair_name}")
        print(f"[Done] pair_avg_normalized_metric = {pair_avg_normalized_metric}")
        print(f"[Done] results saved to: {results_csv}")
        if task_type == "TASKS_blip_base":
            print(f"[Done] VLM caption results saved to: {vlm_results_csv}")

    except Exception as e:
        status = "failed"
        error_msg = traceback.format_exc()
        print(error_msg)

        regmean_cfg_str = f"alpha={regmean_alpha}|eps={regmean_eps}|samples={samples_num}"

        append_csv_row(
            registry_csv,
            [
                experiment_id,
                "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise",
                method_name,
                model_name,
                pair_name,
                "|".join(task_targets),
                regmean_cfg_str,
                rank,
                "|".join(map(str, lora_alpha)),
                "", "", "", "", "", "", status, error_msg,
            ]
        )

        raise e


if __name__ == "__main__":
    main()
