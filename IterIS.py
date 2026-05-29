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
from sklearn.metrics import f1_score
from eval_model import eval_iteris_model
from get_midfeatures import T5WithHooks, BartWithHooks, BlipWithHook
from torch.optim.lr_scheduler import StepLR
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, BartForConditionalGeneration, AutoTokenizer, AutoProcessor
from get_midfeatures import get_all_midfeatures, get_samples, get_pretrain_matrix, get_lora_matrix
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

    if 't5' in model_name and task_type == "GLUE_t5":
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

        # 下面两个先保留，后续如果做 FlickrStyle10k 再用；
        # 如果没有 roman/humor LoRA，不影响 positive/negative。
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
            if not os.path.exists(csv_path):
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
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


def main():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--config', type=str, default="config/methods-config/iteris-config.yaml", help="Path to the config file")
    parser.add_argument('--task_type', type=str, choices=['GLUE_t5', 'EMOTION_t5_large', 'TASKS_blip_base'], 
                        default='GLUE_t5', help="Choose a task type from the list of options.")
    args = parser.parse_args()
    task_type = args.task_type
    with open(args.config, 'r') as file:
        config_data = yaml.safe_load(file)
    set_seed(config_data['seed'])
    model_name = config_data[task_type]['model_name']
    task_targets = config_data[task_type]['task_targets']
    pair_name = "_".join(task_targets)
    experiment_id = f"{task_type}_{pair_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)

    registry_csv = os.path.join(results_dir, "experiment_registry.csv")
    results_csv = os.path.join(results_dir, "pair_merge_results.csv")
    vlm_results_csv = os.path.join(results_dir, "vlm_caption_results.csv")

    merged_model_dir = f"merged_model/{pair_name}"
    log_file = os.environ.get("LOG_FILE", f"logs/{pair_name}.log")
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lora_path = [get_loras_path(task_type, model_name)[item] for item in task_targets]
    with_pretrain_matrix = config_data[task_type]['with_pretrain_matrix']
    tokenizer = AutoTokenizer.from_pretrained(model_name) if 'blip' not in model_name else AutoProcessor.from_pretrained(model_name)
    save = config_data[task_type]['save']
    pair_name = "_".join(task_targets)
    save_dir = f"merged_model/{pair_name}"

    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ensure_csv_header(
    registry_csv,
        [
        "experiment_id","stage","method","base_model","pair_name","task_a","task_b",
        "lora_a_path","lora_b_path","config_path","seed","max_iter","alpha_1","alpha_2",
        "reg_ceof","rank","samples_num","if_balance","shuffle","select_long",
        "with_pretrain_matrix","save_merged_model","start_time","end_time",
        "fusion_total_time_sec","fusion_iter_time_avg_sec","fusion_iter_time_max_sec",
        "fusion_peak_vram_avg_mb","fusion_peak_vram_max_mb","pair_avg_normalized_metric",
        "merged_model_dir","log_file","status","notes"
        ]
    )

    ensure_csv_header(
        results_csv,
        [
            "experiment_id","method","pair_name","task_a","task_b","evaluated_task",
            "primary_metric_name","primary_metric_value","normalized_metric",
            "eval_accuracy","eval_mcc","eval_f1","eval_loss","eval_runtime",
            "eval_samples_per_second","eval_steps_per_second","eval_peak_vram_mb",
            "split","merged_model_dir","log_file","notes"
        ]
    )

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
            ]
        )

        # IterIS algorithm
    start_time = time.time()
    status = "done"
    notes = ""
    fusion_stats = {
        "fusion_iter_time_avg_sec": "",
        "fusion_iter_time_max_sec": "",
        "fusion_peak_vram_avg_mb": "",
        "fusion_peak_vram_max_mb": "",
    }
    pair_norm_metrics = []
    model = None
    wrote_result_row = False

    try:
        model, fusion_stats = update_param(
            task_targets=task_targets,
            lora_path=lora_path,
            model_name=model_name,
            with_pretrain_matrix=with_pretrain_matrix,
            max_iter=config_data[task_type]['max_iter'],
            max_length=config_data[task_type]['max_length'],
            lora_alpha=config_data[task_type]['lora_alpha'],
            alpha_1=config_data[task_type]['alpha_1'],
            alpha_2=config_data[task_type]['alpha_2'],
            reg_ceof=config_data[task_type]['reg_ceof'],
            rank=config_data[task_type]['rank'],
            samples_num=config_data[task_type]['samples_num'],
            manual_ceof=config_data[task_type]['manual_ceof'],
            if_divide=config_data[task_type]['if_divide'],
            if_balance=config_data[task_type]['if_balance'],
            inner_num=config_data[task_type]['inner_num'],
            outer_num=config_data[task_type]['outer_num'],
            seed=config_data['seed'],
            select_long=config_data[task_type]['select_long'],
            shuffle=config_data[task_type]['shuffle'],
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(elapsed_time)

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()

        # model evaluation
        for task_name in task_targets:
            try:
                eval_results = eval_iteris_model(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    task_name=task_name,
                    max_length=config_data[task_type]['max_length'],
                    per_device_eval_batch_size=config_data[task_type]['per_device_eval_batch_size'],
                )

                if task_type == "TASKS_blip_base":
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
                            "IterIS",
                            pair_name,
                            task_targets[0],
                            task_targets[1],
                            task_name,
                            eval_results.get("acc", ""),
                            eval_results.get("cider", ""),
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
                        ]
                    )

                print(f"[Debug][EvalResults][{task_name}] {eval_results}", flush=True)

                eval_accuracy = eval_results.get("eval_accuracy", "")
                eval_mcc = eval_results.get("eval_MCC", "")
                eval_f1 = eval_results.get("eval_f1-score", "")
                eval_loss = eval_results.get("eval_loss", "")
                eval_runtime = eval_results.get("eval_runtime", eval_results.get("eval_wall_time_sec", ""))
                eval_sps = eval_results.get("eval_samples_per_second", "")
                eval_stepsps = eval_results.get("eval_steps_per_second", "")
                eval_peak_vram_mb = eval_results.get("eval_peak_vram_mb", "")

                if task_type == "TASKS_blip_base":
                    # BLIP/SentiCap 评估不是 GLUE accuracy/MCC。
                    # 不同版本 eval_model.py 可能返回 acc/style_acc/CIDEr/BLEU 等不同 key，
                    # 所以这里做兼容式解析。
                    style_keys = [
                        "eval_accuracy",
                        "accuracy",
                        "acc",
                        "style_acc",
                        "style_accuracy",
                        "eval_style_acc",
                        "eval_style_accuracy",
                        "senti_acc",
                        "eval_senti_acc",
                    ]

                    primary_metric_value = ""
                    primary_metric_name = "style_accuracy"

                    for k in style_keys:
                        if k in eval_results and eval_results[k] not in ["", None]:
                            primary_metric_value = eval_results[k]
                            primary_metric_name = k
                            break

                    # 如果没有 style accuracy，就退而求其次用 CIDEr；至少不要写 NaN。
                    if primary_metric_value == "":
                        cider_keys = ["CIDEr", "cider", "eval_CIDEr", "eval_cider"]
                        for k in cider_keys:
                            if k in eval_results and eval_results[k] not in ["", None]:
                                primary_metric_value = eval_results[k]
                                primary_metric_name = k
                                break

                    # 兼容原 CSV：把 VLM style acc 也放到 eval_accuracy 这一列里，方便汇总。
                    eval_accuracy = primary_metric_value
                    normalized_metric = primary_metric_value

                elif task_name == "cola":
                    primary_metric_name = "MCC"
                    primary_metric_value = eval_mcc
                    normalized_metric = (eval_mcc + 1) / 2 if eval_mcc != "" else ""
                else:
                    primary_metric_name = "accuracy"
                    primary_metric_value = eval_accuracy
                    normalized_metric = eval_accuracy

                if normalized_metric != "":
                    pair_norm_metrics.append(normalized_metric)

                append_csv_row(
                    results_csv,
                    [
                        experiment_id, "IterIS", pair_name, task_targets[0], task_targets[1],
                        task_name, primary_metric_name, primary_metric_value, normalized_metric,
                        eval_accuracy, eval_mcc, eval_f1, eval_loss, eval_runtime,
                        eval_sps, eval_stepsps, eval_peak_vram_mb,
                        "validation", merged_model_dir, log_file, ""
                    ]
                )
                wrote_result_row = True

            except Exception as e:
                # 评测失败也要写一行，避免 results_csv 为空表
                append_csv_row(
                    results_csv,
                    [
                        experiment_id, "IterIS", pair_name, task_targets[0], task_targets[1],
                        task_name, "", "", "",
                        "", "", "", "", "",
                        "", "", "",
                        "validation", merged_model_dir, log_file,
                        f"EVAL_FAILED: {type(e).__name__}: {str(e)}"
                    ]
                )
                wrote_result_row = True
                raise

        # save merged model after evaluation
        if save == 1:
            os.makedirs(merged_model_dir, exist_ok=True)
            model.save_pretrained(merged_model_dir)
            tokenizer.save_pretrained(merged_model_dir)
            print(f"Merged model saved to: {merged_model_dir}")

        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        gc.collect()

    except Exception as e:
        status = "failed"
        notes = traceback.format_exc()

        # 如果融合阶段就挂了，连一行结果都没写过，那就补一行失败记录
        if not wrote_result_row:
            append_csv_row(
                results_csv,
                [
                    experiment_id, "IterIS", pair_name, task_targets[0], task_targets[1],
                    "__merge_failed__", "", "", "",
                    "", "", "", "", "",
                    "", "", "",
                    "validation", merged_model_dir, log_file,
                    f"MERGE_FAILED: {type(e).__name__}: {str(e)}"
                ]
            )

    finally:
        end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed_time = round(time.time() - start_time, 4)
        pair_avg_normalized_metric = round(float(np.mean(pair_norm_metrics)), 6) if len(pair_norm_metrics) > 0 else ""

        append_csv_row(
            registry_csv,
            [
                experiment_id, "GLUE_pairwise", "IterIS", model_name, pair_name,
                task_targets[0], task_targets[1],
                lora_path[0], lora_path[1],
                args.config, config_data["seed"], config_data[task_type]["max_iter"],
                config_data[task_type]["alpha_1"], config_data[task_type]["alpha_2"],
                config_data[task_type]["reg_ceof"], config_data[task_type]["rank"],
                config_data[task_type]["samples_num"], config_data[task_type]["if_balance"],
                config_data[task_type]["shuffle"], config_data[task_type]["select_long"],
                with_pretrain_matrix, save, start_dt, end_dt,
                elapsed_time,
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_max_sec"],
                fusion_stats["fusion_peak_vram_avg_mb"],
                fusion_stats["fusion_peak_vram_max_mb"],
                pair_avg_normalized_metric,
                merged_model_dir, log_file, status, notes.replace("\n", " | ")[:5000]
            ]
        )

        if status == "done":
            print(f"\n[Done] IterIS finished for pair: {pair_name}")
            print(f"[Done] pair_avg_normalized_metric = {pair_avg_normalized_metric}")
            print(f"[Done] results saved to: {results_csv}")
            print(f"[Done] registry saved to: {registry_csv}")
        else:
            print(f"\n[Fail] IterIS failed for pair: {pair_name}")
            print(f"[Fail] status = {status}")
            print(f"[Fail] notes = {str(notes)[:1000]}")

        if status == "failed":
            raise RuntimeError(f"Experiment failed: {pair_name}. See log: {log_file}")
    

if __name__ == "__main__":
    main()
