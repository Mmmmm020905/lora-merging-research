#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import itertools
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from multiprocessing import Process, Manager

import pandas as pd
import yaml

# TASKS = ["mnli", "cola"]
TASKS = ["mnli", "rte", "cola", "sst2", "qqp", "qnli", "mrpc", "wnli"]
RESULT_FILENAMES = [
    "pair_merge_results.csv",
    "experiment_registry.csv",
    "drc_alpha_search_results.csv",
    "dare_search_results.csv",
]


def backup_if_exists(path: Path, batch_dir: Path):
    if path.exists():
        batch_dir.mkdir(parents=True, exist_ok=True)
        dst = batch_dir / path.name
        if dst.exists():
            dst = batch_dir / f"old_{datetime.now().strftime('%H%M%S')}_{path.name}"
        shutil.move(str(path), str(dst))
        print(f"[Backup] {path} -> {dst}")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_method_token(value: str) -> str:
    return str(value).strip().lower().replace("_", "-") if value is not None else ""


def get_dare_best_model_dir(pair_name: str, task_cfg: dict) -> str:
    """
    返回 DARE baseline 搜索/融合后保存的稳定 coarse model 路径。
    必须和 DARE.py 中保存 best coarse model 的默认命名保持一致。
    """
    if task_cfg.get("dare_tals_coarse_model_dir"):
        return str(task_cfg["dare_tals_coarse_model_dir"])

    merge_method = normalize_method_token(task_cfg.get("dare_merge_method", "dare"))
    if merge_method == "dare-ties":
        return f"merged_model/DARE_TIES_best_{pair_name}"
    return f"merged_model/DARE_best_{pair_name}"


def get_knots_ties_best_model_dir(pair_name: str, task_cfg: dict) -> str:
    """
    返回 KnOTS-TIES baseline 搜索/融合后保存的稳定 coarse model 路径。
    必须和 KnOTS_TIES.py 中保存 best coarse model 的默认命名保持一致。
    """
    if task_cfg.get("knots_ties_tals_coarse_model_dir"):
        return str(task_cfg["knots_ties_tals_coarse_model_dir"])
    return f"merged_model/KnOTS_TIES_best_{pair_name}"


def make_pair_config(
    base_cfg: dict,
    task_a: str,
    task_b: str,
    method: str,
    dare_drop_rate=None,
    dare_seed=None,
) -> dict:
    """
    为每个 GLUE pair 生成独立 config。

    支持:
      GLUE_t5:
        method_configs:
          linear_tals_drc:
            ...
          iteris_tals_drc:
            ...
          regmean_tals_drc:
            ...
          knots_ties_tals_drc:
            ...
          ties_tals_drc:
            ...
          dare_tals_drc:
            ...

    逻辑：
    1) deep copy base config；
    2) 如果当前 method 在 method_configs 中存在，就把该 method 配置展开到 GLUE_t5 顶层；
    3) 删除 tals_common / method_configs，避免临时 pair config 臃肿；
    4) 写入当前 pair；
    5) 对 DARE 类方法，可选写入当前 grid 的 dare_drop_rate / dare_seed；
    6) 自动生成当前 pair 的 merged_model_dir，避免所有 pair 共用 mnli_cola 路径。
    """
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))  # deep copy
    pair_name = f"{task_a}_{task_b}"

    task_cfg = cfg["GLUE_t5"]

    method_configs = task_cfg.get("method_configs", {})
    if method in method_configs:
        selected_method_cfg = method_configs[method]
        for k, v in selected_method_cfg.items():
            task_cfg[k] = v

    # 临时 config 只保留当前 method 的 flat 参数，避免多个 method 配置共存造成排查困难。
    task_cfg.pop("tals_common", None)
    task_cfg.pop("method_configs", None)


    task_cfg["task_targets"] = [task_a, task_b]
    task_cfg["lora_alpha"] = [32, 32]
    task_cfg["manual_ceof"] = [1, 1]
    
    # Linear baseline：根据 LoRA 来源保存到不同目录
    if method == "linear":
        lora_source = str(task_cfg.get("lora_source", "default")).lower().strip()

        if lora_source == "osrm":
            task_cfg["linear_method_name"] = task_cfg.get("linear_method_name", "OSRM_Linear")
            task_cfg["linear_merged_model_dir"] = f"merged_model/OSRM_Linear_{pair_name}"
        else:
            task_cfg["linear_method_name"] = task_cfg.get("linear_method_name", "Linear")
            task_cfg["linear_merged_model_dir"] = f"merged_model/Linear_{pair_name}"

    if dare_drop_rate is not None:
        task_cfg["dare_drop_rate"] = float(dare_drop_rate)
    if dare_seed is not None:
        task_cfg["dare_seed"] = int(dare_seed)

    if method in ["linear_drc", "linear_drc_pca", "linear_tals_drc"]:
        lora_source = str(task_cfg.get("lora_source", "default")).lower().strip()

        task_cfg["linear_tals_load_coarse_from_dir"] = True

        # 如果配置里显式给了 prefix，就优先使用
        existing_prefix = str(task_cfg.get("linear_tals_existing_coarse_prefix", "")).strip()

        if existing_prefix:
            coarse_dir = f"{existing_prefix}{pair_name}"
        elif lora_source == "osrm":
            coarse_dir = f"merged_model/OSRM_Linear_{pair_name}"
        else:
            coarse_dir = f"merged_model/Linear_{pair_name}"

        task_cfg["linear_tals_coarse_model_dir"] = coarse_dir
        task_cfg["linear_merged_model_dir"] = coarse_dir

    # IterIS-TALS 保存/记录 coarse merged model 的目录
    if method in ["iteris_drc", "iteris_tals_drc"]:
        task_cfg["iteris_drc_merged_model_dir"] = f"merged_model/IterIS_TALS_DRC_{pair_name}"

    # RegMean-TALS 保存/记录 coarse merged model 的目录
    if method == "regmean_tals_drc":
        task_cfg["regmean_tals_merged_model_dir"] = f"merged_model/RegMean_TALS_DRC_{pair_name}"

    # DARE baseline：搜索后保存 best coarse model 到稳定 pair-specific 路径
    if method == "dare":
        task_cfg["dare_merged_model_dir"] = get_dare_best_model_dir(pair_name, task_cfg)
        task_cfg.setdefault("save_best_model", True)
        task_cfg.setdefault("save", 1)

    # KnOTS-TIES baseline：搜索后保存 best coarse model 到稳定 pair-specific 路径
    if method == "knots_ties":
        task_cfg["knots_ties_merged_model_dir"] = get_knots_ties_best_model_dir(pair_name, task_cfg)
        task_cfg.setdefault("save_best_model", True)
        task_cfg.setdefault("save", 1)

    # KnOTS-TIES-TALS：不再默认重跑固定参数 coarse model，而是加载 baseline 已保存的 best coarse model
    if method == "knots_ties_tals_drc":
        cfg["GLUE_t5"]["knots_ties_tals_load_coarse_from_dir"] = True
        cfg["GLUE_t5"]["knots_ties_tals_coarse_model_dir"] = f"merged_model/KnOTS-TIES_{pair_name}"

    # TIES-TALS 保存/记录 coarse merged model 的目录
    if method == "ties_tals_drc":
        task_cfg["ties_tals_merged_model_dir"] = f"merged_model/TIES_TALS_DRC_{pair_name}"

    # DARE-TALS：不再默认重跑固定参数 coarse model，而是加载 baseline 已保存的 best coarse model
    if method == "dare_tals_drc":
        task_cfg["dare_tals_load_coarse_from_dir"] = True

        # 优先使用已有 DARE-TIES coarse model 前缀。
        # 例如:
        #   dare_tals_existing_coarse_prefix: merged_model/DARE_TIES_p0p9_s420_k20_
        # 则 mnli_cola 会自动变成:
        #   merged_model/DARE_TIES_p0p9_s420_k20_mnli_cola
        existing_prefix = str(task_cfg.get("dare_tals_existing_coarse_prefix", "")).strip()

        if existing_prefix:
            task_cfg["dare_tals_coarse_model_dir"] = f"{existing_prefix}{pair_name}"
        else:
            task_cfg["dare_tals_coarse_model_dir"] = get_dare_best_model_dir(pair_name, task_cfg)

        # 仅作为 TALS 结果/日志记录目录；真正 coarse model 从上面的 coarse_model_dir 读取
        task_cfg["dare_tals_merged_model_dir"] = f"merged_model/DARE_TALS_DRC_{pair_name}"

    # LOT 保存/记录 coarse merged model 的目录
    if method == "lot":
        task_cfg["lot_merged_model_dir"] = f"merged_model/LOT_{pair_name}"

    return cfg

def save_yaml(cfg: dict, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def read_csv_if_valid(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[Warn] 读取 CSV 失败，跳过: {path}; error={e}")
        return None


def aggregate_pair_result_files(pair_results_root: Path, results_dir: Path, summary_dir: Path):
    """
    将每个 pair 独立 RESULTS_DIR 下的 CSV 合并成全局 results/*.csv。

    这是本版最关键的结构变化：
    - 每个 pair 子进程只写自己的 pair_results/<pair_name>/*.csv；
    - 所有子进程结束后，由主进程串行合并；
    - 避免多进程同时 append 同一个 CSV 导致丢行/错行。
    """
    ensure_dir(results_dir)
    ensure_dir(summary_dir)

    aggregated = {}
    for filename in RESULT_FILENAMES:
        dfs = []
        for csv_path in sorted(pair_results_root.glob(f"*/{filename}")):
            df = read_csv_if_valid(csv_path)
            if df is None or len(df) == 0:
                continue
            df.insert(0, "_source_pair_dir", csv_path.parent.name)
            dfs.append(df)

        if not dfs:
            aggregated[filename] = None
            continue

        merged = pd.concat(dfs, ignore_index=True)

        # 去掉辅助列后写到 results；summary 里保留一份 raw 文件也方便排查。
        merged_out = merged.drop(columns=["_source_pair_dir"], errors="ignore")
        out_path = results_dir / filename
        merged_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        merged.to_csv(summary_dir / filename, index=False, encoding="utf-8-sig")
        aggregated[filename] = out_path
        print(f"[Done] 聚合 {filename}: {len(merged_out)} rows -> {out_path}")

    return aggregated


def parse_config_blob(raw_value):
    raw = "" if pd.isna(raw_value) else str(raw_value).strip()
    parsed = {"raw_config": raw}
    if not raw:
        return parsed

    # 一些脚本会把 linear_weights 写成: [0.5, 0.5]|inject_position=...
    # 先保留原始字符串，再尽量解析 key=value。
    if "=" not in raw:
        tokens = [x.strip() for x in raw.split("|") if x.strip()]
        if tokens:
            parsed["weights"] = "|".join(tokens)
        return parsed

    parts = [p.strip() for p in raw.split("|") if p.strip()]
    i = 0
    while i < len(parts):
        part = parts[i]
        if "=" not in part:
            parsed.setdefault("unnamed_tokens", []).append(part)
            i += 1
            continue

        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "selected_alpha_by_task":
            values = [value]
            j = i + 1
            while j < len(parts) and "=" not in parts[j] and ":" in parts[j]:
                values.append(parts[j])
                j += 1
            parsed[key] = "|".join(values)
            i = j
            continue

        parsed[key] = value
        i += 1

    return parsed


def summarize_selected_hparams(registry_csv: Path, out_dir: Path):
    if not registry_csv.exists():
        print(f"[Warn] {registry_csv} 不存在，无法生成超参汇总。")
        return

    reg = pd.read_csv(registry_csv)
    if len(reg) == 0:
        print("[Warn] experiment_registry.csv 为空，跳过超参汇总。")
        return

    rows = []
    for _, r in reg.iterrows():
        config_blob = r.get("linear_weights", "")
        if (pd.isna(config_blob) or str(config_blob).strip() == "") and "notes" in reg.columns:
            config_blob = r.get("notes", "")
        parsed = parse_config_blob(config_blob)

        rows.append({
            "experiment_id": r.get("experiment_id", ""),
            "method": r.get("method", ""),
            "pair_name": r.get("pair_name", ""),
            "task_targets": r.get("task_targets", ""),
            "status": r.get("status", ""),
            "pair_avg_normalized_metric": r.get("pair_avg_normalized_metric", ""),
            "weights": parsed.get("weights", ""),
            "drc_inject_position": parsed.get("inject_position", ""),
            "drc_alpha": parsed.get("alpha", ""),
            "drc_alpha_search": parsed.get("alpha_search", ""),
            "drc_alpha_candidates": parsed.get("alpha_candidates", ""),
            "drc_selected_alpha_by_task": parsed.get("selected_alpha_by_task", ""),
            "drc_samples_per_task": parsed.get("samples_per_task", ""),
            "drc_target_part": parsed.get("target_part", ""),
            "drc_target_modules": parsed.get("target_modules", ""),
            "drc_target_layers": parsed.get("target_layers", ""),
            "drc_normalize": parsed.get("normalize", ""),
            "drc_use_hidden_norm_scale": parsed.get("use_hidden_norm_scale", ""),
            "raw_config": parsed.get("raw_config", ""),
        })

    summary_df = pd.DataFrame(rows)
    if len(summary_df) > 0 and "pair_name" in summary_df.columns:
        summary_df = summary_df.sort_values(["method", "pair_name"], na_position="last")

    out_path = out_dir / "selected_hparams_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[Done] 超参汇总已保存到: {out_path}")


def summarize_results_excluding_wnli(pair_summary: pd.DataFrame, out_dir: Path):
    tasks7 = ["mnli", "rte", "cola", "sst2", "qqp", "qnli", "mrpc"]

    if pair_summary is None or len(pair_summary) == 0 or "pair_name" not in pair_summary.columns:
        rows = []
        for task in tasks7:
            rows.append({
                "evaluated_task": task,
                "runs": 0,
                "avg_primary_metric": pd.NA,
                "avg_normalized_metric": pd.NA,
                "avg_eval_loss": pd.NA,
                "avg_eval_runtime": pd.NA,
                "avg_eval_peak_vram_mb": pd.NA,
            })
        pd.DataFrame(rows).to_csv(
            out_dir / "task_average_results_excluding_wnli_pairs.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("[Warn] pair_summary 为空，excluding-wnli 汇总为空。")
        return

    ps7 = pair_summary[~pair_summary["pair_name"].str.contains("wnli", na=False)].copy()
    rows = []
    for task in tasks7:
        metric_col = f"{task}_metric_value"
        norm_col = f"{task}_normalized_metric"
        loss_col = f"{task}_eval_loss"
        runtime_col = f"{task}_eval_runtime"
        vram_col = f"{task}_eval_peak_vram_mb"

        if metric_col not in ps7.columns:
            rows.append({
                "evaluated_task": task,
                "runs": 0,
                "avg_primary_metric": pd.NA,
                "avg_normalized_metric": pd.NA,
                "avg_eval_loss": pd.NA,
                "avg_eval_runtime": pd.NA,
                "avg_eval_peak_vram_mb": pd.NA,
            })
            continue

        valid = ps7[metric_col].notna()
        sub = ps7.loc[valid]
        rows.append({
            "evaluated_task": task,
            "runs": int(valid.sum()),
            "avg_primary_metric": sub[metric_col].mean(),
            "avg_normalized_metric": sub[norm_col].mean() if norm_col in sub.columns else pd.NA,
            "avg_eval_loss": sub[loss_col].mean() if loss_col in sub.columns else pd.NA,
            "avg_eval_runtime": sub[runtime_col].mean() if runtime_col in sub.columns else pd.NA,
            "avg_eval_peak_vram_mb": sub[vram_col].mean() if vram_col in sub.columns else pd.NA,
        })

    task_avg_7 = pd.DataFrame(rows).sort_values("evaluated_task")
    task_avg_7.to_csv(
        out_dir / "task_average_results_excluding_wnli_pairs.csv",
        index=False,
        encoding="utf-8-sig",
    )
    overall_avg = task_avg_7["avg_normalized_metric"].mean()
    print(f"[Done] excluding-wnli 7-task average normalized metric = {overall_avg:.6f}")


def summarize_results(results_csv: Path, registry_csv: Path, out_dir: Path):
    ensure_dir(out_dir)
    if not results_csv.exists():
        print(f"[Warn] {results_csv} 不存在，无法汇总。")
        return None

    df = pd.read_csv(results_csv)
    if len(df) == 0:
        print(f"[Warn] {results_csv} 为空，无法汇总。")
        return None

    task_avg = (
        df.groupby("evaluated_task", as_index=False)
        .agg(
            runs=("evaluated_task", "count"),
            avg_primary_metric=("primary_metric_value", "mean"),
            avg_normalized_metric=("normalized_metric", "mean"),
            avg_eval_loss=("eval_loss", "mean"),
            avg_eval_runtime=("eval_runtime", "mean"),
            avg_eval_peak_vram_mb=("eval_peak_vram_mb", "mean"),
        )
        .sort_values("evaluated_task")
    )
    task_avg.to_csv(out_dir / "task_average_results.csv", index=False, encoding="utf-8-sig")

    rows = []
    group_cols = ["pair_name"]
    if "method" in df.columns and df["method"].nunique(dropna=True) > 1:
        group_cols = ["method", "pair_name"]

    groupby_arg = group_cols[0] if len(group_cols) == 1 else group_cols

    for group_key, g in df.groupby(groupby_arg, dropna=False):
        if len(group_cols) == 2:
            if not isinstance(group_key, tuple) or len(group_key) != 2:
                raise ValueError(f"Unexpected group_key for {group_cols}: {group_key}")
            method_name, pair_name = group_key
            row = {"method": method_name, "pair_name": pair_name}
        else:
            if isinstance(group_key, tuple):
                if len(group_key) != 1:
                    raise ValueError(f"Unexpected group_key for {group_cols}: {group_key}")
                pair_name = group_key[0]
            else:
                pair_name = group_key
            row = {"pair_name": pair_name}

        for _, r in g.iterrows():
            task = r["evaluated_task"]
            row[f"{task}_metric_name"] = r["primary_metric_name"]
            row[f"{task}_metric_value"] = r["primary_metric_value"]
            row[f"{task}_normalized_metric"] = r["normalized_metric"]
            row[f"{task}_eval_loss"] = r["eval_loss"]
            row[f"{task}_eval_runtime"] = r["eval_runtime"]
            row[f"{task}_eval_peak_vram_mb"] = r["eval_peak_vram_mb"]
        rows.append(row)

    if rows:
        sort_cols = ["method", "pair_name"] if "method" in rows[0] else ["pair_name"]
        pair_summary = pd.DataFrame(rows).sort_values(sort_cols)
    else:
        pair_summary = pd.DataFrame(columns=["pair_name"])
    pair_summary.to_csv(out_dir / "pair_summary_results.csv", index=False, encoding="utf-8-sig")
    summarize_results_excluding_wnli(pair_summary, out_dir)

    matrix = pd.DataFrame("-", index=TASKS, columns=TASKS)
    for pair_name, g in df.groupby("pair_name"):
        tasks = sorted(g["evaluated_task"].tolist())
        if len(tasks) != 2:
            continue
        t1, t2 = tasks
        r1 = g[g["evaluated_task"] == t1].iloc[0]
        r2 = g[g["evaluated_task"] == t2].iloc[0]
        cell = f"{t1}:{r1['primary_metric_value']:.6f} | {t2}:{r2['primary_metric_value']:.6f}"
        matrix.loc[t1, t2] = cell
        matrix.loc[t2, t1] = cell
    matrix.to_csv(out_dir / "pair_metric_matrix.csv", encoding="utf-8-sig")

    if registry_csv.exists():
        reg = pd.read_csv(registry_csv)
        keep_cols = [
            "pair_name",
            "fusion_total_time_sec",
            "fusion_iter_time_avg_sec",
            "fusion_iter_time_max_sec",
            "fusion_peak_vram_avg_mb",
            "fusion_peak_vram_max_mb",
            "pair_avg_normalized_metric",
            "status",
        ]
        reg = reg[[c for c in keep_cols if c in reg.columns]]
        reg.to_csv(out_dir / "fusion_registry_summary.csv", index=False, encoding="utf-8-sig")

    print(f"[Done] 汇总完成，结果保存在: {out_dir}")
    return pair_summary


def validate_alpha_search(results_csv: Path, alpha_csv: Path, out_dir: Path):
    """
    检查 alpha search 明细和最终 pair_merge_results 是否一致。

    正常情况：
    - 每个 pair-task 在 pair_merge_results 只有一行 best 结果；
    - drc_alpha_search_results 中同一 pair-task 的 max(normalized_metric)
      与 pair_merge_results 的 normalized_metric 完全一致。
    """
    report_path = out_dir / "alpha_search_consistency_report.csv"
    summary_path = out_dir / "alpha_search_consistency_summary.txt"

    if not results_csv.exists() or not alpha_csv.exists():
        msg = "[Info] 没有检测到 alpha search 文件，跳过一致性检查。"
        print(msg)
        summary_path.write_text(msg + "\n", encoding="utf-8")
        return

    res = pd.read_csv(results_csv)
    alpha = pd.read_csv(alpha_csv)
    if len(alpha) == 0:
        msg = "[Warn] drc_alpha_search_results.csv 为空。"
        print(msg)
        summary_path.write_text(msg + "\n", encoding="utf-8")
        return

    base_group_cols = ["pair_name", "evaluated_task"]
    group_cols = ["method"] + base_group_cols if ("method" in res.columns and "method" in alpha.columns) else base_group_cols
    idx = alpha.groupby(group_cols)["normalized_metric"].idxmax()
    best = alpha.loc[idx, group_cols + ["alpha", "normalized_metric"]].copy()
    best = best.rename(columns={"alpha": "best_alpha", "normalized_metric": "best_normalized_metric"})

    final = res[group_cols + ["normalized_metric"]].copy()
    final = final.rename(columns={"normalized_metric": "final_normalized_metric"})

    m = pd.merge(best, final, on=group_cols, how="outer")
    m["diff"] = m["final_normalized_metric"] - m["best_normalized_metric"]
    m["match"] = m["diff"].abs() <= 1e-9
    m.to_csv(report_path, index=False, encoding="utf-8-sig")

    candidate_counts = alpha.groupby(group_cols)["alpha"].nunique().reset_index(name="num_alpha_candidates")
    candidate_counts.to_csv(out_dir / "alpha_search_candidate_counts.csv", index=False, encoding="utf-8-sig")

    total_result_groups = final.groupby(group_cols).ngroups
    total_search_groups = best.groupby(group_cols).ngroups
    total_rows = len(alpha)
    mismatch_rows = int((~m["match"].fillna(False)).sum())
    min_candidates = int(candidate_counts["num_alpha_candidates"].min()) if len(candidate_counts) else 0
    max_candidates = int(candidate_counts["num_alpha_candidates"].max()) if len(candidate_counts) else 0

    msg = (
        f"alpha_search_rows={total_rows}\n"
        f"result_pair_task_groups={total_result_groups}\n"
        f"alpha_search_pair_task_groups={total_search_groups}\n"
        f"min_candidates_per_pair_task={min_candidates}\n"
        f"max_candidates_per_pair_task={max_candidates}\n"
        f"mismatch_rows={mismatch_rows}\n"
        f"report={report_path}\n"
    )
    summary_path.write_text(msg, encoding="utf-8")
    print("[AlphaSearchCheck]\n" + msg)



def summarize_drc_activation(alpha_csv: Path, out_dir: Path):
    """
    统计 DRC 启用率。

    定义：
    - 对每个 pair-task，从 drc_alpha_search_results.csv 中选择 normalized_metric 最高的 alpha；
    - best_alpha > 0 表示 DRC 被启用；
    - best_alpha == 0 表示 DRC 被关闭，等价于退回原始 merged model。

    输出：
    1) drc_best_alpha_by_pair_task.csv
    2) drc_activation_overall.csv
    3) drc_activation_by_task.csv
    4) drc_activation_by_pair.csv
    5) drc_best_alpha_distribution_by_task.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not alpha_csv.exists():
        print("[Info] 没有检测到 drc_alpha_search_results.csv，跳过 DRC 启用率统计。")
        return

    alpha = pd.read_csv(alpha_csv)
    if len(alpha) == 0:
        print("[Warn] drc_alpha_search_results.csv 为空，跳过 DRC 启用率统计。")
        return

    required_cols = {"pair_name", "evaluated_task", "alpha", "normalized_metric"}
    missing_cols = required_cols - set(alpha.columns)
    if missing_cols:
        print(f"[Warn] drc_alpha_search_results.csv 缺少列 {missing_cols}，跳过 DRC 启用率统计。")
        return

    base_group_cols = ["pair_name", "evaluated_task"]
    group_cols = ["method"] + base_group_cols if "method" in alpha.columns else base_group_cols

    # 1) 每个 method/pair-task 选择 best alpha
    idx_best = alpha.groupby(group_cols)["normalized_metric"].idxmax()
    best = alpha.loc[idx_best].copy()

    best = best.rename(
        columns={
            "alpha": "best_alpha",
            "normalized_metric": "best_normalized_metric",
        }
    )
    best["best_alpha"] = best["best_alpha"].astype(float)
    best["drc_activated"] = best["best_alpha"].abs() > 1e-12

    keep_cols = []
    if "method" in best.columns:
        keep_cols.append("method")
    keep_cols += [
        "pair_name",
        "evaluated_task",
        "best_alpha",
        "drc_activated",
        "best_normalized_metric",
    ]
    optional_cols = [
        "primary_metric_name",
        "primary_metric_value",
        "eval_accuracy",
        "eval_mcc",
        "eval_f1",
        "eval_loss",
        "eval_runtime",
        "eval_peak_vram_mb",
    ]
    keep_cols += [c for c in optional_cols if c in best.columns]

    best_out = best[keep_cols].sort_values(["pair_name", "evaluated_task"])
    best_out.to_csv(
        out_dir / "drc_best_alpha_by_pair_task.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 2) 总体启用率
    total = len(best_out)
    activated = int(best_out["drc_activated"].sum())
    inactive = total - activated
    activation_rate = activated / total if total > 0 else 0.0

    overall = pd.DataFrame(
        [
            {
                "total_pair_tasks": total,
                "activated_pair_tasks": activated,
                "inactive_pair_tasks": inactive,
                "activation_rate": activation_rate,
                "avg_best_alpha": best_out["best_alpha"].mean(),
                "avg_best_normalized_metric": best_out["best_normalized_metric"].mean(),
            }
        ]
    )
    overall.to_csv(
        out_dir / "drc_activation_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 3) 按 evaluated_task 统计
    by_task = (
        best_out.groupby("evaluated_task", as_index=False)
        .agg(
            total_pair_tasks=("drc_activated", "count"),
            activated_pair_tasks=("drc_activated", "sum"),
            avg_best_alpha=("best_alpha", "mean"),
            avg_best_normalized_metric=("best_normalized_metric", "mean"),
        )
    )
    by_task["activated_pair_tasks"] = by_task["activated_pair_tasks"].astype(int)
    by_task["inactive_pair_tasks"] = by_task["total_pair_tasks"] - by_task["activated_pair_tasks"]
    by_task["activation_rate"] = by_task["activated_pair_tasks"] / by_task["total_pair_tasks"]
    by_task = by_task[
        [
            "evaluated_task",
            "total_pair_tasks",
            "activated_pair_tasks",
            "inactive_pair_tasks",
            "activation_rate",
            "avg_best_alpha",
            "avg_best_normalized_metric",
        ]
    ].sort_values(["activation_rate", "evaluated_task"], ascending=[False, True])
    by_task.to_csv(
        out_dir / "drc_activation_by_task.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 4) 按 pair 统计
    pair_group_cols = ["method", "pair_name"] if "method" in best_out.columns else ["pair_name"]
    by_pair = (
        best_out.groupby(pair_group_cols, as_index=False)
        .agg(
            total_tasks=("drc_activated", "count"),
            activated_tasks=("drc_activated", "sum"),
            avg_best_alpha=("best_alpha", "mean"),
            avg_best_normalized_metric=("best_normalized_metric", "mean"),
        )
    )
    by_pair["activated_tasks"] = by_pair["activated_tasks"].astype(int)
    by_pair["inactive_tasks"] = by_pair["total_tasks"] - by_pair["activated_tasks"]
    by_pair["activation_rate"] = by_pair["activated_tasks"] / by_pair["total_tasks"]

    alpha_str = (
        best_out.assign(
            alpha_str=best_out["evaluated_task"] + ":" + best_out["best_alpha"].astype(str)
        )
        .groupby(pair_group_cols)["alpha_str"]
        .apply(lambda x: "|".join(x))
        .reset_index(name="selected_alpha_by_task")
    )
    by_pair = pd.merge(by_pair, alpha_str, on=pair_group_cols, how="left")
    by_pair_cols = []
    if "method" in by_pair.columns:
        by_pair_cols.append("method")
    by_pair_cols += [
        "pair_name",
        "total_tasks",
        "activated_tasks",
        "inactive_tasks",
        "activation_rate",
        "avg_best_alpha",
        "avg_best_normalized_metric",
        "selected_alpha_by_task",
    ]
    by_pair = by_pair[by_pair_cols].sort_values(["activation_rate", "pair_name"], ascending=[False, True])
    by_pair.to_csv(
        out_dir / "drc_activation_by_pair.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 5) 每个任务的 best alpha 分布
    dist = pd.crosstab(best_out["evaluated_task"], best_out["best_alpha"])
    dist.to_csv(
        out_dir / "drc_best_alpha_distribution_by_task.csv",
        encoding="utf-8-sig",
    )

    print(
        "[DRCActivation] "
        f"activation_rate={activation_rate:.6f}, "
        f"activated={activated}/{total}, "
        f"summary={out_dir / 'drc_activation_overall.csv'}"
    )


def summarize_existing_batch_activation(batch_or_summary_dir: Path):
    """
    对已经跑完的 batch 直接补算 DRC 启用率，不重新跑实验。

    用法：
        python run_all_glue_pairs.py --summarize_existing_batch batch_runs/glue_pairs_xxx

    也支持直接传 summary 目录：
        python run_all_glue_pairs.py --summarize_existing_batch batch_runs/glue_pairs_xxx/summary
    """
    batch_or_summary_dir = Path(batch_or_summary_dir)
    if (batch_or_summary_dir / "summary").exists():
        summary_dir = batch_or_summary_dir / "summary"
    else:
        summary_dir = batch_or_summary_dir

    alpha_csv = summary_dir / "drc_alpha_search_results.csv"
    results_csv = summary_dir / "pair_merge_results.csv"

    print(f"[ExistingBatch] summary_dir = {summary_dir}")
    print(f"[ExistingBatch] alpha_csv = {alpha_csv}")
    print(f"[ExistingBatch] results_csv = {results_csv}")

    if results_csv.exists() and alpha_csv.exists():
        validate_alpha_search(
            results_csv=results_csv,
            alpha_csv=alpha_csv,
            out_dir=summary_dir,
        )

    summarize_drc_activation(alpha_csv=alpha_csv, out_dir=summary_dir)


def worker_run(
    gpu_id: str,
    job_list,
    repo_root: Path,
    base_cfg: dict,
    configs_dir: Path,
    logs_dir: Path,
    pair_results_root: Path,
    python_bin: str,
    continue_on_error: bool,
    failed_shared,
    method: str,
):
    script_map = {
        "iteris": "IterIS.py",
        "iteris_drc": "IterIS_DRC.py",
        "iteris_tals_drc": "IterIS_TALS_DRC.py",
        "linear": "Linear.py",
        "linear_drc": "Linear_DRC.py",
        "linear_tals_drc": "Linear_TALS_DRC.py",
        "linear_drc_pca": "Linear_DRC_PCA.py",
        "lot": "LOT.py",
        "regmean": "RegMean.py",
        "regmean_tals_drc": "RegMean_TALS_DRC.py",
        "ties": "TIES.py",
        "ties_tals_drc": "TIES_TALS_DRC.py",
        "knots": "KnOTS.py",
        "knots_ties": "KnOTS_TIES.py",
        "knots_ties_tals_drc": "KnOTS_TIES_TALS_DRC.py",
        "dare": "DARE.py",
        "dare_tals_drc": "DARE_TALS_DRC.py",
    }
    if method not in script_map:
        raise ValueError(f"Unsupported method: {method}")

    for idx, job in enumerate(job_list, start=1):
        if method in ["dare", "dare_tals_drc"] and len(job) == 4:
            task_a, task_b, dare_drop_rate, dare_seed = job
            pair_name = f"{task_a}_{task_b}"
            job_name = f"{pair_name}__p{str(dare_drop_rate).replace('.', 'p')}__s{dare_seed}"
        else:
            task_a, task_b = job
            dare_drop_rate, dare_seed = None, None
            pair_name = f"{task_a}_{task_b}"
            job_name = pair_name

        print(f"[GPU {gpu_id}] [{idx}/{len(job_list)}] 开始: {job_name}", flush=True)

        cfg = make_pair_config(
            base_cfg=base_cfg,
            task_a=task_a,
            task_b=task_b,
            method=method,
            dare_drop_rate=dare_drop_rate,
            dare_seed=dare_seed,
        )
        cfg_path = configs_dir / f"{job_name}.yaml"
        save_yaml(cfg, cfg_path)

        log_path = logs_dir / f"{job_name}.log"
        pair_results_dir = pair_results_root / job_name
        ensure_dir(pair_results_dir)

        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        # 关键：让支持 RESULTS_DIR 的脚本写入 pair 专属目录。
        # 修改后的 Linear_DRC.py 会读取该变量。
        env["RESULTS_DIR"] = str(pair_results_dir)
        env["LOG_FILE"] = str(log_path)

        if method in ["linear_drc", "linear_drc_pca", "linear_tals_drc"]:
            linear_model_dir = repo_root / cfg["GLUE_t5"].get(
                "linear_tals_coarse_model_dir",
                cfg["GLUE_t5"].get("linear_merged_model_dir", f"merged_model/Linear_{pair_name}"),
            )

            if not (linear_model_dir / "config.json").exists():
                failed_shared.append({
                    "gpu_id": gpu_id,
                    "pair_name": pair_name,
                    "log_file": str(log_path),
                    "return_code": -1,
                    "error": f"Missing Linear/OSRM_Linear coarse model: {linear_model_dir}",
                })
                print(
                    f"[GPU {gpu_id}] [Skip] {pair_name} 缺少 Linear/OSRM_Linear coarse model: {linear_model_dir}",
                    flush=True,
                )
                if not continue_on_error:
                    break
                else:
                    continue

        if method == "dare_tals_drc":
            coarse_model_dir = repo_root / cfg["GLUE_t5"]["dare_tals_coarse_model_dir"]

            if not (coarse_model_dir / "config.json").exists():
                failed_shared.append({
                    "gpu_id": gpu_id,
                    "pair_name": pair_name,
                    "log_file": str(log_path),
                    "return_code": -1,
                    "error": f"Missing existing DARE-TIES coarse model: {coarse_model_dir}",
                })
                print(
                    f"[GPU {gpu_id}] [Skip] {pair_name} 缺少已有 DARE-TIES coarse model: {coarse_model_dir}",
                    flush=True,
                )
                if not continue_on_error:
                    break
                else:
                    continue

        if method == "knots_ties_tals_drc":
            coarse_model_dir = repo_root / "merged_model" / f"KnOTS-TIES_{pair_name}"
            if not (coarse_model_dir / "config.json").exists():
                failed_shared.append({
                    "gpu_id": gpu_id,
                    "pair_name": pair_name,
                    "log_file": str(log_path),
                    "return_code": -1,
                    "error": f"Missing existing KnOTS-TIES coarse model: {coarse_model_dir}",
                })
                print(
                    f"[GPU {gpu_id}] [Skip] {pair_name} 缺少已有 KnOTS-TIES coarse model: {coarse_model_dir}",
                    flush=True,
                )
                if not continue_on_error:
                    break
                else:
                    continue

        cmd = [
            python_bin,
            script_map[method],
            "--task_type", "GLUE_t5",
            "--config", str(cfg_path),
        ]

        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                text=True,
            )

        if proc.returncode != 0:
            failed_shared.append({
                "gpu_id": gpu_id,
                "pair_name": pair_name,
                "log_file": str(log_path),
                "return_code": proc.returncode,
            })
            print(f"[GPU {gpu_id}] [Fail] {job_name} 失败，日志见: {log_path}", flush=True)
            if not continue_on_error:
                break
        else:
            print(f"[GPU {gpu_id}] [OK] {job_name} 完成，日志见: {log_path}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/methods-config/iteris-config.yaml")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--python", type=str, default="python")
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument(
        "--summarize_existing_batch",
        type=str,
        default=None,
        help="Only summarize DRC activation for an existing batch or summary directory, without rerunning experiments.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="iteris",
        choices=[
            "iteris", "iteris_drc", "iteris_tals_drc",
            "linear", "linear_drc", "linear_drc_pca", "linear_tals_drc",
            "lot",
            "regmean", "regmean_tals_drc",
            "ties", "ties_tals_drc", "knots", "knots_ties", "knots_ties_tals_drc",
            "dare", "dare_tals_drc"
        ],
        help="Merging method, including *_tals_drc variants such as dare_tals_drc.",
    )
    args = parser.parse_args()

    if args.summarize_existing_batch is not None:
        summarize_existing_batch_activation(Path(args.summarize_existing_batch))
        return

    repo_root = Path(__file__).resolve().parent
    base_config_path = repo_root / args.config
    with open(base_config_path, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = repo_root / "batch_runs" / f"glue_pairs_{batch_id}"
    configs_dir = batch_dir / "configs"
    logs_dir = batch_dir / "logs"
    summary_dir = batch_dir / "summary"
    pair_results_root = batch_dir / "pair_results"

    for d in [configs_dir, logs_dir, summary_dir, pair_results_root]:
        ensure_dir(d)

    results_dir = repo_root / "results"
    ensure_dir(results_dir)
    for filename in RESULT_FILENAMES:
        backup_if_exists(results_dir / filename, batch_dir)

    pairs = list(itertools.combinations(TASKS, 2))

    if args.method in ["dare", "dare_tals_drc"]:
        dare_cfg = base_cfg["GLUE_t5"].get("method_configs", {}).get(args.method, base_cfg["GLUE_t5"])
        dare_grid_search = bool(dare_cfg.get("dare_grid_search", False))
        drop_rates = dare_cfg.get("dare_drop_rates", [dare_cfg.get("dare_drop_rate", 0.1)])
        dare_seeds = dare_cfg.get("dare_seeds", [dare_cfg.get("dare_seed", base_cfg.get("seed", 42))])

        if dare_grid_search:
            jobs = []
            for task_a, task_b in pairs:
                for p in drop_rates:
                    for s in dare_seeds:
                        jobs.append((task_a, task_b, p, s))
        else:
            jobs = pairs
    else:
        jobs = pairs

    gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_list:
        raise ValueError("没有可用 GPU，请用 --gpus 指定，例如 --gpus 0,1,2,3")

    gpu_to_jobs = {gpu: [] for gpu in gpu_list}
    for i, job in enumerate(jobs):
        gpu = gpu_list[i % len(gpu_list)]
        gpu_to_jobs[gpu].append(job)

    manager = Manager()
    failed_shared = manager.list()
    processes = []

    for gpu_id in gpu_list:
        job_list = gpu_to_jobs[gpu_id]
        if not job_list:
            continue
        p = Process(
            target=worker_run,
            args=(
                gpu_id,
                job_list,
                repo_root,
                base_cfg,
                configs_dir,
                logs_dir,
                pair_results_root,
                args.python,
                args.continue_on_error,
                failed_shared,
                args.method,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    failed = list(failed_shared)
    failed_df = pd.DataFrame(failed)
    failed_df.to_csv(summary_dir / "failed_pairs.csv", index=False, encoding="utf-8-sig")

    aggregated = aggregate_pair_result_files(pair_results_root, results_dir, summary_dir)

    # 兼容未改 RESULTS_DIR 的旧脚本：如果 pair_results 没有结果，就尝试使用全局 results。
    results_csv = aggregated.get("pair_merge_results.csv") or (results_dir / "pair_merge_results.csv")
    registry_csv = aggregated.get("experiment_registry.csv") or (results_dir / "experiment_registry.csv")
    alpha_csv = aggregated.get("drc_alpha_search_results.csv")

    summarize_results(results_csv=results_csv, registry_csv=registry_csv, out_dir=summary_dir)
    summarize_selected_hparams(registry_csv=registry_csv, out_dir=summary_dir)
    if args.method in [
        "linear_drc",
        "linear_drc_pca",
        "linear_tals_drc",
        "iteris_drc",
        "iteris_tals_drc",
        "regmean_tals_drc",
        "ties_tals_drc",
        "knots_ties_tals_drc",
        "dare_tals_drc",
    ]:
        validate_alpha_search(results_csv=results_csv, alpha_csv=alpha_csv, out_dir=summary_dir)
        summarize_drc_activation(alpha_csv=alpha_csv, out_dir=summary_dir)
    else:
        print(f"[Info] method={args.method} 不是 DRC/TALS 方法，跳过 alpha search 一致性检查和 DRC 启用率统计。")

    print(f"\nBatch 目录: {batch_dir}")
    print(f"[Info] pair_results 目录: {pair_results_root}")
    print(f"[Info] summary 目录: {summary_dir}")
    if failed:
        print(f"[Warn] 共有 {len(failed)} 个 pair 失败，见 {summary_dir / 'failed_pairs.csv'}")
    else:
        print("[Done] 所有 pair 已跑完。")


if __name__ == "__main__":
    main()
