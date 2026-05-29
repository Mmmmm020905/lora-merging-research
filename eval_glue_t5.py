import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from peft import PeftModel
from sklearn.metrics import f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, T5ForConditionalGeneration, T5Tokenizer


GLUE_KEY_SINGLE = {
    "cola": {"input": "sentence", "label": "label"},
    "sst2": {"input": "sentence", "label": "label"},
}

GLUE_KEY_DOUBLE = {
    "mnli": {"input": ["premise", "hypothesis"], "label": "label"},
    "mnli-mm": {"input": ["premise", "hypothesis"], "label": "label"},
    "mrpc": {"input": ["sentence1", "sentence2"], "label": "label"},
    "qnli": {"input": ["question", "sentence"], "label": "label"},
    "qqp": {"input": ["question1", "question2"], "label": "label"},
    "rte": {"input": ["sentence1", "sentence2"], "label": "label"},
    "wnli": {"input": ["sentence1", "sentence2"], "label": "label"},
}

LABEL2TEXT = {
    "cola": {1: "yes", 0: "no"},
    "sst2": {1: "yes", 0: "no"},
    "mnli": {0: "yes", 1: "maybe", 2: "no"},
    "mnli-mm": {0: "yes", 1: "maybe", 2: "no"},
    "rte": {0: "yes", 1: "no"},
    "wnli": {1: "yes", 0: "no"},
    "qqp": {1: "yes", 0: "no"},
    "mrpc": {1: "yes", 0: "no"},
    "qnli": {0: "yes", 1: "no"},
}

TEXT2LABEL = {
    task: {value: key for key, value in mapping.items()}
    for task, mapping in LABEL2TEXT.items()
}

VALID_TASKS = {"mnli", "rte", "cola", "sst2", "qqp", "qnli", "mrpc", "wnli"}
DEFAULT_BASE_MODEL = "/data2/centrai/mizijie_intern/IterIS-merging-main/flan-t5-base"
DEFAULT_GLUE_ROOT = "/data2/centrai/mizijie_intern/IterIS-merging-main/glue_local"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prompt_text(input1: str, task_name: str, input2: str = None) -> str:
    if task_name == "cola":
        return f"""
Instruction: Is the following sentence grammatically correct? Answer acceptable or unacceptable.
Input: {input1}
Answer:
"""
    if task_name == "sst2":
        return f"""
Instruction: Does the following sentence express a positive or negative sentiment? Answer positive or negative.
Input: {input1}
Answer:
"""
    if task_name in ["rte", "wnli"]:
        return f"""
Instruction: Does Sentence1 imply Sentence2? Please answer yes or no.
Input: Sentence1: {input1}; Sentence2: {input2}
Answer:
"""
    if task_name in ["mnli", "mnli-mm"]:
        return f"""
Instruction: Does Sentence1 imply Sentence2? Please answer yes, no or maybe.
Input: Sentence1: {input1}; Sentence2: {input2}
Answer:
"""
    if task_name == "mrpc":
        return f"""
Instruction: Is Sentence1 equivalent to Sentence2? Please answer yes or no.
Input: Sentence1: {input1}; Sentence2: {input2}
Answer:
"""
    if task_name == "qnli":
        return f"""
Instruction: Given a question and a sentence, does the sentence contain the answer to the question? Please answer yes or no.
Input: Question: {input1}; Sentence: {input2}
Answer:
"""
    if task_name == "qqp":
        return f"""
Instruction: Are Question 1 and Question 2 semantically equivalent? Please answer yes or no.
Input: Question1: {input1}; Question2: {input2}
Answer:
"""
    raise ValueError(f"Unsupported task_name: {task_name}")


def normalize_prediction(text: str, task_name: str) -> int:
    text = text.strip().lower()
    if task_name not in TEXT2LABEL:
        return -1

    for label_text, label_id in TEXT2LABEL[task_name].items():
        if text.startswith(label_text):
            return label_id

    first_token = text.split(" ")[0] if text else ""
    return TEXT2LABEL[task_name].get(first_token, -1)


def get_glue_split(task_name: str, split: str) -> Tuple[str, str]:
    task_actual = "mnli" if task_name == "mnli-mm" else task_name

    if split == "validation":
        if task_name == "mnli":
            return task_actual, "validation_matched"
        if task_name == "mnli-mm":
            return task_actual, "validation_mismatched"
        return task_actual, "validation"

    if split == "test":
        if task_name == "mnli":
            return task_actual, "test_matched"
        if task_name == "mnli-mm":
            return task_actual, "test_mismatched"
        return task_actual, "test"

    raise ValueError("split must be one of: validation, test")


def build_prompts(examples: Dict[str, List], task_name: str) -> List[str]:
    if task_name in GLUE_KEY_DOUBLE:
        key1, key2 = GLUE_KEY_DOUBLE[task_name]["input"]
        return [prompt_text(a, task_name, b) for a, b in zip(examples[key1], examples[key2])]

    if task_name in GLUE_KEY_SINGLE:
        key1 = GLUE_KEY_SINGLE[task_name]["input"]
        return [prompt_text(a, task_name) for a in examples[key1]]

    raise ValueError(f"Unsupported task_name: {task_name}")


def load_eval_dataset(task_name: str, split: str, glue_root: str):
    task_actual, split_name = get_glue_split(task_name, split)
    local_task_path = os.path.join(glue_root, task_actual)

    if os.path.exists(local_task_path):
        print(f"[Info] Loading local GLUE dataset from: {local_task_path}")
        return load_from_disk(local_task_path)[split_name]

    print(f"[Info] Loading GLUE dataset from Hugging Face Hub: {task_actual}")
    return load_dataset("glue", task_actual)[split_name]


def detect_model_type(model_dir: Path) -> str:
    if (model_dir / "adapter_config.json").exists():
        return "peft_adapter"
    if (model_dir / "config.json").exists():
        return "dense_model"
    raise FileNotFoundError(
        f"Cannot detect model type for {model_dir}. Need either adapter_config.json or config.json."
    )


def load_model(model_dir: Path, base_model_path: str, device: str):
    model_type = detect_model_type(model_dir)

    if model_type == "dense_model":
        print(f"[Load] Dense merged model: {model_dir}")
        model = T5ForConditionalGeneration.from_pretrained(str(model_dir))
    else:
        print(f"[Load] PEFT adapter: {model_dir}")
        base_model = T5ForConditionalGeneration.from_pretrained(base_model_path)
        model = PeftModel.from_pretrained(base_model, str(model_dir))

    model.eval()
    model.to(device)
    return model, model_type


def evaluate_task(
    model,
    tokenizer,
    task_name: str,
    split: str,
    glue_root: str,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    device: str,
    prediction_file: Path,
):
    dataset = load_eval_dataset(task_name, split, glue_root)

    def preprocess_function(examples):
        prompts = build_prompts(examples, task_name)
        model_inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        if "label" in examples:
            model_inputs["raw_label"] = examples["label"]
        return model_inputs

    tokenized = dataset.map(preprocess_function, batched=True, batch_size=1024)
    keep_cols = ["input_ids", "attention_mask"]
    if "raw_label" in tokenized.column_names:
        keep_cols.append("raw_label")
    tokenized = tokenized.remove_columns([col for col in tokenized.column_names if col not in keep_cols])

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    dataloader = DataLoader(
        tokenized,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    pred_ids_all = []
    pred_text_all = []
    label_ids_all = []
    has_labels = False

    with torch.no_grad():
        for batch in dataloader:
            raw_labels = None
            if "raw_label" in batch:
                raw_labels = batch.pop("raw_label")
                has_labels = True

            batch = {key: value.to(device) for key, value in batch.items() if value is not None}

            generated = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
            )

            pred_text = tokenizer.batch_decode(generated, skip_special_tokens=True)
            pred_text_all.extend(pred_text)
            pred_ids_all.extend([normalize_prediction(text, task_name) for text in pred_text])

            if raw_labels is not None:
                label_ids_all.extend(raw_labels.cpu().tolist())

    prediction_file.parent.mkdir(parents=True, exist_ok=True)
    with open(prediction_file, "w", encoding="utf-8") as handle:
        for idx, (pred_id, pred_text) in enumerate(zip(pred_ids_all, pred_text_all)):
            row = {
                "idx": idx,
                "task_name": task_name,
                "pred_label_id": int(pred_id),
                "pred_text": pred_text,
            }
            if has_labels and idx < len(label_ids_all):
                row["gold_label_id"] = int(label_ids_all[idx])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "accuracy": "",
        "f1_weighted": "",
        "mcc": "",
        "parsed": 0,
        "total": len(pred_ids_all),
    }

    if has_labels:
        valid = [(pred, gold) for pred, gold in zip(pred_ids_all, label_ids_all) if pred != -1 and gold != -1]
        result["parsed"] = len(valid)

        if valid:
            pred_valid = [pred for pred, _ in valid]
            gold_valid = [gold for _, gold in valid]
            result["accuracy"] = float(np.mean(np.array(pred_valid) == np.array(gold_valid)))
            result["f1_weighted"] = float(f1_score(gold_valid, pred_valid, average="weighted"))

            if task_name == "cola":
                result["mcc"] = float(matthews_corrcoef(gold_valid, pred_valid))

    return result


def summarize_primary_metric(task_name: str, result: Dict[str, object]) -> Tuple[str, object, object]:
    if task_name == "cola" and result["mcc"] != "":
        primary_metric_name = "matthews_correlation"
        primary_metric_value = result["mcc"]
        normalized_metric = (float(result["mcc"]) + 1.0) / 2.0
    else:
        primary_metric_name = "accuracy"
        primary_metric_value = result["accuracy"]
        normalized_metric = result["accuracy"]
    return primary_metric_name, primary_metric_value, normalized_metric


def parse_pair_from_dirname(dirname: str, prefix: str) -> Tuple[str, str]:
    name = dirname
    if prefix and name.startswith(prefix):
        name = name[len(prefix):]

    name = name.strip("_-")
    parts = name.split("_")

    if len(parts) == 2 and parts[0] in VALID_TASKS and parts[1] in VALID_TASKS:
        return parts[0], parts[1]

    if len(parts) >= 2 and parts[-2] in VALID_TASKS and parts[-1] in VALID_TASKS:
        return parts[-2], parts[-1]

    raise ValueError(
        f"Cannot parse pair from dirname={dirname}, after prefix removal={name}. "
        f"Expected suffix like *_mnli_cola or direct mnli_cola."
    )


def evaluate_single(args) -> None:
    model_dir = Path(args.model_dir or args.lora_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    if args.task_name not in LABEL2TEXT:
        raise ValueError(f"Unsupported task_name: {args.task_name}")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    model, model_type = load_model(model_dir, args.base_model, device)

    output_file = Path(args.output_file) if args.output_file else Path(
        f"outputs/eval_glue_t5/{args.task_name}_{args.split}_{model_dir.name}.jsonl"
    )

    result = evaluate_task(
        model=model,
        tokenizer=tokenizer,
        task_name=args.task_name,
        split=args.split,
        glue_root=args.glue_root,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        device=device,
        prediction_file=output_file,
    )
    primary_metric_name, primary_metric_value, normalized_metric = summarize_primary_metric(args.task_name, result)

    print(f"[Info] Model type: {model_type}")
    print(f"[Info] Predictions saved to: {output_file}")
    print(f"[Result] task={args.task_name}, primary={primary_metric_name}:{primary_metric_value}, normalized={normalized_metric}")
    print(f"[Result] accuracy={result['accuracy']}, f1_weighted={result['f1_weighted']}, mcc={result['mcc']}")
    print(f"[Result] parsed={result['parsed']}/{result['total']}")


def evaluate_batch(args) -> None:
    set_seed(args.seed)
    merged_root = Path(args.merged_root)
    if not merged_root.exists():
        raise FileNotFoundError(f"merged_root does not exist: {merged_root}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"outputs/eval_glue_t5_batch_{args.prefix.strip('_-')}_{args.split}"
    )
    pred_dir = out_dir / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(args.base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dirs = sorted([path for path in merged_root.iterdir() if path.is_dir() and path.name.startswith(args.prefix)])

    if not model_dirs:
        raise FileNotFoundError(f"No model dirs found under {merged_root} with prefix={args.prefix}")

    rows = []
    pair_summary = {}

    for model_dir in model_dirs:
        print("=" * 80)
        print(f"[ModelDir] {model_dir}")
        print("=" * 80)

        try:
            task_a, task_b = parse_pair_from_dirname(model_dir.name, args.prefix)
        except Exception as exc:
            print(f"[Skip] Cannot parse pair from {model_dir.name}: {exc}")
            continue

        pair_name = f"{task_a}_{task_b}"
        model, model_type = load_model(model_dir, args.base_model, device)
        pair_summary.setdefault(pair_name, [])

        for task_name in [task_a, task_b]:
            print(f"[Eval] pair={pair_name}, task={task_name}")
            prediction_file = pred_dir / f"{model_dir.name}__{task_name}.jsonl"

            result = evaluate_task(
                model=model,
                tokenizer=tokenizer,
                task_name=task_name,
                split=args.split,
                glue_root=args.glue_root,
                batch_size=args.batch_size,
                max_length=args.max_length,
                max_new_tokens=args.max_new_tokens,
                device=device,
                prediction_file=prediction_file,
            )
            primary_metric_name, primary_metric_value, normalized_metric = summarize_primary_metric(task_name, result)

            if normalized_metric != "":
                pair_summary[pair_name].append(float(normalized_metric))

            rows.append(
                {
                    "model_dir": str(model_dir),
                    "model_type": model_type,
                    "method": args.prefix.strip("_-"),
                    "pair_name": pair_name,
                    "task_a": task_a,
                    "task_b": task_b,
                    "evaluated_task": task_name,
                    "primary_metric_name": primary_metric_name,
                    "primary_metric_value": primary_metric_value,
                    "normalized_metric": normalized_metric,
                    "accuracy": result["accuracy"],
                    "f1_weighted": result["f1_weighted"],
                    "mcc": result["mcc"],
                    "parsed": result["parsed"],
                    "total": result["total"],
                    "prediction_file": str(prediction_file),
                }
            )

            print(
                f"[Result] task={task_name}, primary={primary_metric_name}:{primary_metric_value}, "
                f"normalized={normalized_metric}, parsed={result['parsed']}/{result['total']}"
            )

        del model
        torch.cuda.empty_cache()

    result_csv = out_dir / "pair_task_results.csv"
    with open(result_csv, "w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "model_dir",
            "model_type",
            "method",
            "pair_name",
            "task_a",
            "task_b",
            "evaluated_task",
            "primary_metric_name",
            "primary_metric_value",
            "normalized_metric",
            "accuracy",
            "f1_weighted",
            "mcc",
            "parsed",
            "total",
            "prediction_file",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_csv = out_dir / "pair_summary_results.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pair_name", "pair_avg_normalized_metric"])
        writer.writeheader()
        for pair_name, values in sorted(pair_summary.items()):
            writer.writerow(
                {
                    "pair_name": pair_name,
                    "pair_avg_normalized_metric": float(np.mean(values)) if values else "",
                }
            )

    print("=" * 80)
    print(f"[Done] pair-task results: {result_csv}")
    print(f"[Done] pair summary: {summary_csv}")
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified GLUE/T5 evaluator for single adapters and merged models.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser("single", help="Evaluate one adapter or one dense merged model on one GLUE task.")
    single_parser.add_argument("--task_name", type=str, required=True)
    single_parser.add_argument("--model_dir", type=str, default="", help="Path to a dense model dir or PEFT adapter dir.")
    single_parser.add_argument("--lora_path", type=str, default="", help="Legacy alias of --model_dir.")
    single_parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    single_parser.add_argument("--glue_root", type=str, default=DEFAULT_GLUE_ROOT)
    single_parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    single_parser.add_argument("--batch_size", type=int, default=16)
    single_parser.add_argument("--max_length", type=int, default=256)
    single_parser.add_argument("--max_new_tokens", type=int, default=4)
    single_parser.add_argument("--seed", type=int, default=42)
    single_parser.add_argument("--output_file", type=str, default="")

    batch_parser = subparsers.add_parser("batch", help="Evaluate a directory of merged models pair by pair.")
    batch_parser.add_argument("--merged_root", type=str, required=True)
    batch_parser.add_argument("--prefix", type=str, default="KnOTS-TIES_")
    batch_parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    batch_parser.add_argument("--glue_root", type=str, default=DEFAULT_GLUE_ROOT)
    batch_parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    batch_parser.add_argument("--batch_size", type=int, default=16)
    batch_parser.add_argument("--max_length", type=int, default=256)
    batch_parser.add_argument("--max_new_tokens", type=int, default=4)
    batch_parser.add_argument("--seed", type=int, default=42)
    batch_parser.add_argument("--out_dir", type=str, default="")

    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "single":
        evaluate_single(args)
        return
    if args.command == "batch":
        evaluate_batch(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


def main_single_legacy(argv=None) -> None:
    parser = argparse.ArgumentParser("Evaluate one GLUE LoRA or one dense merged model (FLAN-T5-base)")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True, help="Path containing adapter_model.safetensors or a model dir.")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--glue_root", type=str, default=DEFAULT_GLUE_ROOT)
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="")
    args = parser.parse_args(argv)
    args.model_dir = args.lora_path
    evaluate_single(args)


def main_batch_legacy(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_root", type=str, default="/data2/centrai/mizijie_intern/IterIS-merging-main/merged_model")
    parser.add_argument("--prefix", type=str, default="KnOTS-TIES_")
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--glue_root", type=str, default=DEFAULT_GLUE_ROOT)
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args(argv)
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    evaluate_batch(args)


if __name__ == "__main__":
    main()
