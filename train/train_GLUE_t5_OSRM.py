import os
import re
import yaml
import shutil
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import DataLoader

from datasets import load_dataset, load_from_disk, DatasetDict
from sklearn.metrics import f1_score, matthews_corrcoef
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
)


GLUE_key_single = {
    'cola': {'input': 'sentence', 'label': 'label'},
    'sst2': {'input': 'sentence', 'label': 'label'},
}

GLUE_key_double = {
    'ax': {'input': ['premise', 'hypothesis'], 'label': 'label'},
    'mnli': {'input': ['premise', 'hypothesis'], 'label': 'label'},
    'mrpc': {'input': ['sentence1', 'sentence2'], 'label': 'label'},
    'qnli': {'input': ['question', 'sentence'], 'label': 'label'},
    'qqp': {'input': ['question1', 'question2'], 'label': 'label'},
    'rte': {'input': ['sentence1', 'sentence2'], 'label': 'label'},
    'stsb': {'input': ['sentence1', 'sentence2'], 'label': 'label'},
    'wnli': {'input': ['sentence1', 'sentence2'], 'label': 'label'},
}

label2text = {
    'cola': {1: 'yes', 0: 'no'},
    'sst2': {1: 'yes', 0: 'no'},
    'mnli': {0: 'yes', 1: 'maybe', 2: 'no'},
    'rte':  {0: 'yes', 1: 'no'},
    'wnli': {1: 'yes', 0: 'no'},
    'qqp':  {1: 'yes', 0: 'no'},
    'mrpc': {1: 'yes', 0: 'no'},
    'qnli': {0: 'yes', 1: 'no'},
}

DEFAULT_OSRM_REFERENCE_TASKS = [
    'mnli', 'rte', 'cola', 'sst2', 'qqp', 'qnli', 'mrpc', 'wnli'
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prompt_text(input1, task_name, input2=None):
    assert (task_name in GLUE_key_single) or (task_name in GLUE_key_double)
    if task_name == 'cola':
        return f"""
                    Instruction: Is the following sentence grammatically correct? Answer acceptable or unacceptable.
                    Input: {input1}
                    Answer: 
                """
    elif task_name == 'sst2':
        return f"""
                    Instruction: Does the following sentence express a positive or negative sentiment? Answer positive or negative.
                    Input: {input1}
                    Answer: 
                """
    elif task_name in ['rte', 'wnli']:
        return f"""
                    Instruction: Does Sentence1 imply Sentence2? Please answer yes or no.
                    Input: Sentence1: {input1}; Sentence2: {input2}
                    Answer:
                """
    elif task_name in ['mnli', 'mnli-mm']:
        return f"""
                    Instruction: Does Sentence1 imply Sentence2? Please answer yes, no or maybe.
                    Input: Sentence1: {input1}; Sentence2: {input2}
                    Answer:
                """
    elif task_name == 'mrpc':
        return f"""
                    Instruction: Is Sentence1 equivalent to Sentence2? Please answer yes or no.
                    Input: Sentence1: {input1}; Sentence2: {input2}
                    Answer:
                """
    elif task_name == "qnli":
        return f"""
                    Instruction: Given a question and a sentence, does the sentence contain the answer to the question? Please answer yes or no.
                    Input: Question: {input1}; Sentence: {input2}
                    Answer:
                """
    elif task_name == "qqp":
        return f"""
                    Instruction: Are Question 1 and Question 2 semantically equivalent? Please answer yes or no.
                    Input: Question1: {input1}; Question2: {input2}
                    Answer:
                """
    else:
        raise ValueError(f"Unsupported task for prompt_text: {task_name}")


def preprocess_function(examples, task_name, tokenizer, max_length):
    assert task_name in GLUE_key_double.keys() or task_name in GLUE_key_single.keys()

    if task_name in GLUE_key_double.keys():
        model_inputs = tokenizer(
            [
                prompt_text(input1, task_name, input2)
                for input1, input2 in zip(
                    examples[GLUE_key_double[task_name]['input'][0]],
                    examples[GLUE_key_double[task_name]['input'][1]],
                )
            ],
            truncation=True,
            max_length=max_length,
            padding='max_length',
        )
        label_name = GLUE_key_double[task_name]['label']
    else:
        model_inputs = tokenizer(
            [prompt_text(input1, task_name) for input1 in examples[GLUE_key_single[task_name]['input']]],
            truncation=True,
            max_length=max_length,
            padding='max_length',
        )
        label_name = GLUE_key_single[task_name]['label']

    if task_name not in label2text:
        raise ValueError(
            f"Task {task_name} is not supported by label2text in this text-to-text script. "
            "STS-B needs a separate regression/generation formatting."
        )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            [f"{label2text[task_name][ex]}" for ex in examples[label_name]],
            max_length=max_length,
            padding='max_length',
            truncation=True,
        ).input_ids

    labels = [[(item if item != tokenizer.pad_token_id else -100) for item in label] for label in labels]
    model_inputs['labels'] = labels
    return model_inputs


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    tokenizer = AutoTokenizer.from_pretrained(
        "/data2/centrai/mizijie_intern/IterIS-merging-main/flan-t5-base"
    )
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = labels[:, 0]
    preds = preds[:, 1]
    accuracy = (preds == labels).mean()
    f1 = f1_score(labels, preds, average='weighted')
    mcc = matthews_corrcoef(labels, preds)
    return {"accuracy": accuracy, "f1-score": f1, "MCC": mcc}


def load_glue_dataset(task_name, local_glue_root):
    task_actual = "mnli" if task_name == "mnli-mm" else task_name
    local_task_path = os.path.join(os.path.expanduser(local_glue_root), task_actual)
    if os.path.exists(local_task_path):
        print(f"Loading local GLUE dataset from: {local_task_path}")
        dataset = load_from_disk(local_task_path)
    else:
        print(f"Local dataset not found, loading from Hugging Face Hub: {task_actual}")
        dataset = load_dataset("glue", task_actual)
    return dataset


def build_train_eval_dataset(raw_dataset, task_name):
    validation_key = "validation_mismatched" if task_name == "mnli-mm" else "validation_matched" if task_name == "mnli" else "validation"
    return DatasetDict({'train': raw_dataset['train'], validation_key: raw_dataset[validation_key]}), validation_key


def tokenize_glue_dataset(dataset, task_name, tokenizer, max_length, seed=None):
    tokenized = dataset.map(
        lambda examples: preprocess_function(
            examples,
            task_name=task_name,
            tokenizer=tokenizer,
            max_length=max_length,
        ),
        batched=True,
        batch_size=1024,
    )

    features_to_keep = ['input_ids', 'attention_mask', 'labels']
    for key in tokenized.keys():
        tokenized[key] = tokenized[key].remove_columns(
            [col for col in tokenized[key].column_names if col not in features_to_keep]
        )
        if seed is not None:
            tokenized[key] = tokenized[key].shuffle(seed=seed)
    return tokenized


def safe_name(text):
    text = str(text).replace(os.sep, "_")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def get_osrm_cfg(config_data):
    cfg = config_data.get('osrm', {}) or {}
    return {
        'enabled': bool(cfg.get('enabled', False)),
        'reference_tasks': cfg.get('reference_tasks', DEFAULT_OSRM_REFERENCE_TASKS),
        'exclude_current_task': bool(cfg.get('exclude_current_task', True)),
        'num_samples_per_task': int(cfg.get('num_samples_per_task', 100)),
        'feature_split': cfg.get('feature_split', 'train'),
        'feature_batch_size': int(cfg.get('feature_batch_size', 8)),
        'cache_dir': cfg.get('cache_dir', 'osrm_feature_cache'),
        'rebuild_cache': bool(cfg.get('rebuild_cache', False)),
        'center_features': bool(cfg.get('center_features', False)),
        'target_part': cfg.get('target_part', None),
        'zero_B': bool(cfg.get('zero_B', True)),
        'freeze_A': bool(cfg.get('freeze_A', False)),
        'fallback_to_gaussian': bool(cfg.get('fallback_to_gaussian', False)),
        # Export the final best PEFT adapter to a stable directory that can be
        # consumed directly by Linear.py / run_all_glue_pairs.py.
        'export_best_model': bool(cfg.get('export_best_model', True)),
        'output_root': cfg.get('output_root', 'OSRM_LoRA'),
        'output_name_template': cfg.get('output_name_template', 'T5-{TASK_UPPER}-LoRA'),
        'overwrite_output': bool(cfg.get('overwrite_output', True)),
    }


def is_target_linear_module(module_name, module, target_modules, target_part=None):
    if target_part and not module_name.startswith(str(target_part) + "."):
        return False
    last = module_name.split('.')[-1]
    if last not in set(target_modules):
        return False
    if not hasattr(module, 'weight'):
        return False
    if not torch.is_tensor(module.weight):
        return False
    if module.weight.dim() != 2:
        return False
    return True


def get_target_module_names(model, target_modules, target_part=None):
    names = []
    for name, module in model.named_modules():
        if is_target_linear_module(name, module, target_modules, target_part=target_part):
            names.append(name)
    return sorted(names)


def collect_osrm_features_for_task(
    model,
    tokenizer,
    task_name,
    local_glue_root,
    max_length,
    seed,
    target_modules,
    target_part=None,
    split='train',
    num_samples=100,
    batch_size=8,
    cache_dir='osrm_feature_cache',
    rebuild_cache=False,
    model_tag='flan-t5-base',
):
    """
    Collect one mean latent feature vector per LoRA target module for one GLUE task.

    Hook feature x is the module input with shape [batch, seq_len, hidden].
    For encoder-side features whose sequence length matches input attention_mask,
    we use attention-mask mean pooling; otherwise we use plain token mean pooling.
    The final stored vector is averaged over sampled examples.
    """
    os.makedirs(cache_dir, exist_ok=True)
    target_tag = "_".join(target_modules)
    part_tag = "all" if target_part is None else str(target_part).replace('.', '_')
    cache_path = os.path.join(
        cache_dir,
        f"{safe_name(model_tag)}_{task_name}_{split}_n{num_samples}_seed{seed}_len{max_length}_{part_tag}_{target_tag}.pt"
    )

    if (not rebuild_cache) and os.path.exists(cache_path):
        print(f"[OSRM] Load cached latent features for {task_name}: {cache_path}")
        return torch.load(cache_path, map_location='cpu')

    raw_dataset = load_glue_dataset(task_name, local_glue_root)
    dataset_dict, validation_key = build_train_eval_dataset(raw_dataset, task_name)
    selected_split = validation_key if split in ['validation', 'eval', 'dev'] else 'train'
    selected = DatasetDict({'tmp': dataset_dict[selected_split]})
    tokenized = tokenize_glue_dataset(selected, task_name, tokenizer, max_length=max_length, seed=seed)['tmp']

    if num_samples > 0:
        tokenized = tokenized.select(range(min(len(tokenized), num_samples)))

    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    loader = DataLoader(tokenized, batch_size=batch_size, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model.to(device)

    module_names = get_target_module_names(model, target_modules, target_part=target_part)
    if not module_names:
        raise ValueError(
            f"[OSRM] Cannot find target modules={target_modules}, target_part={target_part} in base model."
        )

    sums = {name: None for name in module_names}
    counts = {name: 0 for name in module_names}
    handles = []
    context = {'attention_mask': None}

    def make_hook(name):
        def hook(module, inputs, outputs):
            if inputs is None or len(inputs) == 0:
                return
            x = inputs[0]
            if not torch.is_tensor(x) or x.dim() != 3:
                return

            x_detached = x.detach().float()
            attn = context.get('attention_mask', None)
            if attn is not None and x_detached.size(1) == attn.size(1):
                mask = attn.to(device=x_detached.device, dtype=x_detached.dtype).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp_min(1.0)
                pooled = (x_detached * mask).sum(dim=1) / denom
            else:
                pooled = x_detached.mean(dim=1)

            vec_sum = pooled.sum(dim=0).detach().cpu()
            if sums[name] is None:
                sums[name] = vec_sum
            else:
                sums[name] = sums[name] + vec_sum
            counts[name] += pooled.size(0)
        return hook

    modules = dict(model.named_modules())
    for name in module_names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            context['attention_mask'] = batch.get('attention_mask', None)
            _ = model(**batch)

    for h in handles:
        h.remove()

    features = {}
    for name in module_names:
        if sums[name] is None or counts[name] == 0:
            print(f"[OSRM][Warn] No feature collected for module: {name}")
            continue
        features[name] = (sums[name] / max(counts[name], 1)).float().cpu()

    torch.save(features, cache_path)
    print(f"[OSRM] Saved latent features for {task_name}: {cache_path}")
    return features


def collect_reference_features(
    base_model,
    tokenizer,
    current_task,
    local_glue_root,
    max_length,
    seed,
    target_modules,
    osrm_cfg,
    model_tag,
):
    ref_tasks = list(osrm_cfg['reference_tasks'])
    if osrm_cfg['exclude_current_task']:
        ref_tasks = [t for t in ref_tasks if t != current_task and not (current_task == 'mnli-mm' and t == 'mnli')]

    if not ref_tasks:
        raise ValueError("[OSRM] reference_tasks is empty after excluding current task.")

    print(f"[OSRM] Current task = {current_task}")
    print(f"[OSRM] Reference tasks = {ref_tasks}")

    all_features = {}
    for ref_task in ref_tasks:
        all_features[ref_task] = collect_osrm_features_for_task(
            model=base_model,
            tokenizer=tokenizer,
            task_name=ref_task,
            local_glue_root=local_glue_root,
            max_length=max_length,
            seed=seed,
            target_modules=target_modules,
            target_part=osrm_cfg['target_part'],
            split=osrm_cfg['feature_split'],
            num_samples=osrm_cfg['num_samples_per_task'],
            batch_size=osrm_cfg['feature_batch_size'],
            cache_dir=osrm_cfg['cache_dir'],
            rebuild_cache=osrm_cfg['rebuild_cache'],
            model_tag=model_tag,
        )
    return all_features


def build_osrm_A_init(reference_features, rank, center_features=False, eps=1e-8):
    """
    For every target module, construct H from reference task mean features and
    initialize LoRA A with the right singular vectors corresponding to the
    smallest singular values.
    """
    module_to_rows = defaultdict(list)
    for task_name, feats in reference_features.items():
        for module_name, vec in feats.items():
            module_to_rows[module_name].append(vec.float().cpu())

    A_init = {}
    stats = {}
    for module_name, rows in module_to_rows.items():
        if not rows:
            continue
        H = torch.stack(rows, dim=0).float()  # [num_reference_tasks, hidden_dim]
        if center_features:
            H = H - H.mean(dim=0, keepdim=True)

        hidden_dim = H.size(1)
        if rank > hidden_dim:
            raise ValueError(f"rank={rank} > hidden_dim={hidden_dim} for module={module_name}")

        try:
            # full_matrices=True is important: when #ref_tasks < hidden_dim,
            # the null-space directions are also available in Vh.
            _, singular_values, Vh = torch.linalg.svd(H, full_matrices=True)
            A = Vh[-rank:, :].contiguous().float()
        except RuntimeError as e:
            raise RuntimeError(f"[OSRM] SVD failed for module={module_name}, H.shape={tuple(H.shape)}: {e}")

        # Safety normalization: rows from SVD should already be orthonormal.
        row_norm = A.norm(dim=1, keepdim=True).clamp_min(eps)
        A = A / row_norm

        A_init[module_name] = A
        stats[module_name] = {
            'num_reference_vectors': int(H.size(0)),
            'hidden_dim': int(hidden_dim),
            'rank': int(rank),
            'max_singular': float(singular_values.max().item()) if singular_values.numel() else 0.0,
            'min_singular': float(singular_values.min().item()) if singular_values.numel() else 0.0,
            'A_row_orth_error': float(torch.norm(A @ A.T - torch.eye(rank)).item()),
        }

    return A_init, stats


def peft_param_to_base_module_name(param_name, suffix):
    name = param_name
    if name.startswith('base_model.model.'):
        name = name[len('base_model.model.'):]
    if name.endswith(suffix):
        name = name[:-len(suffix)]
    return name


def apply_osrm_initialization_to_peft_model(peft_model, A_init, zero_B=True, freeze_A=False, fallback_to_gaussian=False):
    """
    Copy OSRM analytical A into PEFT LoRA A weights.
    Expected PEFT parameter format:
        base_model.model.<module_name>.lora_A.default.weight
        base_model.model.<module_name>.lora_B.default.weight
    """
    missing = []
    applied = []

    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if name.endswith('.lora_A.default.weight'):
                module_name = peft_param_to_base_module_name(name, '.lora_A.default.weight')
                if module_name not in A_init:
                    missing.append(module_name)
                    if not fallback_to_gaussian:
                        raise KeyError(
                            f"[OSRM] Cannot find analytical A for {module_name}. "
                            "Set osrm.fallback_to_gaussian=true to keep PEFT default init for missing modules."
                        )
                    continue
                A = A_init[module_name].to(device=param.device, dtype=param.dtype)
                if tuple(A.shape) != tuple(param.shape):
                    raise ValueError(
                        f"[OSRM] Shape mismatch for {name}: analytical A={tuple(A.shape)}, param={tuple(param.shape)}"
                    )
                param.copy_(A)
                applied.append(module_name)

            if zero_B and name.endswith('.lora_B.default.weight'):
                param.zero_()

    if freeze_A:
        for name, param in peft_model.named_parameters():
            if name.endswith('.lora_A.default.weight'):
                param.requires_grad = False

    print(f"[OSRM] Applied analytical A init to {len(applied)} LoRA A matrices.")
    if missing:
        print(f"[OSRM][Warn] Missing analytical A for {len(missing)} LoRA A matrices.")
        for item in missing[:20]:
            print(f"    missing: {item}")
    return applied


def format_export_name(template, task_name):
    task_upper = task_name.upper().replace('MNLI-MM', 'MNLI-MM')
    return template.format(
        TASK=task_name,
        task=task_name,
        TASK_UPPER=task_upper,
        task_upper=task_upper,
    )


def export_best_adapter(best_model_dir, tokenizer, task_name, osrm_cfg, osrm_enabled, best_info):
    """
    Copy the final PEFT adapter directory to a stable LoRA source directory, e.g.
        OSRM_LoRA/T5-MNLI-LoRA

    This avoids a manual post-training reorganization step. The exported folder
    keeps adapter_model.safetensors / adapter_config.json exactly as saved by PEFT.
    """
    if not osrm_cfg.get('export_best_model', True):
        print('[Export] osrm.export_best_model=false, skip stable LoRA export.')
        return None

    output_root = osrm_cfg.get('output_root', 'OSRM_LoRA')
    template = osrm_cfg.get('output_name_template', 'T5-{TASK_UPPER}-LoRA')
    export_name = format_export_name(template, task_name)
    export_dir = os.path.join(output_root, export_name)
    overwrite = bool(osrm_cfg.get('overwrite_output', True))

    if os.path.exists(export_dir):
        if overwrite:
            shutil.rmtree(export_dir)
        else:
            raise FileExistsError(
                f'[Export] Target export_dir already exists: {export_dir}. '
                'Set osrm.overwrite_output=true or choose another output_root.'
            )

    os.makedirs(os.path.dirname(export_dir), exist_ok=True)
    shutil.copytree(best_model_dir, export_dir)

    # Save tokenizer and metadata again to make the directory self-contained.
    tokenizer.save_pretrained(export_dir)
    export_info = dict(best_info)
    export_info.update({
        'export_dir': export_dir,
        'export_task_name': task_name,
        'export_is_osrm': bool(osrm_enabled),
    })
    with open(os.path.join(export_dir, 'export_info.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(export_info, f, allow_unicode=True, sort_keys=False)

    print('=' * 80)
    print(f'[Export] Stable LoRA adapter exported to: {export_dir}')
    print('[Export] This directory can be used directly by Linear.py get_loras_path().')
    print('=' * 80)
    return export_dir


def main():
    parser = argparse.ArgumentParser(description="T5 GLUE LoRA training with optional OSRM analytical initialization")
    parser.add_argument('--config', type=str, default="config/GLUE-t5-lora-train-config/MNLI-lora-train.yaml")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as file:
        config_data = yaml.safe_load(file)

    seed = config_data['rand_seed']
    set_seed(seed)

    task_name = config_data['task_name']
    local_model_path = config_data.get(
        'model_name_or_path',
        "/data2/centrai/mizijie_intern/IterIS-merging-main/flan-t5-base",
    )
    local_glue_root = config_data.get(
        'glue_root',
        "/data2/centrai/mizijie_intern/IterIS-merging-main/glue_local",
    )
    max_length = int(config_data.get('max_length', 256))

    osrm_cfg = get_osrm_cfg(config_data)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=config_data['lora_config']['r'],
        lora_alpha=config_data['lora_config']['lora_alpha'],
        lora_dropout=config_data['lora_config']['lora_dropout'],
        target_modules=config_data['lora_config']['target_modules'],
    )

    print(f"[Config] task_name = {task_name}")
    print(f"[Config] model = {local_model_path}")
    print(f"[Config] max_length = {max_length}")
    print(f"[Config] target_modules = {config_data['lora_config']['target_modules']}")
    print(f"[Config] OSRM enabled = {osrm_cfg['enabled']}")

    raw_dataset = load_glue_dataset(task_name, local_glue_root)
    dataset, validation_key = build_train_eval_dataset(raw_dataset, task_name)
    dataset[validation_key] = dataset[validation_key].shuffle(seed=seed)
    dataset['train'] = dataset['train'].shuffle(seed=seed)

    tokenizer = T5Tokenizer.from_pretrained(local_model_path)
    tokenized_datasets = tokenize_glue_dataset(dataset, task_name, tokenizer, max_length=max_length)

    # Load base model first. If OSRM is enabled, collect features with this base model
    # before wrapping it by PEFT.
    base_model = T5ForConditionalGeneration.from_pretrained(local_model_path, max_length=max_length)

    A_init = None
    osrm_stats = None
    if osrm_cfg['enabled']:
        reference_features = collect_reference_features(
            base_model=base_model,
            tokenizer=tokenizer,
            current_task=task_name,
            local_glue_root=local_glue_root,
            max_length=max_length,
            seed=seed,
            target_modules=config_data['lora_config']['target_modules'],
            osrm_cfg=osrm_cfg,
            model_tag=Path(local_model_path).name,
        )
        A_init, osrm_stats = build_osrm_A_init(
            reference_features=reference_features,
            rank=config_data['lora_config']['r'],
            center_features=osrm_cfg['center_features'],
        )
        print(f"[OSRM] Built analytical A initialization for {len(A_init)} modules.")

    model = get_peft_model(base_model, lora_config).to('cuda')

    if osrm_cfg['enabled']:
        applied = apply_osrm_initialization_to_peft_model(
            peft_model=model,
            A_init=A_init,
            zero_B=osrm_cfg['zero_B'],
            freeze_A=osrm_cfg['freeze_A'],
            fallback_to_gaussian=osrm_cfg['fallback_to_gaussian'],
        )
        init_save_dir = os.path.join(config_data['training']['output_dir'], 'osrm_init')
        os.makedirs(init_save_dir, exist_ok=True)
        torch.save(
            {
                'task_name': task_name,
                'osrm_cfg': osrm_cfg,
                'applied_modules': applied,
                'stats': osrm_stats,
            },
            os.path.join(init_save_dir, 'osrm_init_info.pt'),
        )
        with open(os.path.join(init_save_dir, 'osrm_init_info.yaml'), 'w', encoding='utf-8') as f:
            yaml.safe_dump(
                {
                    'task_name': task_name,
                    'osrm_cfg': osrm_cfg,
                    'num_applied_modules': len(applied),
                    'applied_modules': applied,
                },
                f,
                allow_unicode=True,
                sort_keys=False,
            )
        print(f"[OSRM] Saved init info to: {init_save_dir}")

    print("[Trainable parameters]")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}, Shape: {tuple(param.shape)}")

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        per_device_train_batch_size=config_data['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config_data['training']['per_device_eval_batch_size'],
        num_train_epochs=config_data['training']['num_train_epochs'],
        learning_rate=config_data['training']['learning_rate'],
        warmup_steps=config_data['training']['warmup_steps'],
        weight_decay=config_data['training']['weight_decay'],
        output_dir=config_data['training']['output_dir'],
        logging_dir=config_data['training']['logging_dir'],
        logging_steps=config_data['training']['logging_steps'],
        do_train=config_data['training']['do_train'],
        evaluation_strategy=config_data['training']['evaluation_strategy'],
        save_strategy=config_data['training']['evaluation_strategy'],
        eval_steps=config_data['training']['eval_steps'],
        save_steps=config_data['training']['save_steps'],
        label_names=config_data['training']['label_names'],
        greater_is_better=config_data['training']['greater_is_better'],
        load_best_model_at_end=config_data['training']['load_best_model_at_end'],
        eval_accumulation_steps=config_data['training']['eval_accumulation_steps'],
        metric_for_best_model=config_data['training']['metric_for_best_model'],
        save_total_limit=config_data['training'].get('save_total_limit', 2),
        predict_with_generate=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets[validation_key],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    print("=" * 80)
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    print(f"Best metric ({training_args.metric_for_best_model}): {trainer.state.best_metric}")
    print("=" * 80)

    best_model_dir = os.path.join(config_data['training']['output_dir'], 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    best_info = {
        'best_model_checkpoint': trainer.state.best_model_checkpoint,
        'best_metric_name': training_args.metric_for_best_model,
        'best_metric_value': trainer.state.best_metric,
        'osrm_enabled': osrm_cfg['enabled'],
        'osrm_cfg': osrm_cfg,
    }
    with open(os.path.join(best_model_dir, 'best_model_info.yaml'), 'w', encoding='utf-8') as f:
        yaml.safe_dump(best_info, f, allow_unicode=True, sort_keys=False)

    print(f"Best model has been saved to: {best_model_dir}")
    print(f"Best model info saved to: {os.path.join(best_model_dir, 'best_model_info.yaml')}")

    export_best_adapter(
        best_model_dir=best_model_dir,
        tokenizer=tokenizer,
        task_name=task_name,
        osrm_cfg=osrm_cfg,
        osrm_enabled=osrm_cfg['enabled'],
        best_info=best_info,
    )


if __name__ == "__main__":
    main()
