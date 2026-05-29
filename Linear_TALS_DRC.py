import os
import gc
import csv
import fcntl
import yaml
import time
import torch
import random
import argparse
import traceback
import hashlib
import numpy as np
from datetime import datetime
from collections import defaultdict

from transformers import (
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)

from peft import PeftModel

from eval_model import eval_iteris_model
from get_midfeatures import (
    get_lora_pos,
    get_samples,
    merge_peft,
)

try:
    from safetensors.torch import load_file as safe_load_file
except Exception:
    safe_load_file = None


GLUE_task_name = [
    "mnli", "rte",
    "cola", "sst2", "qqp",
    "qnli", "mrpc", "wnli",
]


def get_loras_path(task_type, model_name, lora_root="best_LoRA"):
    """
    lora_root:
      best_LoRA -> 普通 LoRA
      OSRM_LoRA -> OSRM 初始化后微调得到的 LoRA
    """
    lora_path_dict = {}

    if "t5" in str(model_name).lower() and task_type == "GLUE_t5":
        lora_path_dict["cola"] = f"{lora_root}/T5-COLA-LoRA"
        lora_path_dict["sst2"] = f"{lora_root}/T5-SST2-LoRA"
        lora_path_dict["rte"]  = f"{lora_root}/T5-RTE-LoRA"
        lora_path_dict["qnli"] = f"{lora_root}/T5-QNLI-LoRA"
        lora_path_dict["qqp"]  = f"{lora_root}/T5-QQP-LoRA"
        lora_path_dict["mrpc"] = f"{lora_root}/T5-MRPC-LoRA"
        lora_path_dict["mnli"] = f"{lora_root}/T5-MNLI-LoRA"
        lora_path_dict["wnli"] = f"{lora_root}/T5-WNLI-LoRA"

    if task_type == "TASKS_blip_base":
        # For SentiCap BLIP LoRAs the directory layout is:
        #   loras/SENTICAP-lora-blip/positive
        #   loras/SENTICAP-lora-blip/negative
        # Keep lora_root configurable so other BLIP LoRA roots remain possible.
        lora_path_dict["positive"] = f"{lora_root}/positive"
        lora_path_dict["negative"] = f"{lora_root}/negative"
        lora_path_dict["roman"] = f"{lora_root}/roman"
        lora_path_dict["humor"] = f"{lora_root}/humor"

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
    """
    并发安全地写 CSV 表头。
    即使多个 pair 进程同时启动，也不会重复写 header。
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                writer = csv.writer(f)
                writer.writerow(header)
                f.flush()
                os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append_csv_row(csv_path, row):
    """
    并发安全地追加一行 CSV。
    现在 run_all_glue_pairs.py 会给每个 pair 设置独立 RESULTS_DIR，
    这里仍保留文件锁，防止未来直接并行写同一个 results 目录。
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def is_blip_model(model_name):
    return "blip" in str(model_name).lower()


def construct_base_model(model_name):
    model_name_l = str(model_name).lower()
    if "t5" in model_name_l:
        return T5ForConditionalGeneration.from_pretrained(model_name)
    elif "bart" in model_name_l:
        return BartForConditionalGeneration.from_pretrained(model_name)
    elif "blip" in model_name_l:
        return BlipForConditionalGeneration.from_pretrained(model_name)
    else:
        raise ValueError(f"[Linear_TALS_DRC] Unsupported model_name: {model_name}")


def load_required_linear_coarse_model(model_name, model_dir, model_label="Linear"):
    """
    加载已经保存好的 Linear / OSRM_Linear coarse merged model。

    注意：
    这里故意严格检查。
    如果目录不存在或不是 HuggingFace dense model，就直接报错，
    不允许静默重新做 Linear merge。
    """
    if model_dir is None or str(model_dir).strip() == "":
        raise ValueError(f"[Linear_TALS_DRC] Empty {model_label} coarse model dir.")

    model_dir = str(model_dir)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"[Linear_TALS_DRC] Required {model_label} coarse model dir not found: {model_dir}\n"
            f"Please run Linear.py first with save=1."
        )

    if not os.path.exists(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(
            f"[Linear_TALS_DRC] {model_label} coarse model dir exists but config.json is missing: {model_dir}\n"
            f"This does not look like a valid HuggingFace saved dense model directory."
        )

    print(f"[Linear_TALS_DRC] Loading required {model_label} coarse model from: {model_dir}")

    model_name_l = str(model_name).lower()
    if "t5" in model_name_l:
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
    elif "bart" in model_name_l:
        model = BartForConditionalGeneration.from_pretrained(model_dir)
    elif "blip" in model_name_l:
        model = BlipForConditionalGeneration.from_pretrained(model_dir)
    else:
        raise ValueError(f"[Linear_TALS_DRC] Unsupported model_name: {model_name}")

    return model


def load_single_lora_dense_model(model_name, lora_path, rank):
    """
    加载 base model，然后把单任务 LoRA 合并进 dense model。
    NLP/T5 保持原来 merge_peft 逻辑；BLIP 使用 PEFT merge_and_unload。
    """
    model = construct_base_model(model_name)
    if is_blip_model(model_name):
        peft_model = PeftModel.from_pretrained(model, lora_path)
        return peft_model.merge_and_unload()
    model = merge_peft(model, model_name, lora_path, rank)
    return model


def normalize_target_modules(value):
    if value is None:
        return ["q", "v"]
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    raise ValueError(f"Unsupported drc_target_modules type: {type(value)}")


def filter_lora_target_keys(
    lora_keys,
    target_part="encoder",
    target_modules=None,
    target_layers=None,
):
    """
    保留 LoRA target module input 注入位置。

    例如:
        encoder.block.1.layer.0.SelfAttention.q
        encoder.block.1.layer.0.SelfAttention.v
    """
    if target_modules is None:
        target_modules = ["q", "v"]

    filtered = []
    for key in lora_keys:
        if target_part and not key.startswith(target_part + "."):
            continue

        last_name = key.split(".")[-1]
        if target_modules and last_name not in target_modules:
            continue

        if target_layers is not None:
            parts = key.split(".")
            layer_id = None
            for i, p in enumerate(parts):
                if p == "block" and i + 1 < len(parts):
                    try:
                        layer_id = int(parts[i + 1])
                    except ValueError:
                        layer_id = None
                    break
            if layer_id is None or layer_id not in target_layers:
                continue

        filtered.append(key)

    return filtered


def build_encoder_block_output_keys(target_layers=None, num_layers=12):
    """
    构造 encoder block output 注入位置。

    例如:
        encoder.block.1
        encoder.block.2
        ...
    """
    if target_layers is None:
        target_layers = list(range(num_layers))

    return [f"encoder.block.{i}" for i in target_layers]


def pooled_mean_feature(x, attention_mask=None):
    """
    x:
        [batch, seq_len, hidden_dim]

    返回:
        [hidden_dim]
    """
    x = x.detach()

    if x.dim() == 3 and attention_mask is not None and x.size(1) == attention_mask.size(1):
        mask = attention_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        denom = mask.sum().clamp_min(1.0)
        pooled = (x * mask).sum(dim=(0, 1)) / denom
    else:
        pooled = x.reshape(-1, x.size(-1)).mean(dim=0)

    return pooled.detach().float().cpu()


def extract_hidden_from_output(output):
    """
    T5Block 的 output 通常是 tuple，第一项是 hidden_states。
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)):
        if len(output) == 0:
            return None
        if torch.is_tensor(output[0]):
            return output[0]

    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state

    return None


def replace_hidden_in_output(output, hidden_new):
    """
    将修改后的 hidden_states 放回原 output 结构。
    """
    if torch.is_tensor(output):
        return hidden_new

    if isinstance(output, tuple):
        return (hidden_new,) + output[1:]

    if isinstance(output, list):
        output = list(output)
        output[0] = hidden_new
        return output

    # 极少数 dataclass 情况暂不强行改，避免破坏结构
    return output


def collect_lora_input_features(
    model,
    lora_keys,
    input_ids,
    attention_mask,
    max_new_tokens=2,
):
    """
    在指定 LoRA target module 上注册 forward hook，收集 module input[0] 的 pooled feature。
    这里仅收集特征，不修改模型。
    """
    model.eval()
    model.to("cuda")

    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")

    modules = dict(model.named_modules())
    sums = {}
    counts = defaultdict(int)
    handles = []

    def make_hook(layer_name):
        def hook_fn(module, inputs, outputs):
            if inputs is None or len(inputs) == 0:
                return

            x = inputs[0]
            if not torch.is_tensor(x):
                return

            pooled = pooled_mean_feature(x, attention_mask=attention_mask)

            if layer_name not in sums:
                sums[layer_name] = pooled
            else:
                sums[layer_name] = sums[layer_name] + pooled
            counts[layer_name] += 1

        return hook_fn

    for key in lora_keys:
        if key not in modules:
            print(f"[Linear_TALS_DRC][Warn] Cannot find module for hook: {key}")
            continue

        handle = modules[key].register_forward_hook(make_hook(key))
        handles.append(handle)

    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

    for h in handles:
        h.remove()

    features = {}
    for key in sums:
        features[key] = sums[key] / max(counts[key], 1)

    return features


def collect_encoder_block_output_features(
    model,
    block_keys,
    input_ids,
    attention_mask,
    max_new_tokens=2,
):
    """
    在 encoder.block.{i} 上注册 forward hook，收集 block output hidden states 的 pooled feature。
    """
    model.eval()
    model.to("cuda")

    input_ids = input_ids.to("cuda")
    attention_mask = attention_mask.to("cuda")

    modules = dict(model.named_modules())
    sums = {}
    counts = defaultdict(int)
    handles = []

    def make_hook(layer_name):
        def hook_fn(module, inputs, outputs):
            hidden = extract_hidden_from_output(outputs)
            if hidden is None or not torch.is_tensor(hidden):
                return

            pooled = pooled_mean_feature(hidden, attention_mask=attention_mask)

            if layer_name not in sums:
                sums[layer_name] = pooled
            else:
                sums[layer_name] = sums[layer_name] + pooled
            counts[layer_name] += 1

        return hook_fn

    for key in block_keys:
        if key not in modules:
            print(f"[Linear_TALS_DRC][Warn] Cannot find encoder block for hook: {key}")
            continue

        handle = modules[key].register_forward_hook(make_hook(key))
        handles.append(handle)

    with torch.no_grad():
        _ = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

    for h in handles:
        h.remove()

    features = {}
    for key in sums:
        features[key] = sums[key] / max(counts[key], 1)

    return features


def get_task_samples(
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
    dataset = get_samples(
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

    input_ids = torch.tensor(dataset["train"]["input_ids"], dtype=torch.long)
    if "attention_mask" in dataset["train"].column_names:
        attention_mask = torch.tensor(dataset["train"]["attention_mask"], dtype=torch.long)
    else:
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

    return input_ids, attention_mask


def normalize_direction(delta, eps=1e-6):
    norm = torch.norm(delta.float())
    if norm < eps:
        return None, float(norm)
    return delta.float() / norm, float(norm)


def select_drc_targets(
    inject_position,
    lora_path_dict,
    task_targets,
    linear_model,
    target_part="encoder",
    target_modules=None,
    target_layers=None,
):
    """
    根据 drc_inject_position 选择 DRC 目标位置。
    """
    if inject_position == "lora_input":
        first_lora_file = os.path.join(
            lora_path_dict[task_targets[0]],
            "adapter_model.safetensors"
        )
        all_lora_keys = get_lora_pos(first_lora_file)

        target_modules = normalize_target_modules(target_modules)
        selected_target_keys = filter_lora_target_keys(
            all_lora_keys,
            target_part=target_part,
            target_modules=target_modules,
            target_layers=target_layers,
        )

        print(f"[Linear_TALS_DRC] drc_inject_position = lora_input")
        print(f"[Linear_TALS_DRC] Found {len(all_lora_keys)} LoRA target modules.")
        print(f"[Linear_TALS_DRC] Selected {len(selected_target_keys)} modules for DRC injection.")

    elif inject_position == "encoder_block_output":
        num_layers = getattr(linear_model.config, "num_layers", 12)
        selected_target_keys = build_encoder_block_output_keys(
            target_layers=target_layers,
            num_layers=num_layers,
        )

        print(f"[Linear_TALS_DRC] drc_inject_position = encoder_block_output")
        print(f"[Linear_TALS_DRC] Selected {len(selected_target_keys)} encoder blocks for DRC injection.")

    else:
        raise ValueError(
            f"Unsupported drc_inject_position={inject_position}. "
            f"Please use 'lora_input' or 'encoder_block_output'."
        )

    for key in selected_target_keys[:10]:
        print(f"[Linear_TALS_DRC]   target: {key}")
    if len(selected_target_keys) > 10:
        print(f"[Linear_TALS_DRC]   ... {len(selected_target_keys) - 10} more")

    return selected_target_keys


def collect_features_by_position(
    model,
    inject_position,
    target_keys,
    input_ids,
    attention_mask,
):
    """
    根据注入位置选择对应的特征收集函数。
    """
    if inject_position == "lora_input":
        return collect_lora_input_features(
            model=model,
            lora_keys=target_keys,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    if inject_position == "encoder_block_output":
        return collect_encoder_block_output_features(
            model=model,
            block_keys=target_keys,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    raise ValueError(f"Unsupported drc_inject_position={inject_position}")



def load_adapter_meta(lora_dir, default_rank=8, default_alpha=32):
    """
    读取 adapter_config.json 中的 LoRA rank / alpha。
    如果不存在，则使用 config 中的默认值。
    """
    import json

    cfg_path = os.path.join(lora_dir, "adapter_config.json")
    r = default_rank
    alpha = default_alpha

    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            r = int(cfg.get("r", r))
            alpha = float(cfg.get("lora_alpha", alpha))
        except Exception as e:
            print(f"[Linear_TALS_DRC][Warn] Failed to read adapter_config.json from {lora_dir}: {e}")

    return r, alpha


def find_lora_ab_keys(state_dict, target_key):
    """
    从 adapter_model.safetensors 的 state_dict 中寻找某个 target module 对应的 LoRA A/B。

    target_key 例子:
        encoder.block.1.layer.0.SelfAttention.q

    常见 PEFT key 例子:
        base_model.model.encoder.block.1.layer.0.SelfAttention.q.lora_A.weight
        base_model.model.encoder.block.1.layer.0.SelfAttention.q.lora_B.weight
    """
    a_candidates = []
    b_candidates = []

    for k in state_dict.keys():
        if target_key not in k:
            continue
        if "lora_A" in k and k.endswith("weight"):
            a_candidates.append(k)
        elif "lora_B" in k and k.endswith("weight"):
            b_candidates.append(k)

    if len(a_candidates) > 1:
        exact = [k for k in a_candidates if k.endswith(target_key + ".lora_A.weight")]
        if len(exact) == 1:
            a_candidates = exact
    if len(b_candidates) > 1:
        exact = [k for k in b_candidates if k.endswith(target_key + ".lora_B.weight")]
        if len(exact) == 1:
            b_candidates = exact

    if len(a_candidates) == 0 or len(b_candidates) == 0:
        return None, None

    return sorted(a_candidates)[0], sorted(b_candidates)[0]


def load_lora_delta_matrices(lora_dir, target_keys, default_rank=8, default_alpha=32):
    """
    为一个单任务 LoRA 还原各 target module 的等效 ΔW。

    ΔW = (lora_alpha / r) * B @ A

    返回:
        delta_dict[target_key] = Tensor[out_dim, in_dim]
    """
    if safe_load_file is None:
        raise ImportError(
            "safetensors is required for Linear_TALS_DRC, but safetensors.torch.load_file is unavailable."
        )

    adapter_path = os.path.join(lora_dir, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Cannot find adapter_model.safetensors: {adapter_path}")

    state_dict = safe_load_file(adapter_path)
    adapter_rank, adapter_alpha = load_adapter_meta(
        lora_dir,
        default_rank=default_rank,
        default_alpha=default_alpha,
    )
    scale = float(adapter_alpha) / float(adapter_rank)

    delta_dict = {}
    missing = []

    for target_key in target_keys:
        a_key, b_key = find_lora_ab_keys(state_dict, target_key)
        if a_key is None or b_key is None:
            missing.append(target_key)
            continue

        A = state_dict[a_key].float().cpu()
        B = state_dict[b_key].float().cpu()

        if A.dim() != 2 or B.dim() != 2:
            print(
                f"[Linear_TALS_DRC][Warn] Invalid LoRA tensor dim at {target_key}: "
                f"A={tuple(A.shape)}, B={tuple(B.shape)}"
            )
            continue

        if B.size(1) != A.size(0):
            print(
                f"[Linear_TALS_DRC][Warn] LoRA shape mismatch at {target_key}: "
                f"A={tuple(A.shape)}, B={tuple(B.shape)}"
            )
            continue

        delta_dict[target_key] = scale * (B @ A)

    if missing:
        print(f"[Linear_TALS_DRC][Warn] Missing LoRA A/B for {len(missing)} targets in {lora_dir}.")
        for item in missing[:10]:
            print(f"    missing: {item}")

    return delta_dict


def build_all_task_lora_deltas(task_targets, lora_path_dict, target_keys, rank, lora_alpha_list=None):
    """
    预加载当前 pair 所有任务在 target_keys 上的 LoRA ΔW。
    """
    all_deltas = {}

    for idx, task_name in enumerate(task_targets):
        default_alpha = 32
        if isinstance(lora_alpha_list, list) and idx < len(lora_alpha_list):
            default_alpha = lora_alpha_list[idx]

        print(f"[Linear_TALS_DRC] Load LoRA ΔW matrices for task={task_name}...")
        all_deltas[task_name] = load_lora_delta_matrices(
            lora_dir=lora_path_dict[task_name],
            target_keys=target_keys,
            default_rank=rank,
            default_alpha=default_alpha,
        )

    return all_deltas


def compute_merged_delta_for_key(all_lora_deltas, task_targets, linear_weights, key):
    """
    计算 Linear merged LoRA ΔW_c = sum_j λ_j ΔW_j。
    """
    merged_delta = None

    for idx, task_name in enumerate(task_targets):
        if key not in all_lora_deltas.get(task_name, {}):
            continue

        w = float(linear_weights[idx]) if idx < len(linear_weights) else 1.0 / len(task_targets)
        cur = all_lora_deltas[task_name][key].float()

        if merged_delta is None:
            merged_delta = w * cur
        else:
            merged_delta = merged_delta + w * cur

    return merged_delta


def stable_int_hash(text, mod=2 ** 31 - 1):
    """
    Python 内置 hash 会受 PYTHONHASHSEED 影响；这里用 md5 保证随机子空间可复现。
    """
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % mod


def make_random_orthogonal_basis(hidden_dim, rank, seed=42, eps=1e-6):
    """
    生成随机正交子空间 Q_k。

    返回:
        Q: [k, hidden_dim]，行向量正交。
    """
    hidden_dim = int(hidden_dim)
    k = int(max(1, min(rank, hidden_dim)))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) % (2 ** 31 - 1))

    A = torch.randn(hidden_dim, k, generator=gen).float()
    Q_col, _ = torch.linalg.qr(A, mode="reduced")
    Q = Q_col.t().contiguous()  # [k, hidden_dim]

    if not torch.isfinite(Q).all() or float(Q.norm()) < eps:
        return None

    return Q


def apply_tals_filter_to_activation_residual(
    activation_residual,
    source_delta_w=None,
    tals_rank=8,
    tals_gamma=0.5,
    tals_eps=1e-6,
    tals_weight_norm="mean",
    tals_svd_center=False,
    tals_subspace_source="missing",
    random_seed=42,
):
    """
    最小版 TALS-DRC 核心:

        r_filtered = V_k G_k V_k^T r_act

    tals_subspace_source:
        missing: V_k 来自 R_W = ΔW_t - ΔW_c，这是当前主方法。
        single : V_k 来自 ΔW_t，用于验证“单任务 LoRA 子空间”是否足够。
        random : V_k 替换为随机正交 Q_k，用于验证是否只是随机降维/投影有效。

    source_delta_w:
        missing 时传入 ΔW_t - ΔW_c
        single  时传入 ΔW_t
        random  时传入 None
    """
    source = str(tals_subspace_source).lower().strip()
    if source not in ["missing", "single", "random"]:
        raise ValueError(
            f"Unsupported tals_subspace_source={tals_subspace_source}. "
            "Please use 'missing', 'single', or 'random'."
        )

    r = activation_residual.detach().float().cpu()
    if r.dim() != 1:
        r = r.reshape(-1)

    if not torch.isfinite(r).all():
        return None, {"reason": "non_finite_activation_residual", "tals_subspace_source": source}

    raw_norm = float(torch.norm(r))
    if raw_norm < tals_eps:
        return None, {
            "reason": "tiny_activation_residual",
            "raw_norm": raw_norm,
            "tals_subspace_source": source,
        }

    # 1. 构造输入侧子空间基 Vk/Qk
    if source == "random":
        k = int(max(1, min(tals_rank, r.numel())))
        Vk = make_random_orthogonal_basis(
            hidden_dim=r.numel(),
            rank=k,
            seed=random_seed,
            eps=tals_eps,
        )
        if Vk is None:
            return None, {
                "reason": "random_basis_failed",
                "raw_activation_norm": raw_norm,
                "tals_subspace_source": source,
                "random_seed": int(random_seed),
            }

        Sk = torch.ones(k).float()
        source_delta_w_fro = 0.0
        missing_delta_w_fro = 0.0
        top_singular = 1.0
        tail_singular = 1.0

    else:
        if source_delta_w is None:
            return None, {
                "reason": "source_delta_w_is_none",
                "raw_activation_norm": raw_norm,
                "tals_subspace_source": source,
            }

        W = source_delta_w.detach().float().cpu()

        if W.dim() != 2:
            return None, {
                "reason": "source_delta_w_not_matrix",
                "W_shape": tuple(W.shape),
                "tals_subspace_source": source,
            }

        if W.size(1) != r.numel():
            return None, {
                "reason": "input_dim_mismatch",
                "residual_dim": int(r.numel()),
                "W_shape": tuple(W.shape),
                "tals_subspace_source": source,
            }

        if not torch.isfinite(W).all():
            return None, {"reason": "non_finite_source_delta_w", "tals_subspace_source": source}

        source_delta_w_fro = float(torch.norm(W))
        missing_delta_w_fro = source_delta_w_fro if source == "missing" else 0.0

        if source_delta_w_fro < tals_eps:
            return None, {
                "reason": "tiny_source_delta_w",
                "source_delta_w_fro": source_delta_w_fro,
                "tals_subspace_source": source,
            }

        X = W
        if tals_svd_center:
            X = X - X.mean(dim=0, keepdim=True)

        min_dim = min(X.size(0), X.size(1))
        k = int(max(1, min(tals_rank, min_dim)))

        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except Exception as e:
            return None, {
                "reason": f"svd_failed:{type(e).__name__}",
                "error": str(e),
                "tals_subspace_source": source,
            }

        if S.numel() == 0:
            return None, {"reason": "empty_singular_values", "tals_subspace_source": source}

        Vk = Vh[:k].contiguous()       # [k, in_dim]
        Sk = S[:k].float().clamp_min(0)
        top_singular = float(Sk[0]) if Sk.numel() > 0 else 0.0
        tail_singular = float(Sk[-1]) if Sk.numel() > 0 else 0.0

    # 2. 不带 G 的子空间投影比例，用于解释
    proj = torch.matmul(torch.matmul(r, Vk.t()), Vk)
    proj_norm = float(torch.norm(proj))
    lora_explainable_ratio = proj_norm / (raw_norm + tals_eps)

    # 3. 构造 G 并做重加权。random 没有奇异值结构，固定使用等权 G=I。
    if source == "random":
        g = torch.ones_like(Sk)
    else:
        if abs(float(tals_gamma)) < 1e-12:
            g = torch.ones_like(Sk)
        else:
            g = (Sk + float(tals_eps)).pow(float(tals_gamma))

        if tals_weight_norm == "mean":
            g = g / g.mean().clamp_min(tals_eps)
        elif tals_weight_norm == "sum":
            g = g / g.sum().clamp_min(tals_eps)
        elif tals_weight_norm == "max":
            g = g / g.max().clamp_min(tals_eps)
        elif tals_weight_norm in ["none", None]:
            pass
        else:
            raise ValueError(f"Unsupported tals_weight_norm={tals_weight_norm}")

    coeff = torch.matmul(r, Vk.t())          # [k]
    coeff = coeff * g
    filtered = torch.matmul(coeff, Vk)       # [in_dim]

    filtered_norm = float(torch.norm(filtered))
    tail_reweighted_ratio = filtered_norm / (raw_norm + tals_eps)

    stats = {
        "raw_activation_norm": raw_norm,
        "source_delta_w_fro": source_delta_w_fro,
        "missing_delta_w_fro": missing_delta_w_fro,
        "svd_rank_used": int(k),
        "tals_gamma": float(tals_gamma),
        "tals_weight_norm": str(tals_weight_norm),
        "tals_svd_center": bool(tals_svd_center),
        "tals_subspace_source": source,
        "random_seed": int(random_seed) if source == "random" else "",
        "top_singular": top_singular,
        "tail_singular": tail_singular,
        "lora_explainable_ratio": float(lora_explainable_ratio),
        "tail_reweighted_ratio": float(tail_reweighted_ratio),
        "filtered_norm": float(filtered_norm),
    }

    if filtered_norm < tals_eps:
        return None, {"reason": "tiny_filtered_residual", **stats}

    return filtered.float().cpu(), stats



def parse_optional_float(value, default=None):
    """
    YAML 中读取可选 float。
    支持 None / "none" / "null" / ""。
    """
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ["", "none", "null"]:
            return default
        return float(value)
    return float(value)


def apply_ler_layerwise_reweight(
    task_direction,
    task_stats,
    score_type="ler_act",
    norm_type="mean_one",
    clip_min=None,
    clip_max=None,
    eps=1e-6,
):
    """
    LER-guided layer-wise injection reweighting.

    当前第一版使用:
        s_{l,m} = LER_{l,m} * ||r_act_{l,m}||

    其中:
        LER = ||V_k V_k^T r_act|| / (||r_act|| + eps)
        ||r_act|| 存在 stats["raw_activation_norm"]

    然后做均值为 1 的归一化:
        omega_{l,m} = N * s_{l,m} / (sum s + eps)

    最后把 omega 直接乘到 direction 上:
        direction_{l,m} <- omega_{l,m} * direction_{l,m}

    这样 hook 代码无需修改，仍然执行:
        x_new = x + alpha * direction

    注意:
        这里的 direction 已经携带 layer/module-wise injection weight。
    """
    if not task_direction:
        return task_direction, task_stats, {}

    score_type = str(score_type).lower().strip()
    norm_type = str(norm_type).lower().strip()

    scores = {}
    score_debug = {}

    for key, direction in task_direction.items():
        st = task_stats.get(key, {})

        ler = float(st.get("lora_explainable_ratio", 0.0))
        raw_act_norm = float(st.get("raw_activation_norm", st.get("raw_norm", 0.0)))
        filtered_norm = float(st.get("filtered_norm", 0.0))
        source_delta_w_fro = float(st.get("source_delta_w_fro", 0.0))

        if score_type == "ler_act":
            score = ler * raw_act_norm
        elif score_type == "ler":
            score = ler
        elif score_type == "act":
            score = raw_act_norm
        elif score_type == "tail_act":
            trr = float(st.get("tail_reweighted_ratio", 0.0))
            score = trr * raw_act_norm
        elif score_type == "ler_act_wfro":
            # 后续增强备选：s = LER * ||r_act|| * ||R_W||_F
            score = ler * raw_act_norm * source_delta_w_fro
        elif score_type == "filtered_norm":
            score = filtered_norm
        else:
            raise ValueError(
                f"Unsupported tals_layer_weight_score={score_type}. "
                "Use 'ler_act', 'ler', 'act', 'tail_act', 'ler_act_wfro', or 'filtered_norm'."
            )

        if (not np.isfinite(score)) or score < 0:
            score = 0.0

        scores[key] = float(score)
        score_debug[key] = {
            "layer_weight_score_raw": float(score),
            "layer_weight_ler": float(ler),
            "layer_weight_raw_activation_norm": float(raw_act_norm),
            "layer_weight_filtered_norm": float(filtered_norm),
            "layer_weight_source_delta_w_fro": float(source_delta_w_fro),
        }

    valid_keys = list(task_direction.keys())
    N = len(valid_keys)
    total_score = float(sum(scores.values()))

    # 如果全部 score 为 0，退化为所有位置权重 1，避免破坏方向。
    if total_score <= eps:
        omegas = {key: 1.0 for key in valid_keys}
        fallback_uniform = True
    else:
        fallback_uniform = False
        if norm_type == "mean_one":
            omegas = {key: float(N * scores[key] / (total_score + eps)) for key in valid_keys}
        elif norm_type == "sum_one":
            omegas = {key: float(scores[key] / (total_score + eps)) for key in valid_keys}
        elif norm_type == "none":
            # 直接使用 score，不推荐作为主实验。
            mean_score = total_score / max(N, 1)
            omegas = {key: float(scores[key] / (mean_score + eps)) for key in valid_keys}
        else:
            raise ValueError(
                f"Unsupported tals_layer_weight_norm={norm_type}. "
                "Use 'mean_one', 'sum_one', or 'none'."
            )

    # 可选 clip。第一版建议 clip_min/max 设为 None，保持纯公式。
    if clip_min is not None or clip_max is not None:
        clipped = {}
        for key, omega in omegas.items():
            v = float(omega)
            if clip_min is not None:
                v = max(v, float(clip_min))
            if clip_max is not None:
                v = min(v, float(clip_max))
            clipped[key] = v

        # 对 mean_one 模式，clip 后再归一化到均值为 1，保持整体注入强度不变。
        if norm_type == "mean_one":
            mean_omega = float(sum(clipped.values()) / max(len(clipped), 1))
            if mean_omega > eps:
                clipped = {key: float(v / mean_omega) for key, v in clipped.items()}
        omegas = clipped

    new_task_direction = {}
    for key, direction in task_direction.items():
        omega = float(omegas.get(key, 1.0))
        new_task_direction[key] = (direction.float().cpu() * omega).cpu()

        if key not in task_stats:
            task_stats[key] = {}
        task_stats[key].update(score_debug.get(key, {}))
        task_stats[key]["layer_weight"] = omega
        task_stats[key]["layer_weight_score_type"] = score_type
        task_stats[key]["layer_weight_norm_type"] = norm_type
        task_stats[key]["layer_weight_fallback_uniform"] = bool(fallback_uniform)

    weights = list(omegas.values())
    summary = {
        "use_layer_weight": True,
        "score_type": score_type,
        "norm_type": norm_type,
        "num_weighted_targets": int(N),
        "fallback_uniform": bool(fallback_uniform),
        "score_sum": float(total_score),
        "omega_min": float(min(weights)) if weights else 0.0,
        "omega_max": float(max(weights)) if weights else 0.0,
        "omega_mean": float(sum(weights) / max(len(weights), 1)) if weights else 0.0,
    }

    return new_task_direction, task_stats, summary




# ======================================================================================
# VLM / BLIP helper functions.
# These helpers are only used when task_type/model_name is TASKS_blip_base/BLIP.
# NLP/T5 keeps the original code path.
# ======================================================================================

def tensor_to_cuda(x):
    if torch.is_tensor(x):
        return x.detach().clone().to("cuda")
    return torch.as_tensor(x).to("cuda")


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
    target_part="encoder",
    target_modules=None,
    target_layers=None,
    model_name=None,
):
    """
    Select DRC target modules.

    For T5/BART this delegates to the original select_drc_targets().
    For BLIP we parse PEFT LoRA keys such as:
        text_decoder.bert.encoder.layer.0.attention.self.query
    and then filter by target_part/target_modules/target_layers.
    """
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
        raise ValueError("[Linear_TALS_DRC][VLM] BLIP currently supports only drc_inject_position='lora_input'.")

    first_task = task_targets[0]
    adapter_file = os.path.join(lora_path_dict[first_task], "adapter_model.safetensors")
    if not os.path.exists(adapter_file):
        raise FileNotFoundError(f"[Linear_TALS_DRC][VLM] adapter_model.safetensors not found: {adapter_file}")

    state_dict = safe_load_file(adapter_file) if safe_load_file is not None else None
    if state_dict is None:
        raise ImportError("[Linear_TALS_DRC][VLM] safetensors.torch.load_file is required.")

    model_modules = dict(linear_model.named_modules())
    target_keys = []

    for full_key in sorted(state_dict.keys()):
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
            print(f"[Linear_TALS_DRC][VLM][Warn] LoRA target not found in dense BLIP modules: {base_name}")
            continue

        target_keys.append(base_name)

    target_keys = sorted(set(target_keys))
    print(f"[Linear_TALS_DRC][VLM] Selected {len(target_keys)} BLIP DRC targets.")
    for key in target_keys[:20]:
        print(f"[Linear_TALS_DRC][VLM]   target: {key}")
    if len(target_keys) > 20:
        print(f"[Linear_TALS_DRC][VLM]   ... {len(target_keys) - 20} more")

    if len(target_keys) == 0:
        raise RuntimeError(
            "[Linear_TALS_DRC][VLM] No BLIP DRC targets selected. "
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
    """
    NLP returns (input_ids, attention_mask).
    BLIP returns a processor batch containing pixel_values/input_ids/attention_mask.
    """
    if is_blip_model(model_name):
        batch = get_samples(
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
            "input_ids": tensor_to_cuda(batch["input_ids"]).long(),
            "attention_mask": tensor_to_cuda(batch["attention_mask"]).long(),
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
    """
    Collect target module input features.

    Important for BLIP:
      module input is usually [batch, seq_len, hidden_dim].
      TALS expects a direction vector whose dim equals LoRA input dim, e.g. 768.
      Therefore BLIP features are pooled over batch/sequence/generation calls to [hidden_dim].
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
        raise ValueError("[Linear_TALS_DRC][VLM] Only drc_inject_position='lora_input' is supported for BLIP.")

    model.eval()
    model.to("cuda")

    modules = dict(model.named_modules())
    sums = {}
    counts = defaultdict(int)
    handles = []

    def _pool_blip_feature(x, mask=None):
        if not torch.is_tensor(x):
            return None
        x = x.detach().float()

        if x.dim() == 3:
            if mask is not None and torch.is_tensor(mask) and x.size(0) == mask.size(0) and x.size(1) == mask.size(1):
                m = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
                denom = m.sum().clamp_min(1.0)
                pooled = (x * m).sum(dim=(0, 1)) / denom
            else:
                pooled = x.reshape(-1, x.size(-1)).mean(dim=0)
            return pooled.detach().cpu()

        if x.dim() == 2:
            return x.mean(dim=0).detach().cpu()

        if x.dim() >= 1:
            return x.reshape(-1, x.size(-1)).mean(dim=0).detach().cpu()

        return None

    def make_hook(layer_name):
        def hook_fn(module, inputs, outputs):
            if inputs is None or len(inputs) == 0:
                return
            mask = sample_batch.get("attention_mask", None) if isinstance(sample_batch, dict) else None
            pooled = _pool_blip_feature(inputs[0], mask=mask)
            if pooled is None:
                return
            if layer_name not in sums:
                sums[layer_name] = pooled
            else:
                sums[layer_name] = sums[layer_name] + pooled
            counts[layer_name] += 1
        return hook_fn

    for key in target_keys:
        if key not in modules:
            print(f"[Linear_TALS_DRC][VLM][Warn] Cannot find module for hook: {key}")
            continue
        handles.append(modules[key].register_forward_hook(make_hook(key)))

    with torch.no_grad():
        _ = model.generate(
            pixel_values=sample_batch["pixel_values"],
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            max_length=max_length,
        )

    for h in handles:
        h.remove()

    features = {}
    for key in sums:
        features[key] = (sums[key] / max(counts[key], 1)).float().cpu()

    shown = 0
    for key, value in features.items():
        print(f"[Linear_TALS_DRC][VLM][FeatureShape] {key}: {tuple(value.shape)}")
        shown += 1
        if shown >= 3:
            break

    return features


def get_primary_metric_any(task_name, eval_result, task_type):
    if task_type == "TASKS_blip_base":
        if "acc" in eval_result and eval_result["acc"] not in ["", None]:
            v = float(eval_result["acc"])
            return "acc", v, v
        for key in ["style_acc", "style_accuracy", "eval_style_acc", "eval_accuracy", "accuracy"]:
            if key in eval_result and eval_result[key] not in ["", None]:
                v = float(eval_result[key])
                return key, v, v
        if "cider" in eval_result and eval_result["cider"] not in ["", None]:
            v = float(eval_result["cider"])
            return "cider", v, v
        return "acc", "", ""

    return get_primary_metric(task_name, eval_result)


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
    eval_result,
    merged_model_dir,
    log_file,
    record_type="best",
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
            alpha,
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
            record_type,
        ],
    )


def build_task_specific_drc_directions(
    model_name,
    tokenizer,
    task_targets,
    lora_path_dict,
    linear_model,
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
    linear_weights=None,
    lora_alpha_list=None,
    tals_rank=8,
    tals_gamma=0.5,
    tals_eps=1e-6,
    tals_weight_norm="mean",
    tals_svd_center=False,
    tals_subspace_source="missing",
    tals_fallback_to_base=False,
    tals_use_layer_weight=False,
    tals_layer_weight_score="ler_act",
    tals_layer_weight_norm="mean_one",
    tals_layer_weight_clip_min=None,
    tals_layer_weight_clip_max=None,
):
    """
    对每个任务单独构造 TALS-DRC 方向:

        r_act = h_single - h_linear

    权重子空间:
        missing: R_W = ΔW_task - ΔW_linear
        single : R_W = ΔW_task
        random : random orthogonal subspace

    VLM/BLIP uses the same method but different input pipeline:
        pixel_values + input_ids + attention_mask,
    and BLIP module features are pooled to [hidden_dim].
    """
    selected_target_keys = select_drc_targets_any(
        inject_position=inject_position,
        lora_path_dict=lora_path_dict,
        task_targets=task_targets,
        linear_model=linear_model,
        target_part=target_part,
        target_modules=target_modules,
        target_layers=target_layers,
        model_name=model_name,
    )

    if inject_position != "lora_input":
        raise ValueError(
            "Linear_TALS_DRC currently supports only drc_inject_position='lora_input', "
            "because TALS uses input-side singular vectors of LoRA target-module ΔW."
        )

    if linear_weights is None:
        linear_weights = [1.0 / len(task_targets)] * len(task_targets)

    all_lora_deltas = build_all_task_lora_deltas(
        task_targets=task_targets,
        lora_path_dict=lora_path_dict,
        target_keys=selected_target_keys,
        rank=rank,
        lora_alpha_list=lora_alpha_list,
    )

    all_task_directions = {}
    direction_stats = {}

    for task_name in task_targets:
        print(f"\n[Linear_TALS_DRC] Building task-specific DRC direction for task = {task_name}")

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
        print(f"[Linear_TALS_DRC] Collect base features on {task_name} samples...")
        base_model = construct_base_model(model_name).to("cuda")
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

        # 2. Single-task LoRA dense model features
        print(f"[Linear_TALS_DRC] Collect single-LoRA features for {task_name}...")
        single_lora_path = lora_path_dict[task_name]
        single_model = load_single_lora_dense_model(
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

        # 3. Linear merged model features
        print(f"[Linear_TALS_DRC] Collect Linear merged features on {task_name} samples...")
        if is_blip_model(model_name):
            linear_features = collect_features_by_position_any(
                model=linear_model,
                model_name=model_name,
                inject_position=inject_position,
                target_keys=selected_target_keys,
                sample_batch=sample_batch,
                max_length=max_length,
            )
        else:
            input_ids, attention_mask = sample_batch
            linear_features = collect_features_by_position_any(
                model=linear_model,
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
            "missing_linear": [],
            "missing_lora_delta": [],
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
            if key not in linear_features:
                missing_report["missing_linear"].append(key)
                continue

            # activation residual: h_single - h_linear
            single_shift = single_features[key] - base_features[key]
            linear_shift = linear_features[key] - base_features[key]
            delta = single_shift - linear_shift

            if key not in all_lora_deltas.get(task_name, {}):
                missing_report["missing_lora_delta"].append((task_name, key))
                continue

            merged_delta_w = compute_merged_delta_for_key(
                all_lora_deltas=all_lora_deltas,
                task_targets=task_targets,
                linear_weights=linear_weights,
                key=key,
            )
            if merged_delta_w is None:
                missing_report["missing_lora_delta"].append(("merged", key))
                continue

            task_delta_w = all_lora_deltas[task_name][key].float()

            if source == "missing":
                source_delta_w = task_delta_w - merged_delta_w.float()
            elif source == "single":
                source_delta_w = task_delta_w
            elif source == "random":
                source_delta_w = None
            else:
                raise ValueError(f"Unsupported tals_subspace_source={source}")

            random_seed = seed + stable_int_hash(f"linear|{task_name}|{key}|{tals_rank}")

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
                        f"[Linear_TALS_DRC][Warn] TALS filter failed at {key}; "
                        "fallback to base DRC direction."
                    )
                    tals_delta = delta.float().cpu()
                    tals_stats = {"fallback_to_base": True, **tals_stats}
                else:
                    continue

            raw_norm = float(torch.norm(tals_delta.float()))

            if normalize:
                delta_normed, used_raw_norm = normalize_direction(tals_delta, eps=tals_eps)
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

        print(f"[Linear_TALS_DRC][Debug] Missing/skip report for task={task_name}:")
        for reason, items in missing_report.items():
            print(f"  - {reason}: {len(items)}")
            for item in items[:10]:
                print(f"      {item}")

        print(
            f"[Linear_TALS_DRC] Task {task_name}: built directions for "
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
            print(f"[Linear_TALS_DRC][LayerWeight] task={task_name}, summary={lw_summary}")
            direction_stats[f"{task_name}__layer_weight_summary"] = lw_summary

        all_task_directions[task_name] = task_direction
        direction_stats[task_name] = task_stats

    return all_task_directions, direction_stats, selected_target_keys


def register_drc_pre_hooks(
    model,
    directions,
    alpha=0.03,
    use_hidden_norm_scale=False,
):
    """
    LoRA target module input 注入。

    对每个 LoRA target module:
        x_new = x + alpha * direction
    """
    modules = dict(model.named_modules())
    handles = []

    def make_pre_hook(layer_name, direction):
        def pre_hook(module, inputs):
            if inputs is None or len(inputs) == 0:
                return inputs

            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs

            if x.size(-1) != direction.numel():
                print(
                    f"[Linear_TALS_DRC][Warn] Shape mismatch at {layer_name}: "
                    f"x hidden={x.size(-1)}, direction={direction.numel()}. Skip."
                )
                return inputs

            d = direction.to(device=x.device, dtype=x.dtype)
            view_shape = [1] * x.dim()
            view_shape[-1] = -1
            d = d.view(*view_shape)

            if use_hidden_norm_scale:
                with torch.no_grad():
                    scale = x.detach().float().norm(dim=-1, keepdim=True).mean()
                x_new = x + float(alpha) * scale.to(device=x.device, dtype=x.dtype) * d
            else:
                x_new = x + float(alpha) * d

            return (x_new,) + tuple(inputs[1:])

        return pre_hook

    for layer_name, direction in directions.items():
        if layer_name not in modules:
            print(f"[Linear_TALS_DRC][Warn] Cannot find module for DRC hook: {layer_name}")
            continue

        handle = modules[layer_name].register_forward_pre_hook(
            make_pre_hook(layer_name, direction)
        )
        handles.append(handle)

    print(f"[Linear_TALS_DRC] Registered {len(handles)} LoRA input DRC pre-hooks.")
    return handles


def register_encoder_block_output_hooks(
    model,
    directions,
    alpha=0.03,
    use_hidden_norm_scale=False,
):
    """
    Encoder block output 注入。

    对每个 encoder.block.{i}:
        h_new = h + alpha * direction
    """
    modules = dict(model.named_modules())
    handles = []

    def make_hook(layer_name, direction):
        def hook_fn(module, inputs, outputs):
            hidden = extract_hidden_from_output(outputs)
            if hidden is None or not torch.is_tensor(hidden):
                return outputs

            if hidden.size(-1) != direction.numel():
                print(
                    f"[Linear_TALS_DRC][Warn] Shape mismatch at {layer_name}: "
                    f"hidden={hidden.size(-1)}, direction={direction.numel()}. Skip."
                )
                return outputs

            d = direction.to(device=hidden.device, dtype=hidden.dtype)
            view_shape = [1] * hidden.dim()
            view_shape[-1] = -1
            d = d.view(*view_shape)

            if use_hidden_norm_scale:
                with torch.no_grad():
                    scale = hidden.detach().float().norm(dim=-1, keepdim=True).mean()
                hidden_new = hidden + float(alpha) * scale.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                ) * d
            else:
                hidden_new = hidden + float(alpha) * d

            return replace_hidden_in_output(outputs, hidden_new)

        return hook_fn

    for layer_name, direction in directions.items():
        if layer_name not in modules:
            print(f"[Linear_TALS_DRC][Warn] Cannot find encoder block for DRC hook: {layer_name}")
            continue

        handle = modules[layer_name].register_forward_hook(
            make_hook(layer_name, direction)
        )
        handles.append(handle)

    print(f"[Linear_TALS_DRC] Registered {len(handles)} encoder block output DRC hooks.")
    return handles


def register_drc_hooks_by_position(
    model,
    directions,
    inject_position,
    alpha=0.03,
    use_hidden_norm_scale=False,
):
    if inject_position == "lora_input":
        return register_drc_pre_hooks(
            model=model,
            directions=directions,
            alpha=alpha,
            use_hidden_norm_scale=use_hidden_norm_scale,
        )

    if inject_position == "encoder_block_output":
        return register_encoder_block_output_hooks(
            model=model,
            directions=directions,
            alpha=alpha,
            use_hidden_norm_scale=use_hidden_norm_scale,
        )

    raise ValueError(f"Unsupported drc_inject_position={inject_position}")


def remove_hooks(handles):
    for h in handles:
        h.remove()


def get_primary_metric(task_name, eval_result):
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



def parse_float_list(value, default=None):
    """
    从 yaml 中读取 alpha candidates。

    支持:
        drc_alpha_candidates: [0.0, 0.03, 0.1]
        drc_alpha_candidates: "0.0,0.03,0.1"
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
    记录每个任务每个候选 alpha 的评测结果。
    注意：这些结果不写入 pair_merge_results.csv，避免汇总时重复统计。
    这里在锁内判断是否需要写 header，避免多进程同时创建文件导致 header/行丢失。
    """
    header = [
        "experiment_id", "method", "pair_name", "evaluated_task",
        "alpha", "primary_metric_name", "primary_metric_value",
        "normalized_metric", "eval_accuracy", "eval_mcc", "eval_f1",
        "eval_loss", "eval_runtime", "eval_peak_vram_mb",
    ]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            need_header = f.tell() == 0
            writer = csv.writer(f)
            if need_header:
                writer.writerow(header)
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def make_cache_path(
    cache_dir,
    method_name,
    pair_name,
    inject_position,
    target_layers,
    target_modules,
):
    layer_tag = (
        "layers_" + "_".join(map(str, target_layers))
        if target_layers is not None
        else "layers_all"
    )

    if inject_position == "lora_input":
        module_tag = "mods_" + "_".join(normalize_target_modules(target_modules))
    else:
        module_tag = "mods_block_output"

    return os.path.join(
        cache_dir,
        f"{method_name}_{pair_name}_{inject_position}_{layer_tag}_{module_tag}.pt"
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
    rank = task_cfg.get("rank", 8)
    linear_weights = task_cfg.get("linear_weights", [1.0 / len(task_targets)] * len(task_targets))

    # LoRA 来源选择：
    # default / best / normal / gaussian -> best_LoRA
    # osrm -> OSRM_LoRA
    lora_source = str(task_cfg.get("lora_source", "default")).lower().strip()

    if lora_source in ["default", "best", "normal", "gaussian"]:
        default_lora_root = "loras/SENTICAP-lora-blip" if task_type == "TASKS_blip_base" else "best_LoRA"
        lora_root = task_cfg.get("lora_root", default_lora_root)
        coarse_prefix_default = "merged_model/Linear_"
        default_method_name = "Linear_TALS_DRC_LER_BLIP" if task_type == "TASKS_blip_base" else "Linear_TALS_DRC_LER"
        coarse_label = "Linear"
    elif lora_source == "osrm":
        lora_root = task_cfg.get("lora_root", "OSRM_LoRA")
        coarse_prefix_default = "merged_model/OSRM_Linear_"
        default_method_name = "OSRM_Linear_TALS_DRC_LER"
        coarse_label = "OSRM_Linear"
    else:
        # 允许用户直接把 lora_source 写成一个目录名
        lora_root = task_cfg.get("lora_root", lora_source)
        coarse_prefix_default = task_cfg.get("linear_tals_coarse_model_prefix", "merged_model/Linear_")
        default_method_name = "Linear_TALS_DRC_LER"
        coarse_label = str(lora_source)

    print(f"[Linear_TALS_DRC] lora_source = {lora_source}")
    print(f"[Linear_TALS_DRC] lora_root = {lora_root}")

    tokenizer = (
        AutoTokenizer.from_pretrained(model_name)
        if "blip" not in model_name
        else AutoProcessor.from_pretrained(model_name)
    )

    lora_path_dict = get_loras_path(
        task_type=task_type,
        model_name=model_name,
        lora_root=lora_root,
    )

    for task in task_targets:
        adapter_path = os.path.join(lora_path_dict[task], "adapter_model.safetensors")
        if not os.path.exists(adapter_path):
            raise FileNotFoundError(
                f"[Linear_TALS_DRC] Cannot find adapter_model.safetensors for task={task}: {adapter_path}\n"
                f"Please check lora_source={lora_source}, lora_root={lora_root}."
            )

    pair_name = "_".join(task_targets)

    method_name = task_cfg.get(
        "linear_tals_method_name",
        task_cfg.get(
            "tals_method_name",
            task_cfg.get("drc_method_name", default_method_name),
        ),
    )
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{method_name}_{pair_name}"

    linear_tals_load_coarse_from_dir = bool(
        task_cfg.get("linear_tals_load_coarse_from_dir", True)
    )

    existing_prefix = str(
        task_cfg.get("linear_tals_existing_coarse_prefix", "")
    ).strip()

    if existing_prefix:
        default_coarse_model_dir = f"{existing_prefix}{pair_name}"
    else:
        default_coarse_model_dir = f"{coarse_prefix_default}{pair_name}"

    linear_tals_coarse_model_dir = task_cfg.get(
        "linear_tals_coarse_model_dir",
        task_cfg.get("linear_merged_model_dir", default_coarse_model_dir),
    )

    if not linear_tals_load_coarse_from_dir:
        raise ValueError(
            "[Linear_TALS_DRC] linear_tals_load_coarse_from_dir must be True. "
            "This script should load an existing Linear/OSRM_Linear coarse model, "
            "not rebuild Linear inside Linear_TALS_DRC.py."
        )

    merged_model_dir = linear_tals_coarse_model_dir

    print(f"[Linear_TALS_DRC] linear_tals_load_coarse_from_dir = {linear_tals_load_coarse_from_dir}")
    print(f"[Linear_TALS_DRC] Expected coarse model dir = {merged_model_dir}")

    model = load_required_linear_coarse_model(
        model_name=model_name,
        model_dir=merged_model_dir,
        model_label=coarse_label,
    ).to("cuda")

    print("[Linear_TALS_DRC] Loaded saved coarse model. No Linear merge is run in this script.")

    # DRC config
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
    drc_normalize_direction = task_cfg.get("drc_normalize_direction", True)
    drc_use_hidden_norm_scale = task_cfg.get("drc_use_hidden_norm_scale", False)
    drc_rebuild_cache = task_cfg.get("drc_rebuild_cache", True)

    # TALS minimal version config:
    # d_TALS = Normalize(V_k G_k V_k^T r_act)
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

    # LER-guided layer-wise reweighting.
    # 第一版:
    #   s_{l,m} = LER_{l,m} * ||r_act_{l,m}||
    #   omega_{l,m} = N * s_{l,m} / (sum s + eps)
    tals_use_layer_weight = bool(task_cfg.get("tals_use_layer_weight", False))
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

    # Keep Linear_TALS cache separate when requested, so it will not collide with IterIS_TALS_DRC.
    cache_dir = task_cfg.get("linear_tals_cache_dir", task_cfg.get("drc_cache_dir", "direction_cache"))
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

    print(f"[Linear_TALS_DRC] drc_inject_position = {drc_inject_position}")
    print(f"[Linear_TALS_DRC] drc_alpha = {drc_alpha}")
    print(f"[Linear_TALS_DRC] drc_alpha_search = {drc_alpha_search}")
    print(f"[Linear_TALS_DRC] drc_alpha_candidates = {drc_alpha_candidates}")
    print(f"[Linear_TALS_DRC] drc_samples_per_task = {drc_samples_per_task}")
    print(f"[Linear_TALS_DRC] drc_target_part = {drc_target_part}")
    print(f"[Linear_TALS_DRC] drc_target_modules = {drc_target_modules}")
    print(f"[Linear_TALS_DRC] drc_target_layers = {drc_target_layers}")
    print(f"[Linear_TALS_DRC] drc_normalize_direction = {drc_normalize_direction}")
    print(f"[Linear_TALS_DRC] tals_rank = {tals_rank}")
    print(f"[Linear_TALS_DRC] tals_gamma = {tals_gamma}")
    print(f"[Linear_TALS_DRC] tals_weight_norm = {tals_weight_norm}")
    print(f"[Linear_TALS_DRC] tals_svd_center = {tals_svd_center}")
    print(f"[Linear_TALS_DRC] tals_subspace_source = {tals_subspace_source}")
    print(f"[Linear_TALS_DRC] tals_fallback_to_base = {tals_fallback_to_base}")
    print(f"[Linear_TALS_DRC] tals_use_layer_weight = {tals_use_layer_weight}")
    print(f"[Linear_TALS_DRC] tals_layer_weight_score = {tals_layer_weight_score}")
    print(f"[Linear_TALS_DRC] tals_layer_weight_norm = {tals_layer_weight_norm}")
    print(f"[Linear_TALS_DRC] tals_layer_weight_clip_min = {tals_layer_weight_clip_min}")
    print(f"[Linear_TALS_DRC] tals_layer_weight_clip_max = {tals_layer_weight_clip_max}")
    print(f"[Linear_TALS_DRC] cache_path = {cache_path}")

    start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    if (not drc_rebuild_cache) and os.path.exists(cache_path):
        print(f"[Linear_TALS_DRC] Load DRC direction cache from: {cache_path}")
        cache = torch.load(cache_path, map_location="cpu")
        drc_directions = cache["directions"]
        direction_stats = cache.get("direction_stats", {})
        selected_target_keys = cache.get(
            "selected_target_keys",
            cache.get("selected_lora_keys", []),
        )
    else:
        drc_directions, direction_stats, selected_target_keys = build_task_specific_drc_directions(
            model_name=model_name,
            tokenizer=tokenizer,
            task_targets=task_targets,
            lora_path_dict=lora_path_dict,
            linear_model=model,
            rank=rank,
            max_length=task_cfg["max_length"],
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
            linear_weights=linear_weights,
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
                "base_method": coarse_label,
                "lora_source": lora_source,
                "lora_root": lora_root,
                "coarse_model_dir": merged_model_dir,
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
        print(f"[Linear_TALS_DRC] Saved DRC direction cache to: {cache_path}")

    fusion_time = round(time.time() - start_time, 4)
    fusion_peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024 ** 2, 2)

    fusion_stats = {
        "fusion_iter_time_avg_sec": fusion_time,
        "fusion_iter_time_max_sec": fusion_time,
        "fusion_peak_vram_avg_mb": fusion_peak_vram_mb,
        "fusion_peak_vram_max_mb": fusion_peak_vram_mb,
    }

    results_dir = os.environ.get("RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"[Linear_TALS_DRC] results_dir = {results_dir}")

    results_csv = os.path.join(results_dir, "pair_merge_results.csv")
    registry_csv = os.path.join(results_dir, "experiment_registry.csv")
    alpha_search_csv = os.path.join(results_dir, "drc_alpha_search_results.csv")
    vlm_results_csv = os.path.join(results_dir, "vlm_caption_results.csv")

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

    log_file = os.environ.get("LOG_FILE", "")
    normalized_metrics = []
    selected_alpha_by_task = {}

    try:
        for task_name in task_targets:
            print(f"\n[Eval] Evaluating Linear + task-specific TALS-DRC on {task_name}...")

            if task_name not in drc_directions:
                raise ValueError(f"Cannot find DRC direction for task: {task_name}")

            if drc_alpha_search:
                candidate_alphas = drc_alpha_candidates
            else:
                candidate_alphas = [drc_alpha]

            best_record = None

            for alpha in candidate_alphas:
                print(f"[Linear_TALS_DRC][AlphaSearch] task={task_name}, alpha={alpha}")

                handles = register_drc_hooks_by_position(
                    model=model,
                    directions=drc_directions[task_name],
                    inject_position=drc_inject_position,
                    alpha=float(alpha),
                    use_hidden_norm_scale=drc_use_hidden_norm_scale,
                )

                eval_result = eval_iteris_model(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    task_name=task_name,
                    max_length=task_cfg["max_length"],
                    per_device_eval_batch_size=task_cfg.get("per_device_eval_batch_size", 8),
                )

                remove_hooks(handles)
                torch.cuda.empty_cache()
                gc.collect()

                eval_accuracy = eval_result.get("eval_accuracy", "")
                eval_mcc = eval_result.get("eval_MCC", "")
                eval_f1 = eval_result.get("eval_f1-score", "")
                eval_loss = eval_result.get("eval_loss", "")
                eval_runtime = eval_result.get("eval_runtime", "")
                eval_sps = eval_result.get("eval_samples_per_second", "")
                eval_stepsps = eval_result.get("eval_steps_per_second", "")
                eval_peak_vram_mb = eval_result.get("eval_peak_vram_mb", "")

                primary_metric_name, primary_metric_value, normalized_metric = get_primary_metric_any(
                    task_name=task_name,
                    eval_result=eval_result,
                    task_type=task_type,
                )
                if task_type == "TASKS_blip_base":
                    # Keep the original CSV schema: store style accuracy in eval_accuracy.
                    eval_accuracy = primary_metric_value

                # 记录每个候选 alpha 的搜索结果。注意：不写入 pair_merge_results.csv。
                append_alpha_search_row(
                    alpha_search_csv,
                    [
                        experiment_id, method_name, pair_name, task_name,
                        float(alpha), primary_metric_name, primary_metric_value,
                        normalized_metric, eval_accuracy, eval_mcc, eval_f1,
                        eval_loss, eval_runtime, eval_peak_vram_mb,
                    ]
                )

                current_record = {
                    "alpha": float(alpha),
                    "eval_result": eval_result,
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

            selected_alpha_by_task[task_name] = best_record["alpha"]
            normalized_metrics.append(best_record["normalized_metric"])

            print(
                f"[Linear_TALS_DRC][AlphaSearch][Best] task={task_name}, "
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
                    eval_result=best_record["eval_result"],
                    merged_model_dir=merged_model_dir,
                    log_file=log_file,
                    record_type="best",
                )

            # pair_merge_results.csv 只写每个任务的最佳 alpha 结果，避免汇总重复统计。
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
                    ""
                ]
            )

        pair_avg_normalized_metric = float(np.mean(normalized_metrics))

        selected_alpha_str = "|".join(
            [f"{task}:{selected_alpha_by_task.get(task, drc_alpha)}" for task in task_targets]
        )

        drc_cfg_str = (
            f"lora_source={lora_source}|"
            f"lora_root={lora_root}|"
            f"coarse_model_dir={merged_model_dir}|"
            f"inject_position={drc_inject_position}|"
            f"alpha_search={drc_alpha_search}|"
            f"alpha_candidates={drc_alpha_candidates}|"
            f"selected_alpha_by_task={selected_alpha_str}|"
            f"samples_per_task={drc_samples_per_task}|"
            f"target_part={drc_target_part}|"
            f"target_modules={normalize_target_modules(drc_target_modules)}|"
            f"target_layers={drc_target_layers}|"
            f"normalize={drc_normalize_direction}|"
            f"use_hidden_norm_scale={drc_use_hidden_norm_scale}|"
            f"tals_rank={tals_rank}|"
            f"tals_gamma={tals_gamma}|"
            f"tals_weight_norm={tals_weight_norm}|"
            f"tals_svd_center={tals_svd_center}|"
            f"tals_subspace_source={tals_subspace_source}|"
            f"tals_fallback_to_base={tals_fallback_to_base}|"
            f"tals_use_layer_weight={tals_use_layer_weight}|"
            f"tals_layer_weight_score={tals_layer_weight_score}|"
            f"tals_layer_weight_norm={tals_layer_weight_norm}|"
            f"tals_layer_weight_clip_min={tals_layer_weight_clip_min}|"
            f"tals_layer_weight_clip_max={tals_layer_weight_clip_max}"
        )

        append_csv_row(
            registry_csv,
            [
                experiment_id, "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise", method_name, model_name, pair_name,
                "|".join(task_targets), f"{linear_weights}|{drc_cfg_str}",
                rank, "|".join(map(str, task_cfg.get("lora_alpha", []))),
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_avg_sec"],
                fusion_stats["fusion_iter_time_max_sec"],
                fusion_stats["fusion_peak_vram_avg_mb"],
                fusion_stats["fusion_peak_vram_max_mb"],
                pair_avg_normalized_metric,
                "success", ""
            ]
        )

        print(f"\n[Done] Linear_TALS_DRC finished for pair: {pair_name}")
        print(f"[Done] selected_alpha_by_task = {selected_alpha_by_task}")
        print(f"[Done] pair_avg_normalized_metric = {pair_avg_normalized_metric}")

    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)

        selected_alpha_str = "|".join(
            [f"{task}:{selected_alpha_by_task.get(task, drc_alpha)}" for task in task_targets]
        ) if "selected_alpha_by_task" in locals() else ""

        drc_cfg_str = (
            f"inject_position={drc_inject_position}|"
            f"alpha_search={drc_alpha_search}|"
            f"alpha_candidates={drc_alpha_candidates}|"
            f"selected_alpha_by_task={selected_alpha_str}|"
            f"samples_per_task={drc_samples_per_task}|"
            f"target_part={drc_target_part}|"
            f"target_modules={normalize_target_modules(drc_target_modules)}|"
            f"target_layers={drc_target_layers}|"
            f"normalize={drc_normalize_direction}|"
            f"use_hidden_norm_scale={drc_use_hidden_norm_scale}|"
            f"tals_rank={tals_rank}|"
            f"tals_gamma={tals_gamma}|"
            f"tals_weight_norm={tals_weight_norm}|"
            f"tals_svd_center={tals_svd_center}|"
            f"tals_subspace_source={tals_subspace_source}|"
            f"tals_fallback_to_base={tals_fallback_to_base}|"
            f"tals_use_layer_weight={tals_use_layer_weight}|"
            f"tals_layer_weight_score={tals_layer_weight_score}|"
            f"tals_layer_weight_norm={tals_layer_weight_norm}|"
            f"tals_layer_weight_clip_min={tals_layer_weight_clip_min}|"
            f"tals_layer_weight_clip_max={tals_layer_weight_clip_max}"
        )

        append_csv_row(
            registry_csv,
            [
                experiment_id, "VLM_pairwise" if task_type == "TASKS_blip_base" else "GLUE_pairwise", method_name, model_name, pair_name,
                "|".join(task_targets), f"{linear_weights}|{drc_cfg_str}",
                rank, "|".join(map(str, task_cfg.get("lora_alpha", []))),
                "", "", "", "", "", "", "failed", error_msg
            ]
        )

        raise e


if __name__ == "__main__":
    main()