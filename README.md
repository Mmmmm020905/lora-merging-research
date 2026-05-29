# LoRA Merging Research

## 1. 项目简介

本仓库用于多 LoRA 模型融合方法的复现、扩展与对比实验，覆盖两类主要实验场景：

1. **NLP 场景**

   * 基座模型：`FLAN-T5-base`
   * 数据集：`GLUE`
   * 目标：验证多个 GLUE 任务 LoRA 的两两融合效果，并比较不同融合方法对任务能力保持的影响。

2. **VLM 场景**

   * 基座模型：`BLIP-image-captioning-base`
   * 数据集：`SENTICAP`
   * 目标：验证 positive / negative 风格化图像描述 LoRA 的融合效果，并观察多模态生成任务中的风格控制能力和生成质量变化。

本仓库主要包含以下工作：

* 复现 `IterIS` 在 GLUE/T5 和 SENTICAP/BLIP 上的 LoRA 融合实验；
* 实现并扩展多种 LoRA 融合 baseline，例如 `Linear`、`RegMean`、`DARE`、`KnOTS`、`KnOTS-TIES`、`KnOTS+Linear` 等；
* 实现融合后补偿方法，例如 `IterIS_TALS_DRC`、`Linear_TALS_DRC`、`RegMean_TALS_DRC`、`DARE_TALS_DRC`、`KnOTS_Linear_TALS_DRC` 等；
* 支持 GLUE 两两任务组合的批量运行、结果保存和汇总分析；
* 支持 BLIP/SENTICAP 风格化 caption 任务的融合与评估。

---

## 2. 仓库结构

当前仓库的核心结构如下：

```text
lora-merging-research/
├── README.md
├── requirements.txt
│
├── eval_glue_t5.py
├── eval_model.py
├── eval_senti.py
├── eval_single_lora_glue_t5.py
├── eval_merged_glue_t5_batch.py
│
├── get_midfeatures.py
├── run_all_glue_pairs.py
├── download_model_assets.py
├── download_text_resources.py
│
├── IterIS.py
├── Linear.py
├── RegMean.py
├── DARE.py
├── KnOTS.py
├── KnOTS_TIES.py
├── KnOTS_Linear.py
│
├── IterIS_TALS_DRC.py
├── Linear_TALS_DRC.py
├── RegMean_TALS_DRC.py
├── DARE_TALS_DRC.py
├── KnOTS_TIES_TALS_DRC.py
├── KnOTS_Linear_TALS_DRC.py
├── Linear_DRC_PCA.py
│
├── config/
│   ├── GLUE-t5-lora-train-config/
│   │   ├── MNLI-lora-train.yaml
│   │   ├── RTE-lora-train.yaml
│   │   ├── COLA-lora-train.yaml
│   │   ├── SST2-lora-train.yaml
│   │   ├── QQP-lora-train.yaml
│   │   ├── QNLI-lora-train.yaml
│   │   ├── MRPC-lora-train.yaml
│   │   └── WNLI-lora-train.yaml
│   └── methods-config/
│       └── iteris-config.yaml
│
├── train/
│   ├── train-GLUE-t5.py
│   ├── train_GLUE_t5_OSRM.py
│   ├── train_blip_positive.py
│   └── train_blip_negative.py
│
├── loras/
│   ├── train_GLUE_t5.py
│   ├── train_BLIP.py
│   ├── train_src/
│   ├── GLUE-lora-t5/
│   └── SENTICAP-lora-blip/
│
├── best_LoRA/
│   ├── T5-MNLI-LoRA/
│   ├── T5-RTE-LoRA/
│   ├── T5-COLA-LoRA/
│   ├── T5-SST2-LoRA/
│   ├── T5-QQP-LoRA/
│   ├── T5-QNLI-LoRA/
│   ├── T5-MRPC-LoRA/
│   └── T5-WNLI-LoRA/
│
├── OSRM_LoRA/
│   ├── T5-MNLI-LoRA/
│   ├── T5-RTE-LoRA/
│   ├── T5-COLA-LoRA/
│   ├── T5-SST2-LoRA/
│   ├── T5-QQP-LoRA/
│   ├── T5-QNLI-LoRA/
│   ├── T5-MRPC-LoRA/
│   └── T5-WNLI-LoRA/
│
├── data/
│   └── SENTICAP/
│       ├── positive/
│       ├── negative/
│       └── val2014/
│
├── merged_model/
├── results/
├── batch_runs/
└── outputs/
```

其中需要重点关注的目录如下：

| 路径                                         | 作用                                        |
| ------------------------------------------ | ----------------------------------------- |
| `config/GLUE-t5-lora-train-config/`        | GLUE 单任务 LoRA 训练配置                        |
| `config/methods-config/iteris-config.yaml` | 多 LoRA 融合、TALS-DRC、VLM 实验的主要配置文件          |
| `best_LoRA/`                               | 普通 GLUE LoRA 存放目录                         |
| `OSRM_LoRA/`                               | OSRM 训练得到的 GLUE LoRA 存放目录                 |
| `loras/SENTICAP-lora-blip/`                | BLIP/SENTICAP positive、negative LoRA 存放目录 |
| `merged_model/`                            | 保存融合后的 dense model 或 adapter model        |
| `results/`                                 | 默认结果输出目录                                  |
| `batch_runs/`                              | 批量 GLUE pair 实验的日志、结果和汇总目录                |
| `outputs/`                                 | 单 LoRA 评估或辅助评估输出目录                        |

---

## 3. 环境安装

建议先创建独立环境，然后安装依赖：

```bash
pip install -r requirements.txt
```

如果服务器需要手动指定缓存目录，可根据实际环境设置：

```bash
export HF_HOME=/data2/centrai/mizijie_intern/IterIS-merging-main/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export NLTK_DATA=/data2/centrai/mizijie_intern/IterIS-merging-main/nltk_data
```

如果模型和数据均已本地准备好，可以使用离线模式运行：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

---

## 4. 资源准备

### 4.1 下载模型相关资源

```bash
python download_model_assets.py
```

该脚本主要用于下载或准备：

* `FLAN-T5-base` / `T5-base` 相关模型资源；
* GLUE LoRA 相关模型资产；
* BLIP 相关模型资源；
* 其他脚本中依赖的 Hugging Face 模型文件。

如果已经手动下载模型，需要在配置文件中将 `model_name` 改为本地路径，例如：

```yaml
model_name: pretrained/flan-t5-base
```

或：

```yaml
model_name: pretrained/blip-image-captioning-base
```

### 4.2 下载文本资源

```bash
python download_text_resources.py
```

该脚本主要用于准备：

* GLUE 数据；
* NLTK 资源；
* SENTICAP 风格评估依赖的文本资源；
* 情感词典或 caption 评估辅助文件。

### 4.3 准备 SENTICAP 图像数据

BLIP/SENTICAP 实验需要 COCO `val2014` 图像：

```bash
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip -d ./data/SENTICAP/val2014
```

最终目录建议保持为：

```text
data/SENTICAP/
├── positive/
├── negative/
└── val2014/
```
### 模型权重与融合结果交付说明

由于本仓库涉及多种 LoRA 融合方法和大量任务组合，合并后产出的 dense model 体积较大。例如在 GLUE/T5 场景中，即使是最简单的 Linear 方法，每个 two-task pair 保存后的融合模型也可能接近 1GB；若对 8 个 GLUE 任务进行两两组合，共包含 28 个 pair，单个方法就会产生数十 GB 的模型文件。若进一步保存 RegMean、DARE、KnOTS、IterIS、TALS-DRC 等多个方法的全部融合模型，整体存储体积会显著增加，不利于仓库管理、传输和复现。

因此，本仓库采用如下交付策略：

1. **单任务 LoRA 微调权重会单独保存和提供**
   单 LoRA 权重体积相对较小，是复现后续融合实验的基础。当前主要包括：

   * GLUE 普通 LoRA；
   * GLUE OSRM LoRA；
   * SENTICAP/BLIP positive 与 negative LoRA。

2. **融合后的 dense model 默认不全部提供**
   对于 Linear、RegMean、DARE、KnOTS、IterIS 以及 TALS-DRC 等融合方法，默认不在仓库中直接保存所有 pair 的合并后模型权重。若需要某个具体 pair 的融合模型，可以根据本仓库提供的脚本、配置文件和命令重新生成。

3. **提供完整可复现的合并脚本和配置文件**
   本仓库保留所有用于复现融合实验的核心脚本和配置文件，包括：

   * `Linear.py`
   * `RegMean.py`
   * `DARE.py`
   * `KnOTS.py`
   * `KnOTS_TIES.py`
   * `KnOTS_Linear.py`
   * `IterIS.py`
   * `Linear_TALS_DRC.py`
   * `RegMean_TALS_DRC.py`
   * `DARE_TALS_DRC.py`
   * `KnOTS_Linear_TALS_DRC.py`
   * `IterIS_TALS_DRC.py`
   * `run_all_glue_pairs.py`
   * `config/methods-config/iteris-config.yaml`
   * `config/GLUE-t5-lora-train-config/*.yaml`

4. **结果文件会保留用于复现实验统计**
   实验结果主要通过 CSV 文件保存，包括：

   * `pair_merge_results.csv`
   * `experiment_registry.csv`
   * `vlm_caption_results.csv`
   * `drc_alpha_search_results.csv`
   * `summary/` 下的汇总结果文件

   这些文件用于记录每次实验的指标、运行时间、显存占用、alpha 搜索结果和 pair-level 平均结果，便于在不直接保存全部融合模型的情况下复现实验结论。

5. **TALS-DRC 类方法的说明**
   TALS-DRC 类方法通常依赖已有 coarse merged model，并在推理阶段构造或加载任务补偿方向。因此，对于这类方法，仓库重点提供：

   * 对应 coarse model 的生成脚本；
   * TALS-DRC 运行脚本；
   * 关键配置文件；
   * alpha 搜索结果；
   * 必要时提供 direction cache 或重新生成 cache 的命令。

   若未直接提供某个 TALS-DRC 的最终模型权重，可通过对应 coarse model、单任务 LoRA 和配置文件重新生成。

建议的模型资源整理方式如下：

```text
lora_merging_artifacts/
├── 01_single_lora_finetuning/
│   ├── 01_GLUE_normal_lora/
│   ├── 02_GLUE_OSRM_lora/
│   └── 03_SENTICAP_BLIP_lora/
│
└── 02_reproduce_merging/
    ├── 01_scripts/
    ├── 02_configs/
    ├── 03_results/
    └── 04_logs_or_cache_optional/
```

其中：

| 目录                           | 内容                                                 |
| ---------------------------- | -------------------------------------------------- |
| `01_GLUE_normal_lora/`       | 普通 GLUE 单任务 LoRA，例如 `T5-MNLI-LoRA`、`T5-RTE-LoRA` 等 |
| `02_GLUE_OSRM_lora/`         | OSRM 训练得到的 GLUE 单任务 LoRA                           |
| `03_SENTICAP_BLIP_lora/`     | BLIP/SENTICAP 的 positive 与 negative LoRA           |
| `01_scripts/`                | 所有融合方法和 TALS-DRC 的复现脚本                             |
| `02_configs/`                | 训练和融合所需配置文件                                        |
| `03_results/`                | 实验结果 CSV、summary 汇总文件                              |
| `04_logs_or_cache_optional/` | 可选日志文件、direction cache 或中间结果                       |

如需复现某个融合模型，可按照 README 中对应方法的运行命令重新生成。例如：

```bash
python Linear.py \
  --task_type GLUE_t5 \
  --config config/methods-config/iteris-config.yaml
```

或批量运行 GLUE 两两融合实验：

```bash
python run_all_glue_pairs.py \
  --gpus 0,1,2,3,4 \
  --method linear \
  --continue_on_error
```

---

## 5. 单 LoRA 训练与评估：FLAN-T5 / GLUE

本节用于训练和评估单个 GLUE 任务 LoRA。GLUE 场景使用 T5 text-to-text 格式，将不同任务统一转换为文本生成式分类任务。

### 5.1 需要关注的配置文件

GLUE 单任务 LoRA 训练主要关注：

```text
config/GLUE-t5-lora-train-config/
```

其中每个任务对应一个配置文件，例如：

```text
MNLI-lora-train.yaml
RTE-lora-train.yaml
COLA-lora-train.yaml
SST2-lora-train.yaml
QQP-lora-train.yaml
QNLI-lora-train.yaml
MRPC-lora-train.yaml
WNLI-lora-train.yaml
```

训练前需要重点检查以下字段：

| 配置项                              | 说明                                                     |
| -------------------------------- | ------------------------------------------------------ |
| `model_name`                     | 基座模型路径，例如 `pretrained/flan-t5-base` 或 Hugging Face 模型名 |
| `task_name`                      | 当前训练任务，例如 `mnli`、`rte`、`cola`                          |
| `output_dir`                     | LoRA 输出目录                                              |
| `rank` / `r`                     | LoRA rank                                              |
| `lora_alpha`                     | LoRA 缩放系数                                              |
| `learning_rate`                  | 学习率                                                    |
| `max_steps` / `num_train_epochs` | 训练步数或 epoch                                            |
| `per_device_train_batch_size`    | 训练 batch size                                          |
| `max_length`                     | 输入最大长度                                                 |
| `target_modules`                 | LoRA 挂载模块，例如 T5 的 `q` / `v`                            |

### 5.2 训练普通 GLUE LoRA

以 RTE 为例：

```bash
cd /data2/centrai/mizijie_intern/IterIS-merging-main

CUDA_VISIBLE_DEVICES=0 python train/train-GLUE-t5.py \
  --config config/GLUE-t5-lora-train-config/RTE-lora-train.yaml \
  2>&1 | tee logs_train_rte_lora.log
```

如果需要训练其他任务，只需要替换配置文件：

```bash
CUDA_VISIBLE_DEVICES=0 python train/train-GLUE-t5.py \
  --config config/GLUE-t5-lora-train-config/MNLI-lora-train.yaml
```

训练完成后，普通 LoRA 通常保存到：

```text
best_LoRA/T5-RTE-LoRA/
```

或配置文件中指定的 `output_dir`。

### 5.3 训练 OSRM GLUE LoRA

OSRM 是训练阶段引入正交子空间约束的 LoRA 训练方式。以 QQP 为例：

```bash
cd /data2/centrai/mizijie_intern/IterIS-merging-main

CUDA_VISIBLE_DEVICES=0 python train/train_GLUE_t5_OSRM.py \
  --config config/GLUE-t5-lora-train-config/QQP-lora-train.yaml \
  2>&1 | tee logs_osrm_lora_train_qqp.log
```

训练完成后，OSRM LoRA 建议保存到：

```text
OSRM_LoRA/T5-QQP-LoRA/
```

运行融合实验前，需要确认每个任务目录中存在：

```text
adapter_model.safetensors
adapter_config.json
```

可以用下面命令检查 OSRM LoRA 是否齐全：

```bash
python - <<'PY'
from pathlib import Path

tasks = ["mnli", "rte", "cola", "sst2", "qqp", "qnli", "mrpc", "wnli"]
root = Path("OSRM_LoRA")

missing = []
for t in tasks:
    p = root / f"T5-{t.upper()}-LoRA" / "adapter_model.safetensors"
    if not p.exists():
        missing.append(str(p))

if missing:
    print("[Missing OSRM LoRA]")
    for x in missing:
        print(x)
else:
    print("[OK] all OSRM LoRA adapter_model.safetensors exist.")
PY
```

### 5.4 评估单个 GLUE LoRA

训练完成后，可以使用 `eval_single_lora_glue_t5.py` 评估单任务 LoRA。

普通 LoRA 示例：

```bash
python eval_single_lora_glue_t5.py \
  --task_name rte \
  --lora_path best_LoRA/T5-RTE-LoRA
```

OSRM LoRA 示例：

```bash
python eval_single_lora_glue_t5.py \
  --task_name qqp \
  --lora_path OSRM_LoRA/T5-QQP-LoRA
```

如果需要批量比较普通 LoRA 与 OSRM LoRA，可以使用自定义 shell 脚本，例如：

```bash
bash eval_gaussian_vs_osrm_loras.sh
```

评估结果通常输出到：

```text
outputs/
```

常见输出包括：

```text
outputs/eval_xxx/
├── logs/
├── predictions/
└── summary.csv
```

具体输出路径以评估脚本中的 `output_dir` 或运行日志为准。

---

## 6. 单 LoRA 训练与评估：BLIP / SENTICAP

本节用于训练和评估 BLIP/SENTICAP 中的单风格 LoRA。SENTICAP 当前主要包含两个风格任务：

```text
positive
negative
```

### 6.1 需要关注的文件

BLIP 单 LoRA 训练主要关注：

```text
train/train_blip_positive.py
train/train_blip_negative.py
```

训练前需要重点检查脚本中的以下内容：

| 项目                              | 说明                                                     |
| ------------------------------- | ------------------------------------------------------ |
| `model_name`                    | BLIP 基座模型路径，例如 `pretrained/blip-image-captioning-base` |
| `data/SENTICAP/positive`        | positive 风格数据路径                                        |
| `data/SENTICAP/negative`        | negative 风格数据路径                                        |
| `data/SENTICAP/val2014`         | COCO val2014 图像路径                                      |
| `output_dir`                    | LoRA 保存目录                                              |
| `rank` / `r`                    | LoRA rank                                              |
| `lora_alpha`                    | LoRA 缩放系数                                              |
| `target_modules`                | BLIP LoRA 挂载模块                                         |
| `max_length` / `max_new_tokens` | caption 输入输出长度                                         |
| `per_device_train_batch_size`   | 训练 batch size                                          |

### 6.2 训练 positive BLIP LoRA

```bash
cd /data2/centrai/mizijie_intern/IterIS-merging-main

CUDA_VISIBLE_DEVICES=0 python train/train_blip_positive.py \
  2>&1 | tee logs_train_blip_positive.log
```

训练完成后，positive LoRA 建议保存到：

```text
loras/SENTICAP-lora-blip/positive/
```

### 6.3 训练 negative BLIP LoRA

```bash
cd /data2/centrai/mizijie_intern/IterIS-merging-main

CUDA_VISIBLE_DEVICES=0 python train/train_blip_negative.py \
  2>&1 | tee logs_train_blip_negative.log
```

训练完成后，negative LoRA 建议保存到：

```text
loras/SENTICAP-lora-blip/negative/
```

### 6.4 检查 BLIP LoRA 是否存在

```bash
ls -lh loras/SENTICAP-lora-blip/positive/adapter_model.safetensors
ls -lh loras/SENTICAP-lora-blip/negative/adapter_model.safetensors
```

### 6.5 评估单个 BLIP LoRA

当前 BLIP 的评估逻辑主要由 `eval_model.py` 中的 BLIP 分支完成，内部会计算：

* style accuracy；
* CIDEr；
* BLEU-1/2/3/4；
* ROUGE-L；
* diversity；
* vocab size。

如果需要评估单个 BLIP LoRA，可以使用如下方式临时调用：

```bash
python - <<'PY'
import torch
from transformers import BlipForConditionalGeneration, AutoProcessor
from peft import PeftModel
from eval_model import eval_iteris_model

model_name = "pretrained/blip-image-captioning-base"
lora_path = "loras/SENTICAP-lora-blip/positive"
task_name = "positive"

processor = AutoProcessor.from_pretrained(model_name)
base_model = BlipForConditionalGeneration.from_pretrained(model_name)
model = PeftModel.from_pretrained(base_model, lora_path).merge_and_unload().to("cuda")

result = eval_iteris_model(
    model=model,
    tokenizer=processor,
    model_name=model_name,
    task_name=task_name,
    max_length=168,
    per_device_eval_batch_size=24,
)

print(result)
PY
```

评估 negative LoRA 时，修改：

```python
lora_path = "loras/SENTICAP-lora-blip/negative"
task_name = "negative"
```

建议后续将以上逻辑整理为独立脚本：

```text
eval_single_lora_blip.py
```

以便和 `eval_single_lora_glue_t5.py` 保持一致。

---

## 7. 多 LoRA 融合实验：GLUE / FLAN-T5

GLUE 场景主要进行两类实验：

1. 单个任务对 pair 的融合实验；
2. 所有 GLUE 任务两两组合的批量融合实验。

### 7.1 需要关注的配置文件

多 LoRA 融合统一使用：

```text
config/methods-config/iteris-config.yaml
```

GLUE 相关配置一般位于：

```yaml
GLUE_t5:
```

需要重点关注以下字段：

| 字段                           | 说明                                        |
| ---------------------------- | ----------------------------------------- |
| `model_name`                 | T5 基座模型路径                                 |
| `task_targets`               | 当前要融合的任务对，例如 `[mnli, rte]`                |
| `lora_alpha`                 | 每个 LoRA 的 alpha                           |
| `rank`                       | LoRA rank                                 |
| `max_length`                 | 输入最大长度                                    |
| `per_device_eval_batch_size` | 评估 batch size                             |
| `save`                       | 是否保存融合模型                                  |
| `lora_source`                | LoRA 来源，普通 LoRA 用 `default`，OSRM 用 `osrm` |
| `lora_root`                  | LoRA 根目录，例如 `best_LoRA` 或 `OSRM_LoRA`     |
| `linear_weights`             | Linear 融合权重                               |
| `method_configs`             | 批量运行时各方法的配置覆盖项                            |

普通 LoRA 融合常用：

```yaml
lora_source: default
lora_root: best_LoRA
```

OSRM LoRA 融合常用：

```yaml
lora_source: osrm
lora_root: OSRM_LoRA
```

### 7.2 GLUE 单个 pair 融合：Linear 示例

先在 `config/methods-config/iteris-config.yaml` 中设置：

```yaml
GLUE_t5:
  task_targets:
  - mnli
  - rte
  model_name: pretrained/flan-t5-base
  lora_source: default
  lora_root: best_LoRA
  linear_method_name: Linear
  linear_weights:
  - 0.5
  - 0.5
  save: 1
```

运行：

```bash
cd /data2/centrai/mizijie_intern/IterIS-merging-main

RESULTS_DIR=results_glue_linear_mnli_rte \
LOG_FILE="" \
CUDA_VISIBLE_DEVICES=0 python Linear.py \
  --task_type GLUE_t5 \
  --config config/methods-config/iteris-config.yaml
```

输出结果：

```text
results_glue_linear_mnli_rte/
├── pair_merge_results.csv
└── experiment_registry.csv
```

如果 `save: 1`，融合模型会保存到：

```text
merged_model/Linear_mnli_rte/
```

或配置文件中指定的 `linear_merged_model_dir`。

### 7.3 GLUE 单个 pair 融合：OSRM+Linear 示例

如果要使用 OSRM LoRA，需要确认：

```text
OSRM_LoRA/T5-MNLI-LoRA/adapter_model.safetensors
OSRM_LoRA/T5-RTE-LoRA/adapter_model.safetensors
```

然后在配置中设置：

```yaml
GLUE_t5:
  task_targets:
  - mnli
  - rte
  lora_source: osrm
  lora_root: OSRM_LoRA
  linear_method_name: OSRM_Linear
  linear_weights:
  - 0.5
  - 0.5
  save: 1
```

运行：

```bash
RESULTS_DIR=results_glue_osrm_linear_mnli_rte \
LOG_FILE="" \
CUDA_VISIBLE_DEVICES=0 python Linear.py \
  --task_type GLUE_t5 \
  --config config/methods-config/iteris-config.yaml
```

输出结果：

```text
results_glue_osrm_linear_mnli_rte/
├── pair_merge_results.csv
└── experiment_registry.csv
```

融合模型通常保存到：

```text
merged_model/OSRM_Linear_mnli_rte/
```

运行后建议检查日志中是否出现：

```text
[Linear] lora_source = osrm
[Linear] lora_root = OSRM_LoRA
```

### 7.4 GLUE 单个 pair 融合：其他方法

不同融合方法对应脚本如下：

| 方法                    | 运行脚本                       |
| --------------------- | -------------------------- |
| IterIS                | `IterIS.py`                |
| Linear                | `Linear.py`                |
| RegMean               | `RegMean.py`               |
| DARE                  | `DARE.py`                  |
| KnOTS                 | `KnOTS.py`                 |
| KnOTS-TIES            | `KnOTS_TIES.py`            |
| KnOTS+Linear          | `KnOTS_Linear.py`          |
| IterIS_TALS_DRC       | `IterIS_TALS_DRC.py`       |
| Linear_TALS_DRC       | `Linear_TALS_DRC.py`       |
| RegMean_TALS_DRC      | `RegMean_TALS_DRC.py`      |
| DARE_TALS_DRC         | `DARE_TALS_DRC.py`         |
| KnOTS_Linear_TALS_DRC | `KnOTS_Linear_TALS_DRC.py` |

例如运行 RegMean：

```bash
RESULTS_DIR=results_glue_regmean_mnli_rte \
LOG_FILE="" \
CUDA_VISIBLE_DEVICES=0 python RegMean.py \
  --task_type GLUE_t5 \
  --config config/methods-config/iteris-config.yaml
```

例如运行 Linear_TALS_DRC：

```bash
RESULTS_DIR=results_glue_linear_tals_mnli_rte \
LOG_FILE="" \
CUDA_VISIBLE_DEVICES=0 python Linear_TALS_DRC.py \
  --task_type GLUE_t5 \
  --config config/methods-config/iteris-config.yaml
```

TALS-DRC 类方法通常需要先保存 coarse merged model，例如：

```text
merged_model/Linear_mnli_rte/
merged_model/OSRM_Linear_mnli_rte/
merged_model/RegMean_mnli_rte/
```

否则会因为找不到已有 coarse model 而失败。

### 7.5 GLUE 批量两两 pair 融合

`run_all_glue_pairs.py` 用于批量运行 GLUE 任务两两组合实验。

示例：批量运行普通 Linear：

```bash
cd /data2/centrai/mizijie_intern/IterIS-merging-main

python run_all_glue_pairs.py \
  --gpus 0,1,2,3,4 \
  --method linear \
  --continue_on_error
```

示例：批量运行 OSRM+Linear：

```bash
python run_all_glue_pairs.py \
  --gpus 0,1,2,3,4 \
  --method linear \
  --continue_on_error
```

使用 OSRM+Linear 前，需要确保配置文件中已经设置：

```yaml
GLUE_t5:
  lora_source: osrm
  lora_root: OSRM_LoRA
  linear_method_name: OSRM_Linear
```

示例：批量运行 Linear_TALS_DRC：

```bash
python run_all_glue_pairs.py \
  --gpus 0,1,2,3,4 \
  --method linear_tals_drc \
  --continue_on_error
```

批量运行输出目录形如：

```text
batch_runs/glue_pairs_YYYYMMDD_HHMMSS/
├── logs/
├── summary/
├── pair_merge_results.csv
├── experiment_registry.csv
└── ...
```

其中：

| 文件/目录                          | 说明                       |
| ------------------------------ | ------------------------ |
| `logs/`                        | 每个 pair 的运行日志            |
| `summary/`                     | 批量运行后的汇总结果               |
| `pair_merge_results.csv`       | 每个 pair、每个任务的评估结果        |
| `experiment_registry.csv`      | 每次实验的总体信息、融合耗时、显存、平均指标   |
| `drc_alpha_search_results.csv` | TALS-DRC 类方法的 alpha 搜索结果 |

### 7.6 检查 GLUE 批量实验是否使用了正确 LoRA

普通 LoRA 应该显示：

```text
lora_root = best_LoRA
```

OSRM LoRA 应该显示：

```text
lora_root = OSRM_LoRA
```

可以用下面命令检查日志：

```bash
grep -R "\[Linear\] lora_source" batch_runs/glue_pairs_*/logs/*.log | tail -20
grep -R "\[Linear\] lora_root" batch_runs/glue_pairs_*/logs/*.log | tail -20
grep -R "\[Linear_TALS_DRC\] lora_root" batch_runs/glue_pairs_*/logs/*.log | tail -20
```

如果日志中出现了错误的 LoRA 根目录，需要停止当前任务并检查 `config/methods-config/iteris-config.yaml`。

---

## 8. 多 LoRA 融合实验：SENTICAP / BLIP

SENTICAP/BLIP 场景主要用于融合 positive 和 negative 两个风格 LoRA。

### 8.1 需要关注的配置文件

BLIP 融合实验同样使用：

```text
config/methods-config/iteris-config.yaml
```

需要关注：

```yaml
TASKS_blip_base:
```

常见配置如下：

```yaml
TASKS_blip_base:
  task_targets:
  - positive
  - negative

  model_name: pretrained/blip-image-captioning-base
  lora_root: loras/SENTICAP-lora-blip

  max_length: 168
  max_new_tokens: 42
  per_device_eval_batch_size: 24

  rank: 32
  lora_alpha:
  - 32
  - 32

  save: 1
```

TALS-DRC 类方法还需要关注：

```yaml
drc_inject_position: lora_input
drc_alpha_search: true
drc_alpha_candidates:
- 0.0
- 0.00005
- 0.0001
- 0.0003
- 0.0005
- 0.001
- 0.005
- 0.007
- 0.01
- 0.02

drc_target_part: text_decoder
drc_target_modules:
- query
- value
drc_target_layers:
- 0
- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
```

### 8.2 运行 BLIP/SENTICAP 单个 pair 融合：IterIS 示例

```bash
cd /data2/centrai/mizijie_intern/IterIS-merging-main

export NLTK_DATA=/data2/centrai/mizijie_intern/IterIS-merging-main/nltk_data

RESULTS_DIR=results_blip_iteris \
LOG_FILE="" \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python IterIS.py \
  --task_type TASKS_blip_base \
  --config config/methods-config/iteris-config.yaml
```

输出结果：

```text
results_blip_iteris/
├── pair_merge_results.csv
├── experiment_registry.csv
└── vlm_caption_results.csv
```

融合模型保存位置通常由配置决定，例如：

```text
merged_model/IterIS_BLIP_positive_negative/
```

或脚本默认目录。

### 8.3 运行 BLIP/SENTICAP：Linear 示例

```bash
RESULTS_DIR=results_blip_linear \
LOG_FILE="" \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python Linear.py \
  --task_type TASKS_blip_base \
  --config config/methods-config/iteris-config.yaml
```

输出结果：

```text
results_blip_linear/
├── pair_merge_results.csv
├── experiment_registry.csv
└── vlm_caption_results.csv
```

融合模型通常保存到：

```text
merged_model/Linear_BLIP_positive_negative/
```

### 8.4 运行 BLIP/SENTICAP：TALS-DRC 示例

以 IterIS_TALS_DRC 为例：

```bash
RESULTS_DIR=results_blip_iteris_tals_drc \
LOG_FILE="" \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python IterIS_TALS_DRC.py \
  --task_type TASKS_blip_base \
  --config config/methods-config/iteris-config.yaml
```

以 Linear_TALS_DRC 为例：

```bash
RESULTS_DIR=results_blip_linear_tals_drc \
LOG_FILE="" \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python Linear_TALS_DRC.py \
  --task_type TASKS_blip_base \
  --config config/methods-config/iteris-config.yaml
```

以 KnOTS_Linear_TALS_DRC 为例：

```bash
RESULTS_DIR=results_blip_knots_linear_tals_drc \
LOG_FILE="" \
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python KnOTS_Linear_TALS_DRC.py \
  --task_type TASKS_blip_base \
  --config config/methods-config/iteris-config.yaml
```

TALS-DRC 输出通常包括：

```text
results_blip_xxx_tals_drc/
├── pair_merge_results.csv
├── experiment_registry.csv
├── vlm_caption_results.csv
└── drc_alpha_search_results.csv
```

其中：

| 文件                             | 说明                                   |
| ------------------------------ | ------------------------------------ |
| `vlm_caption_results.csv`      | positive / negative 风格 caption 的详细指标 |
| `drc_alpha_search_results.csv` | 不同 alpha 候选的搜索结果                     |
| `experiment_registry.csv`      | 方法名、pair、融合耗时、显存、平均指标等               |
| `pair_merge_results.csv`       | pair-level 主指标记录                     |

### 8.5 查看 BLIP/SENTICAP 结果

```bash
cat results_blip_iteris/vlm_caption_results.csv
cat results_blip_iteris/experiment_registry.csv
```

整理成论文表格格式：

```bash
python - <<'PY'
import pandas as pd

path = "results_blip_iteris/vlm_caption_results.csv"
df = pd.read_csv(path)

exp = df["experiment_id"].iloc[-1]
cur = df[df["experiment_id"] == exp]

pos = cur[cur["evaluated_task"] == "positive"].iloc[-1]
neg = cur[cur["evaluated_task"] == "negative"].iloc[-1]

print("experiment_id:", exp)
print(f"Acc(pos,neg) = ({pos['style_acc']:.3f}, {neg['style_acc']:.3f})")
print(f"CIDEr        = {(pos['cider'] + neg['cider']) / 2:.3f}")
print(f"B-1          = {(pos['bleu_1'] + neg['bleu_1']) / 2:.3f}")
print(f"B-2          = {(pos['bleu_2'] + neg['bleu_2']) / 2:.3f}")
print(f"B-3          = {(pos['bleu_3'] + neg['bleu_3']) / 2:.3f}")
print(f"B-4          = {(pos['bleu_4'] + neg['bleu_4']) / 2:.3f}")
PY
```

---

## 9. 已支持的主要方法

| 方法                    | 脚本                         | GLUE/T5 | BLIP/SENTICAP | 说明                                    |
| --------------------- | -------------------------- | ------: | ------------: | ------------------------------------- |
| IterIS                | `IterIS.py`                |       ✅ |             ✅ | 官方核心方法复现与扩展                           |
| Linear                | `Linear.py`                |       ✅ |             ✅ | LoRA 有效更新线性平均                         |
| RegMean               | `RegMean.py`               |       ✅ |             ✅ | 特征统计辅助融合                              |
| DARE                  | `DARE.py`                  |       ✅ |             ✅ | task delta 随机丢弃与重缩放                   |
| KnOTS                 | `KnOTS.py`                 |       ✅ |             ✅ | SVD core-space 融合                     |
| KnOTS-TIES            | `KnOTS_TIES.py`            |       ✅ |             ✅ | KnOTS + TIES sign merge               |
| KnOTS+Linear          | `KnOTS_Linear.py`          |       ✅ |             ✅ | Linear 稳定底座 + KnOTS 低秩修正              |
| IterIS_TALS_DRC       | `IterIS_TALS_DRC.py`       |       ✅ |             ✅ | IterIS coarse model 上的 TALS-DRC       |
| Linear_TALS_DRC       | `Linear_TALS_DRC.py`       |       ✅ |             ✅ | Linear coarse model 上的 TALS-DRC       |
| RegMean_TALS_DRC      | `RegMean_TALS_DRC.py`      |       ✅ |             ✅ | RegMean coarse model 上的 TALS-DRC      |
| DARE_TALS_DRC         | `DARE_TALS_DRC.py`         |       ✅ |             ✅ | DARE coarse model 上的 TALS-DRC         |
| KnOTS_Linear_TALS_DRC | `KnOTS_Linear_TALS_DRC.py` |       ✅ |             ✅ | KnOTS+Linear coarse model 上的 TALS-DRC |

说明：

* TALS-DRC 类脚本通常要求先运行对应基础融合方法并保存 coarse model；
* VLM 场景下部分 TIES-style 方法可能出现 caption 生成坍塌，需要结合 `CIDEr`、`BLEU`、`vocab_size` 判断，不应只看 style accuracy；
* OSRM 主要用于 GLUE/T5 场景，需要先训练 `OSRM_LoRA`，再运行 `OSRM+Linear` 或 `OSRM+Linear_TALS_DRC`。

---

## 10. 常见结果文件说明

### 10.1 GLUE 结果

GLUE 实验常见输出：

```text
pair_merge_results.csv
experiment_registry.csv
drc_alpha_search_results.csv
```

字段含义：

| 文件                             | 说明                  |
| ------------------------------ | ------------------- |
| `pair_merge_results.csv`       | 每个 pair 中每个任务的评估结果  |
| `experiment_registry.csv`      | 每次融合实验的总体信息         |
| `drc_alpha_search_results.csv` | TALS-DRC alpha 搜索记录 |
| `summary/`                     | 批量实验后的汇总统计          |

### 10.2 BLIP/SENTICAP 结果

VLM 实验额外输出：

```text
vlm_caption_results.csv
```

主要指标包括：

| 指标                                        | 说明                  |
| ----------------------------------------- | ------------------- |
| `style_acc`                               | 生成 caption 的风格准确率   |
| `cider`                                   | caption 内容质量指标      |
| `bleu_1` / `bleu_2` / `bleu_3` / `bleu_4` | BLEU 指标             |
| `rougeL`                                  | ROUGE-L 指标          |
| `div_1` / `div_2`                         | 生成多样性               |
| `vocab_size`                              | 生成词表大小，用于辅助判断是否生成坍塌 |

---

## 11. 常见问题与排查

### 11.1 找不到 LoRA 文件

错误示例：

```text
Cannot find adapter_model.safetensors
```

检查对应路径：

```bash
ls -lh best_LoRA/T5-QQP-LoRA/adapter_model.safetensors
ls -lh OSRM_LoRA/T5-QQP-LoRA/adapter_model.safetensors
ls -lh loras/SENTICAP-lora-blip/positive/adapter_model.safetensors
```

如果使用 OSRM，尤其需要确认 8 个 GLUE 任务是否全部存在：

```bash
python - <<'PY'
from pathlib import Path

tasks = ["mnli", "rte", "cola", "sst2", "qqp", "qnli", "mrpc", "wnli"]
root = Path("OSRM_LoRA")

for t in tasks:
    p = root / f"T5-{t.upper()}-LoRA" / "adapter_model.safetensors"
    print(t, "OK" if p.exists() else f"Missing: {p}")
PY
```

### 11.2 配置中 LoRA 来源不对

普通 LoRA 应该使用：

```yaml
lora_source: default
lora_root: best_LoRA
```

OSRM LoRA 应该使用：

```yaml
lora_source: osrm
lora_root: OSRM_LoRA
```

运行后检查日志：

```bash
grep -R "lora_root" batch_runs/glue_pairs_*/logs/*.log | tail -20
```

### 11.3 YAML 中重复 key

如果同一个配置块里重复写了：

```yaml
dare_merged_model_dir:
save_best_model:
```

后面的值通常会覆盖前面的值，导致实际保存路径和预期不一致。修改配置时应避免重复 key。

### 11.4 TALS-DRC 找不到 coarse model

TALS-DRC 类方法通常需要先运行基础融合方法，并保存模型。例如：

```text
merged_model/Linear_mnli_rte/config.json
merged_model/OSRM_Linear_mnli_rte/config.json
merged_model/RegMean_BLIP_positive_negative/config.json
```

如果缺失，需要先运行对应基础融合脚本。

### 11.5 BLIP 评估缺少 NLTK 或情感资源

运行前建议设置：

```bash
export NLTK_DATA=/data2/centrai/mizijie_intern/IterIS-merging-main/nltk_data
```

并确保已经运行：

```bash
python download_text_resources.py
```

---

## 12. 推荐实验流程

### 12.1 GLUE/T5 推荐流程

```text
1. 准备或训练单任务 GLUE LoRA
2. 评估单任务 LoRA
3. 运行 Linear / RegMean / DARE / KnOTS / IterIS 等基础融合方法
4. 保存 coarse merged model
5. 运行对应 TALS-DRC 方法
6. 使用 run_all_glue_pairs.py 批量运行所有 pair
7. 汇总 pair_merge_results.csv 和 summary 结果
```

### 12.2 BLIP/SENTICAP 推荐流程

```text
1. 准备 positive / negative BLIP LoRA
2. 评估单风格 LoRA
3. 运行 Linear / RegMean / IterIS / KnOTS+Linear 等基础融合方法
4. 查看 vlm_caption_results.csv
5. 运行对应 TALS-DRC 方法
6. 对比 style_acc、CIDEr、BLEU、vocab_size
7. 分析是否存在生成坍塌或风格偏置
```

---

## 13. 致谢

本仓库整理与实现过程中参考了多个公开项目和相关方法实现，包括但不限于：

* IterIS
* KnOTS
* DARE
* TIES-Merging
* RegMean
* OSRM

感谢原始论文作者与开源社区提供的方法思路、代码基础和实验设置。
