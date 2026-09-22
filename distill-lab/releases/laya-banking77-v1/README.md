---
license: apache-2.0
language:
- en
base_model: convaiinnovations/laya
base_model_relation: finetune
datasets:
- PolyAI/banking77
pipeline_tag: text-classification
tags:
- laya
- banking77
- intent-classification
- customer-support
- lora
- calibrated-decisions
metrics:
- accuracy
- f1
model-index:
- name: Laya BANKING77 v1
  results:
  - task:
      type: text-classification
      name: Intent classification
    dataset:
      type: PolyAI/banking77
      name: BANKING77
      split: test
    metrics:
    - type: accuracy
      name: Accuracy
      value: 0.8555194805194806
    - type: f1
      name: Macro F1
      value: 0.855314223956437
---

# Laya BANKING77 v1

English banking customer-support intent classifier fine-tuned from Laya (ModernBERT-large).
**85.55% accuracy on the 3,080-example official BANKING77 test set, across all 77 intents.**
This repository contains encoder LoRA weights **and the complete fine-tuned decision head**, a standalone loader,
tokenizer, labels, temperature calibration, evaluation results and provenance.
The loader downloads the exact upstream base checkpoint on first use.

这是基于 Laya 英文底座微调的银行客服意图分类模型，用于将客户消息分流到 BANKING77 的 77 类意图。
它一次前向输出各类别概率，不生成客服回复。训练采用人工标注的 BANKING77 数据，**未使用 Jev 回答进行蒸馏**。

## 运行

需要 Python 3.10+。下载本仓库全部文件后，在该目录运行：

```bash
python -m pip install -r requirements.txt
python predict.py --device cpu --text "I lost my card yesterday. How do I freeze it?"
```

Apple Silicon 可改为 `--device mps`，NVIDIA GPU 可改为 `--device cuda`。
发布包的一致性检查使用 CPU FP32；设备及算子差异可能带来小的数值变化。
首次运行会从 `convaiinnovations/laya` 的固定提交下载英文底座（约 843 MB）；
本仓库的微调权重约 123 MB，加载时必须同时使用底座与微调权重。
如已有底座目录，可传入 `--base /path/to/base`；缓存完毕后可使用 `--local-files-only` 离线运行。

通过 Hugging Face 下载：

```python
from huggingface_hub import snapshot_download

folder = snapshot_download("Cahol/laya-banking77-v1", local_dir="laya-banking77-v1")
print(folder)
```

在下载目录中使用 Python API：

```python
from predict import Banking77Classifier

classifier = Banking77Classifier(device="cpu")
results = classifier.predict([
    "I lost my card yesterday. How do I freeze it?",
    "I was charged twice for the same card payment.",
])
for result in results:
    print(result["label"], result["confidence"], result["top_k"])
```

`probabilities` 包含全部 77 类的校准后概率；`confidence` 是最大类别概率。
本仓库是自定义 **LoRA + 决策头** 检查点格式，不支持直接使用
`AutoModelForSequenceClassification.from_pretrained()`、通用 `pipeline()` 或单独的 PEFT adapter 加载。
请使用随附 `predict.py`，以保持与评测一致的候选编码、参数加载和校准流程。

## 测试结果与数据划分

| 模型 | Accuracy | Macro F1 | Top-3 accuracy |
| --- | ---: | ---: | ---: |
| 原 Laya 英文底座，使用相同的完整 77 类输入协议 | 45.91% | 42.90% | 69.42% |
| 本微调模型 | **85.55%** | **85.53%** | **96.43%** |

温度校准后测试 NLL 为 **0.49064**，15 等宽分箱 ECE 为 **0.01442**。
温度 `1.1343233648166147` 仅在独立校准集上拟合；不改变 top-1 预测。
完整精度数字见 `evaluation.json`，以上为原训练运行的历史评测结果。

| 划分 | 条数 | 用途 |
| --- | ---: | --- |
| Train | 7,964 | 更新 LoRA 与决策头 |
| Development | 998 | 按 NLL 选择检查点 |
| Calibration | 997 | 选模后拟合温度 |
| Official test | 3,080 | 最终效果评测，不参与本轮梯度更新、选模或温度拟合 |

上述前三组来自官方训练集。排除与官方测试集重复/近重复分组关联的 40 条训练样本、
标签冲突的 4 条训练样本，分组后划分，分组跨集合重叠为零。
官方测试集保持原样。这是按本项目重复检测规则进行的隔离，不代表已经证明上游底座的预训练数据不存在重叠。
数据源、固定提交、文件校验值及具体协议见 `data_manifest.json`。

## 微调方式

- 底座：`convaiinnovations/laya` 的英文根目录检查点，提交 `1c5edc17a7acd8701df6fc341c0d179f1c62c982`。
- 编码器：ModernBERT-large；LoRA rank 16、alpha 32，作用于 `Wqkv` 和 `Wo`。
- 同时更新决策头 transformer、类型嵌入和候选评分器，共保存 30,635,009 个训练参数。
- 监督交叉熵，AdamW；LoRA 学习率 `1e-4`，决策头学习率 `5e-5`。
- RTX 5090、BF16、batch 8、梯度累积 4、3 epochs，随机种子 42；按开发集 NLL 选择第 3 轮。
- 训练时随机排列候选；评测与本加载器保持固定类别顺序，完整保留全部候选文字。
- 本次微调没有更新上游 act/escalate 头，加载器也不使用它做自动升级决策。

## 适用范围与限制

- 当前模型仅针对 **英文银行客服意图分类**训练与评测。中文、多领域客服、computer use 和任意动态标签集合尚未在本版本中验证。
- 总输入预算为 **8,192 tokens**，包含问题、77 个候选、客户正文和特殊标记。
  本轮训练实际输入长度为 **575–667 tokens**；8K 是可配置上限，不能据此宣称已有 8K 长文效果。
- 超出预算时加载器明确报错，不静默截断候选或客户正文。长输入的内存开销需要按设备评估。
- 这是 77 类封闭集合分类，无法可靠识别集合以外的意图；最大概率不是正确性保证。
  生产使用应先在目标业务的独立数据上检查分类效果、校准与转人工阈值。
- BANKING77 的效果不能推导为全面优于 Jev 或通用 LLM；此仓库未做这样的结论。

## 文件与复现

`release_config.json` 固定底座提交、校验值、77 类顺序、输入指令和 LoRA 配置。
`trainable.safetensors` 是原检查点的逐字节副本，没有合并或量化转换。
`packaging_verification.json` 记录发布加载器与原实现的 CPU 输入/预测一致性检查，
该检查不替代完整测试集评测。`SHA256SUMS` 可用于核验下载文件。
`training_metadata.json` 与 `data_manifest.json` 提供本轮训练与数据来源记录。

## 许可与致谢

代码和模型微调权重采用 Apache 2.0，见 `LICENSE` 和 `NOTICE`。
感谢 [Laya / Nandha Kishor M / Convai Innovations](https://github.com/NandhaKishorM/laya)
提供模型与架构。`laya_common.py` 来自上游固定提交，未经修改。
[BANKING77 / PolyAI](https://huggingface.co/datasets/PolyAI/banking77) 数据采用 CC BY 4.0，
作者为 Casanueva 等（2020）；详细归属和数据处理说明见 `DATA_LICENSE.md`。
