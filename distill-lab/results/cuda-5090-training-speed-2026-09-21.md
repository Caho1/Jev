# RTX 5090：Laya 长上下文训练测速 · 2026-09-21

8,192 token 的 LoRA 加决策头训练和全量分类微调均已通过实际前向、反向、有限非零梯度与参数更新检查。8k、batch 4 的 BF16 LoRA 配置使用编码器和决策头梯度检查点，平均 3.114 秒/步、1.285 条/秒，PyTorch 分配显存峰值 10.21 GiB。可以把该配置作为后续长上下文训练的起点，但本次没有训练真实业务模型。

## 环境与模型

- NVIDIA GeForce RTX 5090，显存总量 32,607 MiB；驱动 580.105.08。
- Python 3.12.3，PyTorch 2.8.0+cu128，CUDA 12.8，Transformers 4.57.6，PEFT 0.21.0，safetensors 0.8.0；[完整环境记录](cuda-environment.json)。
- 远程工作目录：`/root/autodl-tmp/laya-context-bench-20260921`；隔离虚拟环境复用已有 CUDA PyTorch。
- 检查点：`convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`。模型在服务器上从 Hugging Face 镜像直接下载，权重 SHA256 与本机一致：[校验记录](cuda-weights-integrity.json)。
- 所有配置均使用相同固定上游模型代码、SDPA 和 AdamW。关闭 `reference_compile`；FP32 对照关闭 TF32。模型参数保留 FP32，BF16 通过 autocast 启用。
- LoRA 应用于编码器的 `Wqkv` 和 `Wo`，rank 16、alpha 32，并训练完整决策头，共 30,635,009 个可训练参数。全量分类模式训练 421,029,889 个参数；动作头保留前向但不参与损失及更新。
- GPU 分配器进程预算 28 GiB，未取消显存保护。测量包括前向、反向、优化器更新及同步，排除分词、数据读取、验证、保存和模型初始化。

## BF16 实测

下表除另有说明均预热 3 步、计时 10 步。重算为梯度检查点，反向时重新计算部分激活以减少显存。

| 模式 | Token 数 | Batch | 重算 | 秒/步 | 条/秒 | 分配显存峰值（GiB） | 缓存显存峰值（GiB） |
|---|---:|---:|---|---:|---:|---:|---:|
| [LoRA + 决策头](cuda-5090/lora-b1-l2048-bf16.json) | 2048 | 1 | 无 | 0.142 | 7.050 | 5.45 | 5.79 |
| [LoRA + 决策头](cuda-5090/lora-b1-l4096-bf16.json) | 4096 | 1 | 无 | 0.329 | 3.043 | 8.80 | 9.37 |
| [LoRA + 决策头](cuda-5090/lora-b1-l8192-bf16.json) | 8192 | 1 | 无 | 0.922 | 1.085 | 16.74 | 17.24 |
| [LoRA + 决策头](cuda-5090/lora-b1-l8192-bf16-gc-head.json) | 8192 | 1 | 编码器 + 决策头 | 1.100 | 0.909 | 4.02 | 4.24 |
| [LoRA + 决策头](cuda-5090/lora-b2-l8192-bf16-gc-head.json) | 8192 | 2 | 编码器 + 决策头 | 1.691 | 1.183 | 6.08 | 6.35 |
| [LoRA + 决策头](cuda-5090/lora-b4-l8192-bf16-gc-head.json) | 8192 | 4 | 编码器 + 决策头 | 3.114 | 1.285 | 10.21 | 10.62 |
| [全量分类微调](cuda-5090/full-b1-l8192-bf16-gc-head.json) | 8192 | 1 | 编码器 + 决策头 | 1.079 | 0.927 | 7.21 | 7.65 |

单条输入仍是 8,192 token；batch 4 表示每步处理四条独立输入。较大的 batch 不会增加单条上下文长度。

8k、batch 1 开启重算后，分配显存峰值从 16.74 GiB 降到 4.02 GiB，步时从 0.922 秒变为 1.100 秒。利用释放的显存增加到 batch 4 后，吞吐达到 1.285 条/秒。这只是已测试配置中的结果，未穷举 batch 8 及以上、融合优化器或其他注意力实现。

显存使用 PyTorch CUDA 峰值统计，包含本进程初始化与预热阶段的分配；分配峰值和缓存峰值分别列出，不含所有驱动上下文开销。它与 MPS 步末采样的驱动内存不是相同统计口径。

## 与本机相同训练设置的对照

均为 LoRA、FP32、batch 1、只开启编码器重算。CUDA 关闭 TF32。

| Token 数 | M5 Pro 秒/步 | RTX 5090 秒/步 | 两套环境下的吞吐比 |
|---|---:|---:|---:|
| 2048 | 2.021 | 0.292 | 6.93 倍 |
| 4096 | 6.222 | 0.762 | 8.16 倍 |

本机为 PyTorch 2.14.0/MPS，远程为 PyTorch 2.8.0/CUDA 12.8，底层算子实现与运行环境不同；该表是实际使用这两套环境的结果，不是控制了所有变量的纯硬件比较。本机数据见 [MPS 报告](mps-training-speed-2026-09-21.md)。

## 数据量与训练时间

以 8k、batch 4、BF16 LoRA 加双侧重算的短时吞吐外推：

| 实际训练样本数 | 1 轮纯训练 |
|---|---:|
| 10,000 | 约 2.16 小时 |
| 100,000 | 约 21.62 小时 |

以上假设每条样本都填满 8k，且长时间速度与短测相同，不包括数据管道、校验、保存及评测。一份原始材料拆成多个问题时，应按最终训练样本数计算。实际长短混合训练需要根据真实长度分布再估计。

## 验证范围

共完成 10 个配置，全部检查严格加载同一原始权重、参数及输入设备、有限损失、首个预热步编码器或 LoRA 参数与评分头的非零梯度，以及优化器实际改变权重。[结果清单](cuda-5090/manifest.json)保留原始 JSON 和逐步日志。

输入为重复填充的合成客服文本，仅用于固定长度的计算容量与速度测试；重复的小批次会很快拟合，损失下降不代表业务准确率或长文本推理能力提高。日志中 15,421 token 的警告来自截断前的合成原文；`build_sequence` 将模型实际输入截为 8,192，脚本在创建张量前断言了实际长度和四个候选位置。

正式数据尚未准备，未保存测速后的模型权重。所有测试进程已结束，完成后确认 GPU 无计算进程，显存占用回到 0 MiB。机器未关机。

## 复现

在远程工作目录执行：

```sh
.venv/bin/python scripts/benchmark_training.py \
  --device cuda \
  --weights weights/1c5edc17a7acd8701df6fc341c0d179f1c62c982 \
  --mode lora --precision bf16 --batch-size 4 --seq-len 8192 \
  --gradient-checkpointing --checkpoint-head --memory-limit-gib 28 \
  --warmup 3 --steps 10 --output results/8k-recheck.json
```

整组测试入口为 `scripts/run_cuda_benchmarks.py`。本地保留同一份[测速实现](../scripts/benchmark_training.py)、[批量测试入口](../scripts/run_cuda_benchmarks.py)与[依赖说明](../requirements-cuda-benchmark.txt)。下一阶段按[长上下文方案](../research/long-context-plan.md)准备 4k/8k 业务数据和隔离测试，再评估准确率、校准、证据位置与短任务回退。
