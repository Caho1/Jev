# Computer Use 控件评分：本地第一轮训练

**运行失败记录：** 01:18:36（UTC+8）因 MPS 内存达到 28 GiB 上限中断。最后一条训练进度为 55/259 次更新、440/2,068 条（21.3%）。中断前未到首次模型保存点，仅保留原始 `initial/`；不能从这里继续优化器状态。01:32 已以梯度重算和固定长度桶配置启动新的 `../computer-use-grounding-lora-v2/`。本目录保留原始失败日志，下文为最初运行配置。

2026-09-22 01:15（UTC+8）启动。实时阶段以本目录 `status.json` 为准；`events.jsonl` 保留完整事件序列，结束后生成 `results.json`。训练进程与对话分离，关闭当前消息不会停止进程。

本次使用 Laya 原始基础权重，新建 rank 16、alpha 32 的注意力 LoRA，并训练决策头。没有加载或覆盖 BANKING77 客服适配器。运行配置、输入文件校验值、代码校验值及硬件信息保存在 `run_config.json`。

- 硬件：本机 Apple M5 Pro，48 GiB；PyTorch MPS，FP32。
- 数据：2,068 条控件评分训练对，源自 517 个可采样训练步骤。
- 轮次：1；micro-batch 2，梯度累积 4，常规有效 batch 8，共 259 次优化器更新。
- 学习率：LoRA 5e-5，决策头 2e-5；5% 预热，线性衰减；AdamW，梯度裁剪 1.0。
- 完整输入上限：8,192 tokens；本批实际 314–1,417，中位数 724。没有截断或重复填充，不构成长上下文效果验证。
- 每 100 次更新保存一次独立检查点，整轮完成保存 `epoch-1/`。
- 概率尚未校准，不据此直接设置生产执行阈值。

训练后，先评测新权重，再恢复此次运行保存的原始状态评测基线。开发页面由 ID 的固定哈希顺序预先选定，每个网站 1 页，来自 amtrak、eventbrite、ign、tesla，共 4 页、1,020 个真实候选控件。每页遍历完整候选池，没有强行插入正确目标。

这是 78 个开发步骤中的固定小样本检查，用于第一次本地训练验证。分别报告 Recall@1/8/24/64、目标排名与整页扫描时间，不能将结果称为完整开发集准确率、真实网站任务完成率或生产性能。没有读取校准集和试验留出集进行模型评测或参数更新。

训练样本、页面数据、正确目标仅用于它们各自允许的环节。模型只接收任务、过去的动作、当前页面文本和当前候选的可观察字段。页面中没有目标时，评测计为召回失败。

复现实验时使用新的输出目录，脚本拒绝覆盖已有运行：

```sh
.venv/bin/python scripts/train_computer_grounding.py \
  --output results/computer-use-grounding-lora-reproduction
```

查看当前运行：

```sh
cat results/computer-use-grounding-lora-v1/status.json
tail -f results/computer-use-grounding-lora-v1.log
```

保留的 `initial/` 是微调前的可训练参数状态；`step-*/` 和 `epoch-1/` 是此轮新权重。基础模型仍位于 `checkpoints/banking77-lora-v1/base/`。阶段性文件是模型检查点，没有保存优化器续训状态，不能将新运行误报为精确断点续训。
