# Laya 决策模型微调实验

**新增浏览器接入诊断（2026-09-22）：** 已将 `reference/jev-ultrafast` 接到本地 Laya，在 48 个冻结 DOM 步骤和 8 个闭环任务上比较原始与 Computer Use 微调权重。整步准确率分别为 16.67% 与 18.75%；实际终态达成均为 4/8，正确报告完成分别为 4/8 与 2/8。接口可用，模型效果仍不足；这是小型合成诊断。详见 [接入、结果与复跑说明](results/ultrafast-local-v2/README.md)。

**当前方向（2026-09-22）：中文为主的多场景客服意图识别与分流。** 在已发布的 BANKING77 实验之外，筛选电商、酒店餐饮出行、医疗咨询及政务等原生中文数据。已完成原生中文 CrossWOZ 五场景、35 类的数据准备：训练 26,730、开发 2,982、校准 2,989、封存测试 2,967 条。280 条本地小试完成，同一 103 条开发题准确率从 3.88% 提升到 20.39%；官方多语言底座未微调为 12.62%。这不是正式测试结果，效果尚不足以部署，全量训练未启动。详见 [本轮数据与试验报告](results/customer-intent-zh-pilot-v2/README.md) 和 [魔搭中文客服数据筛选](research/chinese-customer-intent-modelscope-2026-09-22.md)。

此前已在 RTX 5090 完成第一轮 Computer Use 控件评分训练：2,068 条训练对、259 次参数更新，用时 53.92 秒；权重及日志已下载到本机并核对 SHA-256。本地 MPS 训练继续暂停，原有部分权重保留，详见 [暂停记录](results/computer-use-grounding-lora-v2/pause.json)。

Computer Use 实验已导入 Mind2Web 的 109 个任务，完成按网站隔离的数据准备。5090 运行从原始 Laya 权重重新开始，使用 BF16、LoRA rank 16 和有效批量 8，不覆盖客服模型或本地历史结果。详见 [5090 训练报告](results/computer-use-grounding-cuda-v1/README.md) 与 [Computer Use 数据与微调方案](research/computer-use-finetuning-2026-09-22.md)。该实验数据为原始英文，没有翻译生成中文。

全部 78 个独立于训练网站的开发步骤已评完：Recall@24 为原始 Laya 16.67%、本轮微调 11.54%，Recall@64 为 35.90% → 41.03%。首选控件命中均为 1/78，尚未达到可用效果；模型保留作实验检查点。校准集与内部留出集未参与本轮。

此前的客服方向参考 Nimble 的对照样本构造，以及 Kev 的公开数据混合、Mac 训练和评测设计，使用公开业务标签与可验证的规则标签微调 Laya。以下记录保留为已完成的历史实验。
数据调研、Nimble 代码结论及适配方案见 [业务数据与微调方案](research/business-data-and-nimble.md)。
生产场景优先级、公开数据与 LLM 对照方法见 [生产场景与数据筛选](research/production-scenarios-and-datasets.md)。
Kev 的 MPS 训练方式和迁移边界见 [Kev 与 Mac 训练](research/kev-mac-training.md)。
训练上下文上限已确定为 **8,192 token**，长度包括正文、问题、候选项和特殊 token。保留长短样本混合；8k 是上限，不要求把短文本重复填满。训练数据和评测设计见 [长上下文方案](research/long-context-plan.md)。

## 在 VS Code 中使用 Notebook

用 VS Code 打开本目录，安装 Microsoft 的 Python 和 Jupyter 扩展。
依赖安装或恢复命令：

```sh
uv sync
```

环境使用 Python 3.12，依赖版本由 `uv.lock` 固定。
打开 `.ipynb` 后，在右上角 **Select Kernel** 选择本项目的 `.venv/bin/python`；
也可能显示为 **Python (Jev → Laya)**。工作区已设置默认 Python 路径。
VS Code 直接管理内核，无需启动脚本或独立 JupyterLab 服务。

`00_environment_check.ipynb` 用于检查 Python 路径、Jupyter 与 notebook 执行能力。

## 上游参考

`reference/laya_finetune_typed_decisions_2xT4_kaggle.ipynb` 是用户提供的
[官方 notebook](https://github.com/NandhaKishorM/laya/blob/42626c348753fbb17572a813127df2278a1ec527/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)，
基于 commit `42626c348753fbb17572a813127df2278a1ec527` 保存，Markdown 说明和 Python 注释已翻译成中文；
可执行代码保持不变。`reference/source.json` 记录上游版本、原始校验值和翻译后校验值。

该文件使用 CUDA、NCCL 和 `/kaggle/working`，不能直接在 Mac 上完整运行。
其最后部分包含上传 Hugging Face 的代码；阅读原文件不等于运行或发布。

## 数据准备状态

第一批 BANKING77 已完成固定版本导入、精确/近重复分组、冲突标签排除、完整候选编码和独立划分：训练 7,964 条、开发 998 条、校准 997 条、官方测试 3,080 条。各划分的重复组无交集；详细清单及文件校验值位于 `data/banking77-v1/manifest.json`。
正式训练使用 RTX 5090，LoRA rank 16 加决策头、BF16、batch 8、梯度累积 4、3 轮。按开发集 NLL 选择模型，独立校准集拟合温度，最后才评测测试集。完整输入含 77 个候选，训练输入实际 575–667 tokens，8k 是上限，不代表已验证长文本效果。
Nimble 的原始文件仍是研究副本；ESCI、CFPB 等尚未导入。数据盘点见 [当前数据准备状态](research/data-readiness-2026-09-21.md)。
Jev 当前公开客户协议第 2.3(b) 条限制蒸馏及模仿输出训练；本方案不使用 Jev 输出作为训练标签。
条款来源：[TypeSafe Master Customer Agreement](https://typesafe.ai/legal/mca)，核对日期 2026-09-21。

后续数据扩展需要处理：

- 已有 `../decision-lab` 和 `../snake/community` 测评如何保留作回归检查。
- 保留原始业务标签及来源；将类别标签转换为 one-hot 训练目标，不能把它表述成教师概率。
- 对照样本只改变决定标签的事实，原样本及其衍生样本保持在同一数据划分。
- 新增数据继续按关联样本隔离训练、开发、校准与测试。
- 预处理和训练使用相同上下文预算，并检查截断。

第一轮已完成：全流程 8 分 34 秒，独立测试准确率 45.91% → 85.55%。详见 [第一轮训练报告](results/banking77-first-run-2026-09-21.md)。

已完成同一 3,080 条测试集的 Jev 对照：微调 Laya 85.55%，Jev 1.13.0 为 80.71%。本轮仅使用原始类别名，存在标签语义歧义，解释限制与完整协议见 [Jev 对照报告](results/banking77-jev-comparison-2026-09-21.md)。

本地模型已整理到 `checkpoints/banking77-lora-v1/`，含固定基础模型、tokenizer、最佳 LoRA 与决策头权重及概率校准配置，总计约 924 MiB。基础和微调权重均通过训练记录的 SHA-256 校验，文件清单见 `local-manifest.json`，加载说明见[本地模型说明](checkpoints/banking77-lora-v1/README.md)。

## 训练与效果看板

新增「中文本地评测」：已在 M5 Pro / MPS 上评测 MInDS-14 的 502 条中文子集记录及配对英文译文。排除来源混入的 10 条英文原文后，492 条含汉字题目的微调准确率为 63.21%，原始模型为 54.07%。14 类迁移测试与原 BANKING77 77 类基准分开报告。[完整结果与本地调用命令](results/minds14-chinese-local-2026-09-22.md)。

[本机看板](http://127.0.0.1:4173/) 提供训练进度、吞吐、损失曲线、每轮开发集指标、最终测试对比、置信度分流及每类召回。React 源码和可追溯快照位于 `dashboard/`，聚合训练产物保存在 `results/banking77-lora-v1/`。

「Jev 对比」页包含完整 3,080 题的原文、标准标签与三个模型的回答，可按类别、正误/分歧和关键词筛选，点击一行查看前三候选及失败原因。逐题行从已保存的测试预测构建，`scripts/update_jev_dashboard.py --build-status complete` 可重建该视图，不重新调用模型。

恢复已保存的看板（不连接远程、不重新训练）：

```sh
.venv/bin/python scripts/banking77_dashboard.py --serve-only
```

跟踪本次运行时使用 `--watch --serve`。它复用已认证 SSH socket，每 20 秒拉取聚合日志并构建看板；训练结束后停止远程轮询，本地页面仍可查看。编辑/导出期间不自动刷新。连接中断保留最后快照并显示延迟，不将缺失指标填为 0。

正式数据处理、训练及验证脚本分别为 `scripts/banking77_data.py`、`scripts/train_banking77.py` 和 `scripts/test_banking77.py`。正式运行目录拒绝覆盖；不要复用已经评测过测试集的运行来反复调参。

## 本机 MPS 训练测速

实测结果、内存记录和训练时间外推见 [2026-09-21 测速报告](results/mps-training-speed-2026-09-21.md)。

训练依赖放在可选的 `training` 组，安装或恢复时运行：

```sh
uv sync --extra training
```

在 VS Code 的终端中执行短时测速：

```sh
uv run --extra training python scripts/benchmark_mps_training.py \
  --mode lora --batch-size 4 --seq-len 256 --warmup 5 --steps 30 \
  --output results/mps-lora-b4-l256.json
```

`--mode full` 测全量分类微调，`--mode lora` 测编码器注意力 LoRA 加完整决策头训练。
可选序列长度为 128、256、512、1024、2048、4096、8192，且不超过底座配置上限。
实际实现是 `scripts/benchmark_training.py`，旧 MPS 文件名保留为兼容入口。
默认使用 FP32、MPS、SDPA 和 AdamW，关闭不支持算子的 CPU 回退。
`--gradient-checkpointing` 开启编码器重算，`--checkpoint-head` 开启决策头重算；
`--memory-limit-gib` 为该测试进程设置 GPU 内存预算。
每次运行从原始 Laya 检查点重新加载，检查编码器和决策头的梯度与权重更新，测速后的权重不保存。
默认使用本机已经缓存的 `convaiinnovations/laya` 检查点
`1c5edc17a7acd8701df6fc341c0d179f1c62c982`，也可用 `--weights` 指定包含配置、分词器和权重的本地目录；脚本不会联网下载模型。

`reference/laya_common.py` 保留固定上游版本的原始实现，来源与校验值见
`reference/laya_common_source.json`，许可证见 `reference/laya-LICENSE`。
测速使用合成客服四分类输入，不代表真实业务数据的准确率或收敛表现。
上游双 T4 notebook 仍保留原版 CUDA/NCCL 路径；本次测速是独立的单设备脚本。

## RTX 5090 长上下文测速

已完成 2k、4k、8k 训练测速；8k BF16 LoRA 加双侧重算在 batch 4 下约 1.285 条/秒，
分配显存峰值约 10.21 GiB。完整数据及 Mac 对照见 [5090 测速报告](results/cuda-5090-training-speed-2026-09-21.md)。

远程使用隔离虚拟环境并复用已有的 CUDA PyTorch，额外依赖见 `requirements-cuda-benchmark.txt`。
模型在服务器端直接下载，SHA256 与本机缓存权重核对一致；连接密码不写入项目文件。

```sh
python scripts/run_cuda_benchmarks.py \
  --weights weights/1c5edc17a7acd8701df6fc341c0d179f1c62c982 \
  --output-dir results/cuda-5090
```

该脚本顺序执行测试，记录各配置的 JSON、日志和失败清单：
先验证 BF16 基础训练，再测试可与本机对照的 FP32 配置，以及 2k、4k、8k BF16。
超出 28 GiB 显存预算时尝试重算；根据已测显存余量，最多把 8k LoRA 微批量增加到 4。
CUDA 结果记录实际分配与缓存显存峰值，MPS 结果记录采样值，两者不能直接作为同一口径比较。
