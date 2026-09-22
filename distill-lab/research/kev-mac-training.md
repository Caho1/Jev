# Kev 的 Mac 训练方式与 Laya 适配

核对日期 2026-09-21；Kev 源码版本 `4f8110a3f8620cc3a182ae9a708e4398492c4b1a`。仅阅读源码和数据清单，没有执行训练。

## Mac 为什么可以训练

CUDA 是 NVIDIA 的计算后端。PyTorch 通过 MPS/Metal 也能在 Apple GPU 上执行张量运算和自动求导。[PyTorch 官方说明](https://docs.pytorch.org/docs/main/notes/mps.html)

Kev 的训练入口显式支持 cpu、mps、cuda，并在未指定设备时依次探测 CUDA、MPS、CPU。默认小批次配合梯度累积；可开启梯度检查点。`--dtype bf16` 指该实现的 CUDA autocast，代码要求同时指定 CUDA；Mac 默认训练路径使用 FP32，不能机械复制 README 的 CUDA BF16 命令。`--weights_dtype` 是另一项控制骨干权重存储精度的参数。[训练代码](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/kev/train.py#L53-L130)

模型主要更新 LoRA 适配器和指针决策头。它使用骨干隐藏状态打分，没有语言模型词表输出头；MPS/CPU 的模型构造默认采用 eager attention，以兼容其自定义注意力遮罩。[模型代码](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/kev/model.py#L132-L162)

用户截图介绍的是早期 Qwen2.5-0.5B 版本；当前主线是 Qwen3.5 的 0.8B、4B、9B 系列。主线注明 MPS 训练路径可用，但 Qwen3.5 的 DeltaNet 在 Mac 缺少快速内核，运行较慢，当前发布训练配方采用 H100。支持 Mac 并不意味着发布权重都在 Mac 上训练。[当前 README](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/README.md#training)

## 数据和训练方法

`decision-v7` 配方组合 10,000 条公开数据、896 条策略样本和 1,680 条组合规则样本。公开来源含 BANKING77、AG News、Yelp、Amazon reviews、BoolQ、MNLI 等；有单独的训练、开发、校准和测试文件，并冻结来源版本。[清单](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/evals/v7/decision-v7/manifest.json)、[数据适配](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/kev/data.py)

核心损失是候选交叉熵。代码另提供有序分数损失、选项换序一致性、冻结基座概率约束等可选实验项；不能把所有可选项说成发布模型默认都用了。其训练不要求 Jev 的软标签。[损失实现](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/kev/train.py#L20-L39)

可借鉴的数据增强包括选项换序、干扰选项和“正确选项存在/不存在”对照。移除正确选项时应有明确的“以上都不符合”标签，不能把缺失信息直接标成 false。[数据转换代码](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/kev/data.py)

评测代码包括 NLL、Brier、ECE 和温度拟合；存在旧的偶数/奇数样本划分实验，新方案应采用冻结清单中的独立 calibration split，不将测试集用于温度选择。[评测代码](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/kev/benchmark.py)

## 对我们的 Laya 实验意味着什么

Laya 的原始英语模型约 421M 参数，多语言模型约 322M，具备尝试本地小规模微调的合理条件；这是基于模型规模的判断，尚未在这台机器上验证性能。[Laya 模型说明](https://github.com/NandhaKishorM/laya#readme)

后续适配顺序：

1. 先完成业务数据清洗、分组划分、输入长度审查。
2. 将双卡 CUDA/NCCL 流程改为单设备可选的 MPS/CUDA 路径，分别处理精度和设备相关操作。
3. 先验证现有 Laya 决策头的候选交叉熵训练。选 LoRA 时需适配 ModernBERT/mmBERT 的实际模块及保存方式；不复制 Qwen 模块名。
4. 从小批次、短序列开始，测一次前向与反向的耗时、峰值内存、非零梯度及重载结果，再决定本地批量训练是否合适。
5. 保留独立校准和测试集。对照原始 Laya、公开业务标签微调、增加规则对照样本后的效果。

当前仍是数据准备和方法设计阶段，尚未运行以上训练验证。
