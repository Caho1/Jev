# 本地 BANKING77 微调模型

本目录已包含本轮最佳微调检查点及其配套基础模型文件，无需依赖远程 5090 或 Hugging Face 缓存目录来读取权重。

- `best/trainable.safetensors`：第 3 轮选出的 LoRA 参数与训练后的决策头，122,565,852 字节。
- `best/config.json`：训练配置、77 个类别顺序、源代码版本和数据划分记录。
- `best/calibration.json`：独立校准集拟合的温度。
- `base/`：本次训练使用的固定版本 Laya 基础权重、编码器配置及 tokenizer。
- `local-manifest.json`：本地文件大小和 SHA-256 清单。

这是一组「固定基础模型 + LoRA + 决策头」文件，尚未合并成可单独用 `AutoModel.from_pretrained` 加载的通用模型目录。项目的 `scripts/train_banking77.py` 中 `reload_checkpoint` 提供对应加载逻辑；基础权重路径设为本目录下 `base`，检查点路径设为 `best`。加载时须保留 `mode=lora`、`rank=16` 和原训练的决策头包装配置，不能仅加载 LoRA 而忽略决策头。

基础版本：`convaiinnovations/laya@1c5edc17a7acd8701df6fc341c0d179f1c62c982`。
源代码版本：`NandhaKishorM/laya@42626c348753fbb17572a813127df2278a1ec527`，相关实现已保存在项目 `reference/` 与 `scripts/`。

校验结果：基础权重与训练配置中的 SHA-256 相同，微调权重与最终测试锁定的 SHA-256 相同。

本轮官方测试集：3,080 条，准确率 85.55%，Macro F1 85.53%。这是英文银行客服的 77 类闭集分类结果，不代表真实线上效果或 8k 长文本能力。Jev 回答只用于独立评测，没有进入本模型训练。
