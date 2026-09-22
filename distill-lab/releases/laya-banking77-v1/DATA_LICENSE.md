# 数据来源与许可

本模型在 PolyAI 的 BANKING77 英文银行客服意图数据上进行监督微调。
数据集作者为 PolyAI / Iñigo Casanueva 等，论文为
[*Efficient Intent Detection with Dual Sentence Encoders* (2020)](https://aclanthology.org/2020.nlp4convai-1.5/)。

- [原始数据仓库与许可说明](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b)
- [Hugging Face 数据卡](https://huggingface.co/datasets/PolyAI/banking77)
- 数据许可：[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

原训练集经重复与近重复分组、与官方测试集重叠项排除、冲突标签排除后，
划分为训练、开发和校准三组。官方测试集保持不变；具体协议及文件 SHA-256
见 `data_manifest.json`。此仓库包含类别名称和数据清单，不重新分发客户文本。
模型权重与代码的 Apache 2.0 许可不取代原数据集的 CC BY 4.0 许可。
