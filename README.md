# Jev / Laya 实验项目

Jev 决策模型、Laya 微调和浏览器自动化的实验记录，包含本地推理、训练与评测脚本、研究文档和训练效果看板源码。

本仓库是工作目录的公开源码快照。**不上传模型权重、训练/测试数据、逐题预测或看板数据快照**；实验报告、汇总指标、配置、依赖锁文件和第三方来源信息保留。发布范围与完整性清单见 [publication/README.md](publication/README.md)。

## 目录

| 目录 | 内容 |
| --- | --- |
| [distill-lab](distill-lab/README.md) | Laya 数据准备、MPS/CUDA 训练、客服意图识别及 Browser Use 评测 |
| [distill-lab/scripts](distill-lab/scripts) | 数据准备、微调、模型适配、评测及看板更新脚本 |
| [distill-lab/results](distill-lab/results) | 训练配置、汇总指标、运行记录与实验报告 |
| [distill-lab/research](distill-lab/research) | 数据集调研、协议分析、模型与训练方案 |
| [distill-lab/dashboard](distill-lab/dashboard) | React 训练与效果看板源码；实际数据需本地恢复 |
| [browser-lab](browser-lab) | 浏览器和 macOS 操作实验及上游参考源码 |
| [decision-lab](decision-lab/REPORT.md) | 决策准确率、负载测试和评测浏览器模板 |
| [snake](snake/README.md) | Jev / 本地 Laya 贪吃蛇演示与对照实验 |
| [snake-reference](snake-reference) | 蛇演示参考素材 |

## 主要实验记录

- [BANKING77 第一轮微调](distill-lab/results/banking77-first-run-2026-09-21.md)：独立官方测试集 3,080 条，微调准确率 85.55%。这是英语银行客服意图分类实验。
- [同题 Jev 对照](distill-lab/results/banking77-jev-comparison-2026-09-21.md)：记录模型版本、输入协议与标签描述限制，结果只适用于该次实验设置。
- [原生中文客服意图小试](distill-lab/results/customer-intent-zh-pilot-v2/README.md)：CrossWOZ 数据准备与小规模本地微调。
- [Browser Choice 本地训练](distill-lab/results/browser-choice-mps-pilot-v1/README.md)：训练与开发集隔离，记录候选覆盖率、操作与目标联合准确率；当前效果尚不足以用于生产。
- [jev-ultrafast 本地 Laya 接入](distill-lab/results/ultrafast-local-v2/README.md)：冻结 DOM 对照与真实浏览器闭环诊断。

历史报告中指向数据、权重、逐题预测、数据快照或本机服务的链接描述的是原始实验环境；这些文件不属于本次公开提交。具体数据来源、划分方式和复跑入口保留在各报告及准备脚本中。

## 本地使用

Python 环境和 Notebook 使用说明见 [distill-lab/README.md](distill-lab/README.md)。在 VS Code 中打开项目，按需安装对应依赖：

```sh
cd distill-lab
uv sync
# 需要训练/推理时：
uv sync --extra training
```

运行训练或模型评测前，需要自行准备对应的数据和模型；Mac 的 MPS 与 CUDA 的环境、参数分别见各实验文档。不要直接运行 Kaggle/CUDA Notebook 的全部单元格作为 Mac 安装步骤。

看板源码保留原有运行时和构建校验。实际 `distill-lab/dashboard/src/data.json`、生成的 `dist/` 和离线导出未提交；应先从自己的实验结果生成或恢复看板数据，再按看板文档构建。克隆仓库不会自动包含完整实验数据，也不会自动启动训练或调用 Jev。

调用云端服务时通过环境变量提供密钥，不要将密钥写入代码或提交。已发布的客服模型可在 [Cahol/laya-banking77-v1](https://huggingface.co/Cahol/laya-banking77-v1) 查看。

## 上游来源

第三方源码以固定版本工作树快照保留，包含本项目已经做过的局部适配。各目录内原有许可证继续适用；来源、版本与修改文件见 [THIRD_PARTY.md](THIRD_PARTY.md)。本仓库不把第三方模型、数据或代码的许可证统一改为新的许可证。
