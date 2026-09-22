# 业务分类数据与 Laya 微调方案

调研日期：2026-09-21。本文记录来源核对和拟议方案；没有运行训练、调用生成模型或完成正式训练集制作。

## 可用业务数据

规模口径分别为文本、订单或查询—文档判断，不能直接相加。HF 镜像与上游可能有版本差异，导入时应固定版本并核对行数和标签。

| 场景 | 数据与来源 | 规模、标签、真实性 | 许可与接入注意事项 |
|---|---|---|---|
| 银行客服分流 | [PolyAI/BANKING77](https://huggingface.co/datasets/PolyAI/banking77)，[原始 CSV](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data) | 10,003 条训练、3,080 条测试；77 种客服意图，包括退款、重复扣款、卡片未到等。官方称在线银行查询，卡片对最初采集过程披露有限。 | CC BY 4.0。HF 旧加载脚本不兼容新版本 datasets，优先读取官方 CSV。77 个候选需专门检查 Laya 输入预算。 |
| 投诉归类、工单路由 | [CFPB 官方](https://www.consumerfinance.gov/data-research/consumer-complaints/)，[HF Parquet 镜像](https://huggingface.co/datasets/Mouwiya/cfpb-consumer-complaints) | 真实消费者投诉；镜像约 1,656 万条记录，含 product、issue、sub_issue 等标签。只有部分记录含投诉正文，不能把总记录数当成文本训练量。 | 官方公开数据使用说明为准，镜像标记 other。筛选非空正文；按时间划分，并统一历年产品/问题分类。禁止把要预测的标签或事后处理结果放入输入。 |
| 商品搜索相关性 | [Amazon ESCI 官方](https://github.com/amazon-science/esci-data)，[HF 转换版](https://huggingface.co/datasets/tasksource/esci) | 真实购物查询与人工相关性标注；大版约 262 万个判断；小版 1,118,011 个判断。四类：Exact、Substitute、Complement、Irrelevant。 | 官方 Apache 2.0。用官方 split，按 query 分组；产品表按 product_locale + product_id 连接。HF 转换版行数和官方版本不同。 |
| 中文搜索、RAG 相关性 | [THUIR/T2Ranking](https://huggingface.co/datasets/THUIR/T2Ranking) | 搜狗用户查询日志，专家四级相关性标签；258,042 个训练查询、1,613,421 条训练相关性判断、2,303,643 个段落。 | Apache 2.0。使用 qrels.train 做有标签训练；约两亿条检索负例不等于两亿条人工判断。不要将 C-MTEB 测试子集用来训练。 |
| 下单、取消、退款、物流客服 | [Bitext 客服数据](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) | 26,872 条，27 个 intent；有 cancel_order、track_order、get_refund 等。官方明确是混合合成数据，适合作为业务语言补充。 | CDLA-Sharing 1.0。分类输入用 instruction，目标用 intent；response 和 category 不能无意泄漏目标。模板改写需要成组划分。 |
| 订单状态、履约与差评风险 | [Olist 官方发布](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | 约 10 万真实订单，含订单、支付、物流、商品和评价信息。能构造延迟交付、取消、差评等任务，但需要先明确预测时点。 | CC BY-NC-SA 4.0，非商业限制。结构化订单任务适合同时建立表格模型基线；预测下单时风险不能输入实际到货时间等未来信息。 |
| 评论情绪、星级、商品类别 | [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) | 约 5.7154 亿条评论，含文本、星级、商品元数据。可按业务品类抽样，星级可做有序分类；星级并不等同于人工情绪标签。 | 已核对的数据卡未列统一许可证，使用前需进一步核实发布条件。按商品/用户及时间控制泄漏，不建议全量下载。 |
| 社区内容审核 | [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | 新闻站点真实评论及人工审核评分；包含 toxicity 等维度，可以做二分类或学习标注者比例。 | CC0 1.0。人工评分比例应保留来源，它和教师模型概率是两种不同标签。 |

CFPB 正文应使用[官方历史归档](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/)，最晚覆盖到 2026-08-14；[2026 年 9 月 API 发布说明](https://cfpb.github.io/api/ccdb/release-notes.html)确认正文已从在线数据库移除，不能依赖当前主库持续获得新正文。HF 镜像需独立核对快照与正文完整性。公开投诉的产品和问题标签是提交时选择的分类，不应视为已核实的事实或责任认定。

## Nimble 实际如何训练

核对的仓库版本：`f136b3f75721fda4ea961f73993cc50b08488835`。已读取源码、manifest 和完整训练/评测 JSONL 并核对记录数，没有执行仓库脚本。

1. 数据：当前发布集为 2,676 条训练记录，即 1,338 对原始/反事实样本；另有 324 条固定评测记录。训练的 34 个来源家族与评测家族分离。标签是合成且经过模型检查的，没有人工审核。[数据说明](https://github.com/bespokelabsai/nimble/blob/f136b3f75721fda4ea961f73993cc50b08488835/docs/DATASET.md)
2. 来源：manifest 的合并谱系记录了 GPT-5.6 Sol、GPT-5.6 Luna、Claude Sonnet 5 三批数据。生成模型构造和检查证据，代码将业务规则应用到核验后的事实上，得到最终类别。训练器读取 `reference.target`，没有读取 Jev 概率作为目标。[manifest](https://github.com/bespokelabsai/nimble/blob/f136b3f75721fda4ea961f73993cc50b08488835/data/manifest.json)、[标签构造代码](https://github.com/bespokelabsai/nimble/blob/f136b3f75721fda4ea961f73993cc50b08488835/nimble/datasets/evidence_curation.py)
3. 学习目标：将答案映射成 A、B、C 等单 token 代码，从最后一个位置取这些 token 的 logits，计算正确类别的交叉熵。名称中的 contrastive 指数据成对构造；这里没有另加 InfoNCE 或 DPO 损失。[训练器](https://github.com/bespokelabsai/nimble/blob/f136b3f75721fda4ea961f73993cc50b08488835/nimble/training/schema_train.py#L45-L63)
4. 参数更新：Qwen3.5-9B 基座加 LoRA，rank 16、alpha 32、dropout 0.05，作用于语言模型中的线性层；学习率 5e-5、有效 batch 8，训练一轮。LoRA 是更新参数的方式，监督标签可以来自人工、规则或获准使用的合成数据，无需先有教师概率。[LoRA 配置](https://github.com/bespokelabsai/nimble/blob/f136b3f75721fda4ea961f73993cc50b08488835/nimble/training/schema_train.py#L187-L208)、[训练说明](https://github.com/bespokelabsai/nimble/blob/f136b3f75721fda4ea961f73993cc50b08488835/docs/NIMBLE_TRAINING.md)
5. 推理：在允许的答案之间计算 softmax，由程序组装结构化结果。softmax 概率不会自动保证校准。当前每字段最多 26 个候选、提示总长 2,048 tokens；MLX 有共享前缀实现，CUDA 实现逐字段处理。[模型卡](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B)、[评分代码](https://github.com/bespokelabsai/nimble/blob/f136b3f75721fda4ea961f73993cc50b08488835/nimble/scoring/parallel_schema.py)

仓库确实包含 Jev 评测和历史标注工具；这不等于发布模型使用 Jev 蒸馏。部分历史文档提到 2,826 条，当前实际发布数据是 2,676 条，额外 150 条属于另一个本地实验。公开方法也不等于所提到的每个商业生成模型都自动授权我们的训练用途。

## 拟议的 Laya 适配

这里采用其数据构造与监督目标。Laya 使用双向编码器和候选决策头，候选分数来自 `[MASK]` 标记位置；不能直接照搬 Qwen 的下一个 token 评分器或 LoRA 模块名。[Laya 模型代码](https://github.com/NandhaKishorM/laya/blob/42626c348753fbb17572a813127df2278a1ec527/laya/common.py#L84-L118)

数据统一保存：`id`、`source_dataset`、`source_revision`、`source_id`、`group_id`、`language`、`split`、`state`、`questions`、`gold`、`label_origin`。输入和标签分开，保留原始标签，派生样本另存变更事实及核验记录。

- `choice`：类别转成 one-hot，例如 `[0,1,0]`。该向量是监督编码，不是声称正确答案存在 100% 的客观概率。
- `noul`：顺序固定为 `[false,true]`，例如 true 对应 `[0,1]`。
- `score`：只有顺序和含义明确的等级才使用，例如 1～5 星或四级相关性；业务部门和 ESCI 四类不能随意当成有序分数。
- 官方 notebook 已接收 target 分布，one-hot 可以接入。第一阶段拟先用候选交叉熵建立基线，再单独比较是否加入原有 RLCD 奖励。LoRA 作为编码器的可选节省显存方案，需要另行适配并验证；当前 notebook 更新编码器和决策头，没有现成 LoRA 配置。

对照样本示例（人为设计示意，尚未写入训练集）：规则为未使用商品且购买不超过 30 天可退。样本 A：未使用、购买 29 天；样本 B：未使用、购买 31 天。其他输入一致，标签从可退变为不可退。添加费用、商品描述等无关信息时标签应保持不变。规则边界同时覆盖，避免只记住“退款”等关键词。

## 第一批数据准备顺序

以下是建议，不是已生成的数据规模或已完成实验：

1. 先用 BANKING77 建立客服意图基线，再从 ESCI 英语训练部分抽取约 3 万个判断，覆盖四类与难例。第一批先控制在约 4 万条原始标签记录，加少量人工核验的规则对照样本。
2. 加入 CFPB 的非空正文，扩展到长文本投诉分类。Bitext 可补充订单客服表达，其合成来源单独标记。Olist 留作具有明确预测时点的订单任务。
3. 中文相关性用 T2Ranking，基座考虑 Laya multilingual；英语和中文基座分别评测，不把英语基座的能力直接推断到中文。
4. 数据划分为训练、开发、校准、最终测试。保留官方测试集；其余从训练数据中按查询、对话、订单或来源家族划分。原文、改写、反事实对子和翻译放在同一组。
5. 在生成 token 前统一上下文配置，检查状态和候选是否被截断。BANKING77 的 77 个候选可能压缩选项描述，需要实测；采用候选检索时，检索过程必须不使用标准答案，且端到端评测候选召回率。
6. 对比原始 Laya、真实业务标签微调、加入对照样本后的版本。看 macro-F1、每类召回、成对样本同时判对率，以及 NLL/Brier；温度只在独立校准集拟合。

原 notebook 的预处理与训练长度配置不一致，且从训练样本抽取校准数据；正式改造时需要一起修正。官方当前训练脚本依赖 CUDA/NCCL，本地 M5 Pro 先承担清洗、划分、审查和格式验证。训练设备与 LoRA 实现待数据准备后确定。
