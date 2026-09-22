# 中文多场景客服意图：魔搭数据筛选

核对日期：2026-09-22。目标是中文客服意图识别与分流，扩展 BANKING77 以外的业务；优先原生中文，不通过翻译英文数据构造主训练集。本次只调研和检查文件，没有开始训练。

结论：可以开展多领域中文实验，但当前找到的数据不能简单拼成一个“大型真实客服意图库”。需要区分原生中文人工采集、真实业务记录、模型合成、比赛伪标签，以及只提供展示页的商业数据。最先可整理的是 CrossWOZ；电商场景需要补充真实业务标签，不能靠合成数据证明生产效果。

## 已核实的候选

| 数据集 | 场景与规模 | 适配方式 | 使用边界与建议 |
|---|---|---|---|
| [CrossWOZ（魔搭）](https://modelscope.cn/datasets/OpenDataLab/CrossWOZ) / [原作者](https://github.com/thu-coai/CrossWOZ) | 酒店、餐馆、景点、地铁、出租；6,012 个中文会话 | 有对话行为、领域、槽位标注，适合用户多轮意图与需求识别 | **第一优先**。人工扮演任务采集，不是生产客服日志。原仓库 Apache-2.0；本次魔搭镜像只有数据库等文件，已从原作者固定提交下载完整对话 |
| [WWW2025 电商原始训练集](https://modelscope.cn/datasets/smau0441/www2025-train) | 实际文件 1,000 条，其中 **300 条对话意图、700 条图片分类**；对话部分 23 类 | 商品材质、规格、功能、使用方法、补货、质量反馈等 | 适合小规模电商试验。全部记录带图片引用，需筛选纯文本可判断的样本；社区卡片标 Apache-2.0，原比赛数据条款仍需核实 |
| [KUAKE-QIC 魔搭镜像](https://modelscope.cn/datasets/yangzailu/chinese-medical-consult-intent-annotation-dataset) / [CBLUE 原项目](https://github.com/CBLUEbenchmark/CBLUE) | 中文医疗查询意图；实测训练 6,931、开发 1,955，11 类 | 医疗费用、就医建议、疾病表述等咨询类型识别；不是答案生成任务 | 可做独立领域实验。镜像介绍不完整，以实际文件和原项目为准；镜像公开测试 1,994 条全部无标签，不能直接计算准确率。发布或商用前核实原数据协议 |
| [QingshanAI 通用电商](https://modelscope.cn/datasets/QingshanAI/ecom-customer-service-synthetic)、[售后](https://modelscope.cn/datasets/QingshanAI/ecom-after-sale-synthetic)、[物流](https://modelscope.cn/datasets/QingshanAI/ecom-logistics-synthetic) | 实测 992 + 997 + 987 = **2,976 条**，含 scene 标签 | 支付、促销、退换货、投诉、物流停滞、丢件破损 | **仅作辅助训练候选**。作者明确为模型合成；全系列仅 2,340 个完全不同的用户文本，标签需归并、复核；不能作为真实业务效果测试集 |
| [RiSAWOZ](https://modelscope.cn/datasets/OmniData/RiSAWOZ) / [原作者](https://github.com/terryqj0107/RiSAWOZ) | 约 11,200 个中文人工多轮会话，12 个领域，15 万余话语 | 领域、意图、状态、指代省略，多轮能力有价值 | **研究备选**。数据明确 CC BY-NC 4.0，不能把 GitHub 代码的 MIT 当成数据授权 |
| [OralGovQA](https://modelscope.cn/datasets/carina790/OralGovQA) | 卡片标称 8,246 条中文政务咨询，申请条件、流程、材料、政策、时限 5 类 | 政务客服路由；含正式问题及口语化改写 | **研究备选**。来源为政府公开材料及口语化加工，非逐条真实客服日志；CC BY-NC-SA 4.0、仅限学术研究。本次读卡片和文件树，未逐条验收 |
| [数据堂 9 万组客服](https://modelscope.cn/datasets/DatatangBeijing/90000sets-Multi-domainCustomerServiceDialogueTextData) | 卡片标称电信、电商、金融、生活、商业、教育、医疗、娱乐等，9 万组 / 1,826,837 条 | 多行业真实客服语料的商业采购线索，意图标签是否交付未确认 | **不是可直接下载的开放训练集**。本次仓库只有 5 张展示图片及说明；正文写明版权归数据堂、商用数据，与页面 Apache 标识不能混为一谈 |

## 文件核验结果

### CrossWOZ：保留原始会话划分

原作者固定提交：`df82c9fdff91b9b130f2d6b89110d3870ba6260e`。

| 原始划分 | 会话数 | 全部话语 | 用户轮次 |
|---|---:|---:|---:|
| train | 5,012 | 84,692 | 42,346 |
| val | 500 | 8,458 | 4,229 |
| test | 500 | 8,476 | 4,238 |

这些是原始数据规模，还不是完成清洗后的训练样本量。问候、感谢和其他非业务轮次需单独处理。每个用户轮次可能含多个对话行为，不能强行只取第一个作为单标签。输入只能包含当前用户话语和此前历史，不能包含随后客服回复、最终任务目标或标注状态。

魔搭另有 [chinese-travel-scene-dialogue-dataset](https://modelscope.cn/datasets/yangzailu/chinese-travel-scene-dialogue-dataset)，下载后为 44,409 / 4,935 / 4,909 行文本及 158 个组合标签。三份文件分别有 41,744 / 4,627 / 4,614 行文本能在原始 CrossWOZ 中完全匹配。因此不应算作独立新语料，也不和原版混合扩容。其扁平文本缺少原始会话 ID，并混合用户和客服话语；采用原版更便于审计。文本重合含常见寒暄，本次没有据此断言所有跨集重合都是泄漏。

### 电商：原始标签、图片任务、伪标签分开

原始 `train.json` 实测 300 个对话意图样本，覆盖 23 类，每类 4–24 条；另外 700 条是图片类别，不能当成用户意图样本。用于 Laya 文本决策前，需检查去除图片后证据是否充分，去除当前问题之后的客服答复，并复核标签是否仍成立。部分原记录带问题后的客服答复，直接喂给路由模型会产生不符合线上时点的输入。

[17,442 条扩充版](https://modelscope.cn/datasets/smau0441/Intent_Recognition_and_Image_Classification_in_E-commerce_Scenarios)的作者明确说明包含对比赛测试集 1、2 的模型标注及人工核验。该数量也包含图片任务，不能表述为 17,442 条人工金标意图。若将来使用，只能单独标记弱监督来源，不能再用原比赛测试集宣称独立泛化效果。

QingshanAI 三份文件均已逐行解析，原始 scene 标签分别为 58 / 14 / 12 种，存在“咨询/产品咨询/商品咨询”“物流停滞/停滞”等粒度和同义问题。按用户输入去重，而不是按整条含 bot 回复的记录去重；不把 bot 回复作为当前意图输入。售后与物流 LICENSE.txt 另有禁止整体转售、转授权等自定义条件，不能只依据页面 Apache 标识转发整份数据。

### 医疗与政务：单独保留来源和评测口径

KUAKE-QIC 实际 11 类，不能照镜像简介当成 3 类。测试文件有 1,994 条空标签，原项目 README 的测试行数为 1,944，两者不一致，清单采用实际镜像文件数。若做本地实验，应先在训练部分构建开发和校准划分，把原开发集冻结为本地留出评测，并明确它不是官方隐藏测试成绩；正式使用还需要核对镜像和原发布包版本。

OralGovQA 的正式问题与口语化版本应绑定同一 question_id 分组；不能把同一问题的改写分到训练和测试。当前非商用限制使它不进入拟发布的生产模型训练清单。

## 未纳入主训练候选的条目

- [driver_assisant_chat_4](https://modelscope.cn/datasets/tiramisu1125/driver_assisant_chat_4)：train/test/validation 各只有 **1 条记录**，三个文件 SHA-256 完全相同，不能用于独立训练评测。
- [中文外贸意图](https://modelscope.cn/datasets/yangzailu/chinese-foreign-trade-intent-annotation-dataset)：实际文件仅 1,132 字节、约 10 个示例，且 JSON 存在语法错误。只能参考意图名称。
- [Nietsim/wan1103-sn](https://modelscope.cn/datasets/Nietsim/wan1103-sn)：搜索描述提到客服评估，但实际文件树主要为视频，不作为客服标签数据。
- 天猫对话、普通客服问答、评论情感、新闻分类：没有可直接使用的业务意图标签时，不计入本项目已就绪训练量。
- MASSIVE 等由其他语言翻译/本地化而来的中文部分，不作为本轮“原生中文”主来源。

## 魔搭以外值得补齐的真实业务来源

[CSDS](https://github.com/xiaolinAndy/CSDS) 是京东中文客服对话摘要数据，原始结构有 `QA.Topic` 和 `Session_id`，可研究转换为问题类型与分流任务，但不是现成的单句意图分类集。应在业务问题出现的时点截断历史，以主题标注作为候选标签，复核前缀可判定性；商用与衍生发布范围仍需核实。

[Chinese-Ambiguous-Reference](https://github.com/ygan/Chinese-Ambiguous-Reference) 提供中文真实购物沟通和澄清行为标注，仓库 MIT，可作为“是否需要追问”辅助任务的候选。它不是退货、支付等业务意图分类，需要单独定义输出及按会话划分。

[CCL2018 客服意图](https://github.com/nlpjoe/2018-CCL-UIIMCS) 是电信客服方向线索，但本次打开的仓库是比赛方案展示，不能据此宣称已取得完整可授权训练数据。JDDC/CSDS 在本次魔搭关键词检索中未返回同名条目，不代表整个社区绝对不存在镜像。

## 建议的下一轮实验

1. 先整理 CrossWOZ 的原始中文多轮用户意图，完成 Laya 中文基线；电商 300 条先作样本审查和任务设计，不靠它撑起大规模训练。
2. 同时沿 CSDS 等真实电商来源补标签与使用范围；合成电商最多做训练增强，独立记录，并做有/无合成数据的消融对照。
3. 按任务提供中文候选项与明确描述，复用 Laya 的决策训练方式；保留领域命名空间和多意图能力，不把不同领域强制套入 BANKING77 标签。
4. 官方 train/val/test 优先保留；开发和校准需在允许的数据内按会话隔离。先锁定测试，再做训练、增广和阈值选择。同一会话、改写、镜像、近重复组不跨划分；常见寒暄的重复单独报告。
5. 评估每个领域的准确率、macro-F1、长尾召回、置信度与转人工覆盖率；相同输入、候选描述、中文金标下与 Jev 对照。8k 只表示最大预算，应另报告实际长度和长历史分桶效果。

现阶段可准备可追溯的中文实验数据，但还不能宣称已经具备覆盖各行业的生产级客服测试集，或中文效果已对齐 Jev。

## 本地证据

原始卡片、API 搜索结果、文件树、部分实际数据、SHA-256 与计数保存在 [核验目录](modelscope-customer-intent-2026-09-22/)。其中 `crosswoz-verified-statistics.json` 记录原版划分与镜像文本匹配，`sample-inspection.json` 记录小文件下载及哈希。本目录为研究缓存，不是正式训练清单，也不随模型自动发布。
