# 浏览器 DOM 决策数据选型

核查日期：2026-09-22。用途：微调 Laya，接近 jev-ultrafast 的动作与兼容目标选择。
本次只查找数据、读取发布说明和少量样本；未开始训练、调用 Jev 或改变既有划分。
仓库版本、文件校验值见 [manifest.json](manifest.json)，抽样结果见 [sample-verification.json](sample-verification.json)。

## 优先级与用途

| 数据源 | 已核对的规模与内容 | 适合的用途 | 当前边界 |
|---|---|---|---|
| [Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) | 官方训练 1,009 个任务；HTML、正负候选、CLICK/TYPE/SELECT、参数、人工历史 | 首先扩充现有浏览器动作与目标监督 | 以英文为主；保留官方测试与我们已有内部划分；卡片为 CC BY 4.0，另有研究用途说明 |
| [WebChain v2](https://huggingface.co/datasets/webagentlab/webchain) | 31,675 条轨迹、317,682 原始步骤；轨迹/动作元数据、DOM/AX 链接、选择器、截图 | 重点扩量来源，覆盖真实业务网页 | CC BY 4.0；SFT 成品以图像和坐标为主，需从元数据和外部结构文件重建文本候选 |
| [AgentTrek](https://huggingface.co/datasets/xlangai/AgentTrek) | 当前 HF 52,594 对话回合；抽查含网页观察、元素 bid 和动作 | 扩充多步网页决策，作为候选补充 | 自动采集、模型筛选；不是 52,594 个独立任务；HF 卡片未声明数据许可 |
| [Synatra](https://huggingface.co/datasets/oottyy/Synatra) | HF 99,924 步；AX 树、目标、历史、下一步动作 | 合成辅助数据，扩动作和页面覆盖 | 不是人工真实执行轨迹；发布页无 README/许可字段，不能把项目网页的许可自动当数据许可 |
| [WebLINX](https://huggingface.co/datasets/McGill-NLP/WebLINX) | 当前 chat/train 24,418 行；HTML、候选、动作、历史、对话、视口 | 技术上很贴近，多轮纠正/导航研究 | CC BY-NC-SA 4.0，暂不纳入面向生产的训练主线；样本中有 `uid=None` 等需过滤的步骤 |
| [Qwen/WebWorldData](https://huggingface.co/datasets/Qwen/WebWorldData) | 中英网页状态与动作转换；抽查 20 行，3 行含至少 10 个汉字 | 中文网页状态、动作效果和恢复判断的候选补充 | Apache 2.0；主要监督下一页面状态，不能直接把动作当成给定用户目标下的最优决策 |
| [WebRL SFT](https://github.com/THUDM/WebRL/blob/main/LLaMA-Factory/data/web_policy_sft.json) | 官方仓库有约 96 MB 的策略 SFT JSON，已读取首条记录 | WebArena 业务网站的额外策略数据线索 | 非 OpenWebRL；需核对轨迹分组、数据许可及 WebArena 测试任务重叠后再纳入 |

选择建议：先扩充 Mind2Web 的可训练部分，同时对 WebChain 做结构与动作对齐抽样；
AgentTrek、Synatra 可保留为补充候选，先解决许可/来源及重复任务问题。
不因数据量大就把所有来源直接拼到训练集。

## 本次实际检查到什么

- Mind2Web：核对官方卡片与既有本地数据结构。正确控件、当前操作和参数仅作监督，不能进入输入或检索特征。
- WebChain：下载 part_00 的 751 条轨迹元数据；每条都有 DOM/AX URL 列。
  对其中一条轨迹的两个外部结构链接实际请求，均返回 200，分别获得约 1.91 MB HTML 与 1.35 MB AX JSON。
  这只证明该轨迹的结构文件可取，不证明全量链接可用，也尚未验证动作选择器能逐步唯一匹配目标。
  当地样本文件名中的 `first` 仅表示第一条研究样本。
- WebChain 中文覆盖：751 条元数据中有 8 条 query 含汉字，但没有达到 10 个汉字；
  不能把英文任务夹带中文站名计成原生中文任务，本次未确认可用中文任务量。
- AgentTrek、Synatra、WebLINX：各读取 3 条 train 样本，确认实际列结构，未全量下载。
  HF viewer 的样本接口不固定 commit，仓库快照版本另外保存在 manifest。
- WebWorldData：实际样本是 `conversations`，与卡片示意的 `messages` 不同；解析必须以实际文件为准。
  中文内容已经在小样本中确认，原生来源和非翻译属性仍需逐来源过滤，不能从 language 标签推出全部都符合要求。
- WebRL：仅读取首个完整 JSON 对象；没有加载 `.pt` 或执行仓库代码。

## 中文数据的准确结论

已找到有中文内容的网页交互数据，但尚未找到已经核实可直接投入训练的大规模
“原生中文任务目标 + 操作前 DOM + 人工正确动作”的成品。

WebWorldData 的官方来源说明包含 FineWeb、CCI 3.0、自动探索、合成任务和 AgentTrek 等。
它的发布目标是 `(state, action) → next_state`，而我们的目标是 `(goal, state, history) → action/target`。
如果原目标丢失，不能简单把最终助手输出当操作标签，也不能把随机探索动作当最佳动作。
可以进一步寻找保留目标的子集，或在明确的新目标上重新标注，但必须记录这些标签的来源。

卡片所称 1,059,348 条的构成还包含约 548K 通用 QA/对话与多格式转换；
HF viewer 仅展示前 5 GB，并给出部分行数/估算总数。因此本次不把“106 万”算成可直接训练的网页决策数。

[AutoWebBench](https://github.com/THUDM/AutoWebGLM/tree/main/autowebbench) 确实有中文网页评测数据。
已核对官方树中的 `zh/ind/test.json` 与 `zh/ood/test.json`；本次未下载题目内容。
这些是测试文件，不能为补足中文训练量而并入训练。项目也明确将公开数据描述为研究用途。

## 划分与适配规则

1. Mind2Web 的官方 train 也不能整体直接加入：我们当前的内部开发/校准/留出来自已有网站分组。
   扩容时必须保留这些分组，将新任务按既有网站规则归入对应划分。
2. WebChain 的 `test_suite_150` 被描述为从源语料分层选出的子集，不能假设与全部 SFT 自动互斥。
   先取得 test trace UID，从训练候选中明确剔除；同一轨迹的多个窗口必须一起划分。
3. AgentTrek、Synatra 主要发布 train；需恢复任务/轨迹/网站分组，不能随机按行拆分。
   WebWorldData 包含 AgentTrek 等重整数据，跨来源合并要去重。
4. 数据必须有操作前的观察及可映射的目标；不把动作后的状态、当前正确答案或未来轨迹送给模型。
   当前候选中找不到正确目标时单独记录，不在评测输入中补答案。
5. 仅转换真正匹配的操作；hover、goto、复杂键盘、标签页等不能硬改成 CLICK。
   `SELECT` 要恢复选项值；类型转换需保留语义；WAIT/DONE/BLOCKED 不能凭空造标签。
6. 先提取文本与结构。WebChain 的两套全量图像 SFT 各约 193 GB，本轮不下载。
   对我们的纯文本决策模型，优先按轨迹抓取元数据与需要的 DOM/AX 文件。

## 暂不作为训练主数据

- [OpenWebRL SFT](https://huggingface.co/datasets/OpenWebRL/OpenWebRL-SFT-Trajectories)：以视觉网页代理的图像/坐标监督为主，不能直接当 DOM 元素选择数据。
- [WorkArena](https://github.com/ServiceNow/WorkArena)：适合后续业务闭环评测与采集，属于需要 ServiceNow 实例的环境，不等于一份现成训练轨迹包。
- [WebCPM](https://github.com/thunlp/WebCPM)：中文交互搜索有价值，但不是完整网页控件操作监督，不能替代当前 browser grounding 数据。

本目录是研究材料，不是可直接训练的最终数据集。所有新数据尚未写入原训练目录或看板统计。
