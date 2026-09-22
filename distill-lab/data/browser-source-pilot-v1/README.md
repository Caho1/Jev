# Browser 数据小样本核对

日期：2026-09-22。目标是为 Laya 的浏览器动作和目标选择准备数据，参考 jev-ultrafast 的 DOM 路线。
本目录是检查包，**不是新增的训练集**。未启动训练、调用 Jev、执行网页或修改原划分。

## 结果

| 来源 | 实际检查 | 结论 |
|---|---|---|
| Mind2Web | 原有 train 的 72 条轨迹、519 步；另存 64 个检查样本 | 完整候选池包含正确目标 517/519；BM25 top-64 为 222/519（42.77%）。优先改进候选召回 |
| WebChain part_00 | 751 条轨迹元数据；排除本分片内 3 条官方测试轨迹、50 条现有非训练网站轨迹；剩余 698 条未分配，6,423 个源步骤 | 抽取 CLICK/TYPE/SELECT 各 4 步，共 9 条轨迹、8 个站点；12/12 静态 CSS 唯一匹配。观察时序仍未证实，全部待核验 |
| WebWorldData | 固定版本 JSONL 前 4 MiB 中的 49 个完整记录 | 49 条都是网页世界模型格式，0 条含独立目标字段或外层目标标题；9 条在完整对话中含至少 10 个汉字。不能直接当中文策略监督 |

上述抽样都不是总体准确率或代表性语言比例；WebChain 选样要求有 DOM 链接和 selector，
因此 12/12 只说明这些有结构的样本能对齐，不能外推全量可用率。
没有把动作标注指向的节点补进模型候选，也没有用正确标签参与 BM25 排序。

## 已保存的内容

- `mind2web-inspection.jsonl`：30 个 CLICK、20 个 TYPE、14 个 SELECT。`observation` 与 `reference` 分开；仅检查，不是评测集。
- `webchain-alignment.jsonl`：逐步原始操作、结构来源、CSS 匹配计数、文件校验值与待核验原因。
- `webchain-split-exclusions.json`：官方测试 UID 和已有非训练网站的排除记录。
- `webworld-audit.jsonl`：逐行格式、目标字段与汉字检查；不生成或翻译用户目标。
- `summary.json` / `manifest.json`：统计、输入依赖与文件 SHA-256。
- `raw/`：固定版本的元数据、49 条 JSONL 原样记录和 12 个静态 DOM；没有下载整套图像。
  外部 DOM 地址不受 HF commit 锁定，实际内容另存 SHA-256。原始网页文件只作离线解析。

## 使用边界

1. Mind2Web 沿用原有 `split_assignments.json`，脚本核验其 SHA-256；只读取 `train_frames.jsonl.gz`。
   当前中间格式缺少输入参数和完整 SELECT 选项，候选也不等于 jev-ultrafast 的实时可见控件。
2. WebChain 读取官方 150 条测试 UID，在本分片排除命中的 3 条；其余还需跨来源/任务去重及网站分组，
   **没有直接标记为 train**。同一源步骤按 trace UID + step ID 去重。
3. 保留 WebChain 原始动作。该分片过滤后有 288 个 hover 被成品映射为 click、33 个 select 被映射为 type；
   不能直接照搬成品 Seed 动作。SELECT 样本指向 option，后续需映射到所属 select 与选项值。
4. 唯一匹配 selector 不证明 DOM 是操作前快照，也不证明元素在视口内可点击；当前 WebChain 样本全部隔离。
   合成 CoT、操作后状态、当前目标及未来动作不能进入模型输入。
5. WebWorldData 的统计含整个对话中的汉字；原生中文来源未逐条确认，且缺少策略任务目标。
   本次不把它计入可用中文训练量，也不从某个动作反推一个“正确目标”。
6. 数据使用来源说明沿用此前选型报告；本轮没有发布或重新分发原始数据。

## 复现检查

工作目录为 `distill-lab`，使用本地 `.venv`。解析 DOM 新增轻量依赖：

```bash
uv pip install --python .venv/bin/python beautifulsoup4==4.14.3
.venv/bin/python scripts/prepare_browser_source_pilot.py
.venv/bin/python -m unittest discover -s scripts -p test_browser_source_pilot.py -v
```

默认从已经落盘的材料离线重算；`--fetch-dom` 仅补取选出的最多 12 个 DOM。
源数据固定版本/下载地址和 SHA-256 见 `raw/download-manifest.json`、`raw/webworld-sample-manifest.json`。
3 项测试覆盖非训练轨迹拒绝、改变答案不影响模型观察、既有保留网站保护。

## 下一步优先级

先改善 Mind2Web 候选召回并恢复原始参数/下拉选项，再评估在 8K 输入预算下的目标召回。
WebChain 继续核实操作前观察的采集时序、可见性及 SELECT 映射，证据充分后再扩量。
中文策略数据继续寻找独立目标完整、来源可核实的子集；本次没有新增已验证的中文策略样本。

来源：[Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web)、
[WebChain](https://huggingface.co/datasets/webagentlab/webchain)、
[WebChain 项目说明](https://github.com/sicheng-fan/WebChain)、
[WebWorldData](https://huggingface.co/datasets/Qwen/WebWorldData)。
