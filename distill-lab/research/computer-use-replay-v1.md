# Computer use：冻结输入、采集教师回答与离线对照

> 2026-09-22 范围调整：后续只优化浏览器，并以 `browser-use/jev-ultrafast` 为参考。
> 本文记录的 `operation + item` 回放属于早期诊断，尚未对齐其按操作区分的目标题目。
> 桌面 OCR 日志导入器保留为历史工具，不进入后续数据与训练主线。
> 当前实施依据见 [浏览器微调方案](browser-use-jev-ultrafast.md)。

本轮先建立数据与评测流程，不更新模型参数，不接入自动点击。
参考 [typesafe-computer-use](https://github.com/awlevin/typesafe-computer-use) 的
`cc7b5066ae1a07b5e3182e8f87a9b5b6dfdcffc1` 提交，将状态、多个选择题、完整概率、
定位证据和评测标签分开存储。上游代码保持原样，位于 `reference/typesafe-computer-use/`，
其 MIT 许可与 Aaron Levin 的版权声明保留在该目录中。

## 已实现

- `scripts/computer_replay.py`：统一记录、请求 SHA-256、响应校验、评测和训练导出。
- `scripts/prepare_computer_replay.py`：从现有 Mind2Web 完整控件池构造动作与目标两道选择题。
- `scripts/evaluate_computer_replay.py`：原始 Laya、上轮 computer use 检查点和 Jev 使用同一冻结请求。
- `scripts/import_typesafe_replay.py`：导入上游 `payload.txt` 和 `answers.json`，保留原观察及定位证据。
- `scripts/test_computer_replay.py`：检查答案泄漏、候选召回、失败计分和训练/开发隔离。

模型只接收记录中的 `request.state` 和 `request.questions`。
`reference`、`teacher`、划分、任务 ID 和来源信息都不进入模型请求。
每次调用前写入尝试日志，每次响应立即落盘；没有自动重试，输出目录也不能覆盖。
中断后，应先检查 `attempts.jsonl` 与 `responses.jsonl`，不要直接重跑可能已计费的请求。

## 两种动作语义

Mind2Web 的 `CLICK / TYPE / SELECT` 是包含目标选择的原子标注。
桌面项目的 `type_text` 是“当前输入框已经聚焦后填写”，两者不能直接混作同一个标签。
因此 Mind2Web 使用 `operation` 和 `item` 两题；原项目日志保留其 `kind / item / site / offscreen` 题型。
统一回放格式不意味着它们已具有相同的执行语义。

## 固定数据与候选策略

数据来源是已有 `data/computer-use-grounding-v1`，沿用其网站、轨迹与重复任务分组。
当前只打开训练、开发页面和划分元数据；校准与内部留出页面保持未读取。
独立留出划分仍在原数据目录，本版不提前转换或打分。

| 项目 | 训练 | 开发 |
|---|---:|---:|
| 页面步骤 | 519 | 78 |
| 正确目标进入当前候选 | 222 | 40 |
| 候选召回率 | 42.77% | 51.28% |
| 最大编码长度 | 6,236 | 5,237 |

保留最多 64 个控件，按目标文本的固定 BM25 排序，再提供一个 `NONE` 选项。
候选只由可观察控件、目标和预算决定，绝不根据答案插入正确目标。
如果超过 8,192 tokens 或共享 API 字节预算，按检索顺序从尾部删候选并记录最终数量；
模型编码不截断输入。页面正文、控件字段的压缩继承并明确记录于准备代码，不能称为完整 DOM。

519 是原页面数；旧二分类训练只使用其中 517 个能满足正负采样条件的页面，二者统计口径不同。
当前训练页面中超过一半的目标未被检索保留，因此这批回放数据不能被当成已就绪的高质量控件训练集。
下一轮优先改进候选召回、使用更充分的目标上下文，再比较同页候选排序训练。
不应通过增加 `NONE` 样本正确率来宣称控件能力提高。

## 小规模真实验证

开发样本按网站分层、按 ID 哈希固定选取每站 2 步，共 8 步；训练教师采集另外选取 4 步。
选择不依赖目标是否被召回、模型预测或答案。原始 Laya 与 computer use 检查点在本机 MPS 运行，
Jev 使用现有客户端调用 `jev-1.13.0`。本轮没有使用客服 BANKING77 adapter。

- 开发对照：`results/computer-use-replay-smoke-v1/`
- 训练教师采集：`results/computer-use-replay-teacher-smoke-v1/`
- 逐题输入、人工参考与模型输出：各目录的 `cases.jsonl`
- 请求校验值、模型权重校验值和协议：各模型子目录的 `protocol.json`

上述调用已完成：三个模型各 8 步，输入校验值一致；训练教师采集另有 4 步。
已从原始响应重新计算指标并检查训练、开发任务隔离，见
[`local-audit.json`](../results/computer-use-replay-smoke-v1/local-audit.json)。
结果与边界见[诊断记录](../results/computer-use-replay-smoke-v1/README.md)。

本次本地模型每道题分别前向，Jev 将同一步的多题放在一次 HTTP 调用中。
耗时包括调用范围内的编码/推理或网络请求，不包括模型加载，也不包括截图、OCR、执行和界面等待。
因此这些延迟不是闭环操作速度。

## 如何看结果

- `questions.*.accuracy`：每类题目的正确率；失败和未完成请求计为错误。
- `joint_accuracy`：动作与目标两题同时正确，正确选择 `NONE` 也可计入。
- `grounded_joint_accuracy`：目标确实在候选中且两题同时正确；选择 `NONE` 不算成功操作。
- `candidate_recall`：正确目标是否进入候选池，不是模型准确率。
- `median_request_ms`：调用延迟的样本中位数定义采用有序样本中间靠上的值。

8 步只是流程与接口验证，不代表完整开发集或生产性能。
它依赖人工轨迹提供的过去动作，不是模型自主执行后的轨迹；本轮没有闭环成功率。
本地分数未重新校准，也没有沿用 BANKING77 温度或上游项目的执行阈值。

## 教师回答如何进入训练

导出器只接受 `split=train` 和来源明确的 Mind2Web 人工标签。
人工正确答案始终是主监督；教师预测与人工答案一致的题目才附加完整教师概率分布。
教师不一致时仍保留人工标签，教师分布置空。
这只能视为辅助软标签，不能把 Jev 自己的回答称为人工真值。
开发预测只进入对照结果，不写入训练样本。

训练样例准备在 `training_examples.jsonl`，本轮没有执行监督训练或蒸馏优化器。
同一步存在多个可接受目标时，`gold_choices` 保留全部标注；训练器需支持多正例，不能只取第一个。

## 上游日志导入的边界

上游日志保存了 state 和候选描述，但没有完整保存每题的 instructions；导入器从已审查提交的 AST
中读取字符串常量，既不导入也不执行上游程序。
另外，上游在生成 payload 和真正调用时分别计算时间字段，旧教师请求无法证明与重建请求逐字一致。
因此导入记录明确标记 `instructions_reconstructed=true`、`legacy_teacher_request_verified=false`。
旧响应只能用于诊断，不能直接导出为蒸馏监督；应在新冻结输入上重新获得响应。

导入器保留候选框、AX/OCR 来源、焦点字段和屏外候选；有截图时记录文件名与校验值。
它不重新观察当前桌面，也不尝试恢复已经失效的 AX 执行句柄。
同一场景/模板的运行必须使用相同 group ID，在统一数据集组装时保持其划分一致。
`goal_achieved` 是上游模型判断，不作为人工成功标签。无人工标签的日志只报告预测，不报告准确率。

## 复现命令

在 `distill-lab` 下执行；输出必须使用新的目录名：

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_computer_replay.py' -v

.venv/bin/python scripts/prepare_computer_replay.py \
  --output data/computer-use-replay-next

.venv/bin/python scripts/evaluate_computer_replay.py \
  --input data/computer-use-replay-next/development_smoke.jsonl \
  --output results/computer-use-replay-next --device mps

.venv/bin/python scripts/evaluate_computer_replay.py \
  --input data/computer-use-replay-next/train_smoke.jsonl \
  --output results/computer-use-teacher-next --models jev

.venv/bin/python scripts/import_typesafe_replay.py \
  --run /path/to/typesafe-run \
  --output data/imported-desktop-replay \
  --group-id support-search-template --split development
```

Jev 凭据仅从已有环境变量 `TYPESAFE_API_KEY` 读取，不写入协议和日志。
直接使用 `--models laya_base laya_computer` 可以完全离线检查本地两个模型。
训练前还需扩充浏览器任务的动作覆盖，包括等待、滚动、失败恢复和完成判定，
并为每个目标定义独立于模型自评的完成检查。新的浏览器协议不能直接复用本版教师响应。
