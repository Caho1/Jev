# Laya 与 Jev 社区公开测试复测

完成 431 / 431 输入；每模型 831 个判定；请求失败 0。

| 测试 | Laya 准确率 | Jev 准确率 | Laya P50 | Jev P50 |
| --- | ---: | ---: | ---: | ---: |
| JevBench / easy | 95.8% | 100.0% | 16.0 ms | 292.0 ms |
| JevBench / hard | 33.3% | 71.2% | 35.0 ms | 307.6 ms |
| JevBench / standard | 54.2% | 81.9% | 16.4 ms | 318.0 ms |
| Phishing / subset 100 | 50.0% | 64.0% | 31.4 ms | 309.6 ms |
| Reviews / 5 fields | 72.4% | 96.2% | 70.5 ms | 303.6 ms |

## 结论与限制

Laya 在本机延迟更低，Jev 在本次五组测试中准确率更高。不能用贪吃蛇速度证明通用业务能力或生产可用性。困难题 111 个中 57 个存在 Laya 状态截断，1 个存在指令/选项裁剪（与状态截断重叠）。同一未截断子集 54 题：Laya 15/54，Jev 39/54。评论准确率按 500 字段计算，五项全对的评论比例为 Laya 9%，Jev 88%。

Laya 的 MLX 峰值分配 1.45 GiB，进程峰值 RSS 0.93 GiB；这两个统计口径不同，不应称作所有场景固定 1GB。

## 复现

Python 环境使用 snake/.venv，运行 `HF_HUB_OFFLINE=1 snake/.venv/bin/python snake/community/run.py`，在隐藏输入提示提供 TypeSafe 密钥。脚本拒绝覆盖现有 raw.jsonl；复跑前将现有记录移动到独立归档目录。prepare.py 使用克隆至 /tmp 的三个参考仓库，精确 commit 见 sources.json。corpus.json 已保存本轮固定输入与标签，sha256 见 summary.json。模型永远不接收 expected、group 或 provenance。

原始请求每例顺序执行，两个模型的先后顺序交替，使用同一输入和问题。Jev 延迟包含网络；Laya 使用 MLX FP16、batch_size 16。评论每请求 5 问，其余每请求 1 问。各模型 3 次预热排除，没有失败重试和结果缓存。评分沿用原任务定义，评级字段采用原作者范围/取整规则。测试为公开集单轮测量，改写题相关；未计算跨场景泛化置信区间。

## 来源

- https://github.com/fstandhartinger/jevbench · commit `7ce310c7262ed49cc85853339a8a42459298e3f3`
- https://github.com/mameli/jev-vs-luna · commit `b17401dcd982180e70030dcf1575a645e9f3f9a2`
- https://github.com/anisselbd/jev-phishing-bench · commit `1d56e8c64d029a9554a0874e2ef2901ed196e230`

页面：community.html 概览；community-easy.html / community-standard.html / community-hard.html / community-reviews.html / community-phishing.html 独立页面。Tab 导航、分页、筛选、题目展开已通过内置浏览器验证。
## 实时运行

### Typed-Decisions 复测（2026-09-20）

默认实时模型改为 `convaiinnovations/laya-typed-decisions`，并保留 English 下拉切换。Typed checkpoint revision 为 `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2`，421M 参数，1024 token 上下文、256 token head budget，MLX FP16、batch_size 16。Snake 仍使用原 English 模型。

同一 corpus 的 431 个输入、831 个判定全部重新调用 Typed，无请求失败；完整记录见 `typed-results.json`，复现脚本为 `try_typed.py`。以下 English / Jev 数字复用此前同题历史基线，并非本轮同步复测。

| 测试 | English 准确率 | Typed 准确率 | Jev 历史准确率 | Typed P50 |
| --- | ---: | ---: | ---: | ---: |
| 基础判断 | 95.8% | 97.9% | 100.0% | 12.0 ms |
| 路由与规则 | 54.2% | 52.8% | 81.9% | 11.9 ms |
| 困难决策 | 33.3% | 26.1% | 71.2% | 47.6 ms |
| 评论分类（字段） | 72.4% | 77.4% | 96.2% | 35.3 ms |
| 钓鱼邮件 | 50.0% | 50.0% | 64.0% | 21.2 ms |

Typed 的困难题状态截断为 37/111；上下文增加并没有带来本组整体准确率提高。上述结果是这些公开题目的表现，不能代表所有生产工作流。内置浏览器已分别验证 Typed 与 English 的 5 题真实调用、正确模型身份、版本切换和逐题裁剪信息。

五个独立测试页默认显示实时模式，点击按钮后向 `/api/test` 和本机 Laya worker 的 `/api/test` 发起新调用。请求只传固定题目 ID；服务端读取相同原始输入和问题，不把标准答案发给模型。每轮两边并发，各自返回立即更新；下一题等待两边完成。支持前 5、前 20、全部、停止、导出本轮与查看历史。实时轮次包含首次调用，不套用历史预热统计。

内置浏览器验收：五页各完成 5 个输入，两模型均真实成功返回；20 题运行在第 1 题点击停止，完成在途请求后停在 1/20。每次结果包含新的 request_id 和 timestamp；没有结果缓存或自动失败重试。
