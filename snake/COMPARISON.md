# Laya × Jev 同页速度对比

## 打开与重启

页面：http://127.0.0.1:8877/

双击 `start-comparison.command`，或在终端执行：

```sh
cd /Users/bystanders/Desktop/pythonproject/Jev/snake
./start-comparison.command
```

在隐藏提示中输入 TypeSafe 密钥；不保存密钥。Laya 权重已缓存，启动脚本使用离线加载。退出启动终端会停止服务。已运行时不要重复启动同一端口。

首次安装：Python 3.11+ 创建 `.venv`，`pip install laya-mlx==0.1.0`，运行 `load_laya.py` 下载并验证原始模型。

## 两种测试

- **30 秒竞速**：同时开始，两边每步等待一次真实模型推理，各自以最快速度前进。相同初始状态和食物随机种子，模型决策不同会导致后续棋盘不同。没有后台算法代跑。
- **同题 20 轮**：确定性生成同一组 20 个棋盘，逐轮将完全相同的方向特征和问题发给两个模型。没有结果缓存。预热不纳入统计，失败会保留并停止测试，不自动重试。中位数比率仅使用成对成功样本。

两者使用同一 cycle 安全层；原始方向不安全时，执行模型概率最高的安全方向。安全层不替代模型调用。

## 统计含义

- Laya `PREDICT`：Agent.predict 输入处理、分词、GPU 计算、输出处理的墙钟时间。
- Jev `API ROUND TRIP`：TypeSafe HTTPS 请求至完整响应，含网络延迟，复用连接。
- 浏览器 `E2E P50`：浏览器发出请求到解析本机服务响应，包含本机 HTTP 与排队。
- P50/P95：nearest-rank 分位数；只统计成功请求。
- DECISIONS/S：竞速成功决策数 / 本轮实际运行时间；同题测速不显示这个值，避免将成对等待时间混入模型速度。
- 模型加载与预热排除，默认不启用编译优化、不缓存推理结果。

该比较是 **Apple M5 Pro 本机 MLX FP16 与远程 Jev API 部署路径的端到端比较**，不是相同硬件上的纯模型速度比较。20 轮适合演示，不代表广泛工作负载。网页会明确显示上述区别。

## 模型和输入

Laya：`convaiinnovations/laya`，English 421M，原始权重 revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`；MLX 0.32.2，laya-mlx 0.1.0，FP16，GPU，batch_size=1。Jev：`jev-1.13.0`。

相同问题：`Choose the best safe move toward food.`，4 个方向的 choice。选项含碰撞、安全、吃食物、沿环路最佳进展等确定性特征。不是从原始像素学习贪吃蛇策略。

点击「导出结果 JSON」可保存逐条输入、原始概率、耗时、执行方向、种子、模型 revision、硬件和失败记录。

页面来源链接：[Laya 模型](https://huggingface.co/convaiinnovations/laya)、[MLX runtime](https://github.com/mizorewww/laya-mlx)。原有单机页保留在 `/single.html`。

## 本机验收结果

2026-09-20，Apple M5 Pro，20/20 同题样本成功：

| 路径 | P50 | P95 |
| --- | ---: | ---: |
| Laya 本机 MLX | 15.298 ms | 19.343 ms |
| Jev API 含网络 | 298.068 ms | 328.260 ms |

P50 比率 19.48×。原始记录保存在 `benchmark-paired.json`；不是对其它硬件、网络或模型任务的性能承诺。
