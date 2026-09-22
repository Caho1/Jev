# Jev Snake

复刻参考视频的 24×16 贪吃蛇和实时决策面板，无前端依赖。

## 启动

```sh
cd /Users/bystanders/Desktop/pythonproject/Jev
python3 snake/server.py --ask-key
```

在隐藏输入提示里输入 TypeSafe 密钥。也可通过 `TYPESAFE_API_KEY` 环境变量启动 `python3 snake/server.py`。密钥仅存在进程内存，不写文件、不发送到浏览器。访问 http://127.0.0.1:8877 。不提供密钥时仍可使用本地模式。

- Jev 流畅模式：默认目标 20 格/秒，后台请求模型方向概率，本地 cycle 安全层逐步验证和执行；不是每步重新推理。
- 逐步 Jev：每次新模型返回后执行一格，速度取决于远程 API。
- 本地自动：纯确定性规划，右侧明确显示启发式权重。
- 手动：方向键或 WASD，空格暂停，R 重开。自动模式 ↑/↓ 调速。
- API 出错停止请求；流畅模式继续本地执行，点击重连恢复。逐步模式等待恢复。

模型请求复用 HTTPS 连接。概率来自 Jev 原始响应；API ROUND TRIP 包含网络。执行方向旁的 * 表示与最近模型建议不同。安全干预计数包含流畅模式中对旧建议的纠正。方向概率是最近一次采样状态的概率，未声称它代表当前棋盘的新推理。BLOCKED DIRECTIONS 和 FREE BOARD 是棋盘确定性统计，不是模型估计。

## 验证

`node snake/test-engine.mjs`：验证环路、5 个种子完整通关、碰撞、增长、食物不与蛇重叠。

## 参考

参考 [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx) 的蛇演示、紧凑方向特征和显式 cycle 安全层设计。原项目每步本地 MLX 推理；本项目使用远程 Jev，两者速度不可等同。没有复制它的模型权重、视频或记录冒充 Jev 结果。

## Laya × Jev 同页对比

首页现在为双模型对比。启动方式、指标定义和模型版本见 [COMPARISON.md](COMPARISON.md)。双击 `start-comparison.command` 可同时启动 Jev 与本机 Laya 服务。原来的单机版保留在 `/single.html`。
