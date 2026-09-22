#!/bin/zsh
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  print '缺少本地环境，请先按 README.md 安装 laya-mlx。'
  exit 1
fi
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 .venv/bin/python laya_server.py &
worker_pid=$!
trap 'kill "$worker_pid" 2>/dev/null' EXIT INT TERM
.venv/bin/python server.py --ask-key
