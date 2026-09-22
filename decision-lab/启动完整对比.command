#!/bin/zsh
set -e
cd -- "${0:A:h}"
[ ! -r "$HOME/.config/typesafe/env" ] || source "$HOME/.config/typesafe/env"
if [[ -z "$TYPESAFE_API_KEY" ]]; then
  read -s 'TYPESAFE_API_KEY?请输入 TypeSafe API key：'
  print
  export TYPESAFE_API_KEY
fi
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
../snake/.venv/bin/python run.py --models english typed jev --load --resume-load
../snake/.venv/bin/python report.py
read '?测试结束，按回车关闭窗口…'
