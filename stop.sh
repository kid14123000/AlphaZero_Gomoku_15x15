#!/bin/bash
# 停止 v4 训练（主进程 + 工兵 + 控制器）
PROJ=$(cd "$(dirname "$0")" && pwd)
cd "$PROJ" || exit 1

pkill -f "\.venv/bin/python -u parallel_train" 2>/dev/null
sleep 2
pkill -f "[w]orker_selfplay.py" 2>/dev/null
pkill -f "python -u curriculum_ctl" 2>/dev/null
sleep 3

M=$(pgrep -f "\.venv/bin/python -u parallel_train" | wc -l | tr -d ' ')
W=$(pgrep -f "[w]orker_selfplay.py" | wc -l | tr -d ' ')
C=$(pgrep -f "python -u curriculum_ctl" | wc -l | tr -d ' ')
echo "残留: 主 $M / 工兵 $W / 控制器 $C"
if [ "$M" = "0" ] && [ "$W" = "0" ] && [ "$C" = "0" ]; then
  echo "✅ 已全部停止"
else
  echo "⚠ 仍有残留，请检查（可再跑一次，或 ps -eo pid,command | grep -E 'parallel_train|worker_selfplay|curriculum_ctl'）"
fi
