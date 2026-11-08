#!/bin/bash
# 把评估对手从纯 MCTS(100) 提到 400，无损接续当前训练
# 用法: bash _restart_pure400.sh        (先确认当前批次与 lr)
set -e
cd "$(dirname "$0")"

BATCH=$(.venv/bin/python -c "
import json
rows=[json.loads(l) for l in open('logs/metrics.jsonl') if l.strip()]
g=[r for r in rows if r.get('type')=='game']
print(max(x['batch'] for x in g))
")
LR=$(.venv/bin/python -c "
import json
rows=[json.loads(l) for l in open('logs/metrics.jsonl') if l.strip()]
u=[r for r in rows if r.get('type')=='update']
print(round(u[-1]['lr_multiplier'],6))
")
echo "接续: batch=$BATCH lr_multiplier=$LR  → 评估对手 400 sims"

# 停旧进程（只按 train.py 精确匹配，避免误杀）
pkill -f "train.py --board_width 15 --board_height 15 --n_in_row 5 --n_playout 400" || true
pkill -f "caffeinate -i .venv/bin/python train.py" || true
sleep 3
ps -eo pid,command | grep "[t]rain.py --board_width" && { echo "❌ 旧进程还在，中止"; exit 1; } || echo "旧进程已停"

caffeinate -i .venv/bin/python train.py \
  --board_width 15 --board_height 15 --n_in_row 5 \
  --n_playout 400 --game_batch_num $((10000 - BATCH)) --batch_offset $BATCH \
  --check_freq 50 --batch_size 512 --epochs 5 \
  --pure_mcts_playout_num 400 \
  --init_model current_policy.model --init_lr_multiplier $LR \
  --metrics_file logs/metrics.jsonl --save-games logs/games.jsonl \
  --ckpt_dir checkpoints > logs/train_10000d.log 2>&1 &
sleep 25
echo "=== 启动横幅 ==="
grep -vE "Warning|^  " logs/train_10000d.log | head -5
