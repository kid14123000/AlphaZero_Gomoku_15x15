#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 9×9 五连 · 并行续训（1 主进程 + N 自对弈工兵）
#
#   ./resume_parallel_9x9.sh --dry-run     # 只打印命令，逐项核对
#   ./resume_parallel_9x9.sh               # 默认 5 工兵 / 400 sims / 10 万局
#   ./resume_parallel_9x9.sh --total 200000 --workers 5
#
# 为什么并行：MCTS 是单线程的，单进程只吃 1.2 核，机器有 10 逻辑核（4 性能核）。
# 实测（M4，9×9，400 sims，训练好的权重）：
#     单进程  18.4 局/分     ← 3.3s/局
#     4 工兵  43.4 局/分
#     5 工兵  52.2 局/分     ← 最优（4 性能核 + 1 能效核）
#     6 工兵  47.4 局/分     ← 抢核开始回退
#
# 续训三件套自动推导（不手填，填错就静默出问题）：
#   --batch_offset        最新 ckpt 的代数（全局局号，不是「还要跑多少局」）
#   --init_model          最新 ckpt_itN.model（不用 current_policy.model，它是覆盖写）
#   --init_lr_multiplier  metrics.jsonl 最后一条 update 的 lr_multiplier
#                         （此值不存进权重；不恢复会重置成 1.0 ⇒ 学习率瞬间 ×5）
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
TOTAL=100000
WORKERS=5
SIMS=400
CKPT_EVERY=200
PUBLISH_EVERY=10
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --total)   TOTAL="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --sims)    SIMS="$2"; shift 2 ;;
    --ckpt)    CKPT_EVERY="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

if pgrep -f "[p]arallel_train.py" > /dev/null || pgrep -f "[t]rain.py --board_width" > /dev/null; then
  echo "❌ 已有训练进程在跑：" >&2
  pgrep -af "[p]arallel_train.py|[t]rain.py --board_width" >&2
  exit 1
fi
if pgrep -f "[w]orker_selfplay.py" > /dev/null; then
  echo "❌ 还有上次残留的工兵：" >&2
  pgrep -af "[w]orker_selfplay.py" >&2
  exit 1
fi

CKPT=$(ls checkpoints 2>/dev/null | grep -E '^ckpt_it[0-9]+\.model$' \
       | sed 's/ckpt_it//;s/\.model//' | sort -n | tail -1)
if [ -z "${CKPT:-}" ]; then
  echo "❌ checkpoints/ 里没有 ckpt_itN.model" >&2
  exit 1
fi
SZ=$(stat -f%z "checkpoints/ckpt_it${CKPT}.model" 2>/dev/null \
     || stat -c%s "checkpoints/ckpt_it${CKPT}.model")
if [ "${SZ:-0}" -lt 100000 ]; then
  echo "❌ ckpt_it${CKPT}.model 只有 ${SZ}B，疑似写坏；用上一个" >&2
  CKPT=$(ls checkpoints | grep -E '^ckpt_it[0-9]+\.model$' | sed 's/ckpt_it//;s/\.model//' \
         | sort -n | tail -2 | head -1)
  [ -z "${CKPT:-}" ] && exit 1
  SZ=$(stat -f%z "checkpoints/ckpt_it${CKPT}.model" 2>/dev/null \
       || stat -c%s "checkpoints/ckpt_it${CKPT}.model")
fi
OFFSET="$CKPT"

LR=$($PY -c "
import json
lr=None
for line in open('logs/metrics.jsonl'):
    r=json.loads(line)
    if r.get('type')=='update': lr=r.get('lr_multiplier')
print(lr if lr is not None else 1.0)
")

echo "──────────────────────────────────────────────"
echo " 棋盘         9×9   连 5 子"
echo " 续训起点     第 ${OFFSET} 局（ckpt_it${OFFSET}.model，${SZ} B）"
echo " 目标         ${TOTAL} 局  → 还要 $((TOTAL - OFFSET)) 局"
echo " 吞吐预期     约 52 局/分（5 工兵） → ETA 约 $(( (TOTAL - OFFSET) / 52 / 60 )) 小时"
echo " lr_multiplier ${LR}"
echo " 工兵         ${WORKERS}   MCTS ${SIMS} sims / 手   value=q"
echo " 检查点       每 ${CKPT_EVERY} 局"
echo "──────────────────────────────────────────────"

CMD=("$PY" parallel_train.py
  --workers "$WORKERS" \
  --board_width 9 --board_height 9 --n_in_row 5 \
  --n_playout "$SIMS" \
  --game_batch_num "$TOTAL" \
  --check_freq "$CKPT_EVERY" \
  --publish_every "$PUBLISH_EVERY" \
  --batch_size 512 \
  --epochs 5 \
  --value_target q \
  --init_model "checkpoints/ckpt_it${OFFSET}.model" \
  --batch_offset "$OFFSET" \
  --init_lr_multiplier "$LR" \
  --save-games logs/games.jsonl \
  --ckpt_dir checkpoints \
  --metrics_file logs/metrics.jsonl)

if [ "$DRY" = "1" ]; then
  printf 'DRY-RUN:'; printf ' %q' "${CMD[@]}"; echo
  exit 0
fi

rm -rf sync queue
mkdir -p logs checkpoints sync queue
nohup caffeinate -i "${CMD[@]}" >> logs/train_parallel_9x9.log 2>&1 < /dev/null &
sleep 10
PID=$(pgrep -f "[p]arallel_train.py" | head -1)
if [ -z "${PID:-}" ]; then
  echo "❌ 启动失败，看日志：" >&2
  tail -30 logs/train_parallel_9x9.log >&2
  exit 1
fi
NW=$(pgrep -f "[w]orker_selfplay.py" | wc -l | tr -d ' ')
# 变量用 ${} 包起来：写成 "$PID，工兵" 时中文逗号会被 sh 当成变量名的一部分
# （实测报 "PID?: unbound variable"）
echo "已启动主进程 pid=${PID}，工兵 ${NW} 个"
echo "看日志: tail -f logs/train_parallel_9x9.log"
