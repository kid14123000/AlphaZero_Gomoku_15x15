#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 9×9 五连 · 续训到 10 万局
#
#   ./resume_9x9.sh --dry-run              # 只打印要跑的命令，逐项核对
#   ./resume_9x9.sh                        # 按默认 10 万局续跑
#   ./resume_9x9.sh --total 200000         # 改目标
#   ./resume_9x9.sh --sims 800             # 改每手模拟数
#
# 续训三件套全部自动推导，不手填：
#   1. --batch_offset      = 最新 ckpt 的代数（全局局号，不是「还要跑多少局」）
#   2. --init_model        = 最新 ckpt_itN.model（不用 current_policy.model，那个是覆盖写）
#   3. --init_lr_multiplier= metrics.jsonl 里最后一条 update 的 lr_multiplier
#                            （这个值不存进权重文件，不恢复就重置成 1.0 ⇒ 学习率瞬间 ×5）
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
TOTAL=100000
SIMS=400
CKPT_EVERY=200
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --total)  TOTAL="$2"; shift 2 ;;
    --sims)   SIMS="$2"; shift 2 ;;
    --ckpt)   CKPT_EVERY="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

if pgrep -f "[t]rain.py --board_width" > /dev/null; then
  echo "❌ 已有 train.py 在跑，先停干净：" >&2
  pgrep -af "[t]rain.py --board_width" >&2
  exit 1
fi

# ── 1. 找最新 ckpt（按数字排序，不能按文件名/mtime） ──
CKPT=$(ls checkpoints 2>/dev/null | grep -E '^ckpt_it[0-9]+\.model$' \
       | sed 's/ckpt_it//;s/\.model//' | sort -n | tail -1)
if [ -z "${CKPT:-}" ]; then
  echo "❌ checkpoints/ 里没有 ckpt_itN.model，无法续训" >&2
  exit 1
fi
OFFSET="$CKPT"

# ── 2. 确认 ckpt 文件完整（大小 > 100KB，防止上次写坏） ──
SZ=$(stat -f%z "checkpoints/ckpt_it${CKPT}.model" 2>/dev/null || stat -c%s "checkpoints/ckpt_it${CKPT}.model")
if [ "${SZ:-0}" -lt 100000 ]; then
  echo "❌ ckpt_it${CKPT}.model 只有 ${SZ}B，疑似写坏了；退回上一个" >&2
  CKPT=$(ls checkpoints | grep -E '^ckpt_it[0-9]+\.model$' | sed 's/ckpt_it//;s/\.model//' \
         | sort -n | tail -2 | head -1)
  OFFSET="$CKPT"
  [ -z "${CKPT:-}" ] && exit 1
fi

# ── 3. 从 metrics 里取最后的 lr_multiplier ──
LR=$($PY - "$OFFSET" <<'PY'
import json, sys
off = int(sys.argv[1])
lr = None
try:
    for line in open('logs/metrics.jsonl'):
        r = json.loads(line)
        if r.get('type') == 'update':
            lr = r.get('lr_multiplier')
except IOError:
    pass
print(lr if lr is not None else 1.0)
PY
)

echo "──────────────────────────────────────────────"
echo " 棋盘        9×9   连 5 子"
echo " 续训起点    第 ${OFFSET} 局（ckpt_it${OFFSET}.model，${SZ} B）"
echo " 目标        ${TOTAL} 局  → 还要跑 $((TOTAL - OFFSET)) 局"
echo " lr_multiplier ${LR}（从 metrics 恢复，不重置成 1.0）"
echo " MCTS        ${SIMS} sims / 手"
echo " 检查点      每 ${CKPT_EVERY} 局"
echo "──────────────────────────────────────────────"

CMD=("$PY" train.py
  --board_width 9 --board_height 9 --n_in_row 5
  --n_playout "$SIMS" \
  --check_freq "$CKPT_EVERY" \
  --batch_size 512 \
  --epochs 5 \
  --game_batch_num "$TOTAL" \
  --batch_offset "$OFFSET" \
  --init_model "checkpoints/ckpt_it${OFFSET}.model" \
  --init_lr_multiplier "$LR" \
  --value_target q \
  --pure_mcts_playout_num 100 \
  --eval_pure 0 \
  --save-games logs/games.jsonl \
  --ckpt_dir checkpoints \
  --metrics_file logs/metrics.jsonl)

if [ "$DRY" = "1" ]; then
  printf 'DRY-RUN:'; printf ' %q' "${CMD[@]}"; echo
  exit 0
fi

nohup caffeinate -i "${CMD[@]}" >> logs/train_9x9_r5.log 2>&1 < /dev/null &
sleep 8
PID=$(pgrep -f "[t]rain.py --board_width 9" | head -1)
if [ -z "$PID" ]; then
  echo "❌ 启动失败，看 logs/train_9x9_r5.log：" >&2
  tail -25 logs/train_9x9_r5.log >&2
  exit 1
fi
echo "已启动 pid=$PID"
echo "看日志: tail -f logs/train_9x9_r5.log"
