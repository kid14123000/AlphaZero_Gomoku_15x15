#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 断点续训：一条命令从「最后一个检查点」继续跑到目标局数
#
#   ./resume.sh                      # 从最新 ckpt 继续，目标仍取上次的 10000
#   ./resume.sh --total 20000        # 换一个更大的目标
#   ./resume.sh --dry-run            # 只打印将要执行的命令，不启动
#
# 自动推导的三个关键量（不要手填，手填必错）：
#   1. batch_offset   = 最新检查点的局数  → 保证曲线 x 轴连续
#   2. init_lr_multiplier = metrics.jsonl 里最后一条 update 的值 → 保住自适应 lr
#   3. ckpt_it{N}.model → 用它当 init_model，而不是 current_policy.model（可能已损坏）
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"

TOTAL=""
DRY=0
EXTRA=()

while [ $# -gt 0 ]; do
  case "$1" in
    --total)   TOTAL="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *)         EXTRA+=("$1"); shift ;;
  esac
done

METRICS=logs/metrics.jsonl
PY=.venv/bin/python

# ── 1. 最新检查点（按数字排序，不能用 ls -t：mtime 会被同步/拷贝打乱）
CKPT=""
N=0
if [ -d checkpoints ]; then
  for f in $(ls checkpoints/ckpt_it*.model 2>/dev/null | \
             sed -E 's/.*ckpt_it([0-9]+)\.model/\1 &/' | sort -n | awk '{print $2}'); do
    CKPT="$f"
    N=$(echo "$f" | sed -E 's/.*ckpt_it([0-9]+)\.model/\1/')
  done
fi

# 没有归档检查点 → 退回到 current_policy.model，局数从 metrics 里数
if [ -z "$CKPT" ]; then
  if [ ! -f current_policy.model ]; then
    echo "❌ 既没有 checkpoints/ckpt_it*.model 也没有 current_policy.model，无法续训" >&2
    exit 1
  fi
  CKPT=current_policy.model
  N=$($PY -c "
import json
n=0
for line in open('$METRICS'):
    try: r=json.loads(line)
    except ValueError: continue
    if r.get('type')=='game': n=max(n,r.get('batch',0))
print(n)
" 2>/dev/null || echo 0)
  echo "⚠️  无归档检查点，退回 current_policy.model（局数 ${N}）"
fi

# ── 2. 自适应 lr 倍数：取最后一条 update 记录
LR=$($PY -c "
import json
lr=None
try:
    for line in open('$METRICS'):
        try: r=json.loads(line)
        except ValueError: continue
        if r.get('type')=='update': lr=r.get('lr_multiplier')
except FileNotFoundError:
    pass
print(lr if lr is not None else 1.0)
")

# ── 3. 目标局数：默认沿用上一次 start 记录里的目标
if [ -z "$TOTAL" ]; then
  TOTAL=$($PY -c "
import json
t=10000
try:
    for line in open('$METRICS'):
        try: r=json.loads(line)
        except ValueError: continue
        if r.get('type')=='start': t=r.get('game_batch_num') or t
except FileNotFoundError:
    pass
print(t)
")
fi

if [ "$N" -ge "$TOTAL" ]; then
  echo "✅ 已完成 $N / $TOTAL 局，无需续跑（要加量用 --total N）"
  exit 0
fi

# ── 4. 单局耗时：用最近 40 局的实测中位数，别写死常数
#    （手数会变：40 手/局约 12s，14 手/局约 5s，写死会让 ETA 差 2~3 倍）
SEC_PER_GAME=$($PY -c "
import json, statistics
ts=[]
try:
    for line in open('$METRICS'):
        try: r=json.loads(line)
        except ValueError: continue
        if r.get('type')=='game': ts.append((r['t'], r.get('batch',0)))
except FileNotFoundError:
    pass
ts.sort()
ts=ts[-41:]
d=[ts[i][0]-ts[i-1][0] for i in range(1,len(ts))
   if ts[i][1]==ts[i-1][1]+1 and 0 < ts[i][0]-ts[i-1][0] < 120]
print('%.2f' % (statistics.median(d) if d else 12.0))
" 2>/dev/null || echo 12.0)

CMD=("$PY" train.py
  --board_width 15 --board_height 15 --n_in_row 5
  --n_playout 400 --check_freq 50 --batch_size 512 --epochs 5
  --pure_mcts_playout_num 100
  --eval_pure 0                      # 对纯 MCTS 评估已饱和(长期100%)，关闭省机时
  --save-games logs/games.jsonl      # 白棋开局面板靠它，丢了 dashboard 会没数据
  --ckpt_dir checkpoints
  --game_batch_num "$TOTAL" --batch_offset "$N"
  --init_model "$CKPT" --init_lr_multiplier "$LR"
  --metrics_file "$METRICS")

echo "──────────────────────────────────────────────"
echo " 从检查点   : $CKPT  (已完成 $N / $TOTAL 局)"
echo " 恢复 lr×   : $LR"
echo " 本次要跑   : $((TOTAL - N)) 局"
echo " 纯MCTS评估 : 关闭（改看 Rapfi 锚点）"
echo " 单局耗时   : ${SEC_PER_GAME}s（最近 40 局实测中位）"
$PY -c "
rem = $TOTAL - $N
spg = float('$SEC_PER_GAME')
print(' 预计耗时   : 约 %.1f 小时（无评估开销）' % (rem * spg / 3600))
"
echo "──────────────────────────────────────────────"

if [ "$DRY" = "1" ]; then
  printf 'DRY-RUN:'; printf ' %q' "${CMD[@]}" ${EXTRA[@]+"${EXTRA[@]}"}; echo
  exit 0
fi

if ps -Ao command | grep -q "[t]rain.py --board_width"; then
  echo "❌ 已有 train.py 在跑，先停掉再续训（避免两个进程抢写同一个 model）" >&2
  exit 1
fi

exec caffeinate -i "${CMD[@]}" ${EXTRA[@]+"${EXTRA[@]}"}
