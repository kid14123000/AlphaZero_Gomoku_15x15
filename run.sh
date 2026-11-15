#!/bin/bash
# ============================================================================
# v4 15×15 训练：唯一启动入口
#   bash run.sh            # 正常启动
#   bash run.sh --dry      # 只打印解析后的配置，不启动
#   bash stop.sh           # 停止
#
# 设计原则：所有默认值只在这里定义一次；任何一项都可用环境变量覆盖。
#   WORKERS=6 SIMS=400 bash run.sh     ← 临时改单项
# ============================================================================
set -uo pipefail

PROJ=$(cd "$(dirname "$0")" && pwd)
cd "$PROJ" || exit 1
DRY=0
[ "${1:-}" = "--dry" ] && DRY=1

# ── 1. 线程 / 课程学习环境变量（唯一真相源，缺一个补偿就静默失效）────────────
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export GOMOKU_CLASS_WEIGHT_FILE=${GOMOKU_CLASS_WEIGHT_FILE:-sync/class_weight.json}
export GOMOKU_WW_MIN_LEN=${GOMOKU_WW_MIN_LEN:-20}
export GOMOKU_WW_FILE=${GOMOKU_WW_FILE:-sync/ww_reservoir.npz}
export GOMOKU_WW_RESERVOIR=${GOMOKU_WW_RESERVOIR:-24000}

# ── 2. 训练参数（默认值 = 当前主线配方）─────────────────────────────────────
WORKERS=${WORKERS:-8}
BOARD=${BOARD:-15}
ROW=${ROW:-5}
SIMS=${SIMS:-800}
C_PUCT=${C_PUCT:-1.25}
EPS=${EPS:-0.25}
ALPHA=${ALPHA:-0.03}
VALUE_TARGET=${VALUE_TARGET:-z}
PUBLISH_EVERY=${PUBLISH_EVERY:-40}
CHECK_FREQ=${CHECK_FREQ:-50}
BATCH_SIZE=${BATCH_SIZE:-512}
EPOCHS=${EPOCHS:-5}
GAME_BATCH_NUM=${GAME_BATCH_NUM:-100000}
SEED_MODEL=${SEED_MODEL:-}          # 空 = 自动取 checkpoints/ 里最新的
CKPT_DIR=${CKPT_DIR:-checkpoints}
METRICS=${METRICS:-logs/metrics.jsonl}
GAMES=${GAMES:-logs/games.jsonl}
ARCHIVE_OLD=${ARCHIVE_OLD:-1}       # 1 = 启动前把旧日志归档
START_CTL=${START_CTL:-1}           # 1 = 同时拉起课程学习控制器

# ── 3. 前置检查 ─────────────────────────────────────────────────────────────
if [ "$DRY" = "0" ] && pgrep -f "\.venv/bin/python -u parallel_train" >/dev/null 2>&1; then
  echo "❌ 已有训练在跑（PID: $(pgrep -f '\.venv/bin/python -u parallel_train' | tr '\n' ' ')）"
  echo "   先执行： bash stop.sh"
  exit 1
fi

[ -x .venv/bin/python ] || { echo "❌ 找不到 .venv/bin/python"; exit 1; }
[ -f "$GOMOKU_CLASS_WEIGHT_FILE" ] || echo "{\"B_win\":0.35,\"B_lose\":0.10,\"W_win\":0.22,\"W_lose\":0.33}" > "$GOMOKU_CLASS_WEIGHT_FILE"

# 种子权重：默认取**代次最大**的检查点
# 🚨 不能用 ls -t：mtime 会被 scp/拷贝/同步打乱，实测过一次 ls -t 选中旧代次、
#    训练直接从 it16098 退回 it15750（白丢 348 局）。一律按文件名里的数字排序。
if [ -z "$SEED_MODEL" ]; then
  SEED_MODEL=$(ls "$CKPT_DIR"/ckpt_it*.model 2>/dev/null \
               | sed -E 's/.*ckpt_it([0-9]+)\.model/\1 &/' | sort -n | tail -1 | awk '{print $2}')
  [ -z "$SEED_MODEL" ] && { echo "❌ $CKPT_DIR 里没有检查点，请用 SEED_MODEL=... 指定"; exit 1; }
fi
[ -f "$SEED_MODEL" ] || { echo "❌ 种子权重不存在: $SEED_MODEL"; exit 1; }
OFFSET=$(basename "$SEED_MODEL" | sed -E 's/.*ckpt_it([0-9]+)\.model/\1/')
case "$OFFSET" in ''|*[!0-9]*) echo "❌ 无法从 $SEED_MODEL 解析 batch 号"; exit 1 ;; esac

# ── 续训 lr：从 metrics 里最后一条 update 取，**绝不用默认 1.0** ──────────────
# 不传 --init_lr_multiplier 时 argparse 默认 1.0，而稳态是 0.0878（12 倍）。
# 实测后果：首更 lr 冲到 0.667、KL 0.8~1.8、policy 熵从 2.25 被推到 3.4~4.1，
# 要 5~6 次 update 才压回来 —— 这段是「被打伤的网络上打的」，指标不可用。
# 已在 07:34 与 09:22 两次踩到，所以这里强制推导，并允许 LR=... 覆盖。
# 注意：ARCHIVE_OLD 会把 $METRICS 清空，所以当前文件没有 update 时，
#       回落到**最新归档目录**里的 metrics.jsonl（run.sh 自己刚归档那份也在内）。
if [ -z "${LR:-}" ]; then
  LR=$(.venv/bin/python -c "
import glob, json, os
def last_lr(path):
    v = None
    try:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('type') == 'update' and r.get('lr_multiplier'):
                v = float(r['lr_multiplier'])
    except (FileNotFoundError, OSError):
        pass
    return v

v = None
# 优先用**归档**里的最后一条（= 上一次跑完时的稳态 lr，通常是 0.0878 这类平台值）；
# 当前 live 文件可能刚重启、只跑了一两次更新（lr 还在 0.667→0.444 的下降途中），
# 拿它当起点会人为制造一次冲击。
cands = sorted(glob.glob(os.path.expanduser('~/archive/*/metrics.jsonl')),
               key=lambda p: os.path.getmtime(p), reverse=True)
for p in cands:
    v = last_lr(p)
    if v is not None:
        break
if v is None:
    v = last_lr('$METRICS')
print('%.12g' % (v if v is not None else 1.0))
" 2>/dev/null || echo 1.0)
fi

# 白赢储库：文件不存在但有棋谱 → 自动重建
if [ ! -f "$GOMOKU_WW_FILE" ] && [ -f "$GAMES" ] && [ -f tools/rebuild_ww.py ]; then
  echo "… 白赢储库缺失，从棋谱重建"
  .venv/bin/python tools/rebuild_ww.py || echo "⚠ 重建失败，将以空储库启动"
fi
WW_N=$([ -f "$GOMOKU_WW_FILE" ] && .venv/bin/python -c "
import numpy as np,sys
try: print(len(np.load('$GOMOKU_WW_FILE')['zs']))
except Exception: print(0)" || echo 0)

echo "=================================================================="
echo " v4 15×15 训练"
echo "  工兵 $WORKERS ｜ ${BOARD}×${BOARD}/$ROW 连 ｜ $SIMS sims ｜ c_puct $C_PUCT"
echo "  根先验噪声 ε=$EPS α=$ALPHA ｜ value 目标 $VALUE_TARGET"
echo "  每 $PUBLISH_EVERY 局发权重 ｜ 每 $CHECK_FREQ 局存检查点 ｜ batch $BATCH_SIZE × $EPOCHS epoch"
echo "  种子 $SEED_MODEL  (batch_offset $OFFSET)"
echo "  续训 lr_multiplier $LR  $([ "$LR" = "1.0" ] && echo '⚠ 取自默认值（无可读 metrics），首更会被 KL 自适应压回来')"
echo "  分层采样 $GOMOKU_CLASS_WEIGHT_FILE ｜ 白赢储库 $GOMOKU_WW_FILE ($WW_N 条)"
echo "  白赢最短手数 $GOMOKU_WW_MIN_LEN ｜ 控制器 START_CTL=$START_CTL"
echo "=================================================================="

if [ "$DRY" = "1" ]; then echo "(--dry 模式，不启动)"; exit 0; fi

# ── 4. 归档旧日志 ───────────────────────────────────────────────────────────
if [ "$ARCHIVE_OLD" = "1" ] && [ -s "$METRICS" ]; then
  TS=$(date +%m%d_%H%M)
  BK="$HOME/archive/v4_run1_logs_$TS"
  mkdir -p "$BK"
  cp -f "$METRICS" "$GAMES" logs/*.log "$BK"/ 2>/dev/null
  : > "$METRICS"; : > "$GAMES"
  echo "旧日志归档: $BK"
fi

# ── 5. 启动主进程 ───────────────────────────────────────────────────────────
nohup .venv/bin/python -u parallel_train.py \
  --workers "$WORKERS" \
  --board_width "$BOARD" --board_height "$BOARD" --n_in_row "$ROW" \
  --n_playout "$SIMS" --game_batch_num "$GAME_BATCH_NUM" \
  --check_freq "$CHECK_FREQ" --publish_every "$PUBLISH_EVERY" \
  --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" \
  --value_target "$VALUE_TARGET" \
  --dirichlet_eps "$EPS" --dirichlet_alpha "$ALPHA" \
  --handicap_file sync/handicap.json \
  --init_model "$SEED_MODEL" --batch_offset "$OFFSET" \
  --init_lr_multiplier "$LR" \
  --save-games "$GAMES" --ckpt_dir "$CKPT_DIR" --metrics_file "$METRICS" \
  > logs/train_v4.log 2>&1 &
sleep 3

if pgrep -f "\.venv/bin/python -u parallel_train" >/dev/null 2>&1; then
  echo "✅ 主进程已起 (PID $(pgrep -f '\.venv/bin/python -u parallel_train' | head -1))"
else
  echo "❌ 主进程启动失败，日志尾部："; tail -20 logs/train_v4.log; exit 1
fi

# ── 6. 控制器 ───────────────────────────────────────────────────────────────
if [ "$START_CTL" = "1" ]; then
  nohup .venv/bin/python -u curriculum_ctl.py > logs/curriculum_ctl.log 2>&1 &
  sleep 2
  echo "✅ 控制器已起 (PID $(pgrep -f 'python -u curriculum_ctl' | head -1))"
fi

# ── 7. 启动后自检（环境变量是否真进了进程）─────────────────────────────────
sleep 25
P=$(pgrep -f "\.venv/bin/python -u parallel_train" | head -1)
echo "--- 自检 ---"
echo "工兵进程数: $(pgrep -f '[w]orker_selfplay.py' | wc -l | tr -d ' ')"
ps -E -p "$P" 2>/dev/null | tr ' ' '\n' | grep -E "^GOMOKU_" | sed 's/^/  env /'
if ps -E -p "$P" 2>/dev/null | tr ' ' '\n' | grep -q "^GOMOKU_CLASS_WEIGHT_FILE="; then
  echo "✅ 分层采样 env 已注入"
else
  echo "❌ 分层采样 env 缺失（补偿会静默失效！）"
fi
echo "（训练指标见 ${METRICS}；前 40 局后确认 cls_batch 非 [0,0,0,0]）"
