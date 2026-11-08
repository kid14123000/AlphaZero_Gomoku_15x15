#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# 9×9 五连训练启动器（从零开始，value 目标用 q 标签）
#
#   ./run_9x9.sh --dry-run          # 只打印命令
#   ./run_9x9.sh                    # 正式跑（后台 + caffeinate 防休眠）
#   ./run_9x9.sh --total 5000       # 改目标局数
#   ./run_9x9.sh --sims 800         # 改每手模拟数
#   ./run_9x9.sh --value_target mix # value 目标换档：z / q / mix
#
# 口径：9×9 棋盘 / 五连 / 0.33M 小网（131,132 参数）/ 400 sims / value=q
#
# 为什么五连：四连在 9×9 上防不住（攻方三个子就成活四），实测均手只有 9.5，
# 棋局退化成「先手即胜」，白胜率掉到 35.8%。五连把均手拉回 30-45，双方都有防守空间。
#
# 为什么 value=q：最终胜负 ±1 对同一局面只有一个答案，学不到「险胜 vs 碾压」；
# 改用 MCTS 根节点搜索价值 Q 做软标签，相当于用搜索蒸馏网络。
# ⚠️ 已知风险：从零开始时网络输出接近 0，800 次模拟里只有摸到终局的少数样本给 ±1，
#    Q 被稀释到 |q|≈0.03，value 头可能学不到东西。看 metrics 里 q_absmean 的走势：
#    一直贴在 0.03 不涨 ⇒ 换 `--value_target mix`（q/z 各半）。
#
# 旧谱系归档：
#   legacy/9x9_4/    9×9 四连 1698 局 + 33 个 ckpt
#   legacy/15x15_5/  15×15 五连 8260 代 + 161 个 ckpt（在 ~/AlphaZero_Gomoku 下）
# ─────────────────────────────────────────────────────────────────────────────
set -u
cd "$(dirname "$0")"

PY=${PY:-.venv/bin/python}
SIMS=400
TOTAL=3000
VALUE_TARGET=q
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --sims)         SIMS="$2"; shift 2 ;;
    --total)        TOTAL="$2"; shift 2 ;;
    --value_target) VALUE_TARGET="$2"; shift 2 ;;
    --dry-run)      DRY=1; shift ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

if pgrep -f "[t]rain.py --board_width" > /dev/null; then
  echo "❌ 已有 train.py 在跑：" >&2
  pgrep -af "[t]rain.py --board_width" >&2
  exit 1
fi

CMD=("$PY" train.py
  --board_width 9 --board_height 9 --n_in_row 5
  --n_playout "$SIMS" \
  --check_freq 50 \
  --batch_size 512 \
  --epochs 5 \
  --game_batch_num "$TOTAL" \
  --batch_offset 0 \
  --init_lr_multiplier 1.0 \
  --value_target "$VALUE_TARGET" \
  --pure_mcts_playout_num 100 \
  --eval_pure 0 \
  --save-games logs/games.jsonl \
  --ckpt_dir checkpoints \
  --metrics_file logs/metrics.jsonl)

echo "──────────────────────────────────────────────"
echo " 棋盘        9×9   连 5 子"
echo " 网络        0.33M 小网（131,132 参数）"
echo " MCTS        ${SIMS} sims / 手"
echo " value 目标  ${VALUE_TARGET}  (z=胜负±1 / q=搜索价值 / mix=各半)"
echo " 目标        ${TOTAL} 局（从零开始）"
echo " 检查点      每 50 局 → checkpoints/ckpt_itN.model"
echo "──────────────────────────────────────────────"

if [ "$DRY" = "1" ]; then
  printf 'DRY-RUN:'; printf ' %q' "${CMD[@]}"; echo
  exit 0
fi

mkdir -p logs checkpoints
# caffeinate -i：防止 Mac 空闲休眠把训练冻住。setsid 不可用于 macOS，
# 用 nohup 即可（进程靠 caffeinate 保活，不依赖启动会话）。
nohup caffeinate -i "${CMD[@]}" > logs/train_9x9_r5.log 2>&1 < /dev/null &
sleep 6
PID=$(pgrep -f "[t]rain.py --board_width 9" | head -1)
if [ -z "$PID" ]; then
  echo "❌ 启动失败，看 logs/train_9x9_r5.log：" >&2
  tail -20 logs/train_9x9_r5.log >&2
  exit 1
fi
echo "已启动 pid=$PID"
echo "看日志: tail -f logs/train_9x9_r5.log"
