#!/usr/bin/env bash
# 15x15 gomoku (5 in a row) self-play training — local M4 machine, CPU
# Usage:  ./run_15x15.sh                 # defaults below
#         ./run_15x15.sh --n_playout 800 # any train.py flag can be appended
set -u
cd "$(dirname "$0")"
exec ./.venv/bin/python train.py \
  --board_width 15 --board_height 15 --n_in_row 5 \
  --n_playout "${N_PLAYOUT:-400}" \
  --value_target "${VALUE_TARGET:-q}" \
  --game_batch_num 1500 \
  --check_freq 100 \
  --batch_size 512 \
  --epochs 5 \
  --pure_mcts_playout_num 100 \
  "$@"
