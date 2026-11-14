#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""9×9 对 Rapfi 评测（随机首手版）— 统计【胜/和/负】三列

为什么单列和棋：9×9 五连很容易满盘 81 手无连五（双方都堵死），
把和棋算成"非胜"会严重低估水平。

用法: python eval_rapfi_open.py <权重> [局数] [depth] [sims] [seed]
"""
import os
import random
import sys
import time

import numpy as np

PROJ = os.path.expanduser("~/AlphaZero_Gomoku_15x15_v4")
sys.path.insert(0, PROJ)
from game import Board, Game                              # noqa: E402
from mcts_alphaZero import MCTSPlayer                    # noqa: E402
from rapfi_player import RapfiPlayer                     # noqa: E402
from policy_value_net_pytorch import PolicyValueNet      # noqa: E402

N, ROW = 9, 5
ckpt = sys.argv[1]
NG = int(sys.argv[2]) if len(sys.argv) > 2 else 20
DEPTH = int(sys.argv[3]) if len(sys.argv) > 3 else 5
NSIM = int(sys.argv[4]) if len(sys.argv) > 4 else 400
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 7

pv = PolicyValueNet(N, N, model_file=ckpt, use_gpu=False)
pv.policy_value_net.eval()
BIN = os.path.expanduser("~/rapfi_run/pbrain-rapfi")
rng = random.Random(SEED)
CENTER = [(r, c) for r in range(3, 6) for c in range(3, 6)]


class FirstMove:
    """只改第一手（天元 3×3 随机），其余委托原 player，保持 game.start_play 原循环"""

    def __init__(self, base, mv):
        self.b, self.mv, self.used = base, mv, False

    def set_player_ind(self, p):
        self.b.set_player_ind(p)

    def reset_player(self):
        self.b.reset_player()

    def get_action(self, board):
        if not self.used and len(board.states) == 0:
            self.used = True
            return self.mv
        return self.b.get_action(board)


# [胜, 和, 负]
stat = {"B": [0, 0, 0], "W": [0, 0, 0]}
lens = []
t0 = time.time()
print("9x9/5连 | net %d sims vs Rapfi d%d | %d 局 | 首手随机(天元3x3) seed=%d"
      % (NSIM, DEPTH, NG, SEED), flush=True)

for i in range(NG):
    net_black = (i % 2 == 0)
    fm = rng.choice(CENTER)
    mv = fm[0] * N + fm[1]
    board = Board(width=N, height=N, n_in_row=ROW)
    game = Game(board)
    npl = MCTSPlayer(pv.policy_value_fn, c_puct=5, n_playout=NSIM)
    rap = RapfiPlayer(binary=BIN, depth=DEPTH, size=N)
    try:
        if net_black:
            w = game.start_play(FirstMove(npl, mv), rap, start_player=0, is_shown=0)
        else:
            w = game.start_play(FirstMove(rap, mv), npl, start_player=0, is_shown=0)
    finally:
        rap.close()

    key = "B" if net_black else "W"
    if w == -1:
        stat[key][1] += 1
        res = "和"
    elif (w == 1 and net_black) or (w == 2 and not net_black):
        stat[key][0] += 1
        res = "胜"
    else:
        stat[key][2] += 1
        res = "负"
    lens.append(len(board.states))
    b, ww = stat["B"], stat["W"]
    print("  局%2d %s 首手%s %s 手数%2d | 黑 %d胜%d和%d负 | 白 %d胜%d和%d负"
          % (i + 1, "执黑" if net_black else "执白",
             "ABCDEFGHI"[fm[1]] + str(fm[0]), res, len(board.states),
             b[0], b[1], b[2], ww[0], ww[1], ww[2]), flush=True)

b, w = stat["B"], stat["W"]
tot_w, tot_d, tot_l = b[0] + w[0], b[1] + w[1], b[2] + w[2]
tot = NG
print("\n=== %d 局 | 胜 %d (%.0f%%) / 和 %d (%.0f%%) / 负 %d (%.0f%%) | 不败率 %.0f%% ==="
      % (tot, tot_w, 100 * tot_w / tot, tot_d, 100 * tot_d / tot, tot_l, 100 * tot_l / tot,
         100 * (tot_w + tot_d) / tot))
print("  执黑: %d胜 %d和 %d负 | 执白: %d胜 %d和 %d负"
      % (b[0], b[1], b[2], w[0], w[1], w[2]))
print("  平均手数 %.1f | 用时 %.1f 分钟" % (np.mean(lens), (time.time() - t0) / 60))
