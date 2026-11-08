#!/usr/bin/env python3
"""堵率（修正版）：只统计"对手冲四、只有一个必堵点"的强制局面

三种口径并列：
  net    = policy top1 是否落在必堵点
  net3   = policy top3 是否含必堵点
  mcts   = MCTS(400) 最终选点是否为必堵点
  actual = 自对弈时实际走的那一手是否堵
"""
import json
import numpy as np
import torch

from game import Board
from policy_value_net_pytorch import PolicyValueNet
from mcts_alphaZero import MCTSPlayer

N = 15
AL = "ABCDEFGHIJKLMNO"


def grid(b):
    g = np.zeros((N, N), dtype=int)
    for m, p in b.states.items():
        g[int(m) // N, int(m) % N] = p
    return g


def four_points(b, player):
    """player 的四连的所有可成五点"""
    g = grid(b); pts = set()
    for r in range(N):
        for c in range(N):
            if g[r, c] != player:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + k * dr, c + k * dc) for k in range(4)]
                if any(not (0 <= x < N and 0 <= y < N) for x, y in cells):
                    continue
                if any(g[x, y] != player for x, y in cells):
                    continue
                for er, ec in ((r - dr, c - dc), (r + 3 * dr, c + 3 * dc)):
                    if 0 <= er < N and 0 <= ec < N and g[er, ec] == 0:
                        pts.add(er * N + ec)
    return pts


recs = [json.loads(l) for l in open('logs/games.jsonl') if l.strip()]

pv = PolicyValueNet(N, N, model_file='current_policy.model', use_gpu=False)
net = pv.policy_value_net
net.eval()
mplayer = MCTSPlayer(pv.policy_value_fn, c_puct=5, n_playout=400)

n = pol1 = pol3 = mhit = ahit = 0
rows = []
for r in recs:
    b = Board(width=N, height=N, n_in_row=5); b.init_board(0)
    for idx, (r_, c_) in enumerate(r['moves'][:60]):
        pts = four_points(b, 3 - b.current_player)
        if len(pts) == 1:
            st = np.array(b.current_state(), dtype=np.float32, copy=True)
            with torch.no_grad():
                logp, _ = net(torch.from_numpy(st[None]))
            p = torch.exp(logp)[0].numpy()
            order = np.argsort(p)[::-1]
            mv = mplayer.get_action(b, temp=1e-3)
            actual = r_ * N + c_
            n += 1
            pol1 += int(order[0] in pts); pol3 += int(any(o in pts for o in order[:3]))
            mhit += int(mv in pts); ahit += int(actual in pts)
            if n <= 6:
                f = lambda q: f"{AL[q % N]}{N - q // N}"
                rows.append((r['batch'], idx, f(order[0]), f(mv), f(actual), f(list(pts)[0])))
        if b.game_end()[0]:
            break
        b.do_move(r_ * N + c_)

print(f"强制堵点局面（对手冲四、唯一必堵点）: {n} 个 / {len(recs)} 局")
print(f"  net  top1 命中 {pol1}/{n} = {100*pol1/max(n,1):.1f}%")
print(f"  net  top3 命中 {pol3}/{n} = {100*pol3/max(n,1):.1f}%")
print(f"  MCTS(400) 命中 {mhit}/{n} = {100*mhit/max(n,1):.1f}%")
print(f"  实战走法命中 {ahit}/{n} = {100*ahit/max(n,1):.1f}%")
print(f"{'局':>5}{'手':>4}{'net top1':>10}{'MCTS':>8}{'实走':>8}{'必堵点':>8}")
for row in rows:
    print(f"{row[0]:>5}{row[1]:>4}{row[2]:>10}{row[3]:>8}{row[4]:>8}{row[5]:>8}")
