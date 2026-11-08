#!/usr/bin/env python3
"""补两块拼图：
1) 同执子方内部的价值判别力（AUC/Brier）—— 排除"谁走谁赢"混杂
2) 同样 18 个四连威胁局面，看 MCTS(400) 选点是否堵 —— 区分"网络不会" vs "实战不会"
"""
import json
import numpy as np
import torch

from game import Board
from policy_value_net_pytorch import PolicyValueNet
from mcts_alphaZero import MCTSPlayer

N = 15
AL = "ABCDEFGHIJKLMNO"


def load(path):
    pv = PolicyValueNet(N, N, model_file=path, use_gpu=False)
    pv.policy_value_net.eval()
    return pv.policy_value_net


def fwd(net, X):
    with torch.no_grad():
        p, v = net(torch.from_numpy(np.ascontiguousarray(X)))
    return torch.exp(p).numpy(), v.numpy().ravel()


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    order = allv.argsort().argsort() + 1
    return float((order[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def grid(b):
    g = np.zeros((N, N), dtype=int)
    for m, p in b.states.items():
        g[int(m) // N, int(m) % N] = p
    return g


def threats(b, player):
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
print(f"棋谱 {len(recs)} 局")

# ---------- 1) 同执子方 AUC ----------
X, Z, T = [], [], []
for r in recs:
    b = Board(width=N, height=N, n_in_row=5); b.init_board(0)
    for r_, c_ in r['moves'][:45]:
        X.append(np.array(b.current_state(), dtype=np.float32, copy=True))
        Z.append(1.0 if b.current_player == r['winner'] else -1.0)
        T.append(b.current_player)
        b.do_move(r_ * N + c_)
X = np.asarray(X); Z = np.asarray(Z); T = np.asarray(T)

print("\n=== 价值判别力（按执子方分组） ===")
for tag, path in (('250代', 'checkpoints/ckpt_it250.model'), ('300代', 'current_policy.model')):
    net = load(path)
    _, v = fwd(net, X)
    for who, lab in ((1, '黑走'), (2, '白走')):
        m = T == who
        vv, zz = v[m], Z[m]
        pos, neg = vv[zz > 0], vv[zz < 0]
        print(f"  {tag} {lab}: n={m.sum():4d} AUC {auc(pos,neg):.3f} | "
              f"胜局 v={pos.mean():+.2f}(n={len(pos)}) 负局 v={neg.mean():+.2f}(n={len(neg)}) | "
              f"MAE {np.abs(vv-zz).mean():.3f}")
    print(f"  {tag} 合并(不分组, 含身份混杂): AUC {auc(v[Z>0], v[Z<0]):.3f}")

# ---------- 2) MCTS 是否堵 ----------
print("\n=== MCTS(400 sims) 在四连威胁局面是否堵 ===")
net = load('current_policy.model')
pv_m = PolicyValueNet(N, N, model_file='current_policy.model', use_gpu=False)
player = MCTSPlayer(pv_m.policy_value_fn, c_puct=5, n_playout=400)
tot = hit = 0
for r in recs:
    b = Board(width=N, height=N, n_in_row=5); b.init_board(0)
    for idx, (r_, c_) in enumerate(r['moves'][:45]):
        pts = threats(b, 3 - b.current_player)
        if pts and idx >= 3:
            a = player.get_action(b, temp=1e-3)
            tot += 1
            hit += 1 if a in pts else 0
            if tot <= 5:
                print(f"   选 {AL[a % N]}{N - a // N} | 堵点 {[AL[q % N] + str(N - q // N) for q in list(pts)[:3]]} | {'✓' if a in pts else '✗'}")
        if b.game_end()[0]:
            break
        b.do_move(r_ * N + c_)
print(f"  MCTS 堵率: {hit}/{tot} = {100.0*hit/max(tot,1):.1f}%")
