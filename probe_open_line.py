#!/usr/bin/env python3
"""沿着网络自己的 argmax 走前几手，逐手报中心区域概率质量"""
import numpy as np, argparse
from game import Board
from policy_value_net_pytorch import PolicyValueNet

AL = "ABCDEFGHIJKLMNO"
C = 7
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="current_policy.model")
ap.add_argument("--plies", type=int, default=6)
a = ap.parse_args()
net = PolicyValueNet(15, 15, model_file=a.model, use_gpu=False)


def rings(p):
    p = np.asarray(p).reshape(15, 15)
    out = []
    for d in range(5):
        m = np.zeros((15, 15), bool)
        for r in range(15):
            for c in range(15):
                if max(abs(r - C), abs(c - C)) == d:
                    m[r, c] = True
        out.append(float(p[m].sum()))
    return out


b = Board(width=15, height=15, n_in_row=5)
b.init_board(0)
moves = []
print(f"{'手':>3} {'落点':>5} {'天元%':>7} {'d<=1':>7} {'d<=2':>7} {'熵占比':>7} {'value':>7}")
for ply in range(a.plies):
    st = b.current_state()
    p, v = net.policy_value(st[np.newaxis])
    p = np.asarray(p[0]).ravel()
    rr = rings(p)
    mv = int(np.argmax(p))
    r, c = mv // 15, mv % 15
    who = "黑" if ply % 2 == 0 else "白"
    print(f"{ply+1:>3} {who}{AL[c]}{15-r:>3} {rr[0]*100:6.1f}% {(rr[0]+rr[1])*100:6.1f}% "
          f"{(rr[0]+rr[1]+rr[2])*100:6.1f}% {(-(p*np.log(p+1e-12)).sum())/np.log(225)*100:6.1f}% "
          f"{float(np.asarray(v).ravel()[0]):+6.2f}")
    b.do_move(mv)
    moves.append((r, c))
    if b.game_end()[0]:
        print("终局")
        break
print("\n手顺:", " ".join(f"{AL[c]}{15-r}" for r, c in moves))
