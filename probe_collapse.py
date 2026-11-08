#!/usr/bin/env python3
"""价值/策略是否塌缩的定量探针（按执子方分组，排除"谁走谁赢"混杂）

1) 价值：同执子方内部的离散度 vs 跨执子方的差值 → 身份捷径判据
2) 策略：中心质量占比 + top1 距中心
3) 堵率：真实对局里对手刚做出四连时，policy top1 是否落在堵点
"""
import json, os, sys
import numpy as np
import torch

from game import Board
from policy_value_net_pytorch import PolicyValueNet

AL = "ABCDEFGHIJKLMNO"
N = 15


def load(path):
    pv = PolicyValueNet(N, N, model_file=path, use_gpu=False)
    pv.policy_value_net.eval()
    return pv.policy_value_net


def fwd(net, X):
    with torch.no_grad():
        p, v = net(torch.from_numpy(np.ascontiguousarray(X)))
    return torch.exp(p).numpy(), v.numpy().ravel()


def name(i):
    return f"{AL[i % N]}{N - i // N}"


def stone_grid(board):
    g = np.zeros((N, N), dtype=int)
    for m, p in board.states.items():
        g[int(m) // N, int(m) % N] = p
    return g


def fours_threats(board, player):
    """player 是否有 4 连（含被堵一端的冲四）：返回必须处理的点集合"""
    g = stone_grid(board)
    pts = set()
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


def walk_games(path, max_games=10 ** 9):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out[:max_games]


def main():
    nets = [('250代', load('checkpoints/ckpt_it250.model')),
            ('300代', load('current_policy.model'))]

    recs = walk_games('logs/games.jsonl')
    print(f"棋谱 {len(recs)} 局")

    # ---------- 收集真实局面（带执子方） ----------
    X, turn = [], []
    for r in recs:
        b = Board(width=N, height=N, n_in_row=5); b.init_board(0)
        for r_, c_ in r['moves'][:45]:
            X.append(np.array(b.current_state(), dtype=np.float32, copy=True))
            turn.append(b.current_player)
            b.do_move(r_ * N + c_)
    X = np.asarray(X); turn = np.asarray(turn)
    print(f"局面 {len(X)}（黑走 {int((turn==1).sum())} / 白走 {int((turn==2).sum())}）")

    print("\n=== 1. 价值：执子方内部离散度 vs 跨执子方差值 ===")
    for tag, net in nets:
        _, v = fwd(net, X)
        vb, vw = v[turn == 1], v[turn == 2]
        within = (vb.std() + vw.std()) / 2
        between = abs(vb.mean() - vw.mean())
        print(f"  {tag}: 黑走 mean {vb.mean():+.3f} std {vb.std():.3f} | "
              f"白走 mean {vw.mean():+.3f} std {vw.std():.3f}")
        print(f"       跨执子方差 {between:.3f} vs 组内标准差 {within:.3f} → 比值 {between/max(within,1e-6):.1f}x"
              f"{'  ← 身份捷径（输出≈只看谁走）' if between/max(within,1e-6) > 3 else ''}")

    print("\n=== 2. 策略：中心质量与 top1 位置 ===")
    cc = np.array([7, 7])
    for tag, net in nets:
        p, _ = fwd(net, X)
        p = p.reshape(len(p), N, N)
        mass9 = p[:, 6:9, 6:9].sum(axis=(1, 2))
        top1 = p.reshape(len(p), -1).argmax(axis=1)
        d = np.abs(np.array(np.unravel_index(top1, (N, N))).T - cc).max(axis=1)
        ent = -(p * np.log(p + 1e-12)).sum(axis=(1, 2)).mean()
        print(f"  {tag}: 中心3x3 质量 {mass9.mean()*100:5.1f}% | top1 落在中心3x3 {np.mean(d<=1)*100:5.1f}% | "
              f"top1 距中心中位 {np.median(d):.0f} 格 | 平均熵 {ent:.2f}/{np.log(225):.2f}")

    print("\n=== 3. 堵率：对手刚做出四连时，我该堵的点排第几 ===")
    for tag, net in nets:
        tot = hit1 = hit3 = 0
        detail = []
        for r in recs:
            b = Board(width=N, height=N, n_in_row=5); b.init_board(0)
            for idx, (r_, c_) in enumerate(r['moves'][:45]):
                m = r_ * N + c_
                # 上一步是对手做的；若对手已有四连，则现在必须应对
                opp = 3 - b.current_player
                pts = fours_threats(b, opp)
                if pts and idx >= 3:
                    st = np.array(b.current_state(), dtype=np.float32, copy=True)
                    p, _ = fwd(net, st[None])
                    order = np.argsort(p[0].ravel())[::-1]
                    tot += 1
                    if order[0] in pts:
                        hit1 += 1
                    if any(o in pts for o in order[:3]):
                        hit3 += 1
                    detail.append((name(order[0]), [name(q) for q in list(pts)][:3]))
                if b.game_end()[0]:
                    break
                b.do_move(m)
        if tot:
            print(f"  {tag}: 四连威胁局面 {tot} 个 | top1 命中堵点 {hit1/tot*100:5.1f}% | top3 含堵点 {hit3/tot*100:5.1f}%")
            for a1, pts in detail[:4]:
                print(f"       top1={a1}  堵点={pts}")
        else:
            print(f"  {tag}: 未捕到四连威胁局面")


if __name__ == '__main__':
    main()
