#!/usr/bin/env python3
"""c_puct 敏感性探针：同一个"白必须堵"的局面，c=1.25 和 c=5 分别会走哪。

核心看三件事：
  1. 选点是不是落在必堵点（防守率）
  2. 根节点访问分布熵（熵高 = 搜索分不清好坏，选点被噪声主导）
  3. 根子节点 Q 的极差（Q 没有区分度时，PUCT 的探索项一旦被削弱，选点就退化成随机）

用法：.venv/bin/python tools/probe_cpuct_defense.py [权重路径]
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(HERE)
sys.path.insert(0, HERE)

from game import Board                                    # noqa: E402
from mcts_alphaZero import MCTSPlayer                      # noqa: E402
from policy_value_net_pytorch import PolicyValueNet        # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else 'sync/current_policy.model'
SIMS = 800

# ── 构造局面：(说明, 黑子列表, 白子列表, 白必堵的点集合) ──
# 白先走（current_player = 2）。黑已经成型，白只有堵住/封住才不输。
CASES = [
    ('黑活三（横），白必堵两端之一',
     [(7, 5), (7, 6), (7, 7)],
     [(8, 8), (6, 9)],
     {(7, 4), (7, 8)}),

    ('黑冲四（横），白只有唯一堵点',
     [(7, 5), (7, 6), (7, 7), (7, 8)],
     [(8, 8), (6, 9), (9, 3)],
     {(7, 4)}),

    ('黑活三（竖），白必堵两端之一',
     [(5, 9), (6, 9), (7, 9)],
     [(8, 7), (4, 10)],
     {(4, 9), (8, 9)}),

    ('黑双三，白必须堵最急的一条',
     [(7, 5), (7, 6), (7, 7), (6, 8), (8, 8)],
     [(9, 9), (5, 4)],
     {(7, 4), (7, 8), (5, 8), (9, 8)}),
]


def build_game(blacks, whites):
    b = Board(width=15, height=15, n_in_row=5)
    b.init_board()
    b.states = {}
    b.availables = list(range(225))      # game.py 里 availables 是 list
    bl = list(blacks)
    wh = list(whites)
    played = []
    while bl or wh:                      # 交替落子，保证黑白子数差 ≤1
        if bl:
            played.append((bl.pop(0), 1))
        if wh:
            played.append((wh.pop(0), 2))
    for (r, c), pl in played:
        m = r * 15 + c
        b.states[m] = pl
        b.availables.remove(m)
    b.current_player = 2                 # 轮到白
    b.last_move = played[-1][0][0] * 15 + played[-1][0][1]
    return b


def entropy(p):
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def run_case(net, desc, blacks, whites, must):
    b = build_game(blacks, whites)
    out = []
    for c in (1.25, 5.0):
        np.random.seed(7)
        pl = MCTSPlayer(net.policy_value_fn, c_puct=c, n_playout=SIMS,
                        is_selfplay=0, root_noise=False)
        acts, probs = pl.mcts.get_move_probs(b, temp=1.0)
        pi = np.asarray(pl.mcts.last_label, dtype=np.float64)
        vis = np.array([pl.mcts._root._children[a]._n_visits for a in acts], dtype=np.int64)
        Q = np.array([pl.mcts._root._children[a]._Q for a in acts], dtype=np.float64)
        root_q = -float(pl.mcts._root._Q) if pl.mcts._root._n_visits else 0.0
        order = np.argsort(-vis)
        top_idx = order[0]
        top_move = acts[top_idx]
        top_rc = (top_move // 15, top_move % 15)
        # 只看被访问过的子节点，Q 的区分度
        m = vis > 0
        q_vis = Q[m]
        out.append(dict(
            c=c,
            top_rc=top_rc,
            defended=top_move in must,
            top_vis_share=float(vis[top_idx] / max(vis.sum(), 1)),
            visited=int(m.sum()),
            H=entropy(pi / max(pi.sum(), 1e-12)),
            q_root=root_q,
            q_min=float(q_vis.min()) if m.sum() else 0.0,
            q_max=float(q_vis.max()) if m.sum() else 0.0,
            q_range=float(q_vis.max() - q_vis.min()) if m.sum() else 0.0,
            top3=[(a // 15, a % 15, int(vis[i])) for i, a in sorted(
                zip(range(len(acts)), acts), key=lambda t: -vis[t[0]])[:3]],
        ))
    return out


def main():
    net = PolicyValueNet(15, 15, model_file=MODEL, use_gpu=False)
    print('权重: %s   模拟数: %d\n' % (MODEL, SIMS))
    rows = []
    for desc, blacks, whites, must in CASES:
        print('=' * 92)
        print(desc)
        o = run_case(net, desc, blacks, whites, must)
        for r in o:
            print('  c=%-5.2f 选 (r%2d,c%2d) 堵住?%-3s | top1访问占比 %.2f 已访问%3d '
                  '熵 %.2f | 根Q %+.3f | 已访问Q区间 [%+.3f, %+.3f] 跨度 %.3f'
                  % (r['c'], r['top_rc'][0], r['top_rc'][1], '是' if r['defended'] else '否',
                     r['top_vis_share'], r['visited'], r['H'], r['q_root'],
                     r['q_min'], r['q_max'], r['q_range']))
            print('            top3: %s' % ', '.join(
                '(r%d,c%d)%d访问' % t for t in r['top3']))
        rows.append(o)

    print('=' * 92)
    a = [x[0] for x in rows]      # c=1.25
    b = [x[1] for x in rows]      # c=5
    print('汇总（%d 个必堵局面）' % len(rows))
    print('  防守成功:      c=1.25 → %d/%d      c=5 → %d/%d'
          % (sum(x['defended'] for x in a), len(a), sum(x['defended'] for x in b), len(b)))
    print('  平均已访问子节点: c=1.25 → %.1f     c=5 → %.1f'
          % (np.mean([x['visited'] for x in a]), np.mean([x['visited'] for x in b])))
    print('  平均访问熵:     c=1.25 → %.2f       c=5 → %.2f'
          % (np.mean([x['H'] for x in a]), np.mean([x['H'] for x in b])))
    print('  平均 top1 占比: c=1.25 → %.2f       c=5 → %.2f'
          % (np.mean([x['top_vis_share'] for x in a]), np.mean([x['top_vis_share'] for x in b])))
    print('  平均 Q 跨度:    c=1.25 → %.3f      c=5 → %.3f'
          % (np.mean([x['q_range'] for x in a]), np.mean([x['q_range'] for x in b])))
    print('  根 Q 均值:      c=1.25 → %+.3f     c=5 → %+.3f'
          % (np.mean([x['q_root'] for x in a]), np.mean([x['q_root'] for x in b])))


if __name__ == '__main__':
    main()
