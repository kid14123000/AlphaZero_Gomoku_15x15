#!/usr/bin/env python3
"""根先验噪声 / c_puct 冒烟验证（只读，不改训练状态）。

跑法：
    cd ~/AlphaZero_Gomoku_15x15_v4 && \
    PYTHONPATH=$PWD .venv/bin/python <本文件>

四组对照（同权重、同开局、同随机种子）：
    (root_noise=0, c=5)     ← 旧行为（线上跑了 13438 局的就是这个）
    (root_noise=1, c=5)     ← 只改噪声位置
    (root_noise=0, c=1.25)  ← 只改 c_puct
    (root_noise=1, c=1.25)  ← 本次要上线的组合
每局走 6 手，逐手记录：根子节点数 / 被访问数 / 标签熵 / top1 / 根先验 ΣP。
"""
import sys

import numpy as np

from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet

MODEL = 'sync/current_policy.model'
SIMS = 200
STEPS = 6
SEED = 20261114


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def run(root_noise, c_puct):
    np.random.seed(SEED)
    board = Board(width=15, height=15, n_in_row=5)
    board.init_board()
    net = PolicyValueNet(15, 15, model_file=MODEL, use_gpu=False)
    p = MCTSPlayer(net.policy_value_fn, c_puct=c_puct, n_playout=SIMS,
                   is_selfplay=1, dirichlet_eps=0.25, dirichlet_alpha=0.03,
                   root_noise=root_noise)
    rows = []
    for step in range(STEPS):
        t = 1.0 if step < 4 else 0.2          # 模拟线上温度调度（前 4 手热）
        mv, _ = p.get_action(board, temp=t, return_prob=1)
        lab = np.asarray(p.mcts.last_label, dtype=np.float64)   # 前进前的标签
        st = p.mcts.last_root_noise_stats or {}
        rows.append(dict(step=step, move=int(mv),
                         legal=int(mv) in board.availables,
                         applied=p.mcts.root_noise_applied,
                         k=int(lab.size),
                         visited=int((lab > 0).sum()),
                         H=entropy(lab), top1=float(lab.max()),
                         psum=float(st.get('sum', float('nan'))),
                         top1_prior=float(st.get('top1_share', float('nan')))))
        board.do_move(mv)
    return p, rows


def main():
    combos = [('旧: rn=0 c=5', 0, 5.0),
              ('只换噪声: rn=1 c=5', 1, 5.0),
              ('只降c: rn=0 c=1.25', 0, 1.25),
              ('本次上线: rn=1 c=1.25', 1, 1.25)]
    res = {}
    for tag, rn, cp in combos:
        p, rows = run(rn, cp)
        res[tag] = (p, rows)
        print('=' * 78)
        print('%-24s MCTS 实际 c_puct=%.2f  root_noise=%s' % (tag, p.mcts._c_puct, p._root_noise))
        for r in rows:
            print('  手%d mv=%3d 合法=%-5s 注入=%d | 子节点%3d 已访问%3d | '
                  '标签H=%.2f top1=%.3f | 根先验ΣP=%.6f top1占比=%.3f'
                  % (r['step'], r['move'], r['legal'], r['applied'], r['k'],
                     r['visited'], r['H'], r['top1'], r['psum'], r['top1_prior']))

    old = res['旧: rn=0 c=5'][1]
    new = res['本次上线: rn=1 c=1.25'][1]
    noise_only = res['只换噪声: rn=1 c=5'][1]
    all_rows = [r for _, rows in res.values() for r in rows]

    checks = [
        ('旧路径全程不注入噪声', all(r['applied'] == 0 for r in old)),
        ('新路径每手注入 1 次', [r['applied'] for r in new] == list(range(1, STEPS + 1))),
        ('噪声只换位置时注入生效', all(r['applied'] > 0 for r in noise_only)),
        ('根先验始终归一化(ΣP≈1, 容差 1e-2)',
         all(abs(r['psum'] - 1.0) < 1e-2 for r in new)),
        ('所有落子合法', all(r['legal'] for r in all_rows)),
        ('根子节点数>0 且 K 单调对应棋盘(15×15, 前几手=225)',
         all(r['k'] > 200 for r in old[:2]) and all(r['k'] > 200 for r in new[:2])),
        ('噪声让搜索更分散(已访问子节点数↑, 噪声组均值)',
         np.mean([r['visited'] for r in noise_only])
         > np.mean([r['visited'] for r in old])),
    ]
    print('=' * 78)
    ok = True
    for name, val in checks:
        print(('  ✅ ' if val else '  ❌ ') + name)
        ok = ok and bool(val)
    print('=' * 78)
    print('均访问子节点数: 旧 %.1f | 只换噪声 %.1f | 只降c %.1f | 上线组合 %.1f'
          % (np.mean([r['visited'] for r in old]),
             np.mean([r['visited'] for r in noise_only]),
             np.mean([r['visited'] for r in res['只降c: rn=0 c=1.25'][1]]),
             np.mean([r['visited'] for r in new])))
    print('结论:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
