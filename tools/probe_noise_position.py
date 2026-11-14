#!/usr/bin/env python3
"""量化「噪声加在落子抽样（上游写法）」vs「加在根先验（论文写法）」的实际差别。

指标（每手都量）：
  π(chosen)   = 落子点在【搜索访问分布】里占的比例。论文写法下它必然 >0（落子就是从 π 采的）；
                上游写法下 25% 的落子来自 Dirichlet 噪声，可能是搜索**一次都没访问过**的点。
  rank        = 落子点按访问数排第几（1=搜索最想走的那个）
  zero_visit  = 落子点在搜索里访问数为 0（= 搜索完全没看过这一手，之后就走进一个没分析过的局面）

跑法：cd ~/AlphaZero_Gomoku_15x15_v4 && PYTHONPATH=$PWD .venv/bin/python <本文件>
"""
import sys

import numpy as np

from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet

MODEL = 'sync/current_policy.model'
SIMS = 200
STEPS = 10
SEED = 20261114


def run(root_noise):
    np.random.seed(SEED)
    board = Board(width=15, height=15, n_in_row=5)
    board.init_board()
    net = PolicyValueNet(15, 15, model_file=MODEL, use_gpu=False)
    p = MCTSPlayer(net.policy_value_fn, c_puct=1.25, n_playout=SIMS,
                   is_selfplay=1, dirichlet_eps=0.25, dirichlet_alpha=0.03,
                   root_noise=root_noise)
    eps, alpha = 0.25, 0.03
    rows = []
    for step in range(STEPS):
        # 隔离变量：两边都用 temp=1.0（上游默认就是 1.0 全程不降温）
        # 复刻 MCTSPlayer.get_action 的分支，但**先量完再 update_with_move**（
        # get_action 结束时 _root 已前移，那之后就量不到本手的访问分布了）
        _rn = (eps, alpha) if root_noise else None
        acts, probs = p.mcts.get_move_probs(board, 1.0, root_noise=_rn)
        pi = np.asarray(p.mcts.last_label, dtype=np.float64)      # 访问数归一化(按 acts 顺序)
        vis = np.array([p.mcts._root._children[a]._n_visits for a in acts],
                       dtype=np.int64)
        if root_noise:
            mv = int(np.random.choice(acts, p=probs))
        else:
            nz = np.random.dirichlet(alpha * np.ones(len(probs)))
            mv = int(np.random.choice(acts, p=(1.0 - eps) * probs + eps * nz))
        order = np.argsort(-vis)
        idx = acts.index(mv)
        rank = int(np.where(order == idx)[0][0]) + 1
        rows.append(dict(step=step, move=mv,
                         share=float(pi[idx]),
                         rank=rank,
                         zero=bool(vis[idx] == 0),
                         visited=int((vis > 0).sum())))
        p.mcts.update_with_move(mv)
        board.do_move(mv)
    return p, rows


def summarize(tag, rows):
    sh = np.array([r['share'] for r in rows])
    rk = np.array([r['rank'] for r in rows])
    z = sum(r['zero'] for r in rows)
    print('%-28s n=%2d  π(chosen) 均值 %.3f 中位 %.3f | <0.5%% 的占 %2d/%2d | '
          '搜索没访问过的落子 %d/%d | rank 中位 %.0f'
          % (tag, len(rows), sh.mean(), np.median(sh),
             int((sh < 0.005).sum()), len(rows), z, len(rows), np.median(rk)))
    return sh, rk, z


def main():
    out = {}
    for tag, rn in (('上游: 噪声混落子抽样', False), ('论文/v24: 噪声进根先验', True)):
        p, rows = run(rn)
        print('=' * 96)
        out[tag] = rows
        for r in rows:
            print('  手%-2d mv=%3d π=%.4f rank=%-3d 访问=0? %-5s (根已访问子节点 %d)'
                  % (r['step'], r['move'], r['share'], r['rank'],
                     r['zero'], r['visited']))
        summarize(tag, rows)

    a = out['上游: 噪声混落子抽样']
    b = out['论文/v24: 噪声进根先验']
    print('=' * 96)
    summary = {'a': summarize('上游: 噪声混落子抽样', a),
               'b': summarize('论文/v24: 噪声进根先验', b)}
    sa = summary['a']
    sc = summary['b']
    print('=' * 96)
    print('Δ: 上游写法「落子不在搜索访问集里」的比例 %d/%d = %.0f%%'
          % (sa[2], len(a), 100 * sa[2] / len(a)))
    print('Δ: 论文写法「落子不在搜索访问集里」的比例 %d/%d = %.0f%%'
          % (sc[2], len(b), 100 * sc[2] / len(b)))
    print('Δ: rank 中位 %.0f → %.0f' % (np.median(sa[1]), np.median(sc[1])))
    return 0


if __name__ == '__main__':
    sys.exit(main())
