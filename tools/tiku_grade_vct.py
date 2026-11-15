# -*- coding: utf-8 -*-
"""题库判分（VCF + VCT 版）—— 能吃下「三手胜」类题目

判定「网络这一步对不对」的口径：
  落 m 之后，攻击方在自由规则下**仍然**存在 VCT 强制胜（深度 depth-1）。
  不要求与求解器首手相同 —— 同一题常有多条成立的杀法。

比 tiku_grade.py 强的地方：那版只认 VCF，三手胜/四三杀题一律判错。
这版用 VCT，depth 给到 3 就能覆盖大部分三手胜。

用法:
  python tools/tiku_grade_vct.py --n 30 --sims 300 --depth 3 --lib 入门题黑先三手詰
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import vct as VT                     # noqa: E402
from game import Board               # noqa: E402
from mcts_alphaZero import MCTS      # noqa: E402
from policy_value_net_pytorch import PolicyValueNet   # noqa: E402

W = H = 15


def fmt(m):
    return '%c%d' % (ord('A') + m % W, H - m // W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', default=os.path.expanduser(
        '~/Downloads/gomoku_tiku/data/tiku_dedup_17095.jsonl'))
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--sims', type=int, default=300)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--lib', default=None, help='只测这个库')
    ap.add_argument('--model', default=os.path.join(ROOT, 'sync/current_policy.model'))
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    if a.lib:
        rows = [r for r in rows if r['lib'] == a.lib]
    net = PolicyValueNet(W, H, model_file=a.model, use_gpu=False)
    pv = net.policy_value_fn

    ok1 = ok3 = tot = 0
    for r in rows:
        if tot >= a.n:
            break
        side = r['side']
        if r.get('colors') and len(r['colors']) == len(r['moves']):
            g = {m: c for m, c in zip(r['moves'], r['colors'])}
        else:
            g = {m: (r['starts'] if i % 2 == 0 else 3 - r['starts']) for i, m in enumerate(r['moves'])}
        if VT.vct(dict(g), side, depth=a.depth, seconds=6) is None:
            continue                      # 这题 VCT 都解不出，不拿来考
        b = Board(width=W, height=H, n_in_row=5)
        b.init_board(start_player=r['starts'] - 1)
        for m in r['moves']:
            b.do_move(m)
        mcts = MCTS(pv, c_puct=1.25, n_playout=a.sims)
        acts, _ = mcts.get_move_probs(b, temp=1.0, root_noise=None)
        share = dict(zip(acts, mcts.last_label))
        top = sorted(share, key=lambda m: -share[m])[:3]

        def good(m):
            if m in g:
                return False
            g[m] = side
            try:
                return VT.vct(dict(g), side, depth=max(1, a.depth - 1), seconds=3) is not None or \
                    VT.is_five(g, m, side)
            finally:
                del g[m]

        h1 = good(top[0])
        h3 = h1 or any(good(m) for m in top[1:])
        ok1 += h1
        ok3 += h3
        tot += 1
        print('  %2d %-18s #%-3d %s先 top1=%-4s %s | %s %s'
              % (tot, r['lib'][:18], r['idx'], '黑白'[side - 1], fmt(top[0]),
                 ' '.join('%.2f' % share[m] for m in top),
                 '✓' if h1 else '✗', '✓' if h3 else '✗'), flush=True)

    print('\n=== sims=%d depth=%d, %d 题 ===' % (a.sims, a.depth, tot))
    if tot:
        print('  top1 命中 %d/%d = %.1f%%' % (ok1, tot, 100.0 * ok1 / tot))
        print('  top3 命中 %d/%d = %.1f%%' % (ok3, tot, 100.0 * ok3 / tot))


if __name__ == '__main__':
    main()
