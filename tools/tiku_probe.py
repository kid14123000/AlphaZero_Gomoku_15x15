# -*- coding: utf-8 -*-
"""把连珠题库跑在我们的网络上：验证「题面 → Board → MCTS」这条链路

用法: python tools/tiku_probe.py [题库jsonl] [--n 20] [--sims 300] [--mode VCF]
不写任何文件，不影响训练。
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from game import Board                      # noqa: E402
from mcts_alphaZero import MCTS             # noqa: E402
from policy_value_net_pytorch import PolicyValueNet  # noqa: E402

W = H = 15
PLAYERS = ['黑', '白']


def build_board(moves, side):
    b = Board(width=W, height=H, n_in_row=5)
    b.init_board()
    for m in moves:
        assert m in b.availables, '非法落点 %d' % m
        b.do_move(m)
    want = side
    got = b.current_player
    assert got == want, '轮次不符 %d vs %d' % (got, want)
    return b


def fmt(m):
    return '%c%d' % (ord('A') + m % W, H - m // W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('jsonl', nargs='?',
                    default=os.path.expanduser('~/.hermes/cache/scratch/gomoku_tiku/out/tiku_all.jsonl'))
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--sims', type=int, default=300)
    ap.add_argument('--mode', default='VCF')
    ap.add_argument('--model', default=os.path.join(ROOT, 'sync/current_policy.model'))
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    pool = [r for r in rows if r['mode_title'] == a.mode]
    print('题库 %s 共 %d 题，抽前 %d 题，sims=%d' % (a.mode, len(pool), a.n, a.sims))
    print('权重 %s' % a.model)

    net = PolicyValueNet(W, H, model_file=a.model, use_gpu=False)

    pv_fn = net.policy_value_fn   # 入参是 Board，自带 availables 过滤

    n_ok = 0
    for r in pool[:a.n]:
        try:
            b = build_board(r['moves'], r['side'])
        except AssertionError as e:
            print('  [跳过] %s #%d  %s' % (r['lib'], r['idx'], e))
            continue
        mcts = MCTS(pv_fn, c_puct=1.25, n_playout=a.sims)
        acts, _ = mcts.get_move_probs(b, temp=1.0, root_noise=None)
        share = dict(zip(acts, mcts.last_label))       # 访问数归一化，等价于 temp=1.0
        top = sorted(share, key=lambda m: -share[m])[:5]
        top1 = top[0]
        print('  %-14s #%-3d %2d子 %s先 → top1=%s  %s  活跃点=%d  根Q=%+.3f'
              % (r['lib'][:12], r['idx'], r['n_stones'], PLAYERS[r['side'] - 1],
                 fmt(top1),
                 ' '.join('%s:%.2f' % (fmt(m), share[m]) for m in top[:3]),
                 len(share), mcts.last_root_q))
        n_ok += 1
    print('\n完成 %d 题' % n_ok)


if __name__ == '__main__':
    main()
