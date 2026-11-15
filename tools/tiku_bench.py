# -*- coding: utf-8 -*-
"""做题准确率评测 —— 同一套题，测任何检查点

两种口径（分开报，含义不同）：
  policy  只用策略头一次前向，取合法点里概率最高的     ← 网络的「直觉」
  search  MCTS 搜索 n_playout 次，取访问数最多的        ← 网络的「棋力」

判定「对」的标准：**落子在「胜利首手集合」里**。
集合由 VCT 求解器枚举（winning_moves），不是跟某个标准答案比 ——
同一道杀题常有好几条成立的路，卡 equals 会把对的判成错的。

🚨 踩过的坑（写在这里免得再犯）：
  1. 判某一步是否成立时，攻击方落子后**轮到对手**，必须让对手应一手再判。
     早期版本直接拿「攻击方再走一手还能赢」当判据，等于让攻击方连走两手，
     结果 96% 的随机落点都被判「对」，指标完全失效。
  2. 每题的胜利首手集合只算一次（缓存），不要每个候选都重算。

用法:
  python tools/tiku_bench.py --bench ~/Downloads/gomoku_tiku/data/bench_v1.jsonl --policy-only
  python tools/tiku_bench.py --bench ... --sims 400
  python tools/tiku_bench.py --bench ... --ckpt 'checkpoints/ckpt_it15*.model' --sims 200
"""
import argparse
import collections
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import vct as VT                                   # noqa: E402
import vcf2 as V2                                  # noqa: E402
from game import Board                             # noqa: E402
from mcts_alphaZero import MCTS                    # noqa: E402
from policy_value_net_pytorch import PolicyValueNet  # noqa: E402

import torch                                        # noqa: E402
# 训练在跑时单进程开多线程会互相抢核（实测 200 sims 从 8.2s/题 降到 ~2.6s/题）
torch.set_num_threads(1)

W = H = 15


def fmt(m):
    return '%c%d' % (ord('A') + m % W, H - m // W)


def load_bench(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def board_of(r):
    b = Board(width=W, height=H, n_in_row=5)
    b.init_board(start_player=r['starts'] - 1)
    for m in r['moves']:
        b.do_move(m)
    return b


def grid_of(r):
    """题面 grid。有 colors 就用它（题面颜色是权威的），没有才按 starts 交替推。"""
    if r.get('colors') and len(r['colors']) == len(r['moves']):
        return {m: c for m, c in zip(r['moves'], r['colors'])}
    s = r['starts']
    return {m: (s if i % 2 == 0 else 3 - s) for i, m in enumerate(r['moves'])}


def solve_set(r, depth, secs):
    """该题的全体胜利首手（含 VCF 与 VCT 两类杀法）"""
    g = grid_of(r)
    side = r['side']
    out = set(VT.winning_moves(dict(g), side, depth=depth + 1, seconds=secs, limit=60))
    if not out:                       # VCT 没枚举出来时退回 VCF 首手
        mv = V2.vcf(dict(g), side, max_depth=8, deadline=time.time() + secs)
        if mv is not None:
            out.add(int(mv))
    return out


def run_one(net, r, sims, policy_only, win_set, topk=5):
    b = board_of(r)
    side = r['side']

    if policy_only:
        probs, _v = net.policy_value(b.current_state()[None].astype('float32'))
        import numpy as np
        p = np.asarray(probs, dtype='float64').reshape(-1)
        order = sorted(b.availables, key=lambda m: -p[m])[:topk]
        share = {m: float(p[m]) for m in order}
    else:
        mcts = MCTS(net.policy_value_fn, c_puct=1.25, n_playout=sims)
        acts, _ = mcts.get_move_probs(b, temp=1.0, root_noise=None)
        share = dict(zip(acts, mcts.last_label))
        order = sorted(share, key=lambda m: -share[m])[:topk]

    top1 = order[0]
    hits = [m in win_set for m in order]
    return {
        'top1': top1,
        'top1_correct': hits[0],
        'topk_correct': any(hits),
        'exact': top1 == r['answer'],
        'rank_of_answer': (order.index(r['answer']) + 1) if r['answer'] in order else None,
        'p_top1': share[top1],
        'n_win_moves': len(win_set),
        'moves_shown': [fmt(m) for m in order[:3]],
    }


def summarize(rows, tag):
    n = len(rows)
    if not n:
        return
    c1 = sum(r['top1_correct'] for r in rows)
    ck = sum(r['topk_correct'] for r in rows)
    ex = sum(r['exact'] for r in rows)
    print('  %-24s n=%3d  top1正确 %5.1f%%  top5正确 %5.1f%%  与答案一致 %5.1f%%'
          % (tag, n, 100.0 * c1 / n, 100.0 * ck / n, 100.0 * ex / n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default=os.path.expanduser(
        '~/Downloads/gomoku_tiku/data/bench_v1.jsonl'))
    ap.add_argument('--ckpt', nargs='+', default=None)
    ap.add_argument('--sims', type=int, default=400)
    ap.add_argument('--policy-only', action='store_true')
    ap.add_argument('--depth', type=int, default=2, help='胜利首手集合的搜索深度（攻击方回合数）')
    ap.add_argument('--solve-secs', type=float, default=4.0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--save', default=None)
    a = ap.parse_args()

    rows = load_bench(a.bench)
    if a.limit:
        rows = rows[:a.limit]

    # 先把每题胜利首手算好（跟权重无关，缓存到磁盘，避免每次重算 15 分钟）
    cache = '%s.winsets_d%d.json' % (os.path.splitext(a.bench)[0], a.depth)
    t0 = time.time()
    if os.path.exists(cache):
        sets = {int(k): set(v) for k, v in json.load(open(cache)).items()}
        print('胜利首手集合从缓存读入 %s（%d 题）' % (os.path.basename(cache), len(sets)))
    else:
        sets, dropped = {}, []
        for i, r in enumerate(rows, 1):
            ws = solve_set(r, a.depth, a.solve_secs)
            if not ws:
                dropped.append(r['bench_id'])
                continue
            sets[r['bench_id']] = ws
            if i % 50 == 0:
                print('  求解 %d/%d  %.0fs' % (i, len(rows), time.time() - t0), flush=True)
        json.dump({str(k): sorted(v) for k, v in sets.items()}, open(cache, 'w'))
        print('可解 %d / %d（%d 题求解器给不出胜利首手，剔除）  耗时 %.0fs  缓存 → %s'
              % (len(sets), len(rows), len(dropped), time.time() - t0, os.path.basename(cache)))
    rows = [r for r in rows if r['bench_id'] in sets]
    avg = sum(len(s) for s in sets.values()) / max(len(sets), 1)
    print('每题平均 %.1f 个胜利首手\n' % avg)

    ckpts = a.ckpt or [os.path.join(ROOT, 'sync/current_policy.model')]
    ckpts = [c for pat in ckpts for c in (sorted(glob.glob(pat)) or [pat])]
    mode = 'policy（策略头一次前向）' if a.policy_only else 'search sims=%d' % a.sims
    print('口径 %s\n' % mode)

    all_out = []
    for ck in ckpts:
        net = PolicyValueNet(W, H, model_file=ck, use_gpu=False)
        t0 = time.time()
        res = []
        for i, r in enumerate(rows, 1):
            o = run_one(net, r, a.sims, a.policy_only, sets[r['bench_id']])
            o.update(bench_id=r['bench_id'], lib=r['lib'], side=r['side'],
                     solver=r['solver'], n_stones=r['n_stones'])
            res.append(o)
            if i % 50 == 0:
                print('    %d/%d  %.0fs' % (i, len(rows), time.time() - t0), flush=True)
        el = time.time() - t0
        print('\n=== %s  (%.0fs, %.2fs/题) ===' % (os.path.basename(ck), el, el / len(rows)))
        summarize(res, '全部')
        summarize([r for r in res if r['side'] == 1], '黑先')
        summarize([r for r in res if r['side'] == 2], '白先')
        summarize([r for r in res if r['solver'] == 'vcf'], 'VCF 类')
        summarize([r for r in res if r['solver'] == 'vct'], 'VCT 类（三手胜等）')
        for lo, hi, nm in ((0, 15, '≤15 子'), (16, 22, '16~22 子'), (23, 999, '23+ 子')):
            summarize([r for r in res if lo <= r['n_stones'] <= hi], nm)
        print('  分库:')
        for lib, _ in collections.Counter(r['lib'] for r in res).most_common():
            summarize([r for r in res if r['lib'] == lib], '  ' + lib[:20])
        all_out.append({'ckpt': ck, 'sims': a.sims, 'policy_only': a.policy_only, 'rows': res})
        print()

    if a.save:
        with open(a.save, 'w') as f:
            for blk in all_out:
                f.write(json.dumps(blk, ensure_ascii=False) + '\n')
        print('逐题结果已写 %s' % a.save)


if __name__ == '__main__':
    main()
