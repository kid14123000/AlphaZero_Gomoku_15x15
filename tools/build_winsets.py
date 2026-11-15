# -*- coding: utf-8 -*-
"""预计算题库的「胜利首手集合」并缓存（多进程）

tiku_bench.py 会读同一个缓存文件，避免每次跑评测都重算 15 分钟。
macOS 用 spawn 起子进程，所以必须写成独立 .py 文件，不能塞在 stdin heredoc 里。

用法: python tools/build_winsets.py --bench ~/Downloads/gomoku_tiku/data/bench_v1.jsonl -j 3 --depth 2
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

_D = {'depth': 2, 'secs': 4.0}


def _init(depth, secs):
    _D['depth'] = depth
    _D['secs'] = secs


def _work(r):
    import torch
    torch.set_num_threads(1)
    import vct as VT
    import vcf2 as V2

    if r.get('colors') and len(r['colors']) == len(r['moves']):
        g = {m: c for m, c in zip(r['moves'], r['colors'])}
    else:
        s = r['starts']
        g = {m: (s if i % 2 == 0 else 3 - s) for i, m in enumerate(r['moves'])}
    side = r['side']
    ws = set(VT.winning_moves(dict(g), side, depth=_D['depth'] + 1,
                              seconds=_D['secs'], limit=60))
    if not ws:
        mv = V2.vcf(dict(g), side, max_depth=8, deadline=time.time() + _D['secs'])
        if mv is not None:
            ws.add(int(mv))
    return r['bench_id'], sorted(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default=os.path.expanduser(
        '~/Downloads/gomoku_tiku/data/bench_v1.jsonl'))
    ap.add_argument('-j', '--jobs', type=int, default=3)
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--secs', type=float, default=4.0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.bench) if l.strip()]
    out, t0 = {}, time.time()
    with Pool(a.jobs, initializer=_init, initargs=(a.depth, a.secs)) as p:
        for i, (bid, ws) in enumerate(p.imap_unordered(_work, rows, chunksize=4), 1):
            if ws:
                out[str(bid)] = ws
            if i % 50 == 0:
                print('  %d/%d  %.0fs' % (i, len(rows), time.time() - t0), flush=True)

    cache = os.path.splitext(a.bench)[0] + '.winsets_d%d.json' % a.depth
    json.dump(out, open(cache, 'w'))
    print('缓存写好：%d/%d 题  耗时 %.0fs  → %s'
          % (len(out), len(rows), time.time() - t0, os.path.basename(cache)))


if __name__ == '__main__':
    main()
