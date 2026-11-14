#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 v4 的 logs/games.jsonl 转成老 Flask 回放页认的 pkl + 索引。

老版（v36–v44）的自对弈数据是「每局一个 pkl + replay_pool_index.json」，回放页
按「代 → 局 → 每步 stones」三级翻。v4 是 junxiaosong 派生，棋谱全在
logs/games.jsonl（一行一局），两者格式完全不同，所以需要这个转换器。

期望的 pkl：pandas DataFrame，列
    board  : (15,15) int8，**每步落子之前**的局面（第 0 行是空盘）
    player : 该行局面下该谁走（+1 黑 / -1 白）
    winner : +1 黑胜 / -1 白胜 / 0 和
（可选 qa 列：每步 MCTS 软标签；v4 的 games.jsonl 没存，故不产出）

索引结构：
    {"iters": {"<代>": {"games": n, "files": [{file, steps, winner, worker, game}]}},
     "iter_range": [最小代, 最大代], "total_games": N}

用法：
    python tools/build_replay_index.py                       # 全量，每 100 局一代
    python tools/build_replay_index.py --limit 2000          # 只转最近 2000 局（试跑）
    python tools/build_replay_index.py --iter-size 200
"""
from __future__ import print_function

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
BOARD = 15
WIN_MAP = {1: 1, 2: -1, -1: 0}          # 我们的 (1=黑,2=白,-1和) → 回放页 (+1/-1/0)


def boards_of(moves, n=BOARD):
    """把落子序列展开成「每步落子前」的局面列表。

    moves[0] 是黑1。第 i 行 = 前 i 手落下之后的局面 ⇒ 第 0 行是空盘。
    回放接口靠「相邻两行做差」反推每一步，最后再补致胜手，
    所以这里**不要**额外加终点局面（否则会多算一手）。
    """
    b = np.zeros((n, n), dtype=np.int8)
    out = [b.copy()]
    for i, (r, c) in enumerate(moves):
        if i == len(moves) - 1:
            break                      # 最后一手留给接口补（它自己会算致胜手）
        b[r, c] = 1 if i % 2 == 0 else -1
        out.append(b.copy())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', default=os.path.join(PROJ, 'logs', 'games.jsonl'))
    ap.add_argument('--out-dir', default=os.path.join(PROJ, 'gomoku_checkpoints', 'worker_data'))
    ap.add_argument('--index', default=os.path.join(PROJ, 'gomoku_checkpoints', 'replay_pool_index.json'))
    ap.add_argument('--iter-size', type=int, default=100,
                    help='多少局算一代（v4 是「局」而非「代」，这里只做分桶便于翻页）')
    ap.add_argument('--limit', type=int, default=0, help='只转最近 N 局（0=全部）')
    ap.add_argument('--clean', action='store_true', help='先清空 out-dir 里旧的 it*.pkl')
    ap.add_argument('--min-steps', type=int, default=6, help='少于这么多手的局跳过')
    a = ap.parse_args()

    t0 = time.time()
    if not os.path.exists(a.games):
        print('❌ 找不到棋谱: %s' % a.games)
        return 1

    games = []
    with open(a.games, encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                g = json.loads(ln)
            except Exception:
                continue
            mv = g.get('moves') or []
            if len(mv) < a.min_steps:
                continue
            games.append(g)
    if a.limit:
        games = games[-a.limit:]
    if not games:
        print('❌ 棋谱为空')
        return 1

    os.makedirs(a.out_dir, exist_ok=True)
    if a.clean:
        n = 0
        for fn in os.listdir(a.out_dir):
            if fn.endswith('.pkl'):
                os.remove(os.path.join(a.out_dir, fn))
                n += 1
        print('🧹 清掉旧 pkl: %d 个' % n)

    iters = {}
    total = 0
    skipped = 0
    for gi, g in enumerate(games):
        try:
            batch = int(g.get('batch') or 0)
            winner = WIN_MAP.get(int(g.get('winner') or 0))
            if winner is None:
                skipped += 1
                continue
            moves = [(int(r), int(c)) for r, c in g['moves']]
            bds = boards_of(moves)
            if len(bds) < 2:
                skipped += 1
                continue
            it = (batch // a.iter_size) * a.iter_size
            fname = 'it%d_g%05d.pkl' % (it, gi)
            df = pd.DataFrame({
                'board': [b for b in bds],
                'player': [1 if i % 2 == 0 else -1 for i in range(len(bds))],
                'winner': [winner] * len(bds),
            })
            df.to_pickle(os.path.join(a.out_dir, fname))
            iters.setdefault(str(it), {'games': 0, 'files': []})
            iters[str(it)]['games'] += 1
            iters[str(it)]['files'].append({
                'file': fname,
                'steps': len(moves),
                'winner': winner,
                'worker': int(g.get('worker', 0) or 0),
                'game': gi,
            })
            total += 1
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print('  ⚠️ 跳过一局: %s' % e)

    keys = sorted(iters, key=int)
    idx = {
        'iters': iters,
        'iter_range': [int(keys[0]), int(keys[-1])] if keys else [0, 0],
        'total_games': total,
        'built_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': os.path.relpath(a.games, PROJ),
        'iter_size': a.iter_size,
        'note': 'v4(games.jsonl) → 老回放页格式；iter 实际是 batch 按 %d 局分桶' % a.iter_size,
    }
    tmp = a.index + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False)
    os.replace(tmp, a.index)

    dt = time.time() - t0
    print('✅ 转换完成  %.1fs' % dt)
    print('   局数     %d（跳过 %d）' % (total, skipped))
    print('   代数     %d 桶（每桶 %d 局），范围 %s..%s' % (
        len(keys), a.iter_size, idx['iter_range'][0], idx['iter_range'][1]))
    print('   pkl 目录 %s' % a.out_dir)
    print('   索引     %s' % a.index)
    return 0


if __name__ == '__main__':
    sys.exit(main())
