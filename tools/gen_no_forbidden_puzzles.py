# -*- coding: utf-8 -*-
"""从自己的对局里造「无禁手题库」—— 规则和我们引擎完全一致，无著作权问题

思路：一局棋下到快结束时，胜方一定握着必胜。把「胜方落最后一手之前的局面」
从 `logs/games.jsonl` 里截出来，用自由规则 VCF 求解器验证「该走方确实有强制胜」，
验证通过才收为一道题，答案就是求解器给出的首手。

好处：
  - 规则天然一致（五连或更长算胜，无三三/四四/长连禁手）
  - 没有连珠题库的抓禁手污染
  - 题目来自真实对局，分布和训练数据同源

用法:
  python tools/gen_no_forbidden_puzzles.py --games 200 --max-puzzles 60 \
      --out ~/Downloads/gomoku_tiku/data/selfplay_no_forbidden.jsonl

代价：每局要跑几次 VCF（0.1~1s），`--games` 越大越慢，默认只抽尾部几手。
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from vcf import vcf, five_points       # noqa: E402

W = H = 15


def replay(moves):
    """返回 board dict(move->player) 的程序列（第 i 项 = 走完 i 手后的局面）"""
    g = {}
    out = [dict(g)]
    for i, m in enumerate(moves):
        g[m] = 1 if i % 2 == 0 else 2
        out.append(dict(g))
    return out


def fmt(m):
    return '%c%d' % (ord('A') + m % W, H - m // W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', default=os.path.join(ROOT, 'logs/games.jsonl'))
    ap.add_argument('--out', default=os.path.expanduser(
        '~/Downloads/gomoku_tiku/data/selfplay_no_forbidden.jsonl'))
    ap.add_argument('--games', type=int, default=200, help='最多扫多少局')
    ap.add_argument('--max-puzzles', type=int, default=60)
    ap.add_argument('--back', type=int, nargs='+', default=[1, 3, 5],
                    help='从终局往回数第几手截题（1=最后一手前）')
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    rows = []
    with open(a.log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:                 # noqa: BLE001
                continue
            if d.get('moves'):
                rows.append(d)
    random.seed(a.seed)
    random.shuffle(rows)
    print('对局池 %d 局，最多扫 %d 局' % (len(rows), a.games), flush=True)

    out, t0 = [], time.time()
    scanned = 0
    for r in rows:
        if scanned >= a.games or len(out) >= a.max_puzzles:
            break
        moves = [(int(h) * W + int(w)) for h, w in r['moves']]
        if len(moves) < 8 or len(set(moves)) != len(moves):
            continue
        scanned += 1
        winner = int(r['winner'])
        if winner not in (1, 2):
            continue
        boards = replay(moves)
        L = len(moves)
        for back in a.back:
            ply = L - back
            if ply < 4:
                continue
            side = 1 if ply % 2 == 0 else 2     # 走完 ply 手后，该第 ply+1 手（0-based）走
            if side != winner:
                continue
            g = boards[ply]
            if five_points(g, side):
                continue                        # 已经能一手五连，太简单，不要
            mv = vcf(dict(g), side, max_depth=a.depth)
            if mv is None:
                continue
            out.append({'lib': 'selfplay-%s' % os.path.basename(a.log),
                        'idx': len(out), 'side': side, 'starts': 1,
                        'moves': moves[:ply], 'answer': int(mv),
                        'answer_xy': fmt(int(mv)),
                        'src_batch': int(r.get('batch') or 0),
                        'episode_len': L})
            if len(out) >= a.max_puzzles:
                break
        if scanned % 25 == 0:
            print('  扫 %d 局，收 %d 题，%.0fs' % (scanned, len(out), time.time() - t0), flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w') as f:
        for p in out:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print('\n扫 %d 局 → 收 %d 题，用时 %.0fs' % (scanned, len(out), time.time() - t0))
    print('已写 %s' % a.out)
    if out:
        c = {}
        for p in out:
            c[p['side']] = c.get(p['side'], 0) + 1
        print('先手分布: 黑 %d / 白 %d' % (c.get(1, 0), c.get(2, 0)))


if __name__ == '__main__':
    main()
