#!/usr/bin/env python3
"""威胁健康探针（移植自老项目 threat_response_rate.py + analyze_leak_v40.py）

纯数据，不载模型，秒级跑完。定义与老项目对齐：
  段长 L（同色连续）+ 两端是否在盘内且为空
    L>=5        → 成五
    L==4 双开   → 活四（已成杀）
    L==4 单开   → 冲四（唯一必堵点）
    L==3 双开   → 活三
    L==3 单开   → 眠三
    L==2 双开   → 活二
  封堵 = 对手下一手落在该段的紧邻空端点之一
  漏堵被杀 = 漏堵后 2 手内对局终止且胜方 == 造威胁方
数据: logs/games.jsonl（黑先交替，moves=[[r,c],...]）
"""
import json
import sys
from collections import Counter, defaultdict

N = 15
DIRS = [(0, 1), (1, 0), (1, 1), (1, -1)]
TYPES = ['活二', '活三', '眠三', '冲四', '活四']


def grid_after(moves, upto):
    g = [[0] * N for _ in range(N)]
    for i, (r, c) in enumerate(moves[:upto], start=1):
        g[r][c] = 1 if i % 2 else 2
    return g


def segs_through(g, r, c, color):
    """枚举经过 (r,c) 的同色段，返回 [(type, [端点...], cells)]"""
    out = []
    for dr, dc in DIRS:
        s = 0
        while True:
            rr, cc = r - dr * (s + 1), c - dc * (s + 1)
            if 0 <= rr < N and 0 <= cc < N and g[rr][cc] == color:
                s += 1
            else:
                break
        e = 0
        while True:
            rr, cc = r + dr * (e + 1), c + dc * (e + 1)
            if 0 <= rr < N and 0 <= cc < N and g[rr][cc] == color:
                e += 1
            else:
                break
        L = s + e + 1
        ends = []
        for k in (s + 1, -(e + 1)):
            rr, cc = r - dr * k, c - dc * k
            ends.append((rr, cc) if (0 <= rr < N and 0 <= cc < N and g[rr][cc] == 0) else None)
        n_open = sum(1 for x in ends if x is not None)
        if L >= 5:
            t = '成五'
        elif L == 4:
            t = '活四' if n_open == 2 else ('冲四' if n_open == 1 else None)
        elif L == 3:
            t = '活三' if n_open == 2 else ('眠三' if n_open == 1 else None)
        elif L == 2:
            t = '活二' if n_open == 2 else None
        else:
            t = None
        if t:
            out.append((t, [x for x in ends if x], [(r - dr * i2, c - dc * i2) for i2 in range(-s, e + 1)]))
    return out


def main(path='logs/games.jsonl', block=25):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    stat = defaultdict(lambda: {'n': 0, 'blocked': 0, 'loose': 0, 'killed': 0})
    order = []
    for rec in recs:
        mv = [tuple(x) for x in rec['moves']]
        L = len(mv)
        for i in range(1, L):                       # 第 i 手后（1-based）
            p = 1 if i % 2 == 1 else 2
            r, c = mv[i - 1]
            g = grid_after(mv, i)
            for t, ends, cells in segs_through(g, r, c, p):
                if t == '成五':
                    continue
                key = (p, t)
                d = stat[key]
                d['n'] += 1
                if i >= L:                          # 最后一手后无对手应对
                    continue
                nr, nc = mv[i]
                if (nr, nc) in ends:
                    d['blocked'] += 1
                else:
                    d['loose'] += 1
                    # 漏堵后 2 手内结束且造威胁方获胜
                    winner = rec['winner']
                    if winner == p and (L - i) <= 2:
                        d['killed'] += 1
        order.append(rec['batch'])

    print(f"棋谱 {len(recs)} 局（batch {order[0]}~{order[-1]}）\n")
    print(f"{'造威胁方':>8}{'威胁类型':>8}{'出现':>6}{'被堵':>6}{'漏堵':>6}{'漏堵率':>8}{'漏堵后仍赢':>10}")
    for p in (1, 2):
        for t in TYPES:
            d = stat.get((p, t))
            if not d or d['n'] == 0:
                continue
            n = d['blocked'] + d['loose']
            tag = '（已成杀，无可堵）' if t == '活四' else ''
            print(f"{'黑' if p==1 else '白':>8}{t:>8}{d['n']:>6}{d['blocked']:>6}{d['loose']:>6}"
                  f"{100*d['loose']/max(n,1):>7.0f}%{100*d['killed']/max(d['loose'],1):>9.0f}%{tag}")

    # ── 自吃率（数据口径）：当前方有"一手成五"，实际走了没有 ──
    win1 = took = 0
    for rec in recs:
        mv = [tuple(x) for x in rec['moves']]
        L = len(mv)
        for i in range(L):                       # i 手已落，轮到 p
            p = 1 if i % 2 == 0 else 2
            g = grid_after(mv, i)
            pts = set()
            for r in range(N):
                for c in range(N):
                    if g[r][c] == p:
                        for t, ends, _ in segs_through(g, r, c, p):
                            if t in ('冲四', '活四'):
                                pts.update(ends)
            if pts:
                win1 += 1
                if tuple(mv[i]) in pts:
                    took += 1
    print(f"\n自吃率（数据口径）：有一手成五的局面 {win1} 个，实际走了制胜点 {took} 个 = "
          f"{100*took/max(win1,1):.1f}%")


    # 近 25 局 vs 更早：漏堵率趋势
    print(f"\n按 {block} 局分块（活三+冲四 的漏堵率）")
    print(f"{'区间':>14}{'黑造次':>7}{'黑漏堵':>8}{'白造次':>7}{'白漏堵':>8}")
    for i in range(0, len(recs), block):
        chunk = recs[i:i + block]
        agg = defaultdict(lambda: [0, 0])
        for rec in chunk:
            mv = [tuple(x) for x in rec['moves']]
            L = len(mv)
            for j in range(1, L):
                p = 1 if j % 2 == 1 else 2
                r, c = mv[j - 1]
                g = grid_after(mv, j)
                for t, ends, _ in segs_through(g, r, c, p):
                    if t not in ('活三', '冲四'):
                        continue
                    agg[p][0] += 1
                    if j >= L:
                        continue
                    if tuple(mv[j]) not in ends:
                        agg[p][1] += 1
        a, b = agg[1], agg[2]
        print(f"{chunk[0]['batch']:>6}-{chunk[-1]['batch']:<7}"
              f"{a[0]:>7}{100*a[1]/max(a[0],1):>7.0f}%{b[0]:>7}{100*b[1]/max(b[0],1):>7.0f}%")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'logs/games.jsonl')
