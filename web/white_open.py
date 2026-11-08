"""白棋（后手）开局指标 —— 从 logs/games.jsonl 的落子手顺里算

⚠️ 口径修正（2026-11）：白1 距黑1 必须按"黑1 是否天元"拆开，否则会把
   "黑不占中心 → 白抢天元（距 4~7 格，正确行为）"混进"白贴身度"，
   看起来像白棋越下越远。两种行为是：
     · 黑1 = 天元  → 白贴身（距 1~2 格）      ← 真·贴身度
     · 黑1 ≠ 天元  → 白1 落天元（抢中心）      ← 单独统计"抢中心率"

指标：
  center.w1d_median    黑1 天元局：白1 与黑1 的切比雪夫距离中位（越小越贴身）
  center.contact_rate  黑1 天元局：白1 落在黑1 周围 2 格内的比例
  center.far_rate      黑1 天元局：白1 距黑1 ≥4 格的比例（"跑边角"）
  center.w2d_median    黑1 天元局：白2 到最近黑子的距离
  center.spread        白1 偏移类分布熵 / ln(5)（越小越集中）
  offcenter.grab_rate  黑1 非天元局：白1 落在天元 1 格内的比例
"""
import json
import math
import os
import statistics
from collections import Counter

N = 15
CENTER = (7, 7)


def set_board(n):
    """按棋盘边长重设天元与格数：9×9 的天元是 (4,4)，15×15 才是 (7,7)。
    写死会导致换盘后「黑1 天元局」永远是 0 局，贴身度整块显示为空。"""
    global N, CENTER
    n = int(n)
    N = n
    CENTER = (n // 2, n // 2)


def load_games(path, limit=3000):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                      # 训练正在追加，最后一行可能半截
    return out[-limit:]


def offset_class(w1, b1):
    key = tuple(sorted((abs(w1[0] - b1[0]), abs(w1[1] - b1[1]))))
    if key == (0, 1):
        return '贴直线'
    if key == (1, 1):
        return '贴斜线'
    if key == (0, 2):
        return '隔一'
    if key in ((1, 2), (2, 2)):
        return '隔二斜'
    return '更远'


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def analyze(recs, window=20, last=50):
    per = []
    for r in recs:
        mv = r.get('moves') or []
        if len(mv) < 4:
            continue
        b1, w1, b2, w2 = mv[0], mv[1], mv[2], mv[3]
        is_center = list(b1) == list(CENTER)
        rec = {'batch': r.get('batch'), 'center': is_center,
               'd': cheb(w1, b1), 'grab': cheb(w1, CENTER) <= 1,
               'contact': cheb(w1, b1) <= 2,
               'cls': offset_class(w1, b1),
               'w2d': min(cheb(w2, b1), cheb(w2, b2))}
        per.append(rec)
    if not per:
        return {'ok': False, 'n': 0}

    cen = [p for p in per if p['center']]
    off = [p for p in per if not p['center']]
    tail = cen[-last:]
    cnt = Counter(p['cls'] for p in tail)
    n = sum(cnt.values()) or 1
    ent = -sum((v / n) * math.log(v / n) for v in cnt.values()) / math.log(5)

    # 滚动序列：天元局距离 / 非天元局抢中心率，各自在自己的子集上滚
    roll_d, roll_grab = [], []
    hist_c, hist_o = [], []
    for p in per:
        if p['center']:
            hist_c.append(p['d'])
        else:
            hist_o.append(1 if p['grab'] else 0)
        w = hist_c[-window:]
        roll_d.append(round(statistics.median(w), 2) if w else None)
        g = hist_o[-window:]
        roll_grab.append(round(100.0 * sum(g) / len(g), 1) if g else None)

    return {
        'ok': True,
        'n': len(per),
        'center': {
            'n': len(cen),
            'share': round(100.0 * len(cen) / len(per), 1),
            'w1d_median': statistics.median([p['d'] for p in tail]) if tail else None,
            'contact_rate': round(100.0 * sum(p['contact'] for p in tail) / len(tail), 1) if tail else None,
            'far_rate': round(100.0 * sum(1 for p in tail if p['d'] >= 4) / len(tail), 1) if tail else None,
            'w2d_median': statistics.median([p['w2d'] for p in tail]) if tail else None,
            'spread': round(ent, 3),
            'classes': {k: round(100.0 * cnt.get(k, 0) / n, 1) for k in
                        ('贴直线', '贴斜线', '隔一', '隔二斜', '更远')},
        },
        'offcenter': {
            'n': len(off),
            'grab_rate': round(100.0 * sum(1 for p in off[-last:] if p['grab']) / max(len(off[-last:]), 1), 1),
        },
        'series': [{'batch': p['batch'], 'center': p['center']} for p in per],
        'roll_dist': roll_d,
        'roll_grab': roll_grab,
    }
