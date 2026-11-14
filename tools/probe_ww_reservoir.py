#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""白赢储库（ww_reservoir）体检：去重 / 抽样覆盖 / 局面质量。

回答三个问题，每个都只依赖 logs/ 下的落盘数据，不碰训练进程：

  1. 去重后到底有多少个**不同**局面？来自几局？
     🚨 `metrics.jsonl` 的 `ww_res` 是**条目数**（4 平面输入 × 8 重 D4 增广），
        真实多样性 = ww_res / 8 - 跨局重复。把它当"池子大小"会系统性高估 8 倍。

  2. 每个局面被抽过几次？单批覆盖池子的几成？
     用每个 update 时刻**真实**的 ww_res 序列算，不用终值（早期池子只有几十条，
     覆盖率 800%，是"把同一批局面反复灌"）。

  3. 白赢样本是"白真下出来的"还是"黑自爆送的"？
     同时看局面阶段分布（近开局的 +1 标签是噪声来源）。

用法：
    .venv/bin/python tools/probe_ww_reservoir.py
    .venv/bin/python tools/probe_ww_reservoir.py --ww-min-len 20 --board 15
    .venv/bin/python tools/probe_ww_reservoir.py --no-net      # 跳过网络估值（快）
    .venv/bin/python tools/probe_ww_reservoir.py --json out.json

判读口径（和 docs/white_collapse_2026-11-14.md 的判据配套）：
    · 池子大小看「去重局面数」，不看 ww_res。
    · 单批覆盖 < 30% ⇒ 白赢配额大半在重复背同一批局面。
    · 近开局（盘上 <=4 子）条目占比高 ⇒ +1 标签在和主 buffer 的 -1 对撞。
"""
import argparse
import collections
import hashlib
import json
import os
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


# ── 数据读取 ────────────────────────────────────────────────
def load_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue        # 训练正在写的半行
    return out


def white_positions(game):
    """一局里「白走」的局面数（白走 = 偶数下标 1,3,5...，落子前）。"""
    return game['episode_len'] // 2


# ── D4 归一化去重 ───────────────────────────────────────────
def canon_key(moves, k, N):
    """落前 k 手的局面 → D4 八变换里最小的那个做指纹（与棋盘朝向无关）。"""
    black, white = {}, {}
    for i, (r, c) in enumerate(moves[:k]):
        (black if i % 2 == 0 else white)[(r, c)] = 1
    cells = [(r, c, 1, 1 if (r, c) in white else 0) for (r, c) in black] + \
            [(r, c, 0, 1) for (r, c) in white if (r, c) not in black]
    best = None
    for t in range(8):
        pts = []
        for r, c, bb, ww in cells:
            rr, cc = (c, r) if t & 1 else (r, c)
            if t & 2:
                rr = N - 1 - rr
            if t & 4:
                cc = N - 1 - cc
            pts.append((rr * N + cc, bb, ww))
        pts.sort()
        s = b''.join(bytes([bb]) + bytes([ww]) + idx.to_bytes(2, 'big')
                     for idx, bb, ww in pts)
        if best is None or s < best:
            best = s
    return hashlib.sha1(best).hexdigest()[:16]


# ── 质量：白是否主动造活四 / 黑是否该堵未堵 ─────────────────
def _wins_at(b, r, c, p, N, WIN):
    for dr, dc in DIRS:
        cnt = 1
        for s in (1, -1):
            rr, cc = r + dr * s, c + dc * s
            while 0 <= rr < N and 0 <= cc < N and b.get((rr, cc)) == p:
                cnt += 1
                rr += dr * s
                cc += dc * s
        if cnt >= WIN:
            return True
    return False


def _must_points(b, p, N, WIN):
    return [(r, c) for r in range(N) for c in range(N)
            if (r, c) not in b and _wins_at(b, r, c, p, N, WIN)]


def _live_four_after(b, r, c, p, N):
    b[(r, c)] = p
    try:
        for dr, dc in DIRS:
            cnt, ends = 1, []
            for s in (1, -1):
                rr, cc = r + dr * s, c + dc * s
                while 0 <= rr < N and 0 <= cc < N and b.get((rr, cc)) == p:
                    cnt += 1
                    rr += dr * s
                    cc += dc * s
                ends.append((rr, cc))
            if cnt == 4 and all(0 <= e[0] < N and 0 <= e[1] < N and e not in b
                                for e in ends):
                return True
        return False
    finally:
        del b[(r, c)]


def game_quality(g, N, WIN):
    """返回 (黑该堵未堵的手序号列表, 白造活四的手序号列表)。"""
    b, blunder, active = {}, [], []
    for i, (r, c) in enumerate(g['moves']):
        mover = 1 if i % 2 == 0 else 2
        if mover == 1 and i >= 6:
            must = _must_points(b, 2, N, WIN)
            if must and (r, c) not in must and not _wins_at(b, r, c, 1, N, WIN):
                blunder.append(i + 1)
        elif mover == 2 and _live_four_after(b, r, c, 2, N):
            active.append(i + 1)
        b[(r, c)] = mover
    return blunder, active


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--board', type=int, default=15)
    ap.add_argument('--win', type=int, default=5)
    ap.add_argument('--ww-min-len', type=int, default=20,
                    help='与 GOMOKU_WW_MIN_LEN 保持一致（反捡漏阈值）')
    ap.add_argument('--no-net', action='store_true', help='跳过网络估值')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    N, WIN = a.board, a.win
    games = load_jsonl(os.path.join(a.root, 'logs/games.jsonl'))
    met = load_jsonl(os.path.join(a.root, 'logs/metrics.jsonl'))
    ups = [r for r in met if r.get('type') == 'update']
    if not games:
        print('没有 logs/games.jsonl（用 --save-games 启动才会有）')
        return 1

    white = [g for g in games if g['winner'] == 2]
    pool = [g for g in white if g['episode_len'] >= a.ww_min_len]
    rejected = [g for g in white if g['episode_len'] < a.ww_min_len]
    out = {'n_games': len(games), 'n_white_win': len(white),
           'n_pool_games': len(pool), 'n_rejected_short': len(rejected),
           'ww_min_len': a.ww_min_len}

    print('=== 0. 规模 ===')
    print('对局 %d | 白胜 %d | 进储库(手数>=%d) %d | 反捡漏划走 %d %s'
          % (len(games), len(white), a.ww_min_len, len(pool), len(rejected),
             sorted(g['episode_len'] for g in rejected)))
    entries = sum(white_positions(g) for g in pool)
    out.update({'white_pos_entries': entries, 'aug_entries': entries * 8})
    print('白走局面条目 %d | ×8 增广 = %d（这就是 metrics 里 ww_res 的口径）'
          % (entries, entries * 8))
    # ── 对账：每个 update 时刻的 ww_res 是否 = 8 × 已完成白赢局面数
    print('\n=== 1. 与 metrics.ww_res 对账 ===')
    ok = bad = 0
    detail = []
    for u in ups:
        ub = int(u.get('batch') or 0)
        calc = 8 * sum(white_positions(g) for g in pool if g['batch'] <= ub)
        actual = u.get('ww_res')
        if actual == calc:
            ok += 1
        else:
            bad += 1
            detail.append({'batch': ub, 'metrics': actual, 'rebuilt': calc})
    print('update 记录 %d 条，精确吻合 %d 条，不吻合 %d 条 %s'
          % (len(ups), ok, bad, detail[:3] if detail else ''))
    out.update({'reconcile_ok': ok, 'reconcile_bad': bad, 'reconcile_detail': detail})

    # ── 去重
    print('\n=== 2. 去重（D4 归一化） ===')
    pos_owner = {}                      # h -> set(局下标)
    per_game = {}
    for gi, g in enumerate(pool):
        keys = []
        for k in range(1, g['episode_len'] + 1):
            if (k - 1) % 2 != 1:        # 只要白走局面
                continue
            h = canon_key(g['moves'], k - 1, N)
            keys.append(h)
            pos_owner.setdefault(h, set()).add(gi)
        per_game[gi] = keys
    distinct = len(pos_owner)
    dup = [(h, s) for h, s in pos_owner.items() if len(s) > 1]
    shared_entries = sum(len(s) * 8 for h, s in dup) if dup else 0
    biggest = max(dup, key=lambda x: len(x[1])) if dup else None
    print('不同局面 %d 个（来自 %d 局，平均每局 %.1f 个白走局面）'
          % (distinct, len(pool), entries / max(1, len(pool))))
    print('跨局重复的局面 %d 个；其中最大一个被 %d 局共享（独占 %d 条目 = %.1f%% 储库）'
          % (len(dup), len(biggest[1]) if biggest else 0,
             len(biggest[1]) * 8 if biggest else 0,
             100.0 * (len(biggest[1]) * 8) / max(1, entries * 8)))
    out.update({'distinct_positions': distinct, 'dup_positions': len(dup),
                'dup_biggest_games': len(biggest[1]) if biggest else 0})

    # ── 阶段分布
    print('\n=== 3. 局面阶段分布（白走局面） ===')
    stage = collections.Counter()
    occ_all = []
    for g in pool:
        for i in range(g['episode_len']):
            if i % 2 == 1:
                occ_all.append(i)
                stage['0-4子' if i <= 4 else ('5-9子' if i <= 9 else
                      ('10-19子' if i <= 19 else '20+子'))] += 1
    for k in ('0-4子', '5-9子', '10-19子', '20+子'):
        n = stage[k]
        print('  盘上%-8s %4d 个 (%.0f%%) → 条目 %d'
              % (k, n, 100.0 * n / len(occ_all), n * 8))
    print('  中位落子序号（盘上已有子数） %d' % statistics.median(occ_all))
    out['stage'] = dict(stage)

    # ── 抽样覆盖（用真实 ww_res 序列）
    print('\n=== 4. 抽样覆盖（按每个 update 的真实 ww_res） ===')
    def p_untouched(res, k):
        if res - 8 < k:
            return 0.0
        p = 1.0
        for i in range(k):
            p *= (res - 8 - i) / (res - i)
        return p

    draws = collections.defaultdict(float)
    rows = []
    for u in ups:
        res = int(u.get('ww_res') or 0)
        cnt = int((u.get('cls_batch') or [0, 0, 0, 0])[2])
        if res <= 0 or cnt <= 0:
            continue
        P = res / 8.0
        rows.append({'batch': u['batch'], 'ww_res': res, 'draws': cnt,
                     'pool_pos': P, 'covered': P * (1 - p_untouched(res, cnt)),
                     'p_each': cnt / P})
        for gi, g in enumerate(pool):
            if g['batch'] <= u['batch']:
                draws[gi] += cnt / res * 8
    if rows:
        print('批次   ww_res  抽白赢  池内局面  单批覆盖局面  每局面每批被抽')
        for r in rows[:3] + rows[-3:]:
            print('  %5d %6d %7d %8.0f %11.1f %13.1f%%'
                  % (r['batch'], r['ww_res'], r['draws'], r['pool_pos'],
                     r['covered'], 100.0 * r['p_each']))
        last = rows[-1]
        print('末批：抽 %d 条 → 覆盖 %.1f/%d 个局面 = %.0f%%'
              % (last['draws'], last['covered'], last['pool_pos'],
                 100.0 * last['covered'] / last['pool_pos']))
    vals = [v for v in draws.values()]
    if vals:
        print('每局面累计被抽：均值 %.1f | 中位 %.1f | 最少 %.1f | 最多 %.1f'
              % (statistics.mean(vals), statistics.median(vals),
                 min(vals), max(vals)))
        early = sorted(draws.items(), key=lambda kv: -kv[1])[:1]
        if early:
            gi = early[0][0]
            print('  最多次数来自局 %d（%d 手）→ %.1f 次'
                  % (pool[gi]['batch'], pool[gi]['episode_len'], early[0][1]))
    out['coverage_last'] = rows[-1] if rows else None
    out['draws_stats'] = ({'mean': statistics.mean(vals),
                           'median': statistics.median(vals),
                           'min': min(vals), 'max': max(vals)} if vals else None)

    # ── 质量
    print('\n=== 5. 质量：白是主动赢还是黑自爆 ===')
    bl_tot = act_tot = 0
    blunder_only = pure_active = both = 0
    late = 0
    per = []
    for g in pool:
        bl, act = game_quality(g, N, WIN)
        bl_tot += len(bl)
        act_tot += len(act)
        blunder_only += bool(bl) and not act
        pure_active += bool(act) and not bl
        both += bool(bl) and bool(act)
        late += any(x >= g['episode_len'] - 5 for x in bl)
        per.append({'batch': g['batch'], 'len': g['episode_len'],
                    'blunder': bl, 'live4': act})
    n = max(1, len(pool))
    print('白造出过活四 %d 局 (%.0f%%) | 纯黑自爆 %d 局 | 两者都有 %d 局'
          % (n - blunder_only, 100.0 * (n - blunder_only) / n,
             blunder_only, both))
    print('黑自爆次数合计 %d | 白造活四次数合计 %d' % (bl_tot, act_tot))
    print('终局前 3 个黑回合内"该堵未堵" %d 局 (%.0f%%)' % (late, 100.0 * late / n))
    print('手数中位 %d | <=24 手 %d 局 | >=35 手 %d 局'
          % (statistics.median([g['episode_len'] for g in pool]),
             sum(1 for g in pool if g['episode_len'] <= 24),
             sum(1 for g in pool if g['episode_len'] >= 35)))
    out.update({'blunder_total': bl_tot, 'live4_total': act_tot,
                'games_pure_blunder': blunder_only, 'games_both': both,
                'games_late_blunder': late, 'per_game': per})

    # ── 校准（可选）
    if not a.no_net:
        print('\n=== 6. value 校准抽查（当前权重） ===')
        try:
            sys.path.insert(0, os.path.join(a.root, 'web'))
            import engine
            net = engine.get_net(None)
            W = engine.WIDTH
            print('权重 %s' % engine.default_model())

            def black_pct(b):
                _p, v, _l = engine.clean_eval(net, b)
                return 50 + 50 * v if b.current_player == 1 else 50 - 50 * v

            b0 = engine.make_board()
            print('空盘(黑走) 黑胜率 %.1f%%' % black_pct(b0))
            p1 = []
            for g in games[-120:]:
                if len(g['moves']) < 4:
                    continue
                b = engine.make_board()
                b.do_move(g['moves'][0][0] * W + g['moves'][0][1])
                p1.append(black_pct(b))
            if p1:
                print('黑1之后(白走) 网络给黑胜率 均值 %.1f%% 中位 %.1f%%（n=%d）'
                      % (statistics.mean(p1), statistics.median(p1), len(p1)))
            real = 100.0 * sum(1 for g in games if g['winner'] == 1) / len(games)
            print('实际黑胜率（全库）%.1f%%' % real)
            out['calib'] = {'empty_black': black_pct(b0),
                            'after_b1_black': statistics.median(p1) if p1 else None,
                            'real_black': real}
        except Exception as e:              # noqa: BLE001
            print('跳过（%s）' % e)

    if a.json:
        with open(a.json, 'w') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print('\n已写 %s' % a.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
