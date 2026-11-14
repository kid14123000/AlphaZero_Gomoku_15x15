# -*- coding: utf-8 -*-
"""v4 训练监控的真实指标 —— 供 /api/training 的 `v4` 字段使用。

为什么单独一个模块
------------------
参考版 app.py 的 /api/training 是「SSH 到训练机跑 shell、解析 v36 风格日志」，
它的指标模型是「代 / 教练 / 门控 / 熔断」。v4 这条线（junxiaosong 派生 + 并行自对弈）
没有教练、没有门控、没有「代」这个概念——只有局。

这里只从本机能拿到的三份产物算，一个数都不编：
    logs/metrics.jsonl  start / game(batch,episode_len,winner,game_seconds) / update(loss,entropy,ev_new,kl)
    logs/games.jsonl    每局完整落子手顺（防守率要用它算）
    checkpoints/        ckpt_itN.model

缓存三层（首页冷启动几秒，之后每个请求都是几十毫秒）
    1. 两个 jsonl 按文件大小增量读，只解析新增的行
    2. 防守率逐局结果存 sidecar JSON（重算一局 12 ms × 几百局太贵）
    3. 汇总结果按 (文件大小, 局数) 记忆
"""
import json
import os
import statistics
import time

BUCKET = 250          # 趋势图每档多少局
RECENT = 200          # 顶部 KPI 的滚动窗口（局）
DEFENSE_WIN = 300    # 防守率只看最近这么多局（每局约 12 ms，冷启 3-4 s；结果落 sidecar）
SIDECAR = 'v4_combat_cache.json'

_state = {'games': {'size': -1, 'off': 0, 'n': 0, 'rows': []},
          'upd': {'size': -1, 'off': 0, 'rows': []},
          'combat': {}, 'hot': None, 'combat_key': None}


# ────────────────────────────── jsonl 增量读 ──────────────────────────────
def _tail_lines(path, st, parse):
    """把 path 里新增的行解析出来追加到 st['rows']；文件变小（被截断/重建）则整体重来。"""
    try:
        size = os.path.getsize(path)
    except OSError:
        return st['rows']
    if size == st['size']:
        return st['rows']
    if size < st['size']:                      # 截断 / 换文件 → 重头
        st.update({'size': -1, 'off': 0, 'n': 0, 'rows': []})
        st['size'] = 0
    try:
        with open(path, 'rb') as f:
            f.seek(st['off'])
            chunk = f.read()
            st['off'] = f.tell()
    except OSError:
        return st['rows']
    st['size'] = size
    for raw in chunk.split(b'\n'):
        if not raw.strip():
            continue                     # 末尾半行/空行：下一轮从 off 之后才读到，安全
        try:
            d = json.loads(raw)
        except Exception:
            continue                     # 训练正在写的半行
        r = parse(d)
        if r is not None:
            r['_idx'] = len(st['rows'])      # 行号：同 batch 可能被续训重跑，只有行号唯一
            st['rows'].append(r)
    st['n'] = len(st['rows'])
    return st['rows']


def _parse_metric(d):
    if d.get('type') == 'start' or 'game_batch_num' in d:
        return {'k': 'start', 't': d.get('t') or 0,
                'target': d.get('game_batch_num'), 'board': d.get('board'),
                'n_in_row': d.get('n_in_row'), 'playout': d.get('n_playout')}
    if d.get('loss') is not None:
        return {'k': 'upd', 't': d.get('t') or 0, 'batch': int(d.get('batch') or 0),
                'loss': float(d['loss']), 'entropy': float(d.get('entropy') or 0),
                'ev_new': float(d.get('ev_new') or 0), 'kl': float(d.get('kl') or 0),
                'lr_mult': float(d.get('lr_multiplier') or 0)}
    if 'winner' in d:
        return {'k': 'game', 't': d.get('t') or 0, 'batch': int(d.get('batch') or 0),
                'winner': int(d['winner']), 'len': int(d.get('episode_len') or 0),
                'sec': float(d.get('game_seconds') or 0), 'worker': d.get('worker')}
    return None


def _parse_game(d):
    mv = d.get('moves') or []
    if not mv:
        return None
    # _idx 在 _tail_lines 里补
    return {'batch': int(d.get('batch') or 0), 'winner': int(d.get('winner') or 0),
            'len': int(d.get('episode_len') or len(mv)), 'moves': mv}


# ────────────────────────────── 防守率 ──────────────────────────────
def _gkey(g):
    """战斗指标的缓存键。batch 会因续训重跑而重复，len 也可能撞，
    所以带上 games.jsonl 的行号（追写文件，行号稳定）。"""
    return '%d_%d_%d' % (g.get('_idx', -1), g['batch'], g['len'])


def _combat_mod(web_dir):
    import sys
    if web_dir not in sys.path:
        sys.path.insert(0, web_dir)
    import combat
    return combat


def _combat_rows(root, games, board, win, limit):
    """对最近 limit 局算战斗指标；结果按 (batch,手数) 存 sidecar，避免每次重算。"""
    web_dir = os.path.join(root, 'web')
    side = os.path.join(root, 'web_flask', SIDECAR)
    cache = _state['combat']
    key = os.path.join(root, 'web_flask', SIDECAR) + ':%s:%s' % (board, win)
    if _state['combat_key'] != key:
        cache = {}
        try:
            with open(side, 'r') as f:
                cache = {k: v for k, v in json.load(f).items()}
        except Exception:
            cache = {}
        _state['combat'] = cache
        _state['combat_key'] = key

    try:
        combat = _combat_mod(web_dir)
        combat.set_board(int(board or 15), int(win or 5))
    except Exception:
        return None

    tail = games[-limit:]
    added = 0
    for g in tail:
        k = _gkey(g)
        if k in cache:
            continue
        try:
            cache[k] = combat.analyze(g['moves'])
        except Exception:
            cache[k] = None
        added += 1
    if added:
        # 只留最近若干局，防止 sidecar 无限长
        keep = {}
        for g in games[-1500:]:
            k = _gkey(g)
            if k in cache:
                keep[k] = cache[k]
        try:
            tmp = side + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(keep, f)
            os.replace(tmp, side)
        except Exception:
            pass
        _state['combat'] = cache

    res = [cache.get(_gkey(g)) for g in tail]
    return [r for r in res if r], added


def _agg_block(rows):
    n = hit = 0
    mn = mh = 0
    tn = th = 0
    for r in rows:
        n += r.get('block_chances') or 0
        hit += r.get('block_hits') or 0
        # must/three 是比例，用机会数加权得不到，这里退回逐局均值（页面标注是均值口径）
        if r.get('must_block') is not None:
            mn += 1; mh += r['must_block']
        if r.get('three_block') is not None:
            tn += 1; th += r['three_block']
    return {'block': (hit / n * 100) if n else None,
            'must_block': (mh / mn * 100) if mn else None,
            'three_block': (th / tn * 100) if tn else None,
            'chances': n, 'games': len(rows)}


# ────────────────────────────── 主入口 ──────────────────────────────
def collect(root, tlog=None):
    mpath = os.path.join(root, 'logs', 'metrics.jsonl')
    gpath = os.path.join(root, 'logs', 'games.jsonl')

    mrows = _tail_lines(mpath, _state['upd'], _parse_metric)
    grows = _tail_lines(gpath, _state['games'], _parse_game)

    starts = [r for r in mrows if r['k'] == 'start']
    ups = [r for r in mrows if r['k'] == 'upd']
    games = [r for r in mrows if r['k'] == 'game']
    start = starts[-1] if starts else {}

    cur = games[-1]['batch'] if games else (grows[-1]['batch'] if grows else 0)
    target = start.get('target')
    board = 15
    win = 5
    if start.get('board'):
        try:
            board = int(str(start['board']).split('x')[0])
        except Exception:
            pass
    win = int(start.get('n_in_row') or 5)

    # 只统计本轮（最后一次 start 之后）；metrics.jsonl 跨轮追加
    t0 = start.get('t') or 0
    cur_g = [g for g in games if g['t'] >= t0] or games

    # ── 吞吐：首选 game_seconds 的中位数（并行下「相邻出局间隔 × 工兵数」也行，
    #    但单局耗时是页面要显示的东西，直接用日志里记的真实值）
    secs = [g['sec'] for g in cur_g[-RECENT:] if g['sec'] > 0]
    per_game = statistics.median(secs) if secs else None
    n_workers = max(1, len({g['worker'] for g in cur_g if g.get('worker') is not None}) or 8)
    rate = (n_workers * 60.0 / per_game) if per_game else None

    # 最近窗口的胜率/手数
    tail = cur_g[-RECENT:]
    bw = sum(1 for g in tail if g['winner'] == 1)
    ww = sum(1 for g in tail if g['winner'] == 2)
    tw = sum(1 for g in tail if g['winner'] == -1)
    n_tail = max(1, len(tail))

    lens = [g['len'] for g in tail if g['len'] > 0]

    # 防守率（用 games.jsonl 的手顺，倒序取最近 DEFENSE_WIN 局）
    defense = None
    if grows:
        pairs, added = _combat_rows(root, grows, board, win, DEFENSE_WIN)
        if pairs:
            defense = _agg_block(pairs)
            defense['computed_now'] = added

    # ── 曲线：按 BUCKET 局分档（用本轮全部局，不截窗口）
    series = []
    for i in range(0, len(cur_g), BUCKET):
        blk = cur_g[i:i + BUCKET]
        if len(blk) < max(20, BUCKET // 5):
            continue
        nB = len(blk)
        L = [g['len'] for g in blk if g['len'] > 0]
        series.append({
            'batch': blk[-1]['batch'],
            'n': nB,
            'black': round(sum(1 for g in blk if g['winner'] == 1) / nB * 100, 1),
            'white': round(sum(1 for g in blk if g['winner'] == 2) / nB * 100, 1),
            'tie': round(sum(1 for g in blk if g['winner'] == -1) / nB * 100, 1),
            'len': round(statistics.mean(L), 1) if L else None,
            'len_med': statistics.median(L) if L else None,
        })
    # 防守率按同样分档（只覆盖最近 DEFENSE_WIN 局所在的最后几档）
    if grows:
        per = []
        cache = _state['combat']
        for g in grows[-DEFENSE_WIN:]:
            r = cache.get(_gkey(g))
            if r:
                per.append((g['batch'], r))
        for si, s in enumerate(series):
            lo = series[si - 1]['batch'] if si > 0 else -1
            rows = [r for b, r in per if lo < b <= s['batch']]
            if not rows:
                continue
            a = _agg_block(rows)
            if a['block'] is not None:
                s['block'] = round(a['block'], 1)

    # loss / entropy：取最近若干次更新，等距抽样到 ~200 点
    u = ups[-600:] if len(ups) > 600 else ups
    step = max(1, len(u) // 200)
    us = u[::step]
    loss_series = [{'batch': r['batch'], 'loss': round(r['loss'], 4),
                    'entropy': round(r['entropy'], 4)} for r in us]
    last_u = ups[-1] if ups else None
    ma = ups[-20:] if len(ups) >= 20 else ups
    loss_ma = round(statistics.mean([r['loss'] for r in ma]), 4) if ma else None
    ent_ma = round(statistics.mean([r['entropy'] for r in ma]), 4) if ma else None

    eta_h = None
    if target and rate and target > cur:
        eta_h = round((target - cur) / rate / 60.0, 1)

    return {
        'running': bool(cur_g and (time.time() - cur_g[-1]['t']) < 900),
        'board': board, 'win': win, 'workers': n_workers,
        'playout': start.get('playout'), 'started_at': start.get('t'),
        'games_cur': cur, 'games_target': target,
        'games_pct': round(cur / target * 100, 2) if target else None,
        'rate_per_min': round(rate, 2) if rate else None,
        'per_game_sec': round(per_game, 1) if per_game else None,
        'eta_hours': eta_h,
        'loss': round(last_u['loss'], 4) if last_u else None,
        'loss_ma20': loss_ma,
        'entropy': round(last_u['entropy'], 4) if last_u else None,
        'entropy_ma20': ent_ma,
        'ev_new': round(last_u['ev_new'], 4) if last_u else None,
        'kl': round(last_u['kl'], 5) if last_u else None,
        'lr_mult': round(last_u['lr_mult'], 4) if last_u else None,
        'n_updates': len(ups),
        'black_wr': round(bw / n_tail * 100, 1),
        'white_wr': round(ww / n_tail * 100, 1),
        'tie_rate': round(tw / n_tail * 100, 1),
        'n_recent': len(tail),
        'len_mean': round(statistics.mean(lens), 1) if lens else None,
        'len_med': statistics.median(lens) if lens else None,
        'defense': defense,
        'series': series,
        'loss_series': loss_series,
        'total_games_logged': len(grows),
    }
