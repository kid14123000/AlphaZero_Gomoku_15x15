#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local dashboard server for the 15x15 AlphaZero Gomoku training run.

Reads logs/metrics.jsonl (written by train.py) and serves:
  GET /                 -> dashboard page
  GET /api/metrics      -> aggregated JSON (rolling averages computed here)

Pure stdlib, no dependencies. Run:
    python3 web/server.py --port 8770
Then open http://localhost:8770
"""
import argparse
import gzip
import json
import os
import statistics
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_LOG = os.path.join(ROOT, 'logs', 'metrics.jsonl')
DEFAULT_GAMES = os.path.join(ROOT, 'logs', 'games.jsonl')

import sys
sys.path.insert(0, HERE)
import white_open  # noqa: E402
from replay_store import GameStore  # noqa: E402

_GAMES = GameStore(DEFAULT_GAMES, combat_cache=os.path.join(ROOT, 'logs', 'combat_cache.json'))


def read_records(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except ValueError:
                # partially written last line while training is appending
                continue
    return recs


def rolling(values, window):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def mean(xs):
    return sum(xs) / len(xs) if xs else None


# ─────────────────────────────────────────────────────────────
# 局面存档（对弈页存 → MCTS / 网络前向页读）
# ─────────────────────────────────────────────────────────────
POSITIONS = os.path.join(HERE, 'positions.json')
_pos_lock = threading.Lock()


def _read_positions():
    if not os.path.exists(POSITIONS):
        return {}
    try:
        with open(POSITIONS, 'r') as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def list_positions():
    d = _read_positions()
    out = [{'name': k, 'n': len(v.get('stones') or []),
            'ts': v.get('ts', 0), 'note': v.get('note', '')} for k, v in d.items()]
    out.sort(key=lambda x: -x['ts'])
    return out


def save_position(name, stones, note=''):
    with _pos_lock:
        d = _read_positions()
        d[name] = {'stones': [{'r': int(s['r']), 'c': int(s['c'])} for s in stones],
                   'note': note, 'ts': time.time()}
        with open(POSITIONS, 'w') as f:
            json.dump(d, f, ensure_ascii=False, indent=1)


def get_position(name):
    return _read_positions().get(name)


def delete_position(name):
    with _pos_lock:
        d = _read_positions()
        if name in d:
            del d[name]
            with open(POSITIONS, 'w') as f:
                json.dump(d, f, ensure_ascii=False, indent=1)


def _read_start_record(path=None):
    """读 metrics.jsonl 里最后一条 start 记录（= 当前这轮的配置）。

    续训会在同一个文件里追加新的 start，所以要取最后一条而不是第一条。
    """
    p = path or DEFAULT_LOG
    last = None
    try:
        with open(p) as f:
            for line in f:
                if '"start"' not in line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get('type') == 'start':
                    last = r
    except IOError:
        pass
    return last or {}


def _sync_board_modules(path=None, start=None):
    """把棋盘规格同步给依赖它的模块（white_open 的天元、combat 的 N/WIN）。

    这些模块里都有写死的默认值（15×15 / 连 5），换盘面后必须显式覆盖，
    否则 9×9 会按 15×15 解析：天元找错、边缘四连被误判成活四。
    """
    if start is None:
        start = _read_start_record(path)
    b = str((start or {}).get('board') or '')
    if 'x' not in b:
        return
    try:
        n = int(b.split('x')[0])
        win = int((start or {}).get('n_in_row') or 5)
    except (ValueError, TypeError):
        return
    white_open.set_board(n)
    try:
        import combat as _combat
        _combat.set_board(n, win)
    except Exception:                   # noqa: BLE001
        pass


def build_payload(path, window=20):
    recs = read_records(path)
    starts = [r for r in recs if r.get('type') == 'start']
    # last start = the session currently running (a resumed run appends a new one);
    # its game_batch_num is the overall target for the whole lineage.
    start = starts[-1] if starts else {}
    games = [r for r in recs if r.get('type') == 'game']
    updates = [r for r in recs if r.get('type') == 'update']
    evals = [r for r in recs if r.get('type') == 'eval']
    eval_games = [r for r in recs if r.get('type') == 'eval_game']

    # 本仓库约定：game_batch_num 是【最终局数编号】（train.py 内部按
    # n_this_run = game_batch_num - batch_offset 计算），所以总量就是它本身。
    total = int(start.get('game_batch_num') or 0)
    batch = max([g.get('batch', 0) for g in games] or [0])
    run_start_t = start.get('t') or 0

    # ---- self-play result stats (winner 1 = black/player1, 2 = white, -1 = tie)
    wc = Counter(g.get('winner') for g in games)
    black, white, tie = wc.get(1, 0), wc.get(2, 0), wc.get(-1, 0)
    played = len(games)
    decided = black + white

    lens = [g.get('episode_len', 0) for g in games]
    last = games[-window:] if games else []
    lens_last = [g.get('episode_len', 0) for g in last]

    # 近 N 局胜率。全量胜率在长跑里被历史锁死（12000 局时永远停在 62%），
    # 看不出「现在」的棋力，所以 KPI 卡改用近 100 局；全量留作副标题参考。
    wr_win = min(100, played)
    wl = games[-wr_win:] if games else []
    wcw = Counter(g.get('winner') for g in wl)
    wr_black, wr_white, wr_tie = wcw.get(1, 0), wcw.get(2, 0), wcw.get(-1, 0)
    wr_decided = wr_black + wr_white

    # 风格指标: 快攻局 = N 手以内解决（黑/白各自计数）
    # 🚨 阈值不能写死 40：9×9 均手只有 18，40 手等于「全部对局」，
    #    覆盖率恒在 98%+，完全没有区分度（9×9 五连理论最短 9 手）。
    #    按棋盘边长自适应，前端直接用 fast_threshold 显示。
    _bw = 15
    try:
        _bw = int(str(start.get('board') or '15x15').split('x')[0])
    except ValueError:
        pass
    FAST = 12 if _bw <= 11 else 20
    fast_black = sum(1 for g in games
                     if g.get('winner') == 1 and g.get('episode_len', 999) <= FAST)
    fast_white = sum(1 for g in games
                     if g.get('winner') == 2 and g.get('episode_len', 999) <= FAST)
    fast_all = fast_black + fast_white

    # rolling black / white win rate (draws excluded from the denominator).
    # With fewer than MIN_N games in the window the estimate is pure noise
    # (1 game => 0% or 100%), so emit null and let the chart start later.
    #
    # 🚨 图表滚动窗口（ROLL）和「近 N 局」KPI（window=20）是两回事：
    #    20 局滚动在 1 万局的跨度上锯齿太密，趋势被噪声吃掉，所以图表用 100 局。
    #    但均手 KPI 卡仍要「近 20 局」—— 那是看当下的短期手感，取 100 会变迟钝。
    ROLL = 100
    MIN_N = max(5, ROLL // 4)        # 保持原来的 25% 门槛，避免小样本算出 0%/100%
    bwr_roll, wwr_roll, len_roll = [], [], []
    for i in range(len(games)):
        lo = max(0, i - ROLL + 1)
        chunk = games[lo:i + 1]
        nb = sum(1 for g in chunk if g.get('winner') == 1)
        nw = sum(1 for g in chunk if g.get('winner') == 2)
        den = nb + nw
        ok = (i + 1) >= MIN_N and den > 0
        bwr_roll.append(round(100.0 * nb / den, 2) if ok else None)
        wwr_roll.append(round(100.0 * nw / den, 2) if ok else None)
        len_roll.append(round(sum(g.get('episode_len', 0) for g in chunk)
                              / len(chunk), 2))

    # ---- timing / ETA
    # Only games from the CURRENT session count for throughput: a resume boundary
    # carries eval + process-startup time that would poison the estimate.
    #
    # 🚨 并行训练不能用「相邻两局的间隔」估速度：5 个工兵会同一秒一起交局，
    #    相邻 dt≈0 而 batch 差 5，算出来接近 0 → ETA 显示 0 小时。
    #    改成整段吞吐：本轮跑过的局数 ÷ 本轮墙钟跨度。
    per_game = None
    ts = sorted([(g.get('batch'), g.get('t')) for g in games
                 if (g.get('t') or 0) >= run_start_t], key=lambda x: x[0])
    if len(ts) >= 10:
        n_span = ts[-1][0] - ts[0][0]
        t_span = ts[-1][1] - ts[0][1]
        if n_span > 0 and t_span > 0:
            per_game = t_span / n_span

    eval_cost = None
    if eval_games:
        groups = {}
        for r in eval_games:
            groups.setdefault(r.get('batch'), []).append(r)
        costs = []
        for b, rs in groups.items():
            if len(rs) >= 2:
                costs.append(rs[-1]['t'] - rs[0]['t'])
        if costs:
            eval_cost = statistics.median(costs)

    elapsed = recs[-1].get('elapsed') if recs else 0
    # wall clock over the whole lineage (spans restarts), vs this process only
    lineage_elapsed = (recs[-1].get('t') - recs[0].get('t')) if len(recs) > 1 else elapsed
    eta = None
    if per_game and total:
        remaining = max(0, total - batch)
        eta = remaining * per_game
        if eval_cost:
            cf = int(start.get('check_freq') or 0)
            if cf:
                eta += (remaining // cf) * eval_cost

    last_t = recs[-1].get('t') if recs else 0
    stale = (time.time() - last_t) if last_t else None
    finished = bool(total) and batch >= total
    # 🚨 判定「是否在训练」的宽限期不能写死：v2 大网 800 sims 实测 219 s/局，
    #    固定 180 s 会让页面在每局中途闪「未在训练」，看着像进程死了。
    #    取 max(180, 3×单局中位)，单局耗时本身来自最近 40 局中位数。
    grace = max(180.0, 3.0 * per_game) if per_game else 180.0
    running = (not finished) and stale is not None and stale < grace

    # white_open / combat 的棋盘参数也要跟着走
    # （9×9 天元是 (4,4)、连子数是 5；combat 的 N/WIN 同样不能写死 15/5）
    _sync_board_modules(start=start)

    # ---- eval series
    eval_pts = [{'batch': e['batch'], 'win_ratio': round(100.0 * e['win_ratio'], 2),
                 'win': e['win'], 'lose': e['lose'], 'tie': e['tie'],
                 'pure_playout': e.get('pure_playout')} for e in evals]

    # ---- training curves
    return {
        'ok': True,
        'config': {k: start.get(k) for k in
                   ('board', 'n_in_row', 'n_playout', 'game_batch_num',
                    'check_freq', 'batch_size', 'epochs', 'pure_mcts_playout_num',
                    'batch_offset', 'init_lr_multiplier', 'eval_pure',
                    'rapfi_every', 'rapfi_depth', 'value_target')},
        'status': {
            'running': running,
            'finished': finished,
            'batch': batch,
            'total': total,
            'elapsed': elapsed,
            'lineage_elapsed': lineage_elapsed,
            'eta': eta,
            'per_game': per_game,
            'eval_cost': eval_cost,
            'last_update_age': stale,
            'window': window,
        },
        'summary': {
            'games': played,
            'black_wins': black,
            'white_wins': white,
            'ties': tie,
            'black_wr': round(100.0 * black / decided, 2) if decided else None,
            'white_wr': round(100.0 * white / decided, 2) if decided else None,
            'wr_window': wr_win,
            'last_black_wins': wr_black,
            'last_white_wins': wr_white,
            'last_ties': wr_tie,
            'last_black_wr': round(100.0 * wr_black / wr_decided, 2) if wr_decided else None,
            'last_white_wr': round(100.0 * wr_white / wr_decided, 2) if wr_decided else None,
            'tie_rate': round(100.0 * tie / played, 2) if played else None,
            'len_avg': round(mean(lens), 2) if lens else None,
            'len_median': statistics.median(lens) if lens else None,
            'len_avg_last': round(mean(lens_last), 2) if lens_last else None,
            'len_median_last': statistics.median(lens_last) if lens_last else None,
            'len_min': min(lens) if lens else None,
            'len_max': max(lens) if lens else None,
            'fast_black': fast_black,
            'fast_white': fast_white,
            'fast_rate': round(100.0 * fast_all / played, 2) if played else None,
            'fast_threshold': FAST,
            'loss_last': updates[-1]['loss'] if updates else None,
            'loss_avg10': round(mean([u['loss'] for u in updates[-10:]]), 4) if updates else None,
            'entropy_last': updates[-1]['entropy'] if updates else None,
            'entropy_avg10': round(mean([u['entropy'] for u in updates[-10:]]), 4) if updates else None,
            'kl_last': updates[-1]['kl'] if updates else None,
            'ev_new_last': updates[-1]['ev_new'] if updates else None,
            'eval_last': eval_pts[-1] if eval_pts else None,
            'updates': len(updates),
        },
        'games': [[g.get('batch'), g.get('episode_len'), g.get('winner')] for g in games],
        'len_roll': len_roll,
        'black_wr_roll': bwr_roll,
        'white_wr_roll': wwr_roll,
        'roll_window': ROLL,
        'updates': [{'batch': u['batch'], 'loss': round(u['loss'], 4),
                     'entropy': round(u['entropy'], 4), 'kl': round(u['kl'], 5),
                     'lr_multiplier': round(u['lr_multiplier'], 3),
                     'ev_old': round(u['ev_old'], 4), 'ev_new': round(u['ev_new'], 4)}
                    for u in updates],
        'evals': eval_pts,
        'rapfi': [{'batch': r['batch'], 'win': r.get('win', 0), 'lose': r.get('lose', 0),
                   'tie': r.get('tie', 0), 'depth': r.get('depth'),
                   'n_games': r.get('n_games'), 'avg_len': r.get('avg_len'),
                   'secs': r.get('secs')}
                  for r in recs if r.get('type') == 'rapfi'],
        'white_open': white_open.analyze(white_open.load_games(DEFAULT_GAMES)),
    }


class Handler(BaseHTTPRequestHandler):
    log_path = DEFAULT_LOG
    window = 20
    session = None           # engine.Session，懒加载

    def _send(self, code, body, ctype='application/json; charset=utf-8'):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        # gzip：看板 1 秒轮询一次，payload 已到 600KB+（3000 局时），
        # 不压缩的话每秒白传 0.6MB，还会挤占训练进程的 CPU/IO。
        # json 压缩率很高，实测 636KB → ~55KB。
        accept = (self.headers.get('Accept-Encoding') or '')
        if 'gzip' in accept.lower() and len(body) > 1024:
            gz = gzip.compress(body, 6)
            if len(gz) < len(body):
                body = gz
                self.send_header('Content-Encoding', 'gzip')
                self.send_header('Vary', 'Accept-Encoding')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False))

    def _err(self, msg, code=200):
        self._json({'ok': False, 'error': str(msg)}, code)

    def _body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode('utf-8'))
        except ValueError:
            return {}

    # ── 页面 ──
    PAGE_ROUTES = {'/': 'index.html', '/dashboard': 'index.html',
                   '/play': 'play.html', '/mcts': 'mcts.html',
                   '/network': 'network.html', '/replay': 'replay.html'}

    def _serve_page(self, name):
        p = os.path.join(HERE, name)
        if not os.path.exists(p):
            self._send(404, '%s missing' % name, 'text/plain')
            return
        with open(p, 'rb') as f:
            self._send(200, f.read(), 'text/html; charset=utf-8')

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/api/config':
            # 棋盘规格：前端按这个画 9×9 或 15×15，不写死
            try:
                import engine
                self._send(200, json.dumps(dict(engine.spec(), ok=True),
                                           ensure_ascii=False))
            except Exception as exc:      # noqa: BLE001
                self._send(500, json.dumps({'ok': False, 'error': str(exc)}))
            return
        if u.path == '/api/metrics':
            qs = parse_qs(u.query)
            win = int(qs.get('window', [self.window])[0])
            try:
                payload = build_payload(self.log_path, window=win)
            except Exception as exc:  # noqa: BLE001
                self._send(500, json.dumps({'ok': False, 'error': str(exc)}))
                return
            self._send(200, json.dumps(payload, ensure_ascii=False))
            return
        if u.path.startswith('/static/'):
            rel = u.path[len('/static/'):]
            p = os.path.normpath(os.path.join(HERE, 'static', rel))
            static_root = os.path.join(HERE, 'static')
            if not p.startswith(static_root) or not os.path.isfile(p):
                self._send(404, 'not found', 'text/plain')
                return
            ctype = ('text/javascript' if p.endswith('.js')
                     else 'text/css' if p.endswith('.css')
                     else 'application/octet-stream')
            with open(p, 'rb') as f:
                self._send(200, f.read(), ctype)
            return
        if u.path in self.PAGE_ROUTES:
            self._serve_page(self.PAGE_ROUTES[u.path])
            return
        if u.path == '/api/models':
            try:
                import engine
                self._json({'ok': True, 'models': engine.list_models()})
            except Exception as exc:  # noqa: BLE001
                self._err(exc)
            return
        if u.path == '/api/game/state':
            self._json({'ok': True, 'state': self._session().state()})
            return
        if u.path == '/api/positions/list':
            self._json({'ok': True, 'positions': list_positions()})
            return
        if u.path == '/api/games/list':
            qs = parse_qs(u.query)

            def _int(name, default=None):
                v = qs.get(name, [None])[0]
                if v in (None, ''):
                    return default
                try:
                    return int(v)
                except ValueError:
                    return default

            r = _GAMES.list(offset=_int('offset', 0) or 0,
                            limit=_int('limit', 40) or 40,
                            winner=_int('winner'),
                            min_batch=_int('min_batch'),
                            max_batch=_int('max_batch'),
                            order=qs.get('order', ['batch_desc'])[0])
            r['ok'] = True
            self._json(r)
            return
        if u.path == '/api/games/values':
            # 按指定权重重算某一局的逐步数值。
            #   method=net  单次前向（快，~40ms/局）—— 网络直觉
            #   method=mcts 每步跑 MCTS 取根 Q（慢，100 sims 约 7s/局 9×9）
            # 用途：同一局换最新权重重算，比较「胜负分野」出现在第几手。
            qs = parse_qs(u.query)
            idx = (qs.get('idx') or [None])[0]
            if idx is None:
                self._err('缺少 idx')
                return
            g = _GAMES.get(idx)
            if g is None:
                self._err('棋谱不存在: idx=%s' % idx)
                return
            method = (qs.get('method') or ['net'])[0]
            try:
                sims = int((qs.get('sims') or ['100'])[0])
            except ValueError:
                sims = 100
            sims = max(1, min(2000, sims))
            model = (qs.get('model') or [None])[0]
            t0 = time.time()
            try:
                import engine
                if method == 'mcts':
                    vals = engine.game_q_series(
                        g.get('moves'), model_file=model, sims=sims,
                        winner=g.get('winner'))
                else:
                    vals = engine.game_value_series(
                        g.get('moves'), model_file=model,
                        winner=g.get('winner'))
            except Exception as exc:      # noqa: BLE001
                self._send(500, json.dumps({'ok': False, 'error': str(exc)}))
                return
            self._json({
                'ok': True,
                'idx': g.get('idx'),
                'batch': g.get('batch'),
                'winner': g.get('winner'),
                'method': method,
                'sims': sims if method == 'mcts' else None,
                'model': model or 'default',
                'secs': round(time.time() - t0, 2),
                'values': vals,
                'divide': engine.find_divide(vals, g.get('winner')),
            })
            return
        if u.path == '/api/games/detail':
            qs = parse_qs(u.query)
            idx = (qs.get('idx') or [None])[0]
            if idx is None:
                self._err('缺少 idx')
                return
            g = _GAMES.get(idx)
            if g is None:
                self._err('棋谱不存在: idx=%s' % idx)
                return
            g['ok'] = True
            # 逐步数值（黑/白胜率）来源优先级：
            #   ① 训练时落盘的逐手 MCTS 根 Q（g.get('qs')）—— 零计算，且是**当时那一手
            #      实际做决策时**的搜索价值，比事后重算更忠实
            #   ② 没有 qs 的老棋谱（本功能上线前采的）才回落到干净前向
            # ?values=0 仍可整体跳过。
            want = (qs.get('values') or ['1'])[0]
            if want not in ('0', 'false', 'no'):
                try:
                    import engine
                    src = 'forward'
                    g['values'] = []
                    if g.get('qs'):
                        g['values'] = engine.q_series_from_qs(
                            g.get('moves'), g.get('qs'), g.get('winner'))
                        if g['values']:
                            src = 'qs'
                    if not g['values']:
                        g['values'] = engine.game_value_series(
                            g.get('moves'), winner=g.get('winner'))
                        src = 'forward'
                    g['values_source'] = src
                    g['divide'] = engine.find_divide(g['values'], g.get('winner'))
                except Exception as exc:      # noqa: BLE001  没权重时不影响棋谱浏览
                    g['values'] = []
                    g['values_error'] = str(exc)
            self._json(g)
            return
        if u.path == '/api/games/stats':
            qs = parse_qs(u.query)
            try:
                bucket = int((qs.get('bucket') or ['100'])[0])
            except ValueError:
                bucket = 100
            r = _GAMES.stats(bucket=bucket)
            r['ok'] = True
            self._json(r)
            return
        if u.path == '/api/combat':
            qs = parse_qs(u.query)
            try:
                bucket = int((qs.get('bucket') or ['100'])[0])
            except ValueError:
                bucket = 100
            # 独立入口也要先同步棋盘参数：combat 的 N/WIN 不能停在默认 15/5，
            # 否则直接访问本接口（没先走 /api/metrics）会按 15×15 算 9×9 的棋谱。
            _sync_board_modules(self.log_path)
            r = _GAMES.combat_series(bucket=bucket)
            if r.get('computed_now'):
                _GAMES.save_combat_cache()
            r['ok'] = True
            self._json(r)
            return
        self._send(404, 'not found', 'text/plain')

    # ── POST ──
    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path == '/api/game/new':
                self._json({'ok': True, 'state': self._session().new(self._body())})
                return
            if u.path == '/api/game/move':
                d = self._body()
                s = self._session()
                ok, err = s.human_move(d.get('row'), d.get('col'))
                self._json({'ok': ok, 'error': err, 'state': s.state()})
                return
            if u.path == '/api/game/ai':
                s = self._session()
                ok, err = s.ai_step()
                self._json({'ok': ok, 'error': err, 'state': s.state()})
                return
            if u.path == '/api/viz/mcts':
                import engine
                d = self._body()
                board = self._board_for(d)
                net = engine.get_net(d.get('model'))
                r = engine.mcts_viz(board, net,
                                    n_playout=int(d.get('n_playout', 100)),
                                    c_puct=float(d.get('c_puct', 5)),
                                    temp=float(d.get('temp', 1.0)))
                r['ok'] = True
                r['model'] = d.get('model') or engine.default_model()
                self._json(r)
                return
            if u.path == '/api/viz/network':
                import engine
                d = self._body()
                board = self._board_for(d)
                net = engine.get_net(d.get('model'))
                r = engine.net_viz(board, net)
                r['ok'] = True
                r['model'] = d.get('model') or engine.default_model()
                self._json(r)
                return
            if u.path == '/api/positions/save':
                d = self._body()
                name = (d.get('name') or '').strip()
                if not name:
                    self._err('名称不能为空')
                    return
                save_position(name, d.get('stones') or [], d.get('note') or '')
                self._json({'ok': True, 'positions': list_positions()})
                return
            if u.path == '/api/positions/load':
                d = self._body()
                p = get_position(d.get('name'))
                if p is None:
                    self._err('局面不存在')
                    return
                self._json({'ok': True, 'position': p})
                return
            if u.path == '/api/positions/delete':
                delete_position(self._body().get('name'))
                self._json({'ok': True, 'positions': list_positions()})
                return
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._err(exc)
            return
        self._send(404, 'not found', 'text/plain')

    # ── 共用 ──
    def _session(self):
        import engine
        if Handler.session is None:
            Handler.session = engine.Session()
        return Handler.session

    def _board_for(self, d):
        """viz 接口的棋盘来源：显式传 stones 就摆谱，否则用当前对弈局面。"""
        import engine
        if d.get('stones') is not None:
            return engine.make_board(d['stones'])
        s = self._session()
        st = s.state()
        if st is None:
            raise RuntimeError('还没有对弈局面，请先在「对弈」页开局，或在左侧摆谱')
        return s.board

    def log_message(self, fmt, *args):
        pass


def parse_board_spec(log_path):
    """从 metrics.jsonl 的 start 记录读棋盘规格。

    start 记录的 board 形如 '9x9' / '15x15'，n_in_row 是连子数。
    读不到就退回 engine 的默认值（15×15/五连），不让看板因为日志缺失起不来。
    """
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get('type') != 'start':
                    continue
                b = str(rec.get('board') or '')
                if 'x' in b:
                    w, h = b.split('x')[:2]
                    return {'width': int(w), 'height': int(h),
                            'n_in_row': int(rec.get('n_in_row') or 5)}
                break
    except IOError:
        pass
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--log', default=DEFAULT_LOG)
    ap.add_argument('--window', type=int, default=20)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()
    Handler.log_path = args.log
    Handler.window = args.window

    # 棋盘规格跟着日志走：9×9/四连 与 15×15/五连 共用这套代码
    # engine 要 import torch，包一层：缺 torch 时看板仍可用，只是三页降级
    try:
        import engine
        sp = parse_board_spec(args.log)
        if sp:
            engine.configure(**sp)
            print('board spec from log: {}x{} / {} in a row'.format(
                sp['width'], sp['height'], sp['n_in_row']), flush=True)
    except ImportError as exc:
        print('engine 不可用（%s）：/play /mcts /network 三页会报错' % exc, flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print('dashboard: http://{}:{}/  (log={})'.format(args.host, args.port, args.log),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('\nquit')


if __name__ == '__main__':
    main()
