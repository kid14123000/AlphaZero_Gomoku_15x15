# -*- coding: utf-8 -*-
"""训练棋谱库（logs/games.jsonl）的读取与索引。

文件由 train.py `--save-games` 追加写，每行：
    {"batch": 313, "winner": 2, "episode_len": 50, "moves": [[r,c], ...]}

特点与对策
----------
* **追写**：训练在跑时最后一行可能写了一半 → 解析失败的行直接跳过。
* **会重复**：断点续训会重跑一小段，同一 batch 可能出现两条（不同对局）。
  所以对外用**行号 idx** 做唯一键，不用 batch。
* **会变长**：10000 局约 2.5MB，全量读进内存 + 按 mtime 失效重载，
  单次 list 只返回摘要（不含 moves），只有点开某一局才读 moves。
"""
import json
import os
import statistics
import hashlib

import combat as _combat

CACHE_VER = 2


def _hash_moves(moves):
    """一局的身份指纹 = 手顺的 sha1 前 16 位。

    🚨 缓存键必须是「这一局棋本身」，不能是行号 idx（2026-11-14 实锤）：
       训练重启会换一个全新的 logs/games.jsonl（batch 从 batch_offset+1 起、
       旧文件直接被覆盖），此时 idx 0..N 已经指向**完全不同的对局**，
       而旧缓存还挂在同样的 idx 上 → 曲线显示的是上一轮开局那几十局的低防守率
       （实测页面显示 5%，真实值 60~87%），KPI 也把已消失的旧局算进 overall。
    """
    h = hashlib.sha1()
    for m in moves:
        h.update(('%d,%d;' % (int(m[0]), int(m[1]))).encode())
    return h.hexdigest()[:16]


class GameStore(object):
    def __init__(self, path, combat_cache=None):
        self.path = path
        self._mtime = None
        self._rows = []        # 摘要: (idx, batch, winner, episode_len, h)
        self._moves = {}       # idx -> moves（懒加载）
        # 战斗指标缓存：**手顺指纹 h** -> dict。持久化到 sidecar，避免每次重算。
        # 🚨 不能按 idx 缓存：训练重启会换新 games.jsonl，idx 相同的其实是另一局。
        self._combat = {}
        self._combat_path = combat_cache
        if combat_cache and os.path.exists(combat_cache):
            try:
                with open(combat_cache, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get('v') == CACHE_VER:
                    self._combat = dict(data.get('items') or {})
                else:
                    self._combat = {}      # 旧格式（idx 为键）= 不可信，整份丢弃
            except (ValueError, OSError):
                self._combat = {}

    # ── 载入/失效 ───────────────────────────────────────────
    def _ensure(self):
        if not os.path.exists(self.path):
            self._rows, self._moves, self._mtime = [], {}, None
            return
        mt = os.stat(self.path).st_mtime
        if self._mtime == mt:
            return
        rows = []
        idx = 0
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue        # 训练正在写的半行
                moves = d.get('moves') or []
                if not moves:
                    continue
                rows.append({'idx': idx,
                             'batch': int(d.get('batch', 0)),
                             'winner': int(d.get('winner', -1)),
                             'len': int(d.get('episode_len') or len(moves)),
                             'h': _hash_moves(moves)})
                idx += 1
        self._rows = rows
        self._moves = {}
        self._mtime = mt

    # ── 查询 ───────────────────────────────────────────────
    def list(self, offset=0, limit=40, winner=None, min_batch=None,
             max_batch=None, order='batch_desc'):
        self._ensure()
        rows = self._rows
        if winner is not None:
            rows = [r for r in rows if r['winner'] == winner]
        if min_batch is not None:
            rows = [r for r in rows if r['batch'] >= min_batch]
        if max_batch is not None:
            rows = [r for r in rows if r['batch'] <= max_batch]
        total = len(rows)

        keys = {
            'batch_desc': lambda r: (-r['batch'], r['idx']),
            'batch_asc': lambda r: (r['batch'], r['idx']),
            'len_asc': lambda r: (r['len'], r['batch']),
            'len_desc': lambda r: (-r['len'], r['batch']),
            'idx_asc': lambda r: r['idx'],
            'idx_desc': lambda r: -r['idx'],
        }
        rows = sorted(rows, key=keys.get(order, keys['batch_desc']))
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 200))
        page = rows[offset:offset + limit]
        return {'total': total, 'offset': offset, 'limit': limit,
                'games': page}

    def get(self, idx):
        self._ensure()
        idx = int(idx)
        if idx < 0 or idx >= len(self._rows):
            return None
        moves = self._load_moves(idx)
        if moves is None:
            return None
        row = dict(self._rows[idx])
        row['moves'] = moves
        row['combat'] = self.combat_of(idx, moves)
        return row

    def _load_moves(self, idx):
        """读第 idx 局的落子手顺（带懒加载缓存；文件里凑不到这么多局时返回 None）。"""
        moves = self._moves.get(idx)
        if moves is not None:
            return moves
        n = 0
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if not (d.get('moves') or []):
                    continue
                if n == idx:
                    moves = [[int(m[0]), int(m[1])] for m in d['moves']]
                    self._moves[idx] = moves
                    return moves
                n += 1
        return None

    # ── 战斗指标（带增量缓存） ──────────────────────────────
    def combat_of(self, idx, moves=None):
        """单局战斗指标。moves 不传时会懒加载该局手顺。

        缓存键 = 手顺指纹（不是 idx）：训练重启换掉 games.jsonl 后，
        同一个 idx 已经是另一局，用 idx 当键会把上一轮的指标算到这一局头上。
        """
        self._ensure()
        idx = int(idx)
        if moves is None:
            moves = self._load_moves(idx)
            if moves is None:
                return None
        if 0 <= idx < len(self._rows):
            h = self._rows[idx]['h']
        else:
            h = _hash_moves(moves)
        c = self._combat.get(h)
        if c is not None:
            return c
        m = _combat.analyze(moves)
        self._combat[h] = m
        return m

    def _fills_missing(self):
        """把还没算过的对局补上（键=手顺指纹，所以「同 idx 换了棋」也能识别出来）。"""
        self._ensure()
        want = {r['h'] for r in self._rows if r['h'] not in self._combat}
        if not want:
            return 0
        n = 0
        done = 0
        with open(self.path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                moves = d.get('moves') or []
                if not moves:
                    continue
                mv = [[int(m[0]), int(m[1])] for m in moves]
                h = _hash_moves(mv)
                if h in want:
                    self._combat[h] = _combat.analyze(mv)
                    done += 1
                n += 1
        self._n_games_seen = n
        return done

    def save_combat_cache(self):
        """落盘（只留当前文件里还在的对局；换轮/重跑后旧指纹自动淘汰）。"""
        if not self._combat_path:
            return
        try:
            keep = {}
            if self._rows:
                keep = {r['h']: self._combat[r['h']]
                        for r in self._rows if r['h'] in self._combat}
            with open(self._combat_path, 'w') as f:
                json.dump({'v': CACHE_VER, 'items': keep}, f)
            self._combat = keep
        except OSError:
            pass

    def combat_series(self, bucket=100):
        """按代分桶的战斗指标趋势（给看板画线）。

        🚨 overall 只统计【当前 games.jsonl 里的对局】：缓存里可能残留上一轮
           其它对局的指纹，把它们算进 overall 会让 KPI（防守率 51.5%）和
           曲线（2~6%）自相矛盾。
        """
        added = self._fills_missing()
        rows = self._rows
        buckets = {}
        used = []
        for r in rows:
            c = self._combat.get(r['h'])
            if not c:
                continue
            used.append(c)
            b = (r['batch'] // bucket) * bucket
            d = buckets.setdefault(b, {'batch': b, 'rows': [], 'lens': []})
            d['rows'].append(c)
            d['lens'].append(r['len'])

        series = []
        for b in sorted(buckets):
            d = buckets[b]
            ag = _combat.aggregate(d['rows'])
            series.append({
                'batch': b,
                'n': ag['n'],
                'composite': ag['composite'],
                'contact': None if ag['contact'] is None else round(ag['contact'] * 100, 2),
                'block': None if ag['block'] is None else round(ag['block'] * 100, 2),
                'must_block': None if ag['must_block'] is None else round(ag['must_block'] * 100, 2),
                'proximity': ag['proximity'],
                'zone': None if ag['zone'] is None else round(ag['zone'] * 100, 2),
                'len_median': statistics.median(d['lens']),
                'block_chances': ag['block_chances'],
                'block_hits': ag['block_hits'],
            })

        overall = _combat.aggregate([c for c in used if c])
        return {'bucket': bucket, 'series': series, 'overall': overall,
                'computed_now': added}

    # ── 统计（给页面配图） ──────────────────────────────────
    def stats(self, bucket=100):
        self._ensure()
        rows = self._rows
        if not rows:
            return {'n': 0}

        lens = [r['len'] for r in rows]
        wc = {1: 0, 2: 0, -1: 0}
        for r in rows:
            wc[r['winner']] = wc.get(r['winner'], 0) + 1
        decided = wc[1] + wc[2]

        # 手数直方图（10 手一档）
        hist = {}
        for L in lens:
            k = (L // 10) * 10
            hist[k] = hist.get(k, 0) + 1
        bins = [{'from': k, 'to': k + 9, 'n': hist[k]}
                for k in sorted(hist)]

        # 按 batch 分桶的白胜率
        buckets = {}
        for r in rows:
            b = (r['batch'] // bucket) * bucket
            d = buckets.setdefault(b, {'batch': b, 'n': 0,
                                       'black': 0, 'white': 0, 'tie': 0,
                                       'lens': []})
            d['n'] += 1
            d['lens'].append(r['len'])
            if r['winner'] == 1:
                d['black'] += 1
            elif r['winner'] == 2:
                d['white'] += 1
            else:
                d['tie'] += 1
        series = []
        for b in sorted(buckets):
            d = buckets[b]
            dec = d['black'] + d['white']
            series.append({
                'batch': b, 'n': d['n'],
                'black_wr': round(100.0 * d['black'] / dec, 1) if dec else None,
                'white_wr': round(100.0 * d['white'] / dec, 1) if dec else None,
                'len_median': statistics.median(d['lens']),
            })

        return {
            'n': len(rows),
            'batch_min': min(r['batch'] for r in rows),
            'batch_max': max(r['batch'] for r in rows),
            'black': wc[1], 'white': wc[2], 'tie': wc[-1],
            'black_wr': round(100.0 * wc[1] / decided, 1) if decided else None,
            'white_wr': round(100.0 * wc[2] / decided, 1) if decided else None,
            'len_avg': round(sum(lens) / len(lens), 2),
            'len_median': statistics.median(lens),
            'len_min': min(lens), 'len_max': max(lens),
            'hist': bins,
            'series': series,
            'bucket': bucket,
        }
