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

import combat as _combat


class GameStore(object):
    def __init__(self, path, combat_cache=None):
        self.path = path
        self._mtime = None
        self._rows = []        # 摘要: (idx, batch, winner, episode_len)
        self._moves = {}       # idx -> moves（懒加载）
        # 战斗指标缓存：idx -> dict。持久化到 sidecar，避免每次重算
        self._combat = {}
        self._combat_path = combat_cache
        if combat_cache and os.path.exists(combat_cache):
            try:
                with open(combat_cache, 'r') as f:
                    self._combat = {int(k): v for k, v in json.load(f).items()}
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
                             'len': int(d.get('episode_len') or len(moves))})
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
        moves = self._moves.get(idx)
        if moves is None:
            moves = None
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
                        break
                    n += 1
            if moves is None:
                return None
            self._moves[idx] = moves
        row = dict(self._rows[idx])
        row['moves'] = moves
        row['combat'] = self.combat_of(idx, moves)
        return row

    # ── 战斗指标（带增量缓存） ──────────────────────────────
    def combat_of(self, idx, moves=None):
        """单局战斗指标。moves 不传时会懒加载该局手顺。"""
        self._ensure()
        idx = int(idx)
        if idx in self._combat:
            return self._combat[idx]
        if moves is None:
            g = self.get(idx)
            if g is None:
                return None
            return g['combat']
        m = _combat.analyze(moves)
        self._combat[idx] = m
        return m

    def _fills_missing(self):
        """把还没算过的 idx 补上（只补缺的：文件是追加写，idx 稳定）。"""
        self._ensure()
        missing = [r['idx'] for r in self._rows if r['idx'] not in self._combat]
        if not missing:
            return 0
        want = set(missing)
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
                moves = d.get('moves') or []
                if not moves:
                    continue
                if n in want:
                    self._combat[n] = _combat.analyze(
                        [[int(m[0]), int(m[1])] for m in moves])
                n += 1
        return len(missing)

    def save_combat_cache(self):
        if not self._combat_path:
            return
        try:
            with open(self._combat_path, 'w') as f:
                json.dump({str(k): v for k, v in self._combat.items()}, f)
        except OSError:
            pass

    def combat_series(self, bucket=100):
        """按代分桶的战斗指标趋势（给看板画线）。"""
        added = self._fills_missing()
        rows = self._rows
        buckets = {}
        for r in rows:
            c = self._combat.get(r['idx'])
            if not c:
                continue
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

        overall = _combat.aggregate(
            [c for c in self._combat.values() if c])
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
