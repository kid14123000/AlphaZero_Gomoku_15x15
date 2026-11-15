# -*- coding: utf-8 -*-
"""VCF 求解器（快版）—— 无禁手自由规则

和 vcf.py 同一套语义，但把「全场 225 点扫一遍」换成「只看经过落子的四条线」，
单次判五连点从 ~9000 次比较降到 ~180 次。

术语：
  五连点 = 落子后立刻形成 >=5 连的空点
  冲四   = 落子后「恰好一个」五连点 → 对手被迫堵
  活四   = 落子后「两个以上」五连点 → 对手堵不住，直接胜
"""
import time

W = H = 15
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _runs_in_window(grid, p, m, dh, dw):
    """以 m 为中心取 9 格窗口，返回窗口内所有能让 p 形成 >=5 连的空点"""
    h, w = m // W, m % W
    # 窗口格：m 沿 d 方向 -4..+4
    cells = []
    for k in range(-4, 5):
        hh, ww = h + dh * k, w + dw * k
        if not (0 <= hh < H and 0 <= ww < W):
            cells.append((k, None))            # 越界当死格
        else:
            mm = hh * W + ww
            cells.append((k, grid.get(mm, 0)))  # 0=空 1/2=子
    out = []
    for i, (k, v) in enumerate(cells):
        if v != 0:
            continue
        # 试在窗口第 i 位放 p，看窗口内是否出现 5 连
        seg = [c[1] if j != i else p for j, c in enumerate(cells)]
        for s in range(0, 5):                   # 窗口宽 9，5 连起点 0..4
            if all(x == p for x in seg[s:s + 5]):
                hh, ww = h + dh * cells[i][0], w + dw * cells[i][0]
                out.append(hh * W + ww)
                break
    return out


def five_points_after(grid, p, m):
    """假设 p 刚落在 m（grid 里已含 m），返回所有新产生的五连点（去重）"""
    seen = set()
    for dh, dw in DIRS:
        for x in _runs_in_window(grid, p, m, dh, dw):
            seen.add(x)
    return list(seen)


def five_points(grid, p):
    """全部五连点（慢，但只在这种一次性场景用）"""
    out = set()
    for m, c in grid.items():
        if c == p:
            for x in five_points_after(grid, p, m):
                if x not in grid:
                    out.add(x)
    return [x for x in out if x not in grid]


def has_five_at(grid, m, p):
    """在 m 落 p 是否立刻成五（只看经过 m 的四条线）"""
    h, w = m // W, m % W
    for dh, dw in DIRS:
        n = 1
        for s in (1, -1):
            hh, ww = h + dh * s, w + dw * s
            while 0 <= hh < H and 0 <= ww < W and grid.get(hh * W + ww) == p:
                n += 1
                hh += dh * s
                ww += dw * s
        if n >= 5:
            return True
    return False


def _near(grid):
    """所有离已有棋子 <=4 的空点（只有这些点才可能形成五连）"""
    out = []
    occ = set(grid)
    for m in range(W * H):
        if m in occ:
            continue
        h, w = m // W, m % W
        hit = False
        for dh, dw in DIRS:
            for s in range(-4, 5):
                if s == 0:
                    continue
                hh, ww = h + dh * s, w + dw * s
                if 0 <= hh < H and 0 <= ww < W and (hh * W + ww) in occ:
                    hit = True
                    break
            if hit:
                break
        if hit:
            out.append(m)
    return out


def vcf(grid, attacker, max_depth=12, deadline=None):
    """返回 attacker 的 VCF 首手（必胜）或 None。deadline = time.time() + 秒"""
    _seen = set()
    return _vcf(grid, attacker, max_depth, _seen, deadline)


def _vcf(grid, attacker, depth, seen, deadline):
    if deadline is not None and time.time() > deadline:
        return None
    key = (frozenset(grid.items()), attacker, depth)
    if key in seen:
        return None
    seen.add(key)

    # 已有五连点 → 直接赢
    for e in _near(grid):
        if has_five_at(grid, e, attacker):
            return e
    if depth <= 0:
        return None

    defender = 3 - attacker
    cands = []
    for m in _near(grid):
        grid[m] = attacker
        try:
            if has_five_at(grid, m, attacker):       # 落 m 直接五连（上面已covered，双保险）
                return m
            fs = [x for x in five_points_after(grid, attacker, m) if x not in grid]
        finally:
            del grid[m]
        if len(fs) >= 2:
            return m                                  # 活四
        if len(fs) == 1:
            cands.append((m, fs[0]))

    for m, block in cands:
        grid[m] = attacker
        grid[block] = defender
        try:
            sub = _vcf(grid, attacker, depth - 1, seen, deadline)
        finally:
            del grid[block]
            del grid[m]
        if sub is not None:
            return m
    return None
