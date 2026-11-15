# -*- coding: utf-8 -*-
"""极简 VCF（连续冲四）求解器 —— 给题库自动判分用，无禁手规则

术语：
  五连点(m)  = 在该点落子后立刻形成 >=5 连
  冲四(m)   = 落 m 后「恰好一个」五连点存在 → 对手被迫堵那一点
  活四(m)   = 落 m 后「两个以上」五连点 → 对手堵不住，直接胜
"""
W = H = 15
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def five_points(grid, p):
    """所有能让 p 立即五连的空点"""
    out = []
    for m in range(W * H):
        if m in grid:
            continue
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
                out.append(m)
                break
    return out


def vcf(grid, attacker, max_depth=12, _seen=None):
    """返回 attacker 的 VCF 首手（必胜）或 None。攻击方每步只下冲四/活四。"""
    if _seen is None:
        _seen = set()
    key = (frozenset(grid.items()), attacker, max_depth)
    if key in _seen:
        return None
    _seen.add(key)

    wins = five_points(grid, attacker)
    if wins:                        # 已经能直接五连
        return wins[0]
    if max_depth <= 0:
        return None

    defender = 3 - attacker
    # 候选：所有能形成四的点
    cands = []
    empty = [m for m in range(W * H) if m not in grid]
    for m in empty:
        grid[m] = attacker
        fs = five_points(grid, attacker)
        del grid[m]
        if len(fs) >= 2:
            return m                # 活四，对手堵不住
        if len(fs) == 1:
            cands.append((m, fs[0]))

    for m, block in cands:
        grid[m] = attacker
        grid[block] = defender
        try:
            sub = vcf(grid, attacker, max_depth - 1, _seen)
        finally:
            del grid[block]
            del grid[m]
        if sub is not None:
            return m
    return None
