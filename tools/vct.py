# -*- coding: utf-8 -*-
"""VCT 求解器（连续威胁取胜）—— 无禁手自由规则

VCF 只会「连续冲四」；VCT 额外允许「活三」作为威胁，所以能吃下
「三手胜」「四三杀」这类题目（题库里占 2,299 + 816 道）。

威胁模型（三类，全部强制）:
  连五  落子本身成五                          → 直接胜
  活四  落子后产生 >=2 个「五连点」            → 对手堵不完，胜（前提：对手自己没有五连点）
  冲四  落子后恰好 1 个五连点                  → 对手只能堵这一点
  活三  落子后存在空点 e，落 e 会产生 >=2 个五连点
        → 对手只能去占 e（占别处 attacker 直接落 e 成活四）
        → e 可能有多个，全部都要防守方试一遍，任一防住就说明这步不成立

**为什么 sound**：每个威胁的防守应手集是完备的 ——
  - 冲四：唯一正解是堵那个五连点；对手想抢先连五这条路，被「对手有五连点则本步作废」单独挡掉。
  - 活三：唯一正解是占 e。占任何别的点都留 e 给 attacker 成活四。
  - 活四：无解。
所以搜索树每个分支都穷尽了，返回「胜」就是真的必胜。

⚠️ 术语坑：`five_points_after()` 返回的是「**再落一子就能成五的空点**」，
不是「这一子落下就成五」。判断「落 m 直接成五」必须用 `is_five()`。
"""
import time

W = H = 15
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def is_five(grid, m, p):
    """在 m 落 p 之后，p 是否形成 >=5 连（只看经过 m 的四条线）"""
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


def _win_cells(grid, p, m, dh, dw):
    """以 m 为中心 9 格窗口内，填 p 之后能成 5 连的空点"""
    h, w = m // W, m % W
    cells = []
    for k in range(-4, 5):
        hh, ww = h + dh * k, w + dw * k
        if 0 <= hh < H and 0 <= ww < W:
            cells.append((k, grid.get(hh * W + ww, 0)))
        else:
            cells.append((k, -1))          # 越界当死格
    out = []
    for i, (k, v) in enumerate(cells):
        if v != 0:
            continue
        seg = [c[1] if j != i else p for j, c in enumerate(cells)]
        for s in range(0, 5):
            if all(x == p for x in seg[s:s + 5]):
                hh, ww = h + dh * k, w + dw * k
                out.append(hh * W + ww)
                break
    return out


def five_points_after(grid, p, m):
    """假设 p 已落在 m（grid 含 m），返回「再落一子就能成五」的空点（去重）"""
    seen = set()
    for dh, dw in DIRS:
        for x in _win_cells(grid, p, m, dh, dw):
            if x not in grid:
                seen.add(x)
    return list(seen)


def five_points(grid, p):
    seen = set()
    for m, c in grid.items():
        if c == p:
            for x in five_points_after(grid, p, m):
                if x not in grid:
                    seen.add(x)
    return list(seen)


def _near(grid, radii=4):
    """离已有棋子 <=radii 的空点（只有这些点才可能形成威胁）"""
    occ = set(grid)
    out = []
    for m in range(W * H):
        if m in occ:
            continue
        h, w = m // W, m % W
        hit = False
        for dh, dw in DIRS:
            for s in range(1, radii + 1):
                for sg in (1, -1):
                    hh, ww = h + dh * s * sg, w + dw * s * sg
                    if 0 <= hh < H and 0 <= ww < W and (hh * W + ww) in occ:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            out.append(m)
    return out


def _line_cands(grid, m, span=5):
    """与 m 共线的空点（含 m 自身）—— 活四落点必然与 m 同线"""
    h, w = m // W, m % W
    out = []
    for dh, dw in DIRS:
        for k in range(-span, span + 1):
            if k == 0:
                continue
            hh, ww = h + dh * k, w + dw * k
            if 0 <= hh < H and 0 <= ww < W:
                mm = hh * W + ww
                if mm not in grid:
                    out.append(mm)
    return out


class VCT(object):
    def __init__(self, nodes_cap=400000, seconds=10.0):
        self.nodes = 0
        self.nodes_cap = nodes_cap
        self.deadline = time.time() + seconds
        self.truncated = 0

    def _abort(self):
        return self.nodes >= self.nodes_cap or time.time() > self.deadline

    # ── 生成威胁手 ────────────────────────────────────────────
    def gen(self, grid, attacker):
        """返回 (fives, fours, threes)
        fives  : [m]                落子即成五
        fours  : [(m, block, n)]    n=2 活四（无解） / n=1 冲四
        threes : [(m, [e...])]      落 m 成活三，防守方须占 e
        """
        near4 = _near(grid, 4)
        near2 = set(_near(grid, 2))
        fives, fours, threes = [], [], []
        for m in near4:
            grid[m] = attacker
            try:
                if is_five(grid, m, attacker):
                    fives.append(m)
                    continue
                fs = five_points_after(grid, attacker, m)
            finally:
                del grid[m]
            if len(fs) >= 2:
                fours.append((m, fs[0], 2))
            elif len(fs) == 1:
                fours.append((m, fs[0], 1))
            elif m in near2:                       # 孤立点造不出活三
                es = []
                for e in _line_cands(grid, m):
                    grid[m] = attacker
                    grid[e] = attacker
                    try:
                        if len(five_points_after(grid, attacker, e)) >= 2:
                            es.append(e)
                    finally:
                        del grid[e]
                        del grid[m]
                if es:
                    threes.append((m, es))
        return fives, fours, threes

    # ── 落一手之后（轮到对手）是否仍必胜 ──────────────────────
    def _after_att_move(self, grid, attacker, m, depth, es=None):
        """attacker 已落 m（grid 已含 m）。判「对手应一手之后是否还赢」。

        🚨 这里是唯一容易写错的地方：攻击方落 m 之后**轮到对手**，
           必须让对手应一手再判。off-by-one 会让「乱下也赢」的假象出现。
        """
        if is_five(grid, m, attacker):
            return True
        if depth <= 0:
            return False
        defender = 3 - attacker
        if five_points(grid, defender):
            return False                       # 对手能连五，来不及
        fs = five_points_after(grid, attacker, m)
        if len(fs) >= 2:
            return True                        # 活四，对手堵不完
        if len(fs) == 1:                       # 冲四，对手只能堵这一点
            block = fs[0]
            grid[block] = defender
            try:
                ok, _ = self.win(grid, attacker, depth)
            finally:
                del grid[block]
            return ok
        if es is None:                         # 活三，防守点要现算
            es = []
            for e in _line_cands(grid, m):
                grid[e] = attacker
                try:
                    if len(five_points_after(grid, attacker, e)) >= 2:
                        es.append(e)
                finally:
                    del grid[e]
        for e in es:
            if e in grid:
                continue
            grid[e] = defender
            try:
                ok, _ = self.win(grid, attacker, depth)
            finally:
                del grid[e]
            if not ok:
                return False
        return bool(es)

    def win(self, grid, attacker, depth):
        """轮到 attacker 走，是否必胜。返回 (bool, 首手)"""
        self.nodes += 1
        if self._abort():
            self.truncated += 1
            return False, None
        if depth <= 0:
            return False, None
        defender = 3 - attacker

        # 能直接连五
        for e in _near(grid, 4):
            if is_five({**grid, e: attacker}, e, attacker):
                return True, e
        # 对手已有五连点 → 对手先手赢
        if five_points(grid, defender):
            return False, None

        fives, fours, threes = self.gen(dict(grid), attacker)
        if fives:
            return True, fives[0]
        for m, block, n in fours:
            if n >= 2:
                return True, m                 # 活四
            grid[m] = attacker
            grid[block] = defender
            try:
                ok, _ = self.win(grid, attacker, depth - 1)
            finally:
                del grid[block]
                del grid[m]
            if ok:
                return True, m
        for m, es in threes:
            grid[m] = attacker
            try:
                ok = self._after_att_move(grid, attacker, m, depth - 1, es=es)
            finally:
                del grid[m]
            if ok:
                return True, m
        return False, None

    # ── 枚举全部必胜首手（判题口径，不要求等于某个标准答案）──
    def all_winning(self, grid, attacker, depth, limit=40):
        if depth <= 0:
            return []
        if five_points(grid, 3 - attacker):
            return []
        fives, fours, threes = self.gen(dict(grid), attacker)
        out = list(fives)
        for m, block, n in fours:
            if len(out) >= limit:
                break
            if n >= 2:
                out.append(m)
                continue
            grid[m] = attacker
            grid[block] = 3 - attacker
            try:
                ok, _ = self.win(grid, attacker, depth - 1)
            finally:
                del grid[block]
                del grid[m]
            if ok:
                out.append(m)
        for m, es in threes:
            if len(out) >= limit:
                break
            grid[m] = attacker
            try:
                ok = self._after_att_move(grid, attacker, m, depth - 1, es=es)
            finally:
                del grid[m]
            if ok:
                out.append(m)
        return out


def vct(grid, attacker, depth=4, seconds=10.0, nodes_cap=400000):
    """返回必胜首手或 None。不改调用方的 grid。"""
    s = VCT(nodes_cap=nodes_cap, seconds=seconds)
    ok, mv = s.win(dict(grid), attacker, depth)
    return mv if ok else None


def winning_moves(grid, attacker, depth=2, seconds=10.0, nodes_cap=400000, limit=40):
    """全部「必胜首手」的集合。判题要用这个，不能用 equals 标准答案。"""
    s = VCT(nodes_cap=nodes_cap, seconds=seconds)
    return s.all_winning(dict(grid), attacker, depth, limit=limit)


# ────────────────────────────────────────────────────────────────
# 独立暴力校验（只用于三手胜口径，给 VCT 做 soundness 交叉验证）
# ────────────────────────────────────────────────────────────────
def brute_three_move(grid, attacker, seconds=20.0):
    """暴力：找 attacker 的一手 m，使得**无论 defender 走哪**，attacker 下一手都能连五。

    不用威胁模型，纯枚举，所以可以拿来给 vct() 的答案做对照。
    只适合「三手胜」（attacker 落 2 子取胜）。
    """
    t_end = time.time() + seconds
    defender = 3 - attacker
    empties = [m for m in range(W * H) if m not in grid]
    if any(is_five({**grid, e: defender}, e, defender) for e in empties):
        return None

    for m in empties:
        if time.time() > t_end:
            return None
        grid[m] = attacker
        try:
            if is_five(grid, m, attacker):
                return m
            good = True
            for d in empties:
                if d == m:
                    continue
                grid[d] = defender
                try:
                    win2 = False
                    for x in empties:
                        if x in grid:
                            continue
                        grid[x] = attacker
                        if is_five(grid, x, attacker) or len(five_points_after(grid, attacker, x)) >= 2:
                            win2 = True
                            del grid[x]
                            break
                        del grid[x]
                    if not win2:
                        good = False
                        break
                finally:
                    del grid[d]
            if good:
                return m
        finally:
            del grid[m]
    return None
