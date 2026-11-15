# -*- coding: utf-8 -*-
"""VCT 答案的独立复核 —— 不用威胁模型，防守方穷举所有空点

VCT 求解器为了速度只枚举「威胁相关的应手」。如果它的应手集不完备，
就会给出「假胜」。这个脚本用相反的口径复核：

  win_in_2(grid, att, m)  = 落 m 后，**对手走任何一点**，att 下一手都能连五
  win_in_3(grid, att, m)  = 落 m 后，**对手走任何一点**，att 都存在一手 n
                            使 win_in_2(...) 成立

对手方穷举 225 个空点，攻击方只用「能不能连五 / 能不能造活四」判定，
不涉及任何威胁分类。慢，只用来抽样验收。
"""
import time

from vct import W, H, is_five, five_points_after


def _att_has_win_now(grid, att):
    """att 是否能一手结束（连五 或 造出活四）"""
    for m in range(W * H):
        if m in grid:
            continue
        grid[m] = att
        try:
            if is_five(grid, m, att) or len(five_points_after(grid, att, m)) >= 2:
                return True
        finally:
            del grid[m]
    return False


def win_in_2(grid, att, m):
    """落 m 之后，无论对手走哪，att 都能一手结束"""
    dfd = 3 - att
    if m in grid:
        return False
    grid[m] = att
    try:
        if is_five(grid, m, att):
            return True
        for d in range(W * H):
            if d in grid:
                continue
            grid[d] = dfd
            try:
                if is_five(grid, d, dfd):
                    return False                     # 对手先连五
                if not _att_has_win_now(grid, att):
                    return False
            finally:
                del grid[d]
        return True
    finally:
        del grid[m]


def win_in_3(grid, att, m, seconds=60.0):
    """落 m 之后，无论对手走哪，att 都有 win_in_2 的一手"""
    t_end = time.time() + seconds
    dfd = 3 - att
    if m in grid:
        return False
    grid[m] = att
    try:
        if is_five(grid, m, att):
            return True
        for d in range(W * H):
            if d in grid:
                continue
            if time.time() > t_end:
                return None                          # 超时，判不了
            grid[d] = dfd
            try:
                if is_five(grid, d, dfd):
                    return False
                cand = None
                for n in range(W * H):
                    if n in grid:
                        continue
                    if win_in_2(grid, att, n):
                        cand = n
                        break
                if cand is None:
                    return False
            finally:
                del grid[d]
        return True
    finally:
        del grid[m]


def verify(grid, att, m, depth):
    """按 VCT 声称的深度复核"""
    if depth <= 2:
        return win_in_2(grid, att, m)
    return win_in_3(grid, att, m)
