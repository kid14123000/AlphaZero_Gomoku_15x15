#!/usr/bin/env python3
"""棋型健康体检（移植自老项目 Gomoku-v0_pt_v44.py: health_check）

三块：
  ① 价值盲测：活三/冲四/活四/五连 ×（我持型 / 对手持型）×（我先手 / 我后手）
     期望：我持型 → +expect；对手持型 → −expect；空盘 |v|<0.3
     expect: 活三 0.6 / 冲四 0.8 / 活四 1.0 / 五连 1.0
  ② 交替局面：主线 + 对手散子（远离主线），避免纯色 OOD
  ③ Policy 关键点：关键应对点概率 vs 随机基线 1/225（判据 >8×）
     ⚠️ 棋型一律放在远离天元的位置，并同时报中心概率，避免"爱下中心"冒充认棋型
适配：本仓库 4 平面 canonical（我方/对手/最后一手/手数奇偶），无绝对颜色。
"""
import argparse
import numpy as np
import torch
from policy_value_net_pytorch import PolicyValueNet

N = 15
CENTER = (7, 7)

# (名字, 持型方子, 关键点, expect) —— 全部远离天元
SHAPES = [
    ('活三横', [(3, 8), (3, 9), (3, 10)], [(3, 7), (3, 11)], 0.6),
    ('活三竖', [(8, 3), (9, 3), (10, 3)], [(7, 3), (11, 3)], 0.6),
    ('活三斜', [(9, 9), (10, 10), (11, 11)], [(8, 8), (12, 12)], 0.6),
    ('冲四横', [(3, 7), (3, 8), (3, 9), (3, 11)], [(3, 10)], 0.8),
    ('冲四竖', [(7, 3), (8, 3), (9, 3), (11, 3)], [(10, 3)], 0.8),
    ('冲四斜', [(7, 7), (8, 8), (9, 9), (11, 11)], [(10, 10)], 0.8),
    ('活四横', [(3, 8), (3, 9), (3, 10), (3, 11)], [(3, 7), (3, 12)], 1.0),
    ('活四斜', [(9, 9), (10, 10), (11, 11), (12, 12)], [(8, 8), (13, 13)], 1.0),
    ('五连横', [(3, 7), (3, 8), (3, 9), (3, 10), (3, 11)], [], 1.0),
    ('五连斜', [(9, 9), (10, 10), (11, 11), (12, 12), (13, 13)], [], 1.0),
]


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def scatter_far(k, shape):
    """挑 k 个远离棋型、彼此也离得开的点（不会自己连成线）"""
    out = []
    for r in range(N):
        for c in range(N):
            if len(out) >= k:
                return out
            p = (r, c)
            if min(cheb(p, s) for s in shape) < 4:
                continue
            if out and min(cheb(p, q) for q in out) < 3:
                continue
            out.append(p)
    return out


def build(my_cells, opp_cells):
    """4 平面 canonical，末尾与 game.py current_state() 一致做上下翻转"""
    planes = np.zeros((4, N, N), dtype=np.float32)
    for r, c in my_cells:
        planes[0][r, c] = 1.0
    for r, c in opp_cells:
        planes[1][r, c] = 1.0
    if opp_cells:
        r, c = opp_cells[-1]
        planes[2][r, c] = 1.0
    if (len(my_cells) + len(opp_cells)) % 2 == 0:
        planes[3][:, :] = 1.0
    return np.ascontiguousarray(planes[:, ::-1, :])


def val(net, st):
    with torch.no_grad():
        _, v = net(torch.from_numpy(st[None]))
    return float(v.reshape(-1)[0])


def pol(net, st):
    with torch.no_grad():
        p, _ = net(torch.from_numpy(st[None]))
    return torch.exp(p).reshape(-1).numpy()


def make_pair(shape_cells, holder_is_me, i_first):
    k = len(shape_cells)
    other = scatter_far(k, shape_cells)
    if holder_is_me:
        my, opp = list(shape_cells), other
    else:
        my, opp = other, list(shape_cells)
    if i_first:                     # 我先手 → 子数相等
        while len(my) < len(opp):
            my += scatter_far(len(opp), set(my) | set(opp))[:1] or [(0, 14)]
        while len(opp) < len(my):
            opp += scatter_far(len(my), set(my) | set(opp))[:1] or [(14, 0)]
    else:                           # 我后手 → 对手多一子
        extra = scatter_far(len(opp) + 1, set(my) | set(opp))[-1]
        opp = opp + [extra]
    return build(my, opp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', default=['current_policy.model'])
    a = ap.parse_args()
    for mp in a.models:
        net = PolicyValueNet(N, N, model_file=mp, use_gpu=False).policy_value_net
        net.eval()
        print(f"\n=== {mp} ===")
        print(f"{'棋型':>8}{'期望':>6} | {'我持·先手':>10}{'我持·后手':>10} | "
              f"{'敌持·先手':>10}{'敌持·后手':>10} | {'通过':>6}")
        okc = tot = 0
        for name, cells, key, exp in SHAPES:
            v = {}
            for holder_me in (True, False):
                for first in (True, False):
                    st = make_pair(cells, holder_me, first)
                    v[(holder_me, first)] = val(net, st)
                    want = exp if holder_me else -exp
                    okc += (v[(holder_me, first)] > 0) == (want > 0) and abs(v[(holder_me, first)]) > 0.3
                    tot += 1
            print(f"{name:>8}{exp:>6.1f} | {v[(True,True)]:>+10.3f}{v[(True,False)]:>+10.3f} | "
                  f"{v[(False,True)]:>+10.3f}{v[(False,False)]:>+10.3f} | {okc:>2}/{tot:<2}")
        ve = val(net, build([], []))
        print(f"{'空盘':>8}{0.0:>6.1f} | v = {ve:+.3f}  {'✓' if abs(ve) < 0.3 else '✗ |v|>=0.3'}")

        print(f"\n{'棋型':>8}{'关键点最大概率':>13}{'基线×':>8}{'中心概率':>9}{'判据':>7}")
        for name, cells, key, exp in SHAPES:
            if not key:
                continue
            st = make_pair(cells, True, True)
            p = pol(net, st)
            hit = max(float(p[r * N + c]) for r, c in key)
            cen = float(p[CENTER[0] * N + CENTER[1]])
            print(f"{name:>8}{hit:>13.4f}{hit*225:>8.1f}{cen:>9.4f}"
                  f"{'✓' if hit > 8/225 else '✗':>7}")
        print(f"价值盲测通过 {okc}/{tot} = {100*okc/tot:.0f}%")


if __name__ == '__main__':
    main()
