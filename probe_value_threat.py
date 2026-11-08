#!/usr/bin/env python3
"""价值健康度探针：黑/白「二连/三连/四连」威胁识别（同 parity 对照）

固定走子方（ch3 恒定），只改棋盘内容：
  Δ我 = v(我连 k 子) - v(我 k 子散开)    期望为正
  Δ敌 = v(对手连 k 子) - v(对手 k 子散开) 期望为负
健康：k=3 时 |Δ| ≥ 0.15，且黑白幅度接近
身份捷径：Δ≈0（只读 ch3），或棋盘变了但 v 完全不变
"""
import numpy as np
import torch
from game import Board
from policy_value_net_pytorch import PolicyValueNet

N = 15
SCAT = [(0, 0), (0, 2), (0, 4), (2, 0), (2, 2), (2, 4), (4, 0), (4, 2),
        (4, 4), (6, 0), (6, 2), (6, 4)]


def load(path):
    pv = PolicyValueNet(N, N, model_file=path, use_gpu=False)
    pv.policy_value_net.eval()
    return pv.policy_value_net


def build(black, white):
    b = Board(width=N, height=N, n_in_row=5)
    b.init_board(0)
    bs, ws = list(black), list(white)
    t = 0
    while bs or ws:
        r, c = bs.pop(0) if t % 2 == 0 else ws.pop(0)
        b.do_move(r * N + c)
        t += 1
    return b


def line(k):
    return [(7, 4 + i) for i in range(k)]


def scatter(k, off=0):
    return [(r + off, c + off) for r, c in SCAT[:k]]


def val(net, b):
    st = np.array(b.current_state(), dtype=np.float32, copy=True)
    with torch.no_grad():
        _, v = net(torch.from_numpy(st[None]))
    return float(v.item())


def main():
    nets = [('250代', load('checkpoints/ckpt_it250.model')),
            ('300代', load('checkpoints/ckpt_it300.model')),
            ('350代', load('current_policy.model'))]
    print(f"{'代':<6}{'k':>3} | {'黑走:我线':>9}{'我散':>7}{'敌线':>7} | "
          f"{'白走:我线':>9}{'我散':>7}{'敌线':>7} | {'Δ我':>7}{'Δ敌':>7}")
    for tag, net in nets:
        for k in (1, 2, 3, 4):
            # 黑走（ch3=1）
            mine_b = val(net, build(line(k), scatter(k)))                 # 我有线
            base_b = val(net, build(scatter(k, 1), scatter(k)))           # 我散开（对照）
            opp_b = val(net, build(scatter(k, 1), line(k)))               # 对手有线
            # 白走（ch3=0）：黑多一子
            mine_w = val(net, build(scatter(k + 1), line(k)))             # 我有线
            base_w = val(net, build(scatter(k + 1, 1), scatter(k)))       # 我散开
            opp_w = val(net, build(line(k) + [(12, 12)], scatter(k)))     # 对手有线
            d_mine = (mine_b - base_b + mine_w - base_w) / 2
            d_opp = (opp_b - base_b + opp_w - base_w) / 2
            print(f"{tag:<6}{k:>3} | {mine_b:>9.3f}{base_b:>7.3f}{opp_b:>7.3f} | "
                  f"{mine_w:>9.3f}{base_w:>7.3f}{opp_w:>7.3f} | {d_mine:>+7.3f}{d_opp:>+7.3f}")
        print()


if __name__ == '__main__':
    main()
