#!/usr/bin/env python3
"""速度对比：默认网(0.33M) vs v24 同构网(2.5M)

测三档：
  1) 单次前向耗时（batch=1）
  2) 单步 MCTS 耗时（400 / 800 sims，真实局面）
  3) 按实测手数外推：单局自对弈耗时、1 万局总时长
注意：测速只看结构，权重随机即可（BN 在 eval 模式下不依赖 batch 统计）
"""
import time

import numpy as np
import torch

from game import Board
from mcts_alphaZero import MCTS
from net_v24 import V24Net
from policy_value_net_pytorch import Net as NetSmall

N = 15


def bench_forward(net, iters=100):
    net.eval()
    x = torch.randn(1, 4, N, N)
    with torch.no_grad():
        for _ in range(10):
            net(x)
        t0 = time.time()
        for _ in range(iters):
            net(x)
    return (time.time() - t0) / iters * 1000      # ms


def real_state():
    """取一局真实自对弈的前 20 手（保证局面未终局）"""
    import json
    for line in open('logs/games.jsonl'):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if len(r['moves']) < 24:
            continue
        b = Board(width=N, height=N, n_in_row=5)
        b.init_board(0)
        for r_, c_ in r['moves'][:20]:
            b.do_move(r_ * N + c_)
        if not b.game_end()[0]:
            return b
    raise SystemExit('没找到可用的中盘局面')


def bench_mcts(net, sims, n_pos=3):
    """把 net 包成 policy_value_fn，跑 MCTS 单步"""
    def pv_fn(board):
        legal = board.availables
        st = np.array(board.current_state(), dtype=np.float32, copy=True)[None]
        with torch.no_grad():
            logp, v = net(torch.from_numpy(np.ascontiguousarray(st)))
        p = torch.exp(logp)[0].numpy()[legal]
        return zip(legal, p), float(v.item())
    b = real_state()
    times = []
    for i in range(n_pos):
        mcts = MCTS(pv_fn, c_puct=5, n_playout=sims)
        t0 = time.time()
        mcts.get_move_probs(b, 1.0)
        times.append(time.time() - t0)
    return sum(times) / len(times)


def main():
    small = NetSmall(N, N)
    big = V24Net(N)
    ns = sum(p.numel() for p in small.parameters())
    nb = sum(p.numel() for p in big.parameters())
    print(f'默认网 参数 {ns:,} | v24 同构 参数 {nb:,}  ({nb/ns:.1f}x)\n')

    fs = bench_forward(small)
    fb = bench_forward(big)
    print(f'单次前向(batch=1): 默认 {fs:6.2f} ms | v24 {fb:7.2f} ms  ({fb/fs:.1f}x)\n')

    print(f'{"sims":>6}{"默认网":>10}{"v24网":>10}{"倍数":>7}')
    for sims in (400, 800):
        ts = bench_mcts(small, sims)
        tb = bench_mcts(big, sims)
        print(f'{sims:>6}{ts:>9.2f}s{tb:>9.2f}s{tb/ts:>6.1f}x')

    # 外推（按当前实测：一局约 20 手；自对弈里每手都要一次 MCTS）
    print('\n外推（20 手/局）:')
    for sims in (400, 800):
        ts = bench_mcts(small, sims); tb = bench_mcts(big, sims)
        print(f'  {sims} sims: 默认 {ts*20:6.1f} s/局 | v24 {tb*20:7.1f} s/局 '
              f'(1万局 ≈ {tb*20*10000/86400:5.1f} 天)')


if __name__ == '__main__':
    main()
