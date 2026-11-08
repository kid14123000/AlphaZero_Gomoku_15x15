#!/usr/bin/env python3
"""Rapfi depth 阶梯摸底：我们的网络 vs Rapfi(depth N)，分先各半

用法: .venv/bin/python probe_rapfi_depth.py --depth 1 --games 4 --sim 400
"""
import argparse
import time

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
from rapfi_player import RapfiPlayer, RapfiError, play_one_game, OPENING_POINTS

N = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='current_policy.model')
    ap.add_argument('--depth', type=int, default=1)
    ap.add_argument('--games', type=int, default=4)
    ap.add_argument('--sim', type=int, default=400)
    ap.add_argument('--who', default='absolute')
    a = ap.parse_args()

    pv = PolicyValueNet(N, N, model_file=a.model, use_gpu=False)
    print(f'网络 {a.model} @ {a.sim} sims  vs  Rapfi depth={a.depth} (who={a.who})')
    print(f'{"局":>3}{"网络执":>6}{"结果":>6}{"手数":>6}{"耗时s":>8}')
    rec = {'黑': [0, 0], '白': [0, 0]}
    lens = []
    t_all = time.time()
    for i in range(a.games):
        net = MCTSPlayer(pv.policy_value_fn, c_puct=5, n_playout=a.sim)
        rapfi = RapfiPlayer(depth=a.depth, who_mode=a.who)
        net_black = (i % 2 == 0)
        fm = OPENING_POINTS[(i * 7) % len(OPENING_POINTS)]   # 轮换开局，避免确定性重复
        t0 = time.time()
        try:
            winner, n_mv = play_one_game(net, rapfi, net_black=net_black, first_move=fm)
        except RapfiError as e:
            print(f'{i+1:>3}  RapfiError: {e}')
            rapfi.close()
            continue
        dt = time.time() - t0
        n = n_mv
        lens.append(n)
        side = '黑' if net_black else '白'
        my = 1 if net_black else 2
        ok = (winner == my)
        rec[side][1] += 1
        rec[side][0] += int(ok)
        print(f'{i+1:>3}{side:>6}{"胜" if ok else ("负" if winner in (1, 2) else "和"):>6}{n:>6}{dt:>8.0f}')
        rapfi.close()
    tw = sum(v[0] for v in rec.values()); tn = sum(v[1] for v in rec.values())
    print(f'\n战绩 黑 {rec["黑"][0]}/{rec["黑"][1]}  白 {rec["白"][0]}/{rec["白"][1]}'
          f'  合计 {tw}/{tn} = {100.0*tw/max(tn,1):.1f}%')
    if lens:
        print(f'手数 {lens} | 均 {sum(lens)/len(lens):.1f}')
    print(f'总用时 {(time.time()-t_all)/60:.1f} 分')


if __name__ == '__main__':
    main()
