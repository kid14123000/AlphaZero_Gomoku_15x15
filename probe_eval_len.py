#!/usr/bin/env python3
"""补测：评估局（网络 400 sims vs 纯 MCTS N sims）的对局长度分布

metrics.jsonl 的 eval_game 不存长度，这里按训练内评估同口径复现几局。
用法: .venv/bin/python probe_eval_len.py --games 4 --pure 1100 --sim 400
"""
import argparse
import time

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure
from policy_value_net_pytorch import PolicyValueNet

N = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='current_policy.model')
    ap.add_argument('--games', type=int, default=4)
    ap.add_argument('--sim', type=int, default=400)
    ap.add_argument('--pure', type=int, default=1100)
    a = ap.parse_args()

    pv = PolicyValueNet(N, N, model_file=a.model, use_gpu=False)
    print(f'网络 {a.model} @ {a.sim} sims  vs  纯MCTS @ {a.pure} sims')
    print(f'{"局":>3}{"网络执":>6}{"结果":>6}{"手数":>6}{"耗时s":>8}')
    lens, rec = [], {'黑': [0, 0], '白': [0, 0]}
    t_all = time.time()
    for i in range(a.games):
        net = MCTSPlayer(pv.policy_value_fn, c_puct=5, n_playout=a.sim)
        pure = MCTS_Pure(c_puct=5, n_playout=a.pure)
        board = Board(width=N, height=N, n_in_row=5)
        game = Game(board)
        net_black = (i % 2 == 0)
        t0 = time.time()
        if net_black:
            winner = game.start_play(net, pure, start_player=0, is_shown=0)
        else:
            winner = game.start_play(pure, net, start_player=0, is_shown=0)
        dt = time.time() - t0
        n = len(board.states)
        lens.append(n)
        side = '黑' if net_black else '白'
        ok = (winner == (1 if net_black else 2))
        rec[side][1] += 1
        rec[side][0] += int(ok)
        print(f'{i+1:>3}{side:>6}{"胜" if ok else "负":>6}{n:>6}{dt:>8.0f}')
    print(f'\n手数: {lens} | 中位 {sorted(lens)[len(lens)//2]} | 均 {sum(lens)/len(lens):.1f}')
    print(f'战绩: 黑 {rec["黑"][0]}/{rec["黑"][1]}  白 {rec["白"][0]}/{rec["白"][1]}'
          f'  总用时 {(time.time()-t_all)/60:.1f} 分')


if __name__ == '__main__':
    main()
