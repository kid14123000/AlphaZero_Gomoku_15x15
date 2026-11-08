#!/usr/bin/env python3
"""who 语义 A/B：让 Rapfi 执白跟我们的网络打一局，看哪种标法下白棋正常

判读：
  · 标法对 → Rapfi 会防守/反击，对局能撑到 20 手以上，甚至赢
  · 标法错 → 它以为自己是黑棋，去连/补黑棋的线，白棋 10 手内崩
用法: .venv/bin/python probe_rapfi_who.py --depth 5 --sim 400
"""
import argparse
import time

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet
from rapfi_player import RapfiPlayer, RapfiError

N = 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='current_policy.model')
    ap.add_argument('--sim', type=int, default=400)
    ap.add_argument('--depth', type=int, default=5)
    a = ap.parse_args()

    pv = PolicyValueNet(N, N, model_file=a.model, use_gpu=False)
    for mode in ('relative', 'absolute'):
        net = MCTSPlayer(pv.policy_value_fn, c_puct=5, n_playout=a.sim)
        rapfi = RapfiPlayer(depth=a.depth, who_mode=mode)
        board = Board(width=N, height=N, n_in_row=5)
        game = Game(board)
        t0 = time.time()
        try:
            # player1=纯MCTS? 不：player1 是黑（我们的网），player2 是白（Rapfi）
            winner = game.start_play(net, rapfi, start_player=0, is_shown=0)
            n = len(board.states)
            dt = time.time() - t0
            res = '赢' if winner == 2 else ('输' if winner == 1 else '和/异常')
            print(f'who_mode={mode:<9} Rapfi执白 {res}  手数 {n}  用时 {dt:.0f}s  winner={winner}')
        except RapfiError as e:
            print(f'who_mode={mode:<9} RapfiError: {e}  （手数 {len(board.states)}）')
        finally:
            rapfi.close()


if __name__ == '__main__':
    main()
