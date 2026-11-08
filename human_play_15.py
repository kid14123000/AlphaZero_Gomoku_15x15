# -*- coding: utf-8 -*-
"""
15x15 Gomoku: human VS AI (PyTorch backend)

Usage:
    python human_play_15.py                      # use ./current_policy.model if present
    python human_play_15.py --model best_policy.model --n_playout 800
    python human_play_15.py --human_first 0      # AI moves first
Input your move as row,col  (e.g. 7,7 for the centre of a 15x15 board)

Based on junxiaosong/AlphaZero_Gomoku (MIT), adapted for 15x15 + PyTorch.
"""
from __future__ import print_function
import argparse
import os
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer


class Human(object):
    """human player"""

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        while True:
            location = input("Your move (row,col): ")
            if isinstance(location, str):
                location = location.strip()
                if location in ('q', 'quit', 'exit'):
                    raise KeyboardInterrupt
            try:
                location = [int(n, 10) for n in location.split(",")]
                move = board.location_to_move(location)
            except Exception:
                move = -1
            if move == -1 or move not in board.availables:
                print("invalid move, try again")
                continue
            return move

    def __str__(self):
        return "Human {}".format(self.player)


class CompactGame(Game):
    """Game with a compact one-line-per-row board (15x15 is too tall otherwise)"""

    def graphic(self, board, player1, player2):
        width, height = board.width, board.height
        print("Player {} with X   |   Player {} with O".format(player1, player2))
        print("   " + "".join("{:>3d}".format(x) for x in range(width)))
        for i in range(height - 1, -1, -1):
            cells = []
            for j in range(width):
                p = board.states.get(i * width + j, -1)
                cells.append('X' if p == player1
                             else ('O' if p == player2 else '.'))
            print("{:>3d}".format(i) + "".join("{:>3s}".format(c) for c in cells))
        print()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=15)
    parser.add_argument('--height', type=int, default=15)
    parser.add_argument('--n_in_row', type=int, default=5)
    parser.add_argument('--n_playout', type=int, default=400)
    parser.add_argument('--model', type=str, default='current_policy.model')
    parser.add_argument('--human_first', type=int, default=1,
                        help='1 = human plays first, 0 = AI plays first')
    parser.add_argument('--pure', action='store_true',
                        help='ignore any model, play against pure MCTS')
    args = parser.parse_args()

    board = Board(width=args.width, height=args.height,
                  n_in_row=args.n_in_row)
    game = CompactGame(board)

    if (not args.pure) and os.path.exists(args.model):
        from policy_value_net_pytorch import PolicyValueNet
        best_policy = PolicyValueNet(args.width, args.height,
                                     model_file=args.model)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=args.n_playout)
        print("AI: AlphaZero net '{}' ({} playouts)".format(
            args.model, args.n_playout))
    else:
        mcts_player = MCTS_Pure(c_puct=5, n_playout=args.n_playout)
        print("AI: pure MCTS ({} playouts), no model loaded".format(
            args.n_playout))

    human = Human()
    # start_player=0 -> player1 (human) moves first
    winner = game.start_play(human, mcts_player,
                             start_player=0 if args.human_first else 1,
                             is_shown=1)
    print("Result:", "human wins" if winner == 1
          else ("AI wins" if winner == 2 else "tie"))


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print('\n\rquit')
