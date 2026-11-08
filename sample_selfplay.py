#!/usr/bin/env python3
"""用当前权重自对弈若干局，记录完整手顺（junxiaosong 版不存棋谱，只能自己跑）

用法: .venv/bin/python sample_selfplay.py --n 12 --model current_policy.model --out games.json
"""
import argparse, json, time
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=12)
ap.add_argument("--model", default="current_policy.model")
ap.add_argument("--playout", type=int, default=400)
ap.add_argument("--temp", type=float, default=1.0)
ap.add_argument("--out", default="games.json")
a = ap.parse_args()

W = H = 15
board = Board(width=W, height=H, n_in_row=5)
game = Game(board)
net = PolicyValueNet(W, H, model_file=a.model, use_gpu=False)
player = MCTSPlayer(net.policy_value_fn, c_puct=5, n_playout=a.playout, is_selfplay=1)

games, t0 = [], time.time()
for g in range(a.n):
    board.init_board(0)          # 黑先
    player.reset_player()        # 每局重置搜索树（与训练一致）
    moves = []
    while True:
        move = int(player.get_action(board, temp=a.temp, return_prob=0))
        r, c = divmod(move, W)
        moves.append([r, c])
        board.do_move(move)
        end, winner = board.game_end()
        if end:
            break
    games.append({"moves": moves, "winner": winner, "n": len(moves)})
    print(f"  第{g+1}局: {len(moves)}手 {'黑胜' if winner==1 else '白胜' if winner==2 else '和'}", flush=True)

json.dump({"model": a.model, "temp": a.temp, "playout": a.playout, "games": games},
          open(a.out, "w"), ensure_ascii=False)
nb = sum(1 for x in games if x["winner"] == 1)
print(f"\n共 {len(games)} 局 | 黑胜 {nb} / 白胜 {len(games)-nb} | 用时 {(time.time()-t0)/60:.1f} 分 -> {a.out}")
