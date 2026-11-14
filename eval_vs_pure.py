#!/usr/bin/env python3
"""独立复测：网络 MCTS(mcts_alphaZero) vs 纯 MCTS(mcts_pure)

与训练内部评估同口径，但局数可加大、并分色给出胜率 + 95% 置信区间。
用法: .venv/bin/python eval_vs_pure.py --model current_policy.model --games 20 --pure-playout 100
"""
import argparse, json, math, time
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure
from policy_value_net_pytorch import PolicyValueNet

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="current_policy.model")
ap.add_argument("--games", type=int, default=20, help="总局数（黑白各一半）")
ap.add_argument("--net-playout", type=int, default=400)
ap.add_argument("--pure-playout", type=int, default=100)
ap.add_argument("--out", default="eval_vs_pure.json")
a = ap.parse_args()

board = Board(width=15, height=15, n_in_row=5)
game = Game(board)
net = PolicyValueNet(15, 15, model_file=a.model, use_gpu=False)
net_player = MCTSPlayer(net.policy_value_fn, c_puct=5, n_playout=a.net_playout)

res = {"black": [], "white": []}
t0 = time.time()
for i in range(a.games):
    start = i % 2          # 0=网络先手(黑) 1=纯MCTS先手(黑，网络执白)
    pure = MCTS_Pure(c_puct=5, n_playout=a.pure_playout)
    net_player.reset_player(); pure.reset_player()
    winner = game.start_play(net_player, pure, start_player=start, is_shown=0)
    color = "black" if start == 0 else "white"
    res[color].append(1 if winner == 1 else (0.5 if winner == -1 else 0))
    n = len(res[color])
    print(f"  第{i+1:>2}局 网络执{'黑' if color=='black' else '白'}: "
          f"{'胜' if winner==1 else '和' if winner==-1 else '负'}  "
          f"（累计 黑 {sum(res['black'])}/{len(res['black'])} 白 {sum(res['white'])}/{len(res['white'])}）", flush=True)

def stat(v):
    n = len(v)
    if n == 0: return 0.0, 0.0, n
    p = sum(v) / n
    se = math.sqrt(max(p * (1 - p), 1e-9) / n)
    return p, 1.96 * se, n

allv = res["black"] + res["white"]
out = {"model": a.model, "net_playout": a.net_playout, "pure_playout": a.pure_playout,
       "games": a.games, "elapsed_s": round(time.time() - t0, 1)}
for k in ("black", "white"):
    p, ci, n = stat(res[k]); out[k] = {"win_rate": p, "ci95": ci, "n": n}
p, ci, n = stat(allv); out["all"] = {"win_rate": p, "ci95": ci, "n": n}
print(f"\n=== 网络 vs 纯MCTS({a.pure_playout}次模拟) ===")
for k in ("black", "white", "all"):
    o = out[k]; print(f"  {k:>5}: {o['win_rate']*100:5.1f}% ±{o['ci95']*100:4.1f}  (n={o['n']})")
print(f"用时 {(time.time()-t0)/60:.1f} 分")
json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1)
