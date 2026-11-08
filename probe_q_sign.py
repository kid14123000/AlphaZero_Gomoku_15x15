#!/usr/bin/env python3
"""探针：重放棋谱，逐步算 MCTS 根节点 Q，看它和最终胜负的符号对不对得上。

复刻训练时的取值路径（get_move_probs 记录 last_root_q → do_move → update_with_move），
只是把 temp 设成 0 附近、不采样，纯粹为了看数值。

判据（以「该步执子方」视角）：
  - 该方最终获胜 ⇒ Q 应收敛到 +1
  - 该方最终落败 ⇒ Q 应收敛到 -1
所以每局的最后 1-2 步是最灵敏的检验点。

用法：
    python probe_q_sign.py --games 3 --sims 400
"""
from __future__ import print_function

import argparse
import json
import os

import numpy as np

from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet

W = H = 9
NROW = 5


def load_games(path, n):
    recs = [json.loads(l) for l in open(path)]
    return recs[-n:] if n else recs


def net_value(net, board):
    """单次前向的 value（该走方视角），作为 Q 的对照"""
    import torch
    st = np.ascontiguousarray(
        board.current_state().reshape(-1, 4, W, H)).astype(np.float32)
    with torch.no_grad():
        _, v = net.policy_value_net(torch.from_numpy(st))
    return float(v.numpy().ravel()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--games', type=int, default=3)
    ap.add_argument('--sims', type=int, default=400)
    ap.add_argument('--tail', type=int, default=5, help='每局打印最后几步')
    ap.add_argument('--model', default='current_policy.model')
    ap.add_argument('--games-file', default='logs/games.jsonl')
    args = ap.parse_args()

    net = PolicyValueNet(W, H, model_file=args.model)
    board = Board(width=W, height=H, n_in_row=NROW)
    player = MCTSPlayer(net.policy_value_fn, c_puct=5,
                        n_playout=args.sims, is_selfplay=1)

    recs = load_games(args.games_file, args.games)
    print('权重 %s   sims %d   局数 %d'
          % (os.path.basename(args.model), args.sims, len(recs)))
    print()

    for g in recs:
        moves = g['moves']
        winner = g['winner']
        board.init_board()
        player.mcts.update_with_move(-1)

        rows = []
        for i, (r, c) in enumerate(moves):
            mv = r * W + c
            if mv not in board.availables:
                break
            cur = board.current_player
            player.mcts.get_move_probs(board, temp=1.0)   # 记录 last_root_q
            q = float(player.mcts.last_root_q)
            v = net_value(net, board)
            rows.append({'n': i + 1, 'p': cur, 'q': q, 'v': v,
                         'rc': (r, c)})
            board.do_move(mv)
            player.mcts.update_with_move(mv)

        print('=' * 84)
        print('batch %s   %d 手   winner=%d (%s)'
              % (g['batch'], len(moves), winner, '黑' if winner == 1 else '白'))
        print('  %-4s %-3s %-8s %9s %9s %9s'
              % ('手', '执子', '落点', 'MCTS Q', '网络 value', '该方最终'))
        tail = rows[-args.tail:] if args.tail else rows
        for x in tail:
            res = '胜' if x['p'] == winner else ('负' if winner in (1, 2) else '和')
            mark = ''
            if res == '胜' and x['q'] < -0.05:
                mark = '  ← 该胜却给负'
            elif res == '负' and x['q'] > 0.05:
                mark = '  ← 该负却给正'
            print('  %-4d %-3s %-8s %9.4f %9.4f %9s%s'
                  % (x['n'], '黑' if x['p'] == 1 else '白',
                     '%d,%d' % x['rc'], x['q'], x['v'], res, mark))

        # 统计口径说明（别用「输方执子时 Q 应<0」来判符号，那个假设是错的）：
        #   自对弈里输的一方多数是「自己走错了」，不是局面必败，
        #   所以它在自己回合的 Q 完全可以是正的（搜索认为还有机会）。
        # 真正能判符号的是「赢方的最后一步」—— 那一步落子即成五，必赢，
        #   Q 必须显著为正；若为负，就是符号反了。
        last = rows[-1] if rows else None
        pos_w = sum(1 for x in rows if x['p'] == winner and x['q'] > 0)
        all_w = sum(1 for x in rows if x['p'] == winner)
        pos_l = sum(1 for x in rows if x['p'] != winner and x['q'] > 0)
        all_l = sum(1 for x in rows if x['p'] != winner)
        verdict = ''
        if last and last['p'] == winner:
            verdict = ('  ✅ 终局手符号正确' if last['q'] > 0
                       else '  ❌ 终局手符号反了')
        print('  ── 全 %d 步：赢方执子时 Q>0 占 %d/%d；输方执子时 Q>0 占 %d/%d'
              % (len(rows), pos_w, all_w, pos_l, all_l))
        print('  ── 最后一步（%s执子，落子即终局）Q=%+.4f%s'
              % ('黑' if last and last['p'] == 1 else '白',
                 last['q'] if last else 0.0, verdict))
    print('=' * 84)


if __name__ == '__main__':
    main()
