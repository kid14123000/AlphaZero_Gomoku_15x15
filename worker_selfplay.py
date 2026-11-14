#!/usr/bin/env python3
"""自对弈工兵（8 进程并行的那个）。

职责单一：加载 sync/current_policy.model → 打一局 → 把 (state, pi, z, moves)
写进 queue/ 目录 → 看权重有没有更新 → 再打一局。

不做训练、不碰主进程的任何状态，坏了自己重来（主进程会重启它）。

用法（一般由 parallel_train.py 拉起，不手跑）：
    python worker_selfplay.py --wid 0 --queue queue --sync sync --n_playout 800
"""
from __future__ import print_function

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet


def read_version(sync):
    try:
        with open(os.path.join(sync, 'policy_version.txt')) as f:
            return int(f.read().strip())
    except Exception:                       # noqa: BLE001  文件不存在/正在被替换
        return -1


def touch(path, text):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wid', type=int, required=True)
    ap.add_argument('--queue', default='queue')
    ap.add_argument('--sync', default='sync')
    ap.add_argument('--board', type=int, default=15)
    ap.add_argument('--n_in_row', type=int, default=5)
    ap.add_argument('--n_playout', type=int, default=800)
    ap.add_argument('--temp', type=float, default=1.0)
    # 2026-11-14: c_puct 5 → 1.25（v24 同款有效值；LZ 是 0.5，公式不同不可直接比）。
    # 5 是 junxiaosong 上游默认，意味着几乎完全跟着网络先验走，800 次模拟的纠错被浪费。
    ap.add_argument('--c_puct', type=float,
                    default=float(os.environ.get('GOMOKU_C_PUCT', '1.25')),
                    help='PUCT 探索常数（v24 用 1.25；上游默认 5）')
    ap.add_argument('--root_noise', type=int,
                    default=int(os.environ.get('GOMOKU_ROOT_NOISE', '1')),
                    help='1=Dirichlet 噪声注入【根先验】(AGZ/v24 式)；0=旧行为(只混落子抽样)')
    ap.add_argument('--dirichlet_eps', type=float, default=0.25,
                    help='自我对弈探索噪声权重 ε；上游 0.25，短局建议 0.10')
    ap.add_argument('--dirichlet_alpha', type=float, default=0.3)
    ap.add_argument('--temp_switch', type=int, default=0,
                    help='前 N 手用 --temp 的温度，之后切 --temp_after（0=不调度，全程 temp）')
    ap.add_argument('--temp_after', type=float, default=None,
                    help='temp_switch 之后的温度，如 0.01 ≈ argmax')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--noise', type=float, default=0.0,
                    help='>0 时把每个 worker 的探索噪声强度错开（暂不使用）')
    args = ap.parse_args()

    wid = args.wid
    # 每个工兵独立随机流：否则 8 个进程会打出一模一样的棋
    seed = 20261108 + wid * 7919
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(args.queue, exist_ok=True)
    os.makedirs(args.sync, exist_ok=True)

    model_path = os.path.join(args.sync, 'current_policy.model')
    stop_path = os.path.join(args.sync, 'STOP')
    hb_path = os.path.join(args.sync, 'heartbeat_%d.txt' % wid)

    use_gpu = (args.device != 'cpu') and torch.cuda.is_available()

    # 等主进程发布第 0 版权重
    while not os.path.exists(model_path):
        if os.path.exists(stop_path):
            print('[w%d] STOP 早于模型发布，退出' % wid, flush=True)
            return 0
        time.sleep(2)

    try:
        net = PolicyValueNet(args.board, args.board, model_file=model_path,
                             use_gpu=use_gpu)
    except Exception as e:                   # noqa: BLE001  可能读到半个文件，重试
        print('[w%d] 首次加载权重失败(%s)，重试' % (wid, e), flush=True)
        time.sleep(3)
        net = PolicyValueNet(args.board, args.board, model_file=model_path,
                             use_gpu=use_gpu)

    cur_ver = read_version(args.sync)
    player = MCTSPlayer(net.policy_value_fn, c_puct=args.c_puct,
                        n_playout=args.n_playout, is_selfplay=1,
                        dirichlet_eps=args.dirichlet_eps,
                        dirichlet_alpha=args.dirichlet_alpha,
                        root_noise=bool(args.root_noise))
    board = Board(width=args.board, height=args.board, n_in_row=args.n_in_row)
    game = Game(board)

    print('[w%d] 就绪  device=%s  par=%s  ver=%d  sims=%d  eps=%.2f  alpha=%.2f  temp=%s/%s@%d'
          '  c_puct=%.2f  root_noise=%s(实际=%s)'
          % (wid, 'cuda' if use_gpu else 'cpu',
             format(sum(p.numel() for p in net.policy_value_net.parameters()), ','),
             cur_ver, args.n_playout, args.dirichlet_eps, args.dirichlet_alpha,
             args.temp, args.temp_after if args.temp_after is not None else '-',
             args.temp_switch,
             args.c_puct, args.root_noise, player._root_noise),
          flush=True)

    seq = 0
    n_done = 0
    _rn_prev = 0            # 上一局结束时的根噪声累计注入次数（算本局增量）
    while True:
        if os.path.exists(stop_path):
            print('[w%d] 收到 STOP，共打 %d 局' % (wid, n_done), flush=True)
            break

        # 权重更新了就重载（主进程用 tmp+rename 原子写，不会读到半个文件）
        ver = read_version(args.sync)
        if ver > cur_ver:
            try:
                net.policy_value_net.load_state_dict(torch.load(model_path))
                cur_ver = ver
                print('[w%d] 换权重 ver=%d' % (wid, ver), flush=True)
            except Exception as e:          # noqa: BLE001
                print('[w%d] 换权重失败(%s)，用旧版继续' % (wid, e), flush=True)

        t0 = time.time()
        try:
            # 总是收集每步 MCTS 根 Q：主进程按 --value_target 决定用 z 还是 q，
            # worker 侧不重复跑一遍，代价只是多存一个 float 数组。
            winner, play_data, qs = game.start_self_play(
                player, temp=args.temp, collect_q=True,
                temp_switch=args.temp_switch, temp_after=args.temp_after)
        except RuntimeError as e:           # 显存/驱动偶发
            print('[w%d] 对局异常: %s' % (wid, e), flush=True)
            time.sleep(5)
            continue
        play_data = list(play_data)
        dt = time.time() - t0

        states = np.asarray([d[0] for d in play_data], dtype=np.float32)
        probs = np.asarray([d[1] for d in play_data], dtype=np.float32)
        zs = np.asarray([d[2] for d in play_data], dtype=np.float32)
        qs = np.asarray(qs, dtype=np.float32)
        moves = np.asarray(list(board.states.keys()), dtype=np.int32)

        tmp = os.path.join(args.queue, '.tmp_%d_%d.npz' % (wid, seq))
        np.savez(tmp, states=states, probs=probs, zs=zs, qs=qs, moves=moves,
                 winner=np.int32(winner), wid=np.int32(wid),
                 seconds=np.float64(dt), version=np.int32(cur_ver))
        os.replace(tmp, os.path.join(args.queue, '%015d_%d_%d.npz'
                                     % (int(t0 * 1000), wid, seq)))
        seq += 1
        n_done += 1
        touch(hb_path, '%f\n' % time.time())
        print('[w%d] 第%d局  %d手  winner=%d  %.0fs  ver=%d'
              % (wid, n_done, len(play_data), winner, dt, cur_ver), flush=True)
        # 行为指纹：根先验噪声应"每手注入一次"。注入次数应 = 本局手数（树被 reset 时
        # 首手需等根展开，仍是 1 次/手）。sum≈1.0 说明噪声按 ε/α 正确归一化。
        if args.root_noise:
            _tot = player.mcts.root_noise_applied
            _st = player.mcts.last_root_noise_stats or {}
            print('[w%d]   根噪声: 本局 %d 次 / 累计 %d  K=%s Σ=%s 最大先验占比=%s'
                  % (wid, _tot - _rn_prev, _tot, _st.get('k'),
                     ('%.4f' % _st['sum']) if _st else '-',
                     ('%.3f' % _st['top1_share']) if _st else '-'), flush=True)
            _rn_prev = _tot

    return 0


if __name__ == '__main__':
    sys.exit(main())
