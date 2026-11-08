#!/usr/bin/env python3
"""并行训练主进程：1 个训练者 + N 个自对弈工兵。

为什么这么干：MCTS 是串行树搜索，一次只送 1 条样本给网络。实测 2080Ti
单进程前向 2.09ms 而 batch=32 时只要 0.09ms/条 —— GPU 95% 时间在等 Python。
开 8 个独立进程把 GPU 填满，聚合吞吐 220 → 1160 sim/s（5.3x，实测）。

分工：
    工兵  worker_selfplay.py   加载权重 → 打局 → 落盘 queue/*.npz
    主进程 本文件              收数据 → 训练 → 发版权重 → 存检查点

权重同步靠 sync/policy_version.txt（单调递增整数）+ current_policy.model
（tmp 写完再 os.replace，工兵不会读到半个文件）。

用法：
    python parallel_train.py --workers 8 --n_playout 800 --game_batch_num 3000 \\
        --use_gpu --save-games logs/games.jsonl --ckpt_dir checkpoints \\
        --metrics_file logs/metrics.jsonl
"""
from __future__ import print_function

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from train import TrainPipeline


SYNC = 'sync'
QUEUE = 'queue'


def atomic_torch_save(net, path):
    tmp = path + '.tmp'
    torch.save(net.get_policy_param(), tmp)
    os.replace(tmp, path)


def publish(trainer, version):
    """发布第 version 版权重：先写模型、再写版本号（工兵看到版本号时才读模型）"""
    atomic_torch_save(trainer.policy_value_net,
                      os.path.join(SYNC, 'current_policy.model'))
    tmp = os.path.join(SYNC, 'policy_version.txt.tmp')
    with open(tmp, 'w') as f:
        f.write('%d\n' % version)
    os.replace(tmp, os.path.join(SYNC, 'policy_version.txt'))


def spawn(wid, args, logdir):
    cmd = [sys.executable, 'worker_selfplay.py',
           '--wid', str(wid),
           '--queue', QUEUE,
           '--sync', SYNC,
           '--board', str(args.board_width),
           '--n_in_row', str(args.n_in_row),
           '--n_playout', str(args.n_playout),
           '--temp', str(args.temp),
           '--device', 'cuda' if args.use_gpu else 'cpu']
    logf = open(os.path.join(logdir, 'worker_%d.log' % wid), 'a')
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                         cwd=os.path.dirname(os.path.abspath(__file__)))
    p._logf = logf              # 保持引用，避免被 GC 关掉
    return p


def ingest(trainer, files, args, games_file):
    """把工兵产的对局吃进 data_buffer，返回吃了几局"""
    got = 0
    prev = int(trainer.game_batch_cur)
    for path in files:
        try:
            with np.load(path) as d:
                states = d['states']
                probs = d['probs']
                zs = d['zs']
                # q 必须在 with 块内读出来：NpzFile 一关闭，d[...] 就会
                # AttributeError: 'NoneType' object has no attribute 'open'
                qs_in = d['qs'] if 'qs' in getattr(d, 'files', []) else None
                moves = d['moves']
                winner = int(d['winner'])
                wid = int(d['wid'])
                secs = float(d['seconds'])
                ver = int(d['version'])
        except Exception as e:              # noqa: BLE001  文件还没写完/损坏
            print('读队列文件失败 %s: %s' % (path, e), flush=True)
            try:
                os.remove(path)
            except OSError:
                pass
            continue

        trainer.game_batch_cur += 1
        n = int(trainer.game_batch_cur)
        trainer.last_winner = winner
        trainer.episode_len = len(probs)

        trainer.metrics.log('game', batch=n, episode_len=int(len(probs)),
                           winner=winner, worker=wid,
                           game_seconds=round(secs, 1), net_version=ver)

        if games_file:
            try:
                mv = [[int(m) // trainer.board_width, int(m) % trainer.board_width]
                      for m in moves]
                with open(games_file, 'a') as gf:
                    gf.write(json.dumps({'batch': n, 'winner': winner,
                                         'episode_len': int(len(probs)),
                                         'moves': mv}, ensure_ascii=False) + '\n')
            except Exception as e:          # noqa: BLE001
                print('save_games failed:', e, flush=True)

        # value 目标：q/mix 档用 worker 收集的 MCTS 根 Q 替换最终胜负 ±1。
        # 与单进程 train.py 的 collect_selfplay_data 同口径（那边也是替换第三元，
        # get_equi_data 原样透传，训练代码不用改）。
        target = zs
        if getattr(trainer, 'value_target', 'z') != 'z' and qs_in is not None:
            qs = np.asarray(qs_in, dtype=np.float32)
            if len(qs) == len(zs):
                if trainer.value_target == 'q':
                    target = qs
                else:
                    target = trainer.mix_lambda * qs + (1.0 - trainer.mix_lambda) * zs
                if n % 25 == 1:
                    print('  q 分布  mean %+.4f  std %.4f  absmean %.4f'
                          % (float(qs.mean()), float(qs.std()),
                             float(np.abs(qs).mean())), flush=True)
        play_data = list(zip(states, probs, target))
        trainer.data_buffer.extend(trainer.get_equi_data(play_data))
        os.remove(path)
        got += 1
        print('局 %d  手数 %d  winner %d  w%d  %.0fs  ver %d  buffer %d'
              % (n, len(probs), winner, wid, secs, ver, len(trainer.data_buffer)),
              flush=True)
    return got, prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--board_width', type=int, default=15)
    ap.add_argument('--board_height', type=int, default=15)
    ap.add_argument('--n_in_row', type=int, default=5)
    ap.add_argument('--n_playout', type=int, default=800)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--game_batch_num', type=int, default=3000)
    ap.add_argument('--check_freq', type=int, default=50)
    ap.add_argument('--publish_every', type=int, default=8,
                    help='每多少局发布一次新权重给工兵（太小=工兵频繁重载）')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=512)
    ap.add_argument('--init_model', type=str, default=None)
    ap.add_argument('--batch_offset', type=int, default=0)
    ap.add_argument('--init_lr_multiplier', type=float, default=1.0)
    ap.add_argument('--ckpt_dir', type=str, default='checkpoints')
    ap.add_argument('--metrics_file', type=str, default='logs/metrics.jsonl')
    ap.add_argument('--save-games', dest='save_games', type=str, default='')
    ap.add_argument('--use_gpu', action='store_true')
    ap.add_argument('--value_target', type=str, default='z',
                    choices=['z', 'q', 'mix'],
                    help='value 头目标：z=胜负±1 / q=MCTS 搜索价值 / mix=各半')
    ap.add_argument('--poll', type=float, default=2.0, help='轮询队列的间隔秒数')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)
    os.makedirs(SYNC, exist_ok=True)
    os.makedirs(QUEUE, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    logdir = 'logs'
    os.makedirs(logdir, exist_ok=True)

    # 清掉上次残留的 STOP / 半成品
    for p in [os.path.join(SYNC, 'STOP')] + glob.glob(os.path.join(SYNC, '*.tmp')):
        if os.path.exists(p):
            os.remove(p)

    print('=' * 66, flush=True)
    print(' 并行训练  工兵 %d  盘 %dx%d  连%d  模拟 %d/手'
          % (args.workers, args.board_width, args.board_height,
             args.n_in_row, args.n_playout), flush=True)
    print(' 目标 %d 局  每 %d 局发版权重  每 %d 局存检查点'
          % (args.game_batch_num, args.publish_every, args.check_freq), flush=True)
    print(' 设备 %s' % ('GPU' if args.use_gpu else 'CPU'), flush=True)
    print('=' * 66, flush=True)

    # use_gpu 是 v2(GPU) 版 TrainPipeline 才有的参数；_9x9 的 CPU 版没有，
    # 按签名条件传，同一份 parallel_train.py 两个 fork 都能用。
    import inspect
    tkw = dict(init_model=args.init_model,
               board_width=args.board_width,
               board_height=args.board_height,
               n_in_row=args.n_in_row,
               n_playout=args.n_playout,
               game_batch_num=args.game_batch_num,
               check_freq=args.check_freq,
               play_batch_size=1,
               epochs=args.epochs,
               batch_size=args.batch_size,
               pure_mcts_playout_num=100,
               metrics_file=args.metrics_file,
               save_games=(args.save_games or None),
               eval_pure=0,
               value_target=args.value_target,
               batch_offset=args.batch_offset,
               init_lr_multiplier=args.init_lr_multiplier,
               ckpt_dir=args.ckpt_dir)
    if 'use_gpu' in inspect.signature(TrainPipeline.__init__).parameters:
        tkw['use_gpu'] = args.use_gpu
    trainer = TrainPipeline(**tkw)
    # 🚨 必须手动对齐全局局号：TrainPipeline.__init__ 把 game_batch_cur 初始化成 0，
    #    单进程版靠 run() 里的 `self.game_batch_cur = self.batch_offset + i + 1` 补上，
    #    但并行版没走 run()（由 ingest 自己按收数据递增）。
    #    不补的话续训后 batch 会从 1 重新数，导致：
    #      · metrics/games 里的局号与历史对不上（看板进度卡在旧值）
    #      · 检查点命名从 ckpt_it200 开始，**覆盖旧谱系同名的 ckpt**
    trainer.game_batch_cur = int(args.batch_offset)
    trainer.metrics.log('parallel', workers=args.workers,
                        publish_every=args.publish_every,
                        poll=args.poll)

    # 第 0 版权重必须先落地，工兵才有东西可加载
    version = 0
    publish(trainer, version)
    print('已发布 ver=0 初始权重', flush=True)

    procs = {}
    for wid in range(args.workers):
        procs[wid] = spawn(wid, args, logdir)
    print('已拉起 %d 个工兵' % len(procs), flush=True)

    stop_path = os.path.join(SYNC, 'STOP')
    started = time.time()
    n_since_publish = 0
    train_steps = 0

    def shutdown(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)

    try:
        while trainer.game_batch_cur < args.game_batch_num:
            # 1) 收数据
            files = sorted(glob.glob(os.path.join(QUEUE, '*.npz')))
            files = [f for f in files if not os.path.basename(f).startswith('.tmp_')]
            if files:
                # 训练进度已到目标就不要再吃，直接退出循环
                budget = args.game_batch_num - trainer.game_batch_cur
                if len(files) > budget:
                    files = files[:budget]
                got, prev = ingest(trainer, files, args, args.save_games or None)
                n_since_publish += got

                # 2) 训练（数据够一个 batch 就更新一次）
                if len(trainer.data_buffer) > trainer.batch_size and got:
                    trainer.policy_update()
                    train_steps += 1

                # 3) 发版权重给工兵
                if n_since_publish >= args.publish_every:
                    version += 1
                    publish(trainer, version)
                    n_since_publish = 0
                    el = time.time() - started
                    print('>>> 发布 ver=%d（已训 %d 步，%.2fh，%.1f 局/分）'
                          % (version, train_steps, el / 3600,
                             trainer.game_batch_cur / max(el / 60, 1e-9)),
                          flush=True)

                # 4) 检查点：跨过 check_freq 的整数倍就存
                if (prev // args.check_freq) != (trainer.game_batch_cur // args.check_freq):
                    atomic_torch_save(trainer.policy_value_net,
                                      os.path.join(args.ckpt_dir,
                                                   'ckpt_it%d.model'
                                                   % trainer.game_batch_cur))
                    # current_policy 也留一份老习惯的路径
                    atomic_torch_save(trainer.policy_value_net,
                                      './current_policy.model')

            # 5) 工兵死了就重启（不是我们要停的情况下）
            for wid, p in list(procs.items()):
                if p.poll() is not None:
                    print('!! 工兵 w%d 退出(code=%s)，重启' % (wid, p.returncode),
                          flush=True)
                    procs[wid] = spawn(wid, args, logdir)

            time.sleep(args.poll)

        print('=' * 66, flush=True)
        print(' 到达目标 %d 局，收尾' % trainer.game_batch_num, flush=True)
    except KeyboardInterrupt:
        print('收到 Ctrl-C，收尾', flush=True)

    # 收尾：先存权重，再通知工兵退出
    version += 1
    publish(trainer, version)
    atomic_torch_save(trainer.policy_value_net,
                      os.path.join(args.ckpt_dir,
                                   'ckpt_it%d.model' % trainer.game_batch_cur))
    atomic_torch_save(trainer.policy_value_net, './current_policy.model')
    open(stop_path, 'w').close()

    deadline = time.time() + 180
    for wid, p in procs.items():
        try:
            p.wait(timeout=max(1, deadline - time.time()))
        except Exception:                   # noqa: BLE001
            p.kill()
    print('全部工兵已退出', flush=True)

    el = time.time() - started
    print('=' * 66, flush=True)
    print(' 完成 %d 局  用时 %.1f 小时  平均 %.1f 局/分钟  训练 %d 步'
          % (trainer.game_batch_cur, el / 3600,
             trainer.game_batch_cur / max(el / 60, 1e-9), train_steps), flush=True)
    trainer.metrics.log('done', batch=trainer.game_batch_cur,
                        hours=round(el / 3600, 2), train_steps=train_steps)
    return 0


if __name__ == '__main__':
    sys.exit(main())
