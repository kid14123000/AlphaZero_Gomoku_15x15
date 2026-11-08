#!/usr/bin/env python3
"""Rapfi 外部锚点：我们的网络 vs Rapfi（pbrain 协议，headless）

协议照抄老项目 rapfi_bench.py（含两个坑）：
  1) BOARD 行 `x,y,who` 里 who 是【相对引擎】的：1=引擎自己 2=对手（不是绝对黑1白2）
  2) 引擎常驻：只发 BOARD...DONE，不重启、不发 EOF（每手重启会被 EOF 逼出仓促着法）

用法:
  .venv/bin/python pk_vs_rapfi.py --model best_weights/it800_pure100_100pct.model \
      --rapfi ~/rapfi/pbrain-rapfi_arm64_neon_dotprod --depth 5 --games 4 --sim 400
"""
import argparse
import os
import select
import subprocess
import sys
import time

import numpy as np
import torch

from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet

N = 15
AL = "ABCDEFGHIJKLMNO"


def nm(r, c):
    return f"{AL[c]}{N - r}"


class Rapfi:
    """pbrain 协议客户端（常驻进程）"""

    def __init__(self, binary, depth=None, config=None, size=N):
        self.binary = os.path.expanduser(binary)
        self.cwd = os.path.dirname(self.binary)
        self.size = size
        self.depth = depth
        self.config = config
        self.p = None
        self.start()

    def start(self):
        self.stop()
        cmd = [self.binary]
        if self.config:
            cmd.append('--config=' + self.config)
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  cwd=self.cwd)
        end = time.time() + 5.0
        while time.time() < end:                      # 读到静默为止（不要固定读 N 行）
            r, _, _ = select.select([self.p.stdout], [], [], 0.25)
            if not r:
                break
            if not self.p.stdout.readline():
                break
        self.p.stdin.write(f'START {self.size}\n')
        self.p.stdin.flush()
        self._read_line(timeout=10)

    def _read_line(self, timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            r, _, _ = select.select([self.p.stdout], [], [], 0.5)
            if r:
                line = self.p.stdout.readline()
                if not line:
                    return None
                line = line.strip()
                if line and not line.startswith('MESSAGE'):
                    return line
            if self.p.poll() is not None:
                return None
        return None

    def move(self, stones, my_color):
        """stones: [(r,c,color)] color∈{1,2}；my_color: 引擎执的颜色"""
        lines = ['BOARD']
        for r, c, col in stones:
            lines.append(f'{c},{r},{1 if col == my_color else 2}')
        lines.append('DONE')
        if self.p is None or self.p.poll() is not None:
            self.start()
        try:
            self.p.stdin.write('\n'.join(lines) + '\n')
            self.p.stdin.flush()
        except Exception:
            return None
        resp = self._read_line()
        if not resp:
            return None
        parts = resp.split(',')
        if len(parts) < 2 or parts[0] == '0':
            return None
        try:
            r, c = int(parts[1]), int(parts[0])       # 回复是 "x,y" → (col,row)
        except ValueError:
            return None
        if any(r == rr and c == cc for rr, cc, _ in stones):
            print(f'[rapfi] 返回已占位 {nm(r, c)} → 判引擎异常')
            return None
        return r, c

    def stop(self):
        if self.p:
            try:
                self.p.kill()
            except Exception:
                pass
            self.p = None


def play_one(net_player, rapfi, net_color, sim, verbose=False):
    """net_color: 1=网络执黑 2=网络执白"""
    b = Board(width=N, height=N, n_in_row=5)
    b.init_board(0)
    net_player.set_player_ind(net_color)
    rand = np.random.RandomState(0)
    stones = []
    while True:
        end, winner = b.game_end()
        if end:
            return winner, len(stones)
        if b.current_player == net_color:
            mv = net_player.get_action(b, temp=1e-3, return_prob=0)
        else:
            got = rapfi.move(stones, my_color=3 - net_color)
            if got is None:
                return ('rapfi_fail', len(stones))
            mv = got[0] * N + got[1]
        r, c = mv // N, mv % N
        if mv not in b.availables:
            return ('illegal', len(stones))
        if verbose:
            print(f'   {len(stones)+1:>2}. {"网" if b.current_player == net_color else "R "} {nm(r, c)}')
        stones.append((r, c, b.current_player))
        b.do_move(mv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='best_weights/it800_pure100_100pct.model')
    ap.add_argument('--rapfi', default='~/rapfi/pbrain-rapfi')
    ap.add_argument('--depth', type=int, default=5)
    ap.add_argument('--games', type=int, default=4, help='总对局（黑白各一半）')
    ap.add_argument('--sim', type=int, default=400)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    pv = PolicyValueNet(N, N, model_file=a.model, use_gpu=False)
    player = MCTSPlayer(pv.policy_value_fn, c_puct=5, n_playout=a.sim)
    rapfi = Rapfi(a.rapfi, depth=a.depth)
    print(f'网络 {a.model} @ {a.sim} sims  vs  Rapfi {a.rapfi} depth={a.depth}')
    wins = {'黑': [0, 0], '白': [0, 0]}          # [胜, 总]
    t0 = time.time()
    for i in range(a.games):
        net_color = 1 if i % 2 == 0 else 2
        side = '黑' if net_color == 1 else '白'
        w, n_mv = play_one(player, rapfi, net_color, a.sim, verbose=a.verbose)
        rapf = Rapfi(a.rapfi, depth=a.depth)      # 每局重开，避免状态残留
        rapfi.stop()
        rapfi = rapf
        ok = (w == net_color)
        if w in (1, 2):
            wins[side][1] += 1
            wins[side][0] += int(ok)
        print(f'第{i+1}局 网络执{side}: {"胜" if ok else "负"} ({n_mv} 手) '
              f'累计 黑 {wins["黑"][0]}/{wins["黑"][1]} 白 {wins["白"][0]}/{wins["白"][1]}')
    rapfi.stop()
    tw = sum(v[0] for v in wins.values())
    tn = sum(v[1] for v in wins.values())
    print(f'\n合计 {tw}/{tn} = {100.0*tw/max(tn,1):.1f}%  用时 {(time.time()-t0)/60:.1f} 分')


if __name__ == '__main__':
    main()
