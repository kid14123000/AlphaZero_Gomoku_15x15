#!/usr/bin/env python3
"""白崩哨兵：只读 sync/current_policy.model，测三个指标，追加到 logs/sentinel.txt。

不改训练代码、不重启、不占核（一次前向 + 轻量统计）。
用法：.venv/bin/python sentinel.py            # 跑一次
     .venv/bin/python sentinel.py 1800        # 每 1800 秒跑一次，共 4 次（约 2 小时）
"""
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

from game import Board                      # noqa: E402
from policy_value_net_pytorch import PolicyValueNet   # noqa: E402

OUT = os.path.join(HERE, 'logs', 'sentinel.txt')
MODEL = os.path.join(HERE, 'sync', 'current_policy.model')


def empty_value(path):
    """空盘 value（黑先视角）。偏正越多 = 越相信"先手/执黑必胜"这个身份捷径。"""
    net = PolicyValueNet(15, 15, model_file=path, use_gpu=False)
    b = Board(width=15, height=15, n_in_row=5)
    b.init_board()
    p, v = net.policy_value(b.current_state()[None].astype(np.float32))
    return float(np.asarray(v).reshape(-1)[0]), float(p[0].max())


def recent_stats(n_games=60):
    """最近 n 局的白胜率 / 均手 / 速杀率，以及最新权重版本。"""
    rows = []
    for line in open(os.path.join(HERE, 'logs', 'metrics.jsonl')):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('type') == 'game':
            rows.append(r)
    rows.sort(key=lambda r: r['batch'])
    s = rows[-n_games:]
    L = [x['episode_len'] for x in s]
    w = sum(1 for x in s if x['winner'] == 2)
    b = sum(1 for x in s if x['winner'] == 1)
    return dict(n=len(s), batch_last=s[-1]['batch'],
                white=100.0 * w / len(s), black=100.0 * b / len(s),
                mean=statistics.mean(L), med=statistics.median(L),
                fast=100.0 * sum(1 for x in L if x <= 11) / len(L),
                ver=s[-1].get('net_version'))


def one_shot(tag=''):
    try:
        v, top1 = empty_value(MODEL)
    except Exception as e:
        print('  读权重失败(可能正在被替换): %s' % e)
        v, top1 = float('nan'), float('nan')
    st = recent_stats()
    ts = time.strftime('%m-%d %H:%M:%S')
    line = ('%s | ver=%-3s 局号=%-6d 近%2d局 白胜=%5.1f%% 黑胜=%5.1f%% 均手=%4.1f '
            '中位=%3.0f 速杀=%4.1f%% | 空盘value=%+.4f top1=%.3f %s'
            % (ts, st['ver'], st['batch_last'], st['n'], st['white'], st['black'],
               st['mean'], st['med'], st['fast'], v, top1, tag))
    with open(OUT, 'a') as f:
        f.write(line + '\n')
    print('  ' + line)
    return st, v


if __name__ == '__main__':
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if interval <= 0:
        one_shot()
    else:
        for i in range(4):
            one_shot('[哨兵%d/4]' % (i + 1))
            if i < 3:
                time.sleep(interval)
