#!/usr/bin/env python3
"""把「人类执白输」那局画成棋盘图。坐标沿用看板的 0 基标签（与页面上看到的一致）。"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Circle, Rectangle

avail = {f.name for f in fm.fontManager.ttflist}
for cand in ('PingFang SC', 'Hiragino Sans GB', 'Heiti TC', 'STHeiti', 'Arial Unicode MS'):
    if cand in avail:
        plt.rcParams['font.family'] = cand
        break
plt.rcParams['axes.unicode_minus'] = False

SRC = sys.argv[1]
OUT = os.path.join(SRC, 'docs', 'img')
os.makedirs(OUT, exist_ok=True)

st = json.load(open(os.path.join(SRC, 'web', 'positions.json')))['人类执白输']['stones']
N = 15

fig, ax = plt.subplots(figsize=(8.6, 9.0))
ax.set_xlim(-1.3, N - 0.3)
ax.set_ylim(-1.8, N - 0.3)

for i in range(N):
    ax.plot([0, N - 1], [i, i], color='#b8a888', lw=1.0, zorder=1)
    ax.plot([i, i], [0, N - 1], color='#b8a888', lw=1.0, zorder=1)
for (r, c) in [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]:
    ax.plot(c, r, marker='o', ms=4, color='#8a7a5c', zorder=2)

R = 0.43
for i, s in enumerate(st, 1):
    r, c = s['r'], s['c']
    black = (i % 2 == 1)
    ax.add_patch(Circle((c, r), R, facecolor='#1c1c1c' if black else 'white',
                        edgecolor='#333' if black else '#555', lw=1.2, zorder=3))
    ax.text(c, r, str(i), ha='center', va='center', fontsize=8.5, zorder=4,
            color='white' if black else '#111',
            fontweight='bold' if i >= 23 else 'normal')

# 黑五连：列 4，行 4..8（0 基）
ax.add_patch(Rectangle((3.48, 3.48), 1.04, 5.04, fill=False,
                       edgecolor='#c0392b', lw=2.4, zorder=5))

ax.set_xticks(range(N))
ax.set_xticklabels(range(N), fontsize=9)
ax.set_yticks(range(N))
ax.set_yticklabels(range(N), fontsize=9)
ax.set_aspect('equal')
ax.set_facecolor('#f3e9d2')
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(length=0)

ax.set_title('对最新权重的实测：人类执白，25 手告负', fontsize=15, pad=14)
ax.text(-1.2, -1.65,
        '黑 = 网络（800 模拟/手）　白 = 人类　棋子上的数字是手顺\n'
        '红框：黑第 23 手在列 4 连成两端都活的四连；白第 24 手只能堵一头，黑第 25 手收五。\n'
        '坐标沿用看板的 0 基标签（0-14）。',
        fontsize=9.2, color='#4a5568', ha='left', va='bottom', linespacing=1.8)

fig.savefig(os.path.join(OUT, 'human_vs_net.png'), bbox_inches='tight', facecolor='white')
print('✅ human_vs_net.png')
