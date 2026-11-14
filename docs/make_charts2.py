#!/usr/bin/env python3
"""布局演化图（第二稿）：去掉没信息量的"天元率"，换成两个距离指标 + 开局热力图。"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

avail = {f.name for f in fm.fontManager.ttflist}
for cand in ('PingFang SC', 'Hiragino Sans GB', 'Heiti TC', 'STHeiti',
             'Songti SC', 'Arial Unicode MS'):
    if cand in avail:
        plt.rcParams['font.family'] = cand
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 130
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

SRC = sys.argv[1]
OUT = os.path.join(SRC, 'docs', 'img')
os.makedirs(OUT, exist_ok=True)

MARKERS = [(2656, '重启\n温度改指数衰减'), (13467, '噪声搬到根先验\nc_puct 5→1.25')]

games = []
for line in open(os.path.join(SRC, 'logs', 'metrics.jsonl')):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get('type') == 'game':
        games.append(r)
moves_by_batch = {}
for line in open(os.path.join(SRC, 'logs', 'games.jsonl')):
    try:
        r = json.loads(line)
    except Exception:
        continue
    moves_by_batch[r['batch']] = r.get('moves') or []
games.sort(key=lambda r: r['batch'])

gb = np.array([g['batch'] for g in games])
glen = np.array([g['episode_len'] for g in games], dtype=float)
gwin = np.array([g['winner'] for g in games])
moves = [moves_by_batch.get(g['batch']) or [] for g in games]

C = 7.0
d1 = np.array([abs(m[0][0] - C) + abs(m[0][1] - C) if m else np.nan for m in moves])
d4 = np.array([np.mean([abs(r - C) + abs(c - C) for r, c in m[:4]]) if len(m) >= 4 else np.nan
               for m in moves])


def sm(y, k):
    c = np.convolve(y, np.ones(k) / k, mode='valid')
    return np.concatenate([np.full(k - 1, np.nan), c])


K = 200
black = (gwin == 1).astype(float)
white = (gwin == 2).astype(float)

fig = plt.figure(figsize=(12, 9.6))
def ax_at(pos):
    return fig.add_axes(pos)

# A 单局手数
axA = ax_at([0.075, 0.700, 0.86, 0.215])
for bx, _ in MARKERS:
    axA.axvline(bx, color='#34495e', ls='--', lw=1.0, alpha=0.7)
axA.plot(gb, glen, color='#8e44ad', lw=0.4, alpha=0.16)
axA.plot(gb, sm(glen, K), color='#8e44ad', lw=2.2, label='均值（200 局滑动）')
axA.plot(gb, sm(np.array([np.median(glen[max(0, i - K):i + 1]) for i in range(len(glen))]), 1),
         color='#e67e22', lw=1.3, ls='--', label='中位数（200 局滑动）')
axA.set_title('单局手数（步均）', loc='left', fontsize=13, pad=6)
axA.set_ylabel('手数', fontsize=10)
axA.set_xlim(gb[0], gb[-1])
axA.set_ylim(0, 62)
axA.legend(fontsize=9.5, loc='upper right', framealpha=0.92)
axA.tick_params(labelsize=9)
axA.annotate('大量速杀局\n（≤11 手）', xy=(700, 14), xytext=(1500, 44),
             fontsize=9, color='#6c3483',
             arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.0))

# B 开局中心
axB = ax_at([0.075, 0.415, 0.86, 0.215])
for bx, _ in MARKERS:
    axB.axvline(bx, color='#34495e', ls='--', lw=1.0, alpha=0.7)
axB.plot(gb, sm(d1, K), color='#16a085', lw=2.1, label='首手距天元（格）')
axB.plot(gb, sm(d4, K), color='#c0392b', lw=1.8, label='前 4 手平均距天元（格）')
axB.set_title('开局离天元多远（越小越靠中心）', loc='left', fontsize=13, pad=6)
axB.set_ylabel('曼哈顿距离（格）', fontsize=10)
axB.set_xlim(gb[0], gb[-1])
axB.set_ylim(2.0, 8.6)
axB.legend(fontsize=9.5, loc='upper right', framealpha=0.92)
axB.tick_params(labelsize=9)

# C 胜率
axC = ax_at([0.075, 0.130, 0.86, 0.215])
for bx, _ in MARKERS:
    axC.axvline(bx, color='#34495e', ls='--', lw=1.0, alpha=0.7)
axC.plot(gb, sm(black, K) * 100, color='#2c3e50', lw=2.0, label='黑胜')
axC.plot(gb, sm(white, K) * 100, color='#b03a2e', lw=2.0, label='白胜')
axC.axhline(50, color='#95a5a6', ls=':', lw=1.2)
axC.set_title('胜负（200 局滑动）', loc='left', fontsize=13, pad=6)
axC.set_ylabel('胜率 %', fontsize=10)
axC.set_xlabel('自对弈局数', fontsize=10)
axC.set_xlim(gb[0], gb[-1])
axC.set_ylim(0, 100)
axC.legend(fontsize=9.5, loc='upper left', framealpha=0.92)
axC.tick_params(labelsize=9)

axA.annotate(MARKERS[0][1], xy=(MARKERS[0][0], 30), xytext=(3300, 46),
             fontsize=9, color='#34495e',
             arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.0))
axA.annotate(MARKERS[1][1], xy=(MARKERS[1][0], 26), xytext=(8300, 44),
             fontsize=9, color='#34495e',
             arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.0))

fig.suptitle('布局演化：步均、开局中心、胜负', fontsize=16, y=0.965)
fig.text(0.5, 0.045,
         '窗口 200 局；灰虚线为两次关键改动（局号 2656 / 13467）。'
         '单局手数原始值超过 62 的被截断，只看滚动线。',
         ha='center', fontsize=9.5, color='#4a5568')
fig.savefig(os.path.join(OUT, 'layout_evolution.png'), bbox_inches='tight')
plt.close(fig)
print('✅ layout_evolution.png')

# ══════════ 开局热力图：早期 vs 最近 ══════════
N_SPLIT = 4000
early = [m for m in moves[:N_SPLIT] if len(m) >= 4]
late = [m for m in moves[-N_SPLIT:] if len(m) >= 4]


def heat(games_moves):
    h = np.zeros((15, 15))
    for m in games_moves:
        for r, c in m[:4]:
            h[r, c] += 1
    return h / max(len(games_moves), 1)


hE, hL = heat(early), heat(late)
vmax = max(hE.max(), hL.max())

fig2, axes = plt.subplots(1, 2, figsize=(11.4, 5.4))
for ax, h, title in ((axes[0], hE, f'前 {len(early)} 局'),
                     (axes[1], hL, f'最近 {len(late)} 局')):
    im = ax.imshow(h, cmap='YlOrRd', vmin=0, vmax=vmax)
    ax.set_title(title + '：前 4 手落点频次', fontsize=13, pad=8)
    ax.set_xticks(range(0, 15, 2))
    ax.set_yticks(range(0, 15, 2))
    ax.set_xticklabels([chr(65 + i) for i in range(0, 15, 2)], fontsize=9)
    ax.set_yticklabels(range(1, 16, 2), fontsize=9)
    ax.grid(False)
    ax.set_aspect('equal')
    ax.plot(7, 7, marker='+', color='#2c3e50', ms=12, mew=1.4)
    for sp in ax.spines.values():
        sp.set_visible(True)
fig2.colorbar(im, ax=axes, fraction=0.028, pad=0.02, label='每局出现概率')
fig2.suptitle('开局前 4 手的落点分布（+ = 天元）', fontsize=15, y=0.99)
fig2.text(0.5, 0.02,
          '两张图同一色标。早期六成首手集中在天元的 4 个正邻点（它们互为 D4 对称像，概率必然相同）；'
          '最近首手摊到全部 225 个点，单点最高只剩 4.5%。',
          ha='center', fontsize=9.5, color='#4a5568')
fig2.savefig(os.path.join(OUT, 'opening_heatmap.png'), bbox_inches='tight')
plt.close(fig2)
print('✅ opening_heatmap.png')

# 数字
def seg(lo, hi):
    m = (gb >= lo) & (gb < hi)
    if not m.sum():
        return
    print('%-14s n=%5d 均手%5.1f 中位%4.1f 黑%4.1f%% 白%4.1f%% ≤11手%4.1f%% 首手距%4.2f 前4手距%4.2f'
          % ('%d-%d' % (lo, hi), m.sum(), np.nanmean(glen[m]), np.nanmedian(glen[m]),
             black[m].mean() * 100, white[m].mean() * 100,
             (glen[m] <= 11).mean() * 100, np.nanmean(d1[m]), np.nanmean(d4[m])))


seg(1, 2656); seg(2656, 13467); seg(13467, 99999)
print('热力图落点峰值：早期 %.4f @%s   最近 %.4f @%s'
      % (hE.max(), np.unravel_index(hE.argmax(), hE.shape),
         hL.max(), np.unravel_index(hL.argmax(), hL.shape)))
