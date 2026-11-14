#!/usr/bin/env python3
"""从 logs/metrics.jsonl + logs/games.jsonl 生成 README 用的两张图。

输出到 <repo>/docs/img/ :
    training_curves.png   loss / 策略熵 / 价值头解释方差
    layout_evolution.png  步均 / 开局中心情况 / 黑白胜率
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ── 中文字体（缺字形会变豆腐块，先挑一个存在的）──
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

SRC = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = os.path.join(SRC, 'docs', 'img')
os.makedirs(OUT, exist_ok=True)

# 关键调整点（局号, 标签）
MARKERS = [
    (2656, '重启\n温度改指数衰减'),
    (13467, '噪声搬到根先验\nc_puct 5→1.25'),
]

# ── 读数据 ──
ups, games = [], []
for line in open(os.path.join(SRC, 'logs', 'metrics.jsonl')):
    try:
        r = json.loads(line)
    except Exception:
        continue
    if r.get('type') == 'update':
        ups.append(r)
    elif r.get('type') == 'game':
        games.append(r)

# 手顺在另一个文件里（metrics 的 game 记录只有手数，没有 moves）
moves_by_batch = {}
gp = os.path.join(SRC, 'logs', 'games.jsonl')
if os.path.exists(gp):
    for line in open(gp):
        try:
            r = json.loads(line)
        except Exception:
            continue
        moves_by_batch[r['batch']] = r.get('moves') or []

ups.sort(key=lambda r: r['batch'])
games.sort(key=lambda r: r['batch'])
ub = np.array([r['batch'] for r in ups])
print(f'update {len(ups)} 条，game {len(games)} 局，局号 {games[0]["batch"]}→{games[-1]["batch"]}')


def smooth(y, k):
    if k <= 1:
        return y
    c = np.convolve(y, np.ones(k) / k, mode='valid')
    pad = np.full(k - 1, np.nan)
    return np.concatenate([pad, c])


# ══════════════ 图 1：训练曲线 ══════════════
fig = plt.figure(figsize=(12, 9.2))

def _ax(pos):
    return fig.add_axes(pos)

panels = [
    (0.075, 0.700, 0.86, 0.215, 'loss', '总 loss', '#c0392b', None),
    (0.075, 0.415, 0.86, 0.215, 'entropy', '策略熵（均匀基线 ln225 = 5.416）', '#2471a3', 5.416),
    (0.075, 0.130, 0.86, 0.215, 'ev_new', '价值头解释方差（新参数）', '#1e8449', None),
]

for left, bottom, w, h, key, title, color, hline in panels:
    ax = _ax([left, bottom, w, h])
    y = np.array([r[key] for r in ups], dtype=float)
    ax.plot(ub, y, color=color, lw=0.6, alpha=0.28)
    ax.plot(ub, smooth(y, 40), color=color, lw=2.0)
    if hline is not None:
        ax.axhline(hline, color='#7f8c8d', ls=':', lw=1.2)
    for bx, lbl in MARKERS:
        ax.axvline(bx, color='#34495e', ls='--', lw=1.0, alpha=0.7)
    ax.set_title(title, loc='left', fontsize=13, pad=6)
    ax.set_xlim(ub[0], ub[-1])
    ax.set_xlabel('自对弈局数', fontsize=10)
    ax.tick_params(labelsize=9)

ax0 = fig.axes[0]
for bx, lbl in MARKERS:
    ax0.annotate(lbl, xy=(bx, ax0.get_ylim()[1] * 0.985),
                 xytext=(bx + 260, ax0.get_ylim()[1] * 0.80),
                 fontsize=9, color='#34495e',
                 arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.0))

fig.suptitle('15×15 自对弈训练曲线（800 模拟/手，batch 512）', fontsize=16, y=0.965)
fig.text(0.5, 0.045,
         '细线=逐次更新，粗线=40 次滑动平均。熵从 ln225≈5.416 起步（初始网络接近均匀），'
         '局号 13467 之后噪声改注入根先验，目标分布本身变宽，loss 与熵同时上抬属预期。',
         ha='center', fontsize=9.5, color='#4a5568', wrap=True)
fig.savefig(os.path.join(OUT, 'training_curves.png'), bbox_inches='tight')
plt.close(fig)
print('✅ training_curves.png')

# ══════════════ 图 2：布局演化 ══════════════
gb = np.array([g['batch'] for g in games])
glen = np.array([g['episode_len'] for g in games], dtype=float)
gwin = np.array([g['winner'] for g in games])

moves = []
for g in games:
    mv = moves_by_batch.get(g['batch']) or []
    moves.append(mv)
n_no_moves = sum(1 for m in moves if not m)
if n_no_moves:
    print(f'    {n_no_moves} 局缺 moves（games.jsonl 未覆盖），中心指标按缺测处理')

CENTER = 7.0
first_d = np.array([abs(m[0][0] - CENTER) + abs(m[0][1] - CENTER) if m else np.nan
                    for m in moves], dtype=float)
tengen = np.array([1.0 if (m and m[0][0] == 7 and m[0][1] == 7) else 0.0 for m in moves])
# 前 4 手到中心的平均曼哈顿距离
d4 = []
for m in moves:
    if len(m) >= 4:
        d4.append(np.mean([abs(r - CENTER) + abs(c - CENTER) for r, c in m[:4]]))
    else:
        d4.append(np.nan)
d4 = np.array(d4, dtype=float)

K = 200
fig = plt.figure(figsize=(12, 9.6))


def series_ax(pos, title, ylabel, hline=None):
    ax = fig.add_axes(pos)
    for bx, _ in MARKERS:
        ax.axvline(bx, color='#34495e', ls='--', lw=1.0, alpha=0.7)
    ax.set_title(title, loc='left', fontsize=13, pad=6)
    ax.set_xlim(gb[0], gb[-1])
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=9)
    if hline is not None:
        ax.axhline(hline, color='#7f8c8d', ls=':', lw=1.2)
    return ax


axA = series_ax([0.075, 0.700, 0.86, 0.215], '单局手数（步均）', '手数')
axA.plot(gb, glen, color='#8e44ad', lw=0.4, alpha=0.18)
axA.plot(gb, smooth(glen, K), color='#8e44ad', lw=2.2, label='均值（200 局滑动）')
roll_med = np.array([np.median(glen[max(0, i - K):i + 1]) for i in range(len(glen))])
axA.plot(gb, roll_med, color='#e67e22', lw=1.3, ls='--', label='中位数（200 局滑动）')
axA.legend(fontsize=9, loc='upper left', framealpha=0.9)

axB = series_ax([0.075, 0.415, 0.86, 0.215], '开局位置：首手落在天元的比例 / 前 4 手距中心', '天元率')
axB.plot(gb, smooth(tengen, K) * 100, color='#16a085', lw=2.2, label='首手 = 天元（%）')
axB.set_ylim(0, 100)
axB2 = axB.twinx()
axB2.plot(gb, smooth(d4, K), color='#c0392b', lw=1.8, label='前 4 手平均曼哈顿距离')
axB2.set_ylabel('距天元距离（格）', fontsize=10, color='#c0392b')
axB2.tick_params(axis='y', labelcolor='#c0392b', labelsize=9)
axB2.grid(False)
axB2.spines['right'].set_visible(True)
ln1, lb1 = axB.get_legend_handles_labels()
ln2, lb2 = axB2.get_legend_handles_labels()
axB.legend(ln1 + ln2, lb1 + lb2, fontsize=9, loc='upper left', framealpha=0.9)

axC = series_ax([0.075, 0.130, 0.86, 0.215], '胜率（200 局滑动）', '胜率 %')
black = np.array([1.0 if w == 1 else 0.0 for w in gwin])
white = np.array([1.0 if w == 2 else 0.0 for w in gwin])
axC.plot(gb, smooth(black, K) * 100, color='#2c3e50', lw=2.0, label='黑胜')
axC.plot(gb, smooth(white, K) * 100, color='#b03a2e', lw=2.0, label='白胜')
axC.axhline(50, color='#95a5a6', ls=':', lw=1.2)
axC.set_ylim(0, 100)
axC.set_xlabel('自对弈局数', fontsize=10)
axC.legend(fontsize=9, loc='upper left', framealpha=0.9)

axA.annotate(MARKERS[0][1], xy=(MARKERS[0][0], axA.get_ylim()[1] * 0.95),
             xytext=(MARKERS[0][0] + 300, axA.get_ylim()[1] * 0.80),
             fontsize=9, color='#34495e',
             arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.0))
axA.annotate(MARKERS[1][1], xy=(MARKERS[1][0], axA.get_ylim()[1] * 0.95),
             xytext=(MARKERS[1][0] - 3600, axA.get_ylim()[1] * 0.62),
             fontsize=9, color='#34495e',
             arrowprops=dict(arrowstyle='->', color='#34495e', lw=1.0))

fig.suptitle('布局演化：步均、开局中心、黑白胜率', fontsize=16, y=0.965)
fig.text(0.5, 0.045,
         '窗口 200 局。灰虚线为两次关键改动：局号 2656（温度改指数衰减）、'
         '13467（Dirichlet 噪声搬到根先验 + c_puct 5→1.25）。最后一段样本仅一百多局，波动大。',
         ha='center', fontsize=9.5, color='#4a5568', wrap=True)
fig.savefig(os.path.join(OUT, 'layout_evolution.png'), bbox_inches='tight')
plt.close(fig)
print('✅ layout_evolution.png')

# ── 给 README 用的数字 ──
def seg(lo, hi):
    mask = (gb >= lo) & (gb < hi)
    n = int(mask.sum())
    if not n:
        return None
    return dict(n=n,
                mean=float(np.nanmean(glen[mask])),
                med=float(np.nanmedian(glen[mask])),
                blk=float(black[mask].mean() * 100),
                wht=float(white[mask].mean() * 100),
                fast=float((glen[mask] <= 11).mean() * 100),
                ten=float(tengen[mask].mean() * 100),
                d1=float(np.nanmean(first_d[mask])),
                d4=float(np.nanmean(d4[mask])))

print()
for lo, hi, name in [(1, 2656, '起手→2656'), (2656, 13467, '2656→13467'),
                     (13467, 99999, '13467→末')]:
    s = seg(lo, hi)
    if s:
        print('%-14s n=%5d 均手%5.1f 中位%4.1f 黑%4.1f%% 白%4.1f%% ≤11手%4.1f%% 天元%5.1f%% 首手距%4.2f 前4手距%4.2f'
              % (name, s['n'], s['mean'], s['med'], s['blk'], s['wht'], s['fast'],
                 s['ten'], s['d1'], s['d4']))
