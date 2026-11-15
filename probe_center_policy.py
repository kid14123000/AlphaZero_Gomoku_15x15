#!/usr/bin/env python3
"""看 policy 输出在【中心区域】的概率分布

口径：
  · 用 Board.current_state() 构造输入（与训练/对弈完全一致，含末尾的[:, ::-1, :]翻转）
  · 网络输出 225 维概率，按【到天元(7,7)的切比雪夫距离 d】分环统计质量
  · 空盘 1 个局面 + 若干真实开局局面（ply0..N-1）的平均
产物：控制台表 + JSON + 空盘概率热图 PNG
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from game import Board
from policy_value_net_pytorch import PolicyValueNet

AL = "ABCDEFGHIJKLMNO"
C = 7
_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((n for n in ["PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"] if n in _avail), "DejaVu Sans")
plt.rcParams["font.sans-serif"] = [FONT, "Heiti SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def rings(prob):
    """返回 {d: 概率质量}，d = 到天元的切比雪夫距离"""
    p = np.asarray(prob).reshape(15, 15)
    out = {}
    for d in range(8):
        mask = np.zeros((15, 15), bool)
        for r in range(15):
            for c in range(15):
                if max(abs(r - C), abs(c - C)) == d:
                    mask[r, c] = True
        out[d] = float(p[mask].sum())
    return out


def board_state(moves):
    b = Board(width=15, height=15, n_in_row=5)
    b.init_board(0)
    for r, c in moves:
        b.do_move(r * 15 + c)
    return b.current_state()


def ent(p):
    p = np.asarray(p, dtype=float).ravel()
    p = p / p.sum()
    return float(-(p * np.log(p + 1e-12)).sum())


ap = argparse.ArgumentParser()
ap.add_argument("--model", default="current_policy.model")
ap.add_argument("--games", default="games_1649.json")
ap.add_argument("--plies", type=int, default=12)
ap.add_argument("--out-png", default="policy_center_heatmap.png")
ap.add_argument("--out-json", default="policy_center.json")
a = ap.parse_args()

net = PolicyValueNet(15, 15, model_file=a.model, use_gpu=False)

# ── 1. 空盘 ──
st0 = board_state([])
p0, v0 = net.policy_value(st0[np.newaxis])
p0 = np.asarray(p0[0])
r0 = rings(p0)
print(f"=== 空盘（黑先） ===")
print(f"  熵 {ent(p0):.3f}（上限 {np.log(225):.3f}，占比 {ent(p0)/np.log(225)*100:.0f}%）| value {float(np.asarray(v0).ravel()[0]):+.3f}")
for d in range(5):
    print(f"  d={d}: {r0[d]*100:5.1f}%   d>={d+1}: {(1-sum(r0[i] for i in range(d+1)))*100:5.1f}%")
top = np.argsort(p0)[::-1][:8]
print("  top8:", [(f"{AL[i%15]}{15-i//15}", round(float(p0[i])*100, 1)) for i in top])

# ── 2. 真实开局局面（ply 0..plies-1）──
games = json.load(open(a.games))["games"]
acc = {d: [] for d in range(4)}
tops = []
for g in games:
    mv = [tuple(x) for x in g["moves"]]
    for k in range(min(a.plies, len(mv) - 1)):
        st = board_state(mv[:k])
        p, _ = net.policy_value(st[np.newaxis])
        p = np.asarray(p[0])
        rr = rings(p)
        for d in range(4):
            acc[d].append(rr[d])
        best = int(np.argmax(p))
        tops.append(max(abs(best // 15 - C), abs(best % 15 - C)))

print(f"\n=== 真实开局局面（{len(tops)} 个，ply0..{a.plies-1}，{len(games)} 局）===")
for d in range(4):
    v = np.mean(acc[d]) * 100
    print(f"  环 d={d} 平均质量 {v:5.1f}%")
print(f"  d<=1 {np.mean(acc[0])*100+np.mean(acc[1])*100:5.1f}% | d<=2 {(np.mean(acc[0])+np.mean(acc[1])+np.mean(acc[2]))*100:5.1f}%")
print(f"  top1 距天元: 中位 {np.median(tops):.0f} 格 | 占比 d<=2: {np.mean(np.array(tops)<=2)*100:.0f}%")

json.dump({"model": a.model, "empty": {"rings": r0, "entropy": ent(p0), "value": float(np.asarray(v0).ravel()[0])},
           "openings": {"mean_rings": {str(d): float(np.mean(acc[d])) for d in range(4)},
                        "n": len(tops), "top1_median_d": float(np.median(tops))}},
          open(a.out_json, "w"), ensure_ascii=False, indent=1)

# ── 3. 空盘热图 ──
P = np.asarray(p0).reshape(15, 15) * 100
fig, ax = plt.subplots(figsize=(9.2, 8.2))
im = ax.imshow(P, cmap="YlOrRd")
ax.set_xticks(range(15)); ax.set_xticklabels(list(AL), fontsize=9)
ax.set_yticks(range(15)); ax.set_yticklabels([str(15 - i) for i in range(15)], fontsize=9)
for r in range(15):
    for c in range(15):
        if P[r, c] >= 0.8:
            ax.text(c, r, f"{P[r,c]:.1f}", ha="center", va="center", fontsize=7.2,
                    color="black" if P[r, c] < P.max() * 0.65 else "white")
ax.add_patch(plt.Rectangle((C - 0.5, C - 0.5), 1, 1, fill=False, ec="#0033cc", lw=2.2))
ax.add_patch(plt.Rectangle((C - 2.5, C - 2.5), 5, 5, fill=False, ec="#0033cc", lw=1.2, ls="--"))
ax.set_title(f"空盘 policy 概率分布（{a.model}）\n"
             f"熵占比 {ent(p0)/np.log(225)*100:.0f}%｜天元 {P[C,C]:.2f}%｜d<=1 合计 {(r0[0]+r0[1])*100:.1f}%",
             fontsize=12)
fig.colorbar(im, ax=ax, fraction=0.045, label="概率 %")
fig.tight_layout()
fig.savefig(a.out_png, dpi=150)
print("\n热图:", a.out_png)
