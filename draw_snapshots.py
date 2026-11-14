#!/usr/bin/env python3
"""自对弈棋谱分阶段图：N 局 × 3 个截面（第12手 / 第30手 / 终局）

坐标口径与前端一致：名(r,c) = LETTERS[c] + (15 - r)  → r=0 是最上行(15)
"""
import json, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

AL = "ABCDEFGHIJKLMNO"

_avail = {f.name for f in font_manager.fontManager.ttflist}
FONT = next((n for n in ["PingFang SC", "Heiti SC", "STHeiti", "Songti SC", "Arial Unicode MS"]
             if n in _avail), "DejaVu Sans")
plt.rcParams["font.sans-serif"] = [FONT, "Heiti SC", "STHeiti", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
print("中文字体 =", FONT)


def draw(ax, moves, upto, title, ring_last=True):
    """moves: [(r,c),...]; upto: 画到第几手（含）"""
    ax.set_xlim(-0.9, 14.9)
    ax.set_ylim(14.9, -0.9)                      # r=0 在顶部
    ax.set_xticks(range(15)); ax.set_xticklabels(list(AL), fontsize=7)
    ax.set_yticks(range(15)); ax.set_yticklabels([str(15 - i) for i in range(15)], fontsize=7)
    ax.set_xticks([x - 0.5 for x in range(1, 15)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, 15)], minor=True)
    ax.grid(which="major", color="#888", lw=0.6)
    ax.grid(which="minor", color="#888", lw=0.3, alpha=0.45)
    ax.tick_params(length=0); ax.set_aspect("equal")

    for i, (r, c) in enumerate(moves[:upto], start=1):
        black = (i % 2 == 1)
        ax.scatter(c, r, s=300, c=("#111" if black else "white"),
                   edgecolors="#111", linewidths=0.9, zorder=3)
        ax.text(c, r, str(i), ha="center", va="center", fontsize=8.2, zorder=4,
                color=("white" if black else "#111"))
    if ring_last and upto >= 1:
        r, c = moves[upto - 1]
        ax.scatter(c, r, s=430, facecolors="none", edgecolors="#d40000", linewidths=1.6, zorder=5)
    ax.set_title(title, fontsize=11, pad=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games")
    ap.add_argument("--out", default="snapshots.png")
    ap.add_argument("--min-len", type=int, default=40, help="只挑手数>=此值的局")
    a = ap.parse_args()

    d = json.load(open(a.games))
    games = [g for g in d["games"] if g["n"] >= a.min_len]
    black = sorted([g for g in games if g["winner"] == 1], key=lambda x: -x["n"])[:3]
    white = sorted([g for g in games if g["winner"] == 2], key=lambda x: -x["n"])[:2]
    rows = [("黑胜", g) for g in black] + [("白胜", g) for g in white]
    if not rows:
        raise SystemExit("没有足够长的局")

    fig, axes = plt.subplots(len(rows), 3, figsize=(12.6, 4.5 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for i, (res, g) in enumerate(rows):
        mv = [tuple(x) for x in g["moves"]]
        cuts = [12, 30, len(mv)]
        for j, k in enumerate(cuts):
            last = (j == 2)
            t = f"第{k}手（终局）" if last else f"第{k}手后"
            draw(axes[i][j], mv, k, t)
        axes[i][0].set_ylabel(f"第{i+1}局\n{res}·{len(mv)}手", fontsize=10, labelpad=10)

    fig.suptitle(f"自对弈棋谱（权重 {d['model']}｜{d['playout']} 次模拟｜temp={d['temp']}）\n"
                 f"共抽 {len(rows)} 局：{len(black)} 黑胜 + {len(white)} 白胜｜红圈=该截面最后一手",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(a.out, dpi=170)
    print("已保存:", a.out, f"({len(rows)} 行 × 3 列)")


if __name__ == "__main__":
    main()
