#!/usr/bin/env python3
"""抽最近几局训练对局画出分阶段局面（黑胜/白胜 × 短局/长局）"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from draw_snapshots import draw, AL

N = 15


def nm(p):
    return f"{AL[p[1]]}{N - p[0]}"


recs = [json.loads(l) for l in open('logs/games.jsonl') if l.strip()][-100:]
model = "current_policy(450代)"
print("可选局数:", len(recs))


def pick(winner, shortest):
    c = [r for r in recs if r['winner'] == winner]
    c.sort(key=lambda r: len(r['moves']), reverse=not shortest)
    return c[0] if c else None


rows = [("黑胜·最短", pick(1, True)), ("黑胜·最长", pick(1, False)),
        ("白胜·最短", pick(2, True)), ("白胜·最长", pick(2, False))]
rows = [(t, g) for t, g in rows if g]

fig, axes = plt.subplots(len(rows), 3, figsize=(12.6, 4.4 * len(rows)))
if len(rows) == 1:
    axes = axes[None, :]

for i, (tag, g) in enumerate(rows):
    mv = [tuple(x) for x in g['moves']]
    n = len(mv)
    cuts = [min(n, max(4, int(n * 0.35))), min(n, max(6, int(n * 0.7))), n]
    for j, k in enumerate(cuts):
        t = f"第{k}手（终局·{'黑胜' if g['winner'] == 1 else '白胜'}）" if j == 2 else f"第{k}手"
        draw(axes[i][j], mv, k, t)
    opening = " ".join(nm(p) for p in mv[:8])
    axes[i][0].set_ylabel(f"局{g['batch']}\n{tag}·{n}手", fontsize=10, labelpad=10)
    axes[i][0].text(0.0, -0.115, "开局: " + opening, transform=axes[i][0].transAxes,
                    fontsize=8.6, color="#333")

fig.suptitle(f"最近训练对局抽样（{model}｜400 模拟｜temp=1.0）\n"
             f"红圈 = 该截面最后一手；棋子上的数字 = 手顺", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "训练对局抽样_450代.png")
fig.savefig(OUT, dpi=170, bbox_inches="tight")
print("已保存 " + OUT)
for tag, g in rows:
    mv = [tuple(x) for x in g['moves']]
    print(f"局{g['batch']} {tag} {len(mv)}手 开局 {' '.join(nm(p) for p in mv[:10])}")
    print(f"    末5手 {' '.join(nm(p) for p in mv[-5:])}")
