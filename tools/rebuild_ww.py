"""用 games.jsonl 里的白胜局重建白赢储库（π=单手 one-hot，z=+1）"""
import json
import os
import sys

import numpy as np

PROJ = os.path.expanduser("~/AlphaZero_Gomoku_15x15_v4")
sys.path.insert(0, PROJ)
from game import Board                     # noqa: E402

N = 15
MINLEN = int(os.environ.get("GOMOKU_WW_MIN_LEN", "20"))
OUT = os.path.join(PROJ, "sync/ww_reservoir.npz")

gs = [json.loads(l) for l in open(os.path.join(PROJ, "logs/games.jsonl")) if l.strip()]
gs = [x for x in gs if x.get("moves")]
ww = [x for x in gs if x["winner"] == 2]
ww_long = [x for x in ww if x["episode_len"] >= MINLEN]
print("本 run %d 局 | 白胜 %d 局 | 其中 >=%d 手 %d 局" % (len(gs), len(ww), MINLEN, len(ww_long)))

board = Board(width=N, height=N, n_in_row=5)
states, probs = [], []
for game in ww_long:
    board.init_board()
    for i, m in enumerate(game["moves"]):
        r, c = m
        mi = r * N + c
        if i % 2 == 1:                      # 白方手
            states.append(np.ascontiguousarray(board.current_state()))
            p = np.zeros(N * N, dtype=np.float32)
            p[mi] = 1.0
            probs.append(p)
        board.do_move(mi)

if not states:
    print("没有样本，放弃")
    sys.exit(1)

st = np.asarray(states, dtype=np.float32)
pb = np.asarray(probs, dtype=np.float32)
zs = np.ones(len(st), dtype=np.float32)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
tmp = OUT + ".tmp"
np.savez(tmp, states=st, probs=pb, zs=zs)
os.replace(tmp + ".npz", OUT)
print("已写 %s | %d 条样本 (来自 %d 局) | %.1f MB"
      % (OUT, len(zs), len(ww_long), os.path.getsize(OUT) / 1e6))
