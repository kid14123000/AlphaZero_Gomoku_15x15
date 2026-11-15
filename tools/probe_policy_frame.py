"""判定 v4 权重的 policy 输出到底在哪个坐标系（翻转 or 真实）。

背景：训练侧 game.py 的 current_state() 把整盘 [:, ::-1, :] 翻转后喂网络，
但 MCTS 的动作索引（训练标签）一直是真实坐标。所以要么
  (A) 网络学会了「翻转输入 → 真实输出」，推理时不该再翻回去；
  (B) 输出也在翻转坐标系里，推理时必须翻回来。

做法：从最近的真实自对弈对局里取样，把某一手之前的局面喂给网络，
看 raw / unflipped 两种读法，哪个的 top1 更常命中那一手真实落子。
正确的那一种命中率应在几十个百分点，错的那种接近随机 (1/225≈0.4%)。
"""
import json, sys, os, importlib.util
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location('gv4', os.path.join(ROOT, 'Gomoku-v0_pt_v4.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

N = 15
CKPT = os.path.join(ROOT, 'checkpoints', 'ckpt_it13256.model')
meta = json.load(open(os.path.join(ROOT, 'gomoku_checkpoints', 'metadata.json')))
kw = meta['config']['net_kwargs']
net = mod.AlphaZeroNet(conv_channels=kw['conv_filters'][0],
                       residual_blocks=len(kw['residual_filters']), board_size=15)
sd = torch.load(CKPT, map_location='cpu', weights_only=False)
if all(k.startswith('net.') for k in sd):
    sd = {k[4:]: v for k, v in sd.items()}
net.load_state_dict(sd)
net.eval()

# 取最近的对局，每局采样若干手（跳过第 1 手：空盘太对称，区分不出坐标系）
games = []
with open(os.path.join(ROOT, 'logs', 'games.jsonl')) as f:
    lines = f.readlines()[-400:]
for line in lines:
    try:
        games.append(json.loads(line))
    except Exception:
        pass

pos = []          # (canonical, last_move, player, actual_move)
for g in games:
    mv = g['moves']
    if len(mv) < 6:
        continue
    for t in range(2, len(mv) - 1, max(1, (len(mv) - 2) // 3)):
        board = np.zeros((N, N), dtype=np.float32)
        for i, (r, c) in enumerate(mv[:t]):
            board[r, c] = 1.0 if i % 2 == 0 else -1.0      # 1=黑(先手) -1=白
        player = 1 if t % 2 == 0 else -1
        last = mv[t - 1]
        pos.append((board, last, player, mv[t]))

print('取样局面数:', len(pos))

hit_raw = hit_uf = 0
rank_raw = []
rank_uf = []
for board, last, player, actual in pos:
    x = np.asarray(mod.make_input(board, last, player), dtype=np.float32)
    with torch.no_grad():
        p_log, v = net(torch.from_numpy(x).unsqueeze(0))
    p = torch.exp(p_log).reshape(N, N).numpy()
    p_raw = p                                  # 原样读
    p_uf = p[::-1, :]                          # 翻回来再读
    a = actual[0] * N + actual[1]
    hit_raw += int(np.unravel_index(p_raw.argmax(), p_raw.shape) == (actual[0], actual[1]))
    hit_uf += int(np.unravel_index(p_uf.argmax(), p_uf.shape) == (actual[0], actual[1]))
    rank_raw.append(int((p_raw.ravel() > p_raw[actual[0], actual[1]]).sum()))
    rank_uf.append(int((p_uf.ravel() > p_uf[actual[0], actual[1]]).sum()))

n = len(pos)
print('── top1 命中真实落子 ──')
print('  raw       : %d/%d = %.1f%%   中位排名 %d' % (hit_raw, n, 100 * hit_raw / n, int(np.median(rank_raw))))
print('  unflipped : %d/%d = %.1f%%   中位排名 %d' % (hit_uf, n, 100 * hit_uf / n, int(np.median(rank_uf))))
print('  随机基线  : %.2f%%' % (100 / 225))
