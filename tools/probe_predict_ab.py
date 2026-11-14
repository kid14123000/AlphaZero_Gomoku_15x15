"""A/B：predict_inference 的输出到底被翻错没有。

probe_policy_frame.py 已证明「网络 policy 输出 = 真实坐标」（raw 32.1% vs 翻回 6.6%）。
但 predict_inference 结尾还有一次 pc[:, ::-1, :]「翻回真实坐标」。
如果它多余，那 MCTS/对弈/摆谱 拿到的就是镜像后的 policy。

判据：predict_inference 的 top1 是否恰好等于 raw 的垂直镜像；去掉那一步后两者是否完全一致。
"""
import json, sys, os, importlib.util
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('gv4', os.path.join(ROOT, 'Gomoku-v0_pt_v4.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

N = 15
meta = json.load(open(os.path.join(ROOT, 'gomoku_checkpoints', 'metadata.json')))
kw = meta['config']['net_kwargs']
net = mod.AlphaZeroNet(conv_channels=kw['conv_filters'][0],
                       residual_blocks=len(kw['residual_filters']), board_size=N)
sd = torch.load(os.path.join(ROOT, 'checkpoints', 'ckpt_it13256.model'),
                map_location='cpu', weights_only=False)
if all(k.startswith('net.') for k in sd):
    sd = {k[4:]: v for k, v in sd.items()}
net.load_state_dict(sd); net.eval()

class Fake:                       # 只为复用 predict_inference 的边界逻辑
    net = net
    _remote = None

import types
mod_agent = types.SimpleNamespace(
    predict_inference=types.MethodType(mod.AlphaZeroAgent.predict_inference, Fake()),
)

games = []
with open(os.path.join(ROOT, 'logs', 'games.jsonl')) as f:
    for line in f.readlines()[-200:]:
        try: games.append(json.loads(line))
        except Exception: pass

pos = []
for g in games:
    mv = g['moves']
    if len(mv) < 5: continue
    for t in range(2, len(mv) - 1, max(1, (len(mv) - 2) // 2)):
        b = np.zeros((N, N), dtype=np.float32)
        for i, (r, c) in enumerate(mv[:t]):
            b[r, c] = 1.0 if i % 2 == 0 else -1.0
        pos.append((b, mv[t - 1], 1 if t % 2 == 0 else -1, mv[t]))

same_pi_raw = 0
pi_is_mirror = 0
n = 0
tv_raw_s, tv_mir_s = 0.0, 0.0
for board, last, player, actual in pos:
    x = np.asarray(mod.make_input(board, last, player), dtype=np.float32)
    with torch.no_grad():
        p_log, _ = net(torch.from_numpy(x).unsqueeze(0))
    # policy 头是 log_softmax 输出 → exp 即概率
    raw = torch.exp(p_log).reshape(N, N).numpy().astype(np.float64)
    raw = raw / raw.sum()
    mir = raw[::-1, :]

    pi, _ = mod_agent.predict_inference(board[None], np.array([last]), np.array([player]))
    pi = pi.reshape(N, N).astype(np.float64)

    tv_raw_s += 0.5 * np.abs(pi - raw).sum()
    tv_mir_s += 0.5 * np.abs(pi - mir).sum()
    if np.unravel_index(pi.argmax(), pi.shape) == np.unravel_index(raw.argmax(), raw.shape):
        same_pi_raw += 1
    if np.unravel_index(pi.argmax(), pi.shape) == np.unravel_index(mir.argmax(), mir.shape):
        pi_is_mirror += 1
    n += 1

print('取样局面:', n)
print('TV(predict_inference, raw)         平均: %.4f' % (tv_raw_s / n))
print('TV(predict_inference, raw 垂直镜像) 平均: %.4f   ← 若≈0 则它输出的是镜像' % (tv_mir_s / n))
print('top1 一致 raw: %d%%   一致镜像: %d%%' % (100*same_pi_raw/n, 100*pi_is_mirror/n))
