#!/usr/bin/env python3
"""权重网络"学没学"体检（对齐 alphazero-gomoku-net-analysis 的四类探针）

A 权重统计 + 跨代位移   B 激活率/死通道   C 表征随局面变化
D 价值幅度与判别力(AUC)   E 一步成五/堵四 探针
用法: .venv/bin/python analyze_net.py [--ckpt-old checkpoints/ckpt_it250.model] [--ckpt-new current_policy.model]
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn

from game import Board
from policy_value_net_pytorch import PolicyValueNet, Net

AL = "ABCDEFGHIJKLMNO"
CONV = ['conv1', 'conv2', 'conv3']
ap = argparse.ArgumentParser()
ap.add_argument('--ckpt-old', default='checkpoints/ckpt_it250.model')
ap.add_argument('--ckpt-new', default='current_policy.model')
ap.add_argument('--games', default='logs/games.jsonl')
ap.add_argument('--max-pos', type=int, default=400)
a = ap.parse_args()


def load(path):
    pv = PolicyValueNet(15, 15, model_file=path, use_gpu=False)
    pv.policy_value_net.eval()
    return pv.policy_value_net


def layer_stats(net):
    out = {}
    for n, m in net.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            w = m.weight.detach()
            out[n] = {'numel': w.numel(), 'mean': float(w.mean()),
                      'std': float(w.std()), 'norm': float(w.norm())}
    return out


def rel_disp(na, nb):
    res = {}
    da, db = dict(na.named_modules()), dict(nb.named_modules())
    for n in da:
        if isinstance(da[n], (nn.Conv2d, nn.Linear)):
            w0, w1 = da[n].weight.detach(), db[n].weight.detach()
            res[n] = float((w1 - w0).norm() / (w0.norm() + 1e-12))
    return res


def positions(path, max_n):
    """真实局面 + z 标签（+1 = 当前走子方最终获胜）"""
    X, Z = [], []
    if not os.path.exists(path):
        return np.zeros((0, 4, 15, 15), np.float32), np.zeros(0, np.float32)
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        w = r.get('winner')
        if w not in (1, 2):
            continue
        b = Board(width=15, height=15, n_in_row=5)
        b.init_board(0)
        for k, (r_, c_) in enumerate(r['moves'][:40]):
            X.append(np.array(b.current_state(), dtype=np.float32, copy=True))
            Z.append(1.0 if b.current_player == w else -1.0)
            b.do_move(r_ * 15 + c_)
            if len(X) >= max_n:
                return np.asarray(X), np.asarray(Z)
    return np.asarray(X), np.asarray(Z)


def make_pos(moves):
    b = Board(width=15, height=15, n_in_row=5)
    b.init_board(0)
    for r, c in moves:
        b.do_move(r * 15 + c)
    return np.array(b.current_state(), dtype=np.float32, copy=True)


def acts(net, X):
    store, hs = {}, []
    for name in CONV:
        def hook(mod, inp, out, name=name):
            store[name] = out.detach()
        hs.append(getattr(net, name).register_forward_hook(hook))
    with torch.no_grad():
        net(torch.from_numpy(np.ascontiguousarray(X)))
    for h in hs:
        h.remove()
    return store


def policy_topk(net, state, k=3):
    with torch.no_grad():
        logp, v = net(torch.from_numpy(np.ascontiguousarray(state[None])))
    p = torch.exp(logp)[0].numpy()
    idx = np.argsort(p)[::-1][:k]
    return [(f"{AL[i % 15]}{15 - i // 15}", float(p[i])) for i in idx], float(v.item())


def values(net, X):
    with torch.no_grad():
        _, v = net(torch.from_numpy(np.ascontiguousarray(X)))
    return v.numpy().ravel()


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg])
    order = allv.argsort().argsort() + 1
    rp = order[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    old, new = load(a.ckpt_old), load(a.ckpt_new)
    rnd = Net(15, 15); rnd.eval()

    print("=== A. 权重统计与跨代位移 ===")
    so, sn = layer_stats(old), layer_stats(new)
    d = rel_disp(old, new)
    print(f"{'层':<10}{'参数量':>9}{'std(250)':>10}{'std(300)':>10}{'|W|(250)':>11}{'|W|(300)':>11}{'相对位移':>10}")
    for n in d:
        print(f"{n:<10}{so[n]['numel']:>9}{so[n]['std']:>10.4f}{sn[n]['std']:>10.4f}"
              f"{so[n]['norm']:>11.2f}{sn[n]['norm']:>11.2f}{d[n]*100:>9.2f}%")
    print(f"{'合计':<10}{sum(so[n]['numel'] for n in d):>9}  平均相对位移 {np.mean(list(d.values()))*100:.2f}%")

    print("\n=== B. 激活率（真实局面） ===")
    X, Z = positions(a.games, a.max_pos)
    print(f"局面样本 {len(X)}")
    if len(X):
        A = acts(new, X)
        for name in CONV:
            t = A[name].numpy()
            flat = t.reshape(t.shape[0], t.shape[1], -1)
            nz = (flat > 0).mean()
            per_ch = (flat > 0).mean(axis=(0, 2))
            ch_std = flat.mean(axis=2).std(axis=0)
            dead = int((per_ch < 0.01).sum()); const = int((ch_std < 1e-4).sum())
            print(f"  {name}: 激活率 {nz*100:5.1f}% | 死通道 {dead}/{len(per_ch)} | "
                  f"常通道 {const}/{len(per_ch)} | 通道激活率 {per_ch.min()*100:.0f}~{per_ch.max()*100:.0f}%")

    print("\n=== C. 表征随局面变化（特征图 L2 距离） ===")
    Xp = np.stack([make_pos([]),
                   make_pos([(7,3),(7,2),(7,4),(0,0),(7,5),(0,1),(7,6),(0,2)]),
                   make_pos([(9,2),(9,3),(0,0),(9,4),(0,1),(9,5),(0,2),(9,6),(0,3),(14,14)])])
    Ap = acts(new, Xp)
    for name in CONV:
        t = Ap[name].numpy().reshape(3, -1)
        print(f"  {name}: 空盘↔黑四 {np.linalg.norm(t[0]-t[1]):6.2f} | 黑四↔白四 {np.linalg.norm(t[1]-t[2]):6.2f} | 空盘↔白四 {np.linalg.norm(t[0]-t[2]):6.2f}")

    if len(X):
        print("\n=== D. 价值幅度与判别力（真实局面，z=当前走子方最终胜） ===")
        for tag, net in (('250代', old), ('300代', new), ('随机初始化', rnd)):
            v = values(net, X)
            pos, neg = v[Z > 0], v[Z < 0]
            print(f"  {tag}: mean {v.mean():+.3f} std {v.std():.3f} range [{v.min():+.2f},{v.max():+.2f}] | "
                  f"胜局 {pos.mean():+.3f}(n={len(pos)}) 负局 {neg.mean():+.3f}(n={len(neg)}) | AUC {auc(pos,neg):.3f}")

    print("\n=== E. 关键局面探针（policy top3 + value） ===")
    cases = [
        ("空盘（黑走）", []),
        ("黑冲四·右端空（黑走，制胜点=第8列）",
         [(7,3),(7,2),(7,4),(0,0),(7,5),(0,1),(7,6),(0,2)]),
        ("白冲四·黑须堵（黑走，堵点=第8列）",
         [(9,2),(9,3),(0,0),(9,4),(0,1),(9,5),(0,2),(9,6),(0,3),(14,14)]),
    ]
    for tag, mv in cases:
        st = make_pos(mv)
        t3, v3 = policy_topk(new, st)
        t2, v2 = policy_topk(old, st)
        print(f"  {tag}")
        print(f"     300代: top3={t3}  value={v3:+.3f}")
        print(f"     250代: top3={t2}  value={v2:+.3f}")


if __name__ == '__main__':
    main()
