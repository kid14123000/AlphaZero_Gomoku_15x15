# -*- coding: utf-8 -*-
"""题库自动判分：VCF 求解器给标准答案，网络 MCTS 的 top1/top3 是否命中

判「命中」的口径：网络落 m 后，对手被迫堵唯一五连点，攻击方仍有 VCF 胜。
（不要求与求解器首手完全相同 —— 同一题常有多个成立的 VCF 起手）
"""
import json, sys, time, os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.expanduser('~/AlphaZero_Gomoku_15x15_v4'))
from vcf import vcf, five_points
from game import Board
from mcts_alphaZero import MCTS
from policy_value_net_pytorch import PolicyValueNet

W = H = 15
ROOT = os.path.expanduser('~/AlphaZero_Gomoku_15x15_v4')


def is_vcf_move(grid, attacker, m, depth=10):
    if m in grid:
        return False
    g = dict(grid)
    g[m] = attacker
    fs = five_points(g, attacker)
    if len(fs) >= 2:
        return True
    if not fs:
        return False
    g[fs[0]] = 3 - attacker
    return vcf(g, attacker, max_depth=depth) is not None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    sims = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = sys.argv[3] if len(sys.argv) > 3 else os.path.join(here, 'data/tiku_dedup_17095.jsonl')
    rows = [json.loads(l) for l in open(src)]
    pool = [r for r in rows if r.get('mode_title') == 'VCF'] or rows
    net = PolicyValueNet(W, H, model_file=os.path.join(ROOT, 'sync/current_policy.model'),
                         use_gpu=False)
    pv = net.policy_value_fn

    ok1 = ok3 = tot = 0
    t0 = time.time()
    for r in pool:
        if tot >= n:
            break
        nst = len(r['moves']); side = r['side']; starts = r['starts']
        g = {m: (starts if j % 2 == 0 else 3 - starts) for j, m in enumerate(r['moves'])}
        if vcf(dict(g), side, max_depth=12) is None:
            continue                      # 求解器都解不出，跳过（可能含禁手）
        b = Board(width=W, height=H, n_in_row=5)
        b.init_board(start_player=starts - 1)
        for m in r['moves']:
            b.do_move(m)
        mcts = MCTS(pv, c_puct=1.25, n_playout=sims)
        acts, _ = mcts.get_move_probs(b, temp=1.0, root_noise=None)
        share = dict(zip(acts, mcts.last_label))
        top = sorted(share, key=lambda m: -share[m])[:3]
        hit1 = is_vcf_move(g, side, top[0])
        hit3 = hit1 or any(is_vcf_move(g, side, m) for m in top[1:])
        ok1 += hit1; ok3 += hit3; tot += 1
        print('  %-2d %-14s #%-3d %2d子 %s先 top1=%-4s %s | top1%s top3%s'
              % (tot, r['lib'][:12], r['idx'], nst, '黑白'[side - 1],
                 '%c%d' % (65 + top[0] % W, H - top[0] // W),
                 ' '.join('%.2f' % share[m] for m in top),
                 '✓' if hit1 else '✗', '✓' if hit3 else '✗'))
    el = time.time() - t0
    print('\n=== sims=%d, %d 题, %.1fs (%.2fs/题) ===' % (sims, tot, el, el / max(tot, 1)))
    print('  top1 命中 %d/%d = %.1f%%' % (ok1, tot, 100 * ok1 / max(tot, 1)))
    print('  top3 命中 %d/%d = %.1f%%' % (ok3, tot, 100 * ok3 / max(tot, 1)))


main()
