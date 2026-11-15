# -*- coding: utf-8 -*-
"""造「固定评测题库」—— 一次造好，以后所有权重都跑同一套，数字可比

筛选口径：求解器（VCF → VCT）能证明「该走方有强制胜」才收，答案一起存。
不要随机抽样，按 (lib, idx) 顺序取，保证可复现。

配比目标：白先尽量多（连珠题库里白先本来就是少数），黑先补到差不多两倍。
白先数量受供给限制——纯白先库加起来只有 971 道，能解出来的更少。

用法:
  python build_bench.py -j 3 --white 200 --black 300 --out bench_v1.jsonl
  nice -n 10 python build_bench.py ...   # 训练在跑时加 nice
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import vcf2 as V2          # noqa: E402
import vct as VT           # noqa: E402

# 纯白先库（胜方=白）—— 供给有限，全扫
WHITE_LIBS = ['白先VCF', '入门题白先三手詰', '坂田吾朗追诘胜-白先', '白先胜100题', '天狗道场2020-白先']
# 黑先为主的库
BLACK_LIBS = ['黑先VCF', '入门题黑先一手詰', '入门题黑先三手詰', '入门题黑先四追い',
              '黑先胜100题', '坂田吾朗追诘胜-黑先', '天狗道场2020-黑先',
              '三手胜1-入门题', '三手胜2-初级题', '三手胜3-中级题', '三手胜4-高级题',
              '入门题黑先五手詰', '高村政则诘连珠', '五子棋阶梯训练第一章']
# 混合库（两边都有，用得上）
MIX_LIBS = ['vcf-batch-1001', 'vcf-batch-2002', 'vcf-batch-3003', '实战VCF',
            'RenjuPortalVCF', '五子茶馆解题大赛11~20届', '連珠苦楽部詰連珠']

_CFG = {}


def _init(depth, vct_secs, vcf_secs):
    _CFG.update(depth=depth, vct_secs=vct_secs, vcf_secs=vcf_secs)


def _solve(r):
    """返回带上 answer 的题，或 None"""
    side = r['side']
    if r.get('colors') and len(r['colors']) == len(r['moves']):
        g = {m: c for m, c in zip(r['moves'], r['colors'])}
    else:
        s = r['starts']
        g = {m: (s if i % 2 == 0 else 3 - s) for i, m in enumerate(r['moves'])}
    mv = V2.vcf(dict(g), side, max_depth=8, deadline=time.time() + _CFG['vcf_secs'])
    kind = 'vcf'
    if mv is None:
        mv = VT.vct(dict(g), side, depth=_CFG['depth'], seconds=_CFG['vct_secs'])
        kind = 'vct'
    if mv is None:
        return None
    out = dict(r)
    out['answer'] = int(mv)
    out['solver'] = kind
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', default=os.path.expanduser(
        '~/Downloads/gomoku_tiku/data/tiku_dedup_17095.jsonl'))
    ap.add_argument('--out', default=os.path.join(HERE, 'bench_v1.jsonl'))
    ap.add_argument('-j', '--jobs', type=int, default=3)
    ap.add_argument('--white', type=int, default=200, help='白先目标数')
    ap.add_argument('--black', type=int, default=300, help='黑先目标数')
    ap.add_argument('--cap', type=int, default=24, help='每个库最多取几题（保证题型多样）')
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--vct-secs', type=float, default=3.0)
    ap.add_argument('--vcf-secs', type=float, default=2.0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    by_lib = {}
    for r in rows:
        by_lib.setdefault(r['lib'], []).append(r)

    t0 = time.time()
    picked = []
    CAP_PER_LIB = a.cap
    with Pool(a.jobs, initializer=_init, initargs=(a.depth, a.vct_secs, a.vcf_secs)) as p:
        for tag, libs, want in (('白先', WHITE_LIBS, a.white), ('黑先', BLACK_LIBS, a.black)):
            got = 0
            # 🚨 必须按库轮流取，不能顺序取满 —— 否则前一个库题量大就把配额吃光，
            #    造出来的题库只有一个库的题型（第一版就踩了：400 题全来自两个 VCF 库）
            queues = []
            for lib in libs + MIX_LIBS:
                pool = sorted([r for r in by_lib.get(lib, [])
                               if r['side'] == (2 if tag == '白先' else 1)],
                              key=lambda r: r['idx'])
                if pool:
                    queues.append([lib, pool[:CAP_PER_LIB * 4], CAP_PER_LIB, 0])
            while got < want and queues:
                for q in list(queues):
                    lib, pool, cap, used = q
                    if got >= want:
                        break
                    if used >= cap or not pool:
                        queues.remove(q)
                        continue
                    # 一次投 12 道，够填满这个库剩下的配额就行
                    batch, pool[:] = pool[:max(12, cap - used)], pool[max(12, cap - used):]
                    if not batch:
                        queues.remove(q)
                        continue
                    for res in p.imap_unordered(_solve, batch, chunksize=4):
                        if res and used < cap and got < want:
                            res['bench_side'] = tag
                            picked.append(res)
                            got += 1
                            used += 1
                    q[3] = used
            print('  %s 目标 %d，实际拿到 %d  %.0fs' % (tag, want, got, time.time() - t0), flush=True)

    picked.sort(key=lambda r: (r['bench_side'], r['lib'], r['idx']))
    for i, r in enumerate(picked):
        r['bench_id'] = i

    with open(a.out, 'w') as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    import collections
    print('\n共 %d 题 → %s' % (len(picked), a.out))
    c = collections.Counter((r['bench_side'], r['lib']) for r in picked)
    for (sd, lib), n in sorted(c.items()):
        print('  %s %-22s %3d' % (sd, lib[:22], n))
    print('  solver: vcf %d / vct %d' % (sum(1 for r in picked if r['solver'] == 'vcf'),
                                        sum(1 for r in picked if r['solver'] == 'vct')))
    print('耗时 %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
