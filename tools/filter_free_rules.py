# -*- coding: utf-8 -*-
"""把题库筛成「无禁手自由规则下确实有解」的子集

对每道题跑 VCF → VCT，能解出的写进输出文件，答案（求解器首手）一起存，
后面判分直接用，不用重算。

⚠️ 全量 17,095 题单核约 7~8 小时。多进程 -j 提速，但训练在跑时别开太多，
   会抢 CPU（我实测 load 18 时训练对局速率没掉，但不建议再高）。

用法:
  python filter_free_rules.py -j 4 -n 0 --out usable.jsonl       # 全量，4 进程
  python filter_free_rules.py -n 300 --out head300.jsonl         # 只跑前 300 题试水
  python filter_free_rules.py --lib 三手胜2-初级题 --out s2.jsonl
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

W = H = 15
_CFG = {}


def _init(depth, vct_secs, vcf_secs):
    _CFG['depth'] = depth
    _CFG['vct_secs'] = vct_secs
    _CFG['vcf_secs'] = vcf_secs


def _one(r):
    side = r['side']
    s = r['starts']
    g = {m: (s if i % 2 == 0 else 3 - s) for i, m in enumerate(r['moves'])}
    t = time.time()
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
    out['solve_s'] = round(time.time() - t, 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', default=os.path.expanduser(
        '~/Downloads/gomoku_tiku/data/tiku_dedup_17095.jsonl'))
    ap.add_argument('--out', default='usable_free_rules.jsonl')
    ap.add_argument('-j', '--jobs', type=int, default=1)
    ap.add_argument('-n', '--limit', type=int, default=0, help='0=全量')
    ap.add_argument('--lib', default=None)
    ap.add_argument('--depth', type=int, default=3)
    ap.add_argument('--vct-secs', type=float, default=3.0)
    ap.add_argument('--vcf-secs', type=float, default=2.0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.jsonl) if l.strip()]
    if a.lib:
        rows = [r for r in rows if r['lib'] == a.lib]
    if a.limit:
        rows = rows[:a.limit]
    print('待跑 %d 题，进程 %d' % (len(rows), a.jobs), flush=True)

    t0 = time.time()
    kept = 0
    with open(a.out, 'w') as fo:
        if a.jobs > 1:
            with Pool(a.jobs, initializer=_init, initargs=(a.depth, a.vct_secs, a.vcf_secs)) as p:
                for i, res in enumerate(p.imap_unordered(_one, rows, chunksize=8), 1):
                    if res:
                        fo.write(json.dumps(res, ensure_ascii=False) + '\n')
                        kept += 1
                    if i % 100 == 0:
                        print('  %d/%d  留下 %d  %.0fs' % (i, len(rows), kept, time.time() - t0), flush=True)
        else:
            _init(a.depth, a.vct_secs, a.vcf_secs)
            for i, r in enumerate(rows, 1):
                res = _one(r)
                if res:
                    fo.write(json.dumps(res, ensure_ascii=False) + '\n')
                    kept += 1
                if i % 100 == 0:
                    print('  %d/%d  留下 %d  %.0fs' % (i, len(rows), kept, time.time() - t0), flush=True)

    print('\n完成：%d 题 → 可用 %d = %.1f%%，用时 %.0fs'
          % (len(rows), kept, 100.0 * kept / max(len(rows), 1), time.time() - t0))
    print('已写 %s' % a.out)


if __name__ == '__main__':
    main()
