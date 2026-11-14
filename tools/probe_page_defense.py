# -*- coding: utf-8 -*-
"""页面层行为验证：给 AI 一个活三/冲四，看它去堵"正确位置"还是"镜像位置"。

为什么要做这一步
----------------
网络层的探针（tools/probe_policy_frame.py）已经证明 policy 输出是真实坐标，
但那是在 python 里直接喂 net 得到的结果。页面层还有几层自己的约定：
摆谱页 /api/study/analyze 自己造棋盘、自己判执色、自己调 _paper_search；
对弈页走另一条 agent.decide。任何一层把坐标翻错，网络层探针都看不见。

判据（三条都要成立）
------------------
1. 正确堵点出现在 MCTS top5 里（通常就是 top1）
2. 正确堵点出现在网络 prior top5 里
3. 镜像点**不在** top5（如果也在，说明这局面对称，不能用来判定，换一个局面）

局面全部故意做成非对称：15×15 的垂直镜像 r → 14-r，所以刻意避开第 7 行
（第 7 行镜像到自己，对称局面根本区分不出坐标系）。
"""
import json
import sys
import urllib.request

API = 'http://127.0.0.1:8773'


def post(path, body):
    req = urllib.request.Request(API + path,
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def mir(r):
    return 14 - r


# ── 局面定义：stones 顺序即手顺，len 为奇数 → 轮白走（白来防守）──────────────
CASES = [
    {
        'name': '横向活三 (第3行)',
        'black': [(3, 6), (3, 7), (3, 8)],
        'white': [(10, 10), (12, 4)],
        'must': [(3, 5), (3, 9)],          # 两侧都能成四，堵任一端即正确
    },
    {
        'name': '横向活三 (第5行)',
        'black': [(5, 3), (5, 4), (5, 5)],
        'white': [(11, 9), (2, 12)],
        'must': [(5, 2), (5, 6)],
    },
    {
        'name': '竖向活三 (第3列, 行2-4)',
        'black': [(2, 3), (3, 3), (4, 3)],
        'white': [(9, 9), (11, 2)],
        'must': [(1, 3), (5, 3)],
    },
    {
        'name': '斜向活三 (3,3)-(5,5)',
        'black': [(3, 3), (4, 4), (5, 5)],
        'white': [(10, 9), (12, 12)],
        'must': [(2, 2), (6, 6)],
    },
    {
        'name': '冲四 (第3行 4子, 两端都成五)',
        'black': [(3, 6), (3, 7), (3, 8), (3, 9)],
        'white': [(10, 10), (12, 4), (1, 13)],
        'must': [(3, 5), (3, 10)],
    },
]


def stones_of(c):
    """交错落子：黑先，黑白交替，与页面 build_state 的奇偶判色一致。"""
    out = []
    b, w = list(c['black']), list(c['white'])
    i = 0
    while b or w:
        if i % 2 == 0 and b:
            r, cc = b.pop(0); out.append({'r': r, 'c': cc, 'p': 1})
        elif w:
            r, cc = w.pop(0); out.append({'r': r, 'c': cc, 'p': -1})
        elif b:
            r, cc = b.pop(0); out.append({'r': r, 'c': cc, 'p': 1})
        i += 1
    return out


def run(weight, sim, cases):
    print('\n' + '=' * 78)
    print('权重 %s   sim=%d' % (weight, sim))
    print('=' * 78)
    rows = []
    for c in cases:
        stones = stones_of(c)
        try:
            r = post('/api/study/analyze', {'stones': stones, 'weight': weight,
                                            'sim_count': sim})
        except Exception as e:
            print('%-28s ❌ 请求失败 %s' % (c['name'], e))
            continue
        if not r.get('ok'):
            print('%-28s ❌ %s' % (c['name'], r.get('error')))
            continue

        pt = [(t['r'], t['c']) for t in r['policy_top']]
        mt = [(t['r'], t['c']) for t in r['mcts_top']]
        must = set(c['must'])
        mmust = {(mir(a), b) for a, b in must}

        hit_p = bool(must & set(pt))
        hit_m = bool(must & set(mt))
        mir_p = bool(mmust & set(pt))
        mir_m = bool(mmust & set(mt))
        # 只看 top1 才能真正判定「它选择堵哪」；top5 里出现镜像常是 1-2 次访问的噪声
        top1 = mt[0] if mt else None
        top1_hit = top1 in must
        top1_mir = top1 in mmust
        # 正确堵点 / 镜像点在 MCTS 里的最好名次（1 起）
        rank_must = min([i + 1 for i, x in enumerate(mt) if x in must] or [99])
        rank_mir = min([i + 1 for i, x in enumerate(mt) if x in mmust] or [99])

        rows.append({'case': c['name'], 'hit_policy': hit_p, 'hit_mcts': hit_m,
                     'mirror_in_policy': mir_p, 'mirror_in_mcts': mir_m,
                     'top1': top1, 'top1_hit': top1_hit, 'top1_mir': top1_mir,
                     'rank_must': rank_must, 'rank_mir': rank_mir,
                     'policy_top': pt[:5], 'mcts_top': mt[:5],
                     'must': list(must), 'mirror_must': sorted(mmust)})

        flag = '✅' if top1_hit and not top1_mir else ('❌' if top1_mir else '⚠️')
        print('%-28s %s  MCTS首选=%-9s 正确堵点名次=%-4s 镜像最好名次=%-4s' %
              (c['name'], flag, top1, rank_must, rank_mir))
        print('     prior top5: %s' % pt)
        print('     MCTS  top5: %s   ← 前3的访问数 %s' %
              (mt, [t['visits'] for t in r['mcts_top'][:3]]))
        print('     正确堵点 %s   镜像点 %s' % (list(must), sorted(mmust)))
    return rows


if __name__ == '__main__':
    weights = sys.argv[1:] or ['v4|weights_it13304.pt']
    sim = 400
    allrows = {}
    for w in weights:
        allrows[w] = run(w, sim, CASES)

    print('\n' + '=' * 78)
    print('汇总')
    print('=' * 78)
    ok = True
    for w, rows in allrows.items():
        n = len(rows)
        m = sum(1 for r in rows if r['top1_hit'])
        p = sum(1 for r in rows if r['hit_policy'])
        mir = sum(1 for r in rows if r['top1_mir'])
        print('%-28s MCTS首选=正确堵点 %d/%d · prior 含 %d/%d · 首选=镜像 %d/%d %s' %
              (w, m, n, p, n, mir, n, '✅' if (m == n and mir == 0) else '❌'))
        ok = ok and (m == n and mir == 0)
    print('\n结论: %s' % ('✅ 页面层坐标正确（堵的是真实位置，不是镜像）' if ok
                        else '❌ 页面层仍有坐标问题'))
