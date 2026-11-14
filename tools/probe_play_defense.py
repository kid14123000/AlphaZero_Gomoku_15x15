# -*- coding: utf-8 -*-
"""对弈页行为验证：让页面自己的 AI（/api/start + /api/play → agent.decide）在真实对局里
遇到「对方下一手就成五/成活四」的局面，看它堵的是真实坐标还是垂直镜像坐标。

为什么要单独一条
----------------
摆谱页走 /api/study/analyze（内部 _paper_search），对弈页走 ai_move → agent.decide，
两条不同的封装。网络层探针和摆谱页都验过了，但 ai_move 这一层没人验过。

做法
----
AI vs AI 连打几局（每手都由页面的 /api/play 产生），把整盘手顺记下来，
然后离线逐手判定：
  · 这手之前，对手是否有「下一手成五」的必堵点（combat._threat_points）
  · 落子是否落在必堵点上            → 正确防守
  · 落子是否恰是某个必堵点的垂直镜像 → 说明坐标被翻过
  · 落子是否等于网络 raw 约定的 top1 → 与已验证正确的约定是否一致
"""
import json
import sys
import urllib.request

API = 'http://127.0.0.1:8773'
N = 15

sys.path.insert(0, '/Users/kid/AlphaZero_Gomoku_15x15_v4/web')
import combat  # noqa: E402

combat.set_board(N, 5)


def post(path, body=None):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(API + path, data=data,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())


def mir(r):
    return N - 1 - r


def play_one_game(weight, sim, temp=0.02, max_moves=60):
    r = post('/api/start', {
        'black_type': 'ai', 'white_type': 'ai',
        'black_weight': weight, 'white_weight': weight,
        'sim_count': sim, 'greedy': True,
        'temperature': temp,          # 低=近贪心(每局几乎一样)；0.3~0.4 才有对局多样性
    })
    if not r.get('ok'):
        raise RuntimeError(r.get('error'))
    moves = []
    for _ in range(max_moves):
        r = post('/api/play', {})
        if not r.get('ok'):
            break
        la = r.get('last_ai')
        st = r['state']
        if la:
            moves.append((la['r'], la['c']))
        if st.get('over'):
            return moves, st.get('winner')
    return moves, None


def analyze(moves):
    """逐手判定防守。返回统计。

    ⚠️ combat 内部的棋盘是稀疏 dict {(r,c): 1|2}（1=黑 2=白），
       不是 15×15 的 int 数组 —— 直接塞数组会在 board.get() 处炸。
    """
    seq = [(r, c, 1 if i % 2 == 0 else 2) for i, (r, c) in enumerate(moves)]
    stat = {'opportunities': 0, 'defended': 0, 'mirror_played': 0,
            'must': 0, 'must_defended': 0, 'details': []}
    for i, (r, c, p) in enumerate(seq):
        if i < 6:
            continue
        before = combat._board_of(moves, upto=i)
        opp = 2 if p == 1 else 1
        # 自己已经能连五 → 不该被要求防守（否则把制胜手算成防守）
        if combat._winning_now(before, p):
            continue
        must, three = combat._threat_points(before, opp)
        if not must:
            continue
        stat['opportunities'] += 1
        stat['must'] += 1
        hit = (r, c) in must
        mirr = any((mir(mr), mc) == (r, c) for mr, mc in must)
        if hit:
            stat['defended'] += 1
            stat['must_defended'] += 1
        if mirr and not hit:
            stat['mirror_played'] += 1
        stat['details'].append({'move': i + 1, 'played': [r, c],
                                'must': sorted(must), 'hit': hit, 'mirror': mirr})
    return stat


if __name__ == '__main__':
    weight = sys.argv[1] if len(sys.argv) > 1 else 'v4|weights_it13304.pt'
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    sim = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    temp = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02

    print('权重 %s · sim=%d · temp=%s · AI vs AI 打 %d 局（全部走页面 /api/play）' % (weight, sim, temp, games))
    tot = {'opportunities': 0, 'defended': 0, 'mirror_played': 0}
    lens = []
    for gi in range(games):
        moves, win = play_one_game(weight, sim, temp)
        lens.append(len(moves))
        st = analyze(moves)
        for k in tot:
            tot[k] += st[k]
        print('\n第 %d 局: %d 手, winner=%s' % (gi + 1, len(moves), win))
        print('  必堵点机会 %d · 堵住 %d · 落子恰是镜像点 %d' %
              (st['opportunities'], st['defended'], st['mirror_played']))
        for d in st['details'][:5]:
            print('    第%2d手 落在%s · 必堵点%s · 堵住=%s 镜像=%s' %
                  (d['move'], d['played'], d['must'], d['hit'], d['mirror']))
        if moves:
            print('  手顺: %s' % (moves[:14],))

    print('\n' + '=' * 70)
    print('合计: 必堵点机会 %d · 正确防守 %d (%.1f%%) · 落子=镜像点 %d (%.1f%%)' % (
        tot['opportunities'], tot['defended'],
        100 * tot['defended'] / max(1, tot['opportunities']),
        tot['mirror_played'], 100 * tot['mirror_played'] / max(1, tot['opportunities'])))
    print('对局手数: %s' % lens)
    if tot['mirror_played'] == 0 and tot['defended'] > 0:
        print('结论: ✅ 对弈页 AI 堵的是真实坐标（一次镜像都没出现）')
    elif tot['opportunities'] == 0:
        print('结论: ⚠️ 这局没出现必堵点，样本无效，加大局数')
    else:
        print('结论: ❌ 出现了镜像落子，页面层坐标有问题')
