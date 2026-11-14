# -*- coding: utf-8 -*-
"""战斗激烈度指标 —— 全部只依赖一局的落子手顺 `moves=[[r,c],...]`。

为什么需要这套指标
------------------
自对弈里黑先、两色都由同一个网络出招。实测对局里出现过
「黑一路连成五子，白在旁边 5 格外交替落子」的局面 —— 双方各自发展，
互相不干扰。用胜率/手数都看不出这种毛病（手数短只说明有一方赢了），
必须直接量化「有没有在跟对手较劲」。

四个子指标（都是 0~1，越大越激烈）
--------------------------------
contact   接触率：落子在敌方棋子切比雪夫距离 ≤1 的位置的比例
proximity 贴身度：1 - median(每手到最近敌方子的距离)/4，夹到 0~1
block     防守率：对手有「下一手就成五」的必堵点时，本方去堵的比例
zone      区域共享：双方棋子包围盒的重叠面积 / 并集面积

composite 战斗激烈度指数 = 上面四项等权平均 × 100
（等权是有意的：不做调参，避免"指标好看"变成拟合出来的）
"""
import statistics

N = 15
WIN = 5


def set_board(n, win=None):
    """按当前棋盘的边长/连子数重设，必须在 analyze() 之前调用。

    🚨 写死 N=15 在 9×9 上会误判「活四」：_makes_live_four 用 `0 <= c < N`
       判断端口是否为空，(0,9) 在 9×9 里越界但当 N=15 时被当成合法空点，
       于是右边缘的 4 连 (0,5)-(0,8) 被算成两端都空的活四。
       后果：威胁点里混进棋盘外坐标（永远不会被落子），防守率分母被撑大 ⇒ 偏低。
    """
    global N, WIN
    N = int(n)
    if win:
        WIN = int(win)


def _cheb(r1, c1, r2, c2):
    return max(abs(r1 - r2), abs(c1 - c2))


def _board_of(moves, upto=None):
    """返回 {(r,c): 1|2}，只放前 upto 手。颜色按顺序交替（黑先）。"""
    b = {}
    for i, (r, c) in enumerate(moves[:upto] if upto else moves):
        b[(r, c)] = 1 if i % 2 == 0 else 2
    return b


def _wins_at(board, r, c, p):
    """在 (r,c) 放 p 子是否能连成 5（不修改 board）。"""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cnt = 1
        for s in (1, -1):
            nr, nc = r + dr * s, c + dc * s
            while 0 <= nr < N and 0 <= nc < N and board.get((nr, nc)) == p:
                cnt += 1
                nr += dr * s
                nc += dc * s
        if cnt >= WIN:
            return True
    return False


def _winning_now(board, p):
    """p 方此刻是否已有「下一手就成五」的点。"""
    for r in range(N):
        for c in range(N):
            if (r, c) in board:
                continue
            if _wins_at(board, r, c, p):
                return True
    return False


def _threat_points(board, p):
    """p 方「下一手成五」的点（必堵点），以及「下一手成活四」的活三点。

    返回 (must_block, live_three_def)
      must_block     : 放这里 p 就连成 5 → 对方必须堵
      live_three_def : 放这里 p 会形成活四（两端可延）→ 对方也该堵
    """
    must, three = [], []
    for r in range(N):
        for c in range(N):
            if (r, c) in board:
                continue
            if _wins_at(board, r, c, p):
                must.append((r, c))
                continue
            # 模拟落子后看是否产生"活四"
            board[(r, c)] = p
            try:
                if _makes_live_four(board, r, c, p):
                    three.append((r, c))
            finally:
                del board[(r, c)]
    return must, three


def _makes_live_four(board, r, c, p):
    """刚落下的 (r,c) 是否让 p 形成「活四」（恰好 4 连，且两端都空）。"""
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        cnt = 1
        ends = []
        for s in (1, -1):
            nr, nc = r + dr * s, c + dc * s
            while 0 <= nr < N and 0 <= nc < N and board.get((nr, nc)) == p:
                cnt += 1
                nr += dr * s
                nc += dc * s
            ends.append((nr, nc))
        if cnt == 4:
            open_ends = [e for e in ends
                         if 0 <= e[0] < N and 0 <= e[1] < N and e not in board]
            if len(open_ends) == 2:
                return True
    return False


def _split(moves):
    """返回 [(r,c,player)]，player 1=黑 2=白。"""
    return [(r, c, 1 if i % 2 == 0 else 2) for i, (r, c) in enumerate(moves)]


def analyze(moves, block_from=6):
    """算一局的战斗指标。moves 为空或太短时返回 None。

    block_from: 前几手之后才开始统计"必堵点"（开局必堵点没有意义，
                而且第 1 手棋盘上没有对手子，白方必堵点会被误判）
    """
    moves = [(int(m[0]), int(m[1])) for m in moves]
    if len(moves) < 4:
        return None

    seq = _split(moves)

    # ── 接触率 / 贴身距离
    contact_hits = 0
    contact_n = 0
    min_dists = []
    for i, (r, c, p) in enumerate(seq):
        enemies = [(rr, cc) for (rr, cc, q) in seq[:i] if q != p]
        if not enemies:
            continue
        d = min(_cheb(r, c, rr, cc) for rr, cc in enemies)
        min_dists.append(d)
        contact_n += 1
        if d <= 1:
            contact_hits += 1

    contact = contact_hits / contact_n if contact_n else None
    if min_dists:
        md = statistics.median(min_dists)
        proximity = max(0.0, min(1.0, 1.0 - md / 4.0))
    else:
        md, proximity = None, None

    # ── 防守率：对手有威胁时本方是否去应对
    #   must : 对手下一手就成五（不堵立刻输）
    #   three: 对手下一手会成活四（不堵两步后被双四收）
    must_n = must_hit = 0
    three_n = three_hit = 0
    for i, (r, c, p) in enumerate(seq):
        if i < block_from:
            continue
        before = _board_of(moves, upto=i)
        opp = 2 if p == 1 else 1
        # 🚨 本方自己下一手就能成五时，不该被要求去防守 —— 直接赢就是了。
        #    不排除这种情况会把"赢家的制胜一手"算成成功防守，虚高防守率。
        #    （实测这种局面每局都会出现一次，就是最后那手）
        if _winning_now(before, p):
            continue
        must, three = _threat_points(before, opp)
        if must:
            must_n += 1
            if (r, c) in must:
                must_hit += 1
            continue      # 有必堵点时，该手只按必堵点判
        if three:
            three_n += 1
            # 堵活三 = 落在对手成活四的点上
            if (r, c) in three:
                three_hit += 1
    must_block = (must_hit / must_n) if must_n else None
    three_block = (three_hit / three_n) if three_n else None
    # 总防守率：所有"该防"的机会里防住了几成
    tot_n = must_n + three_n
    tot_hit = must_hit + three_hit
    block = (tot_hit / tot_n) if tot_n else None

    # ── 区域共享：双方包围盒重叠 / 并集
    zone = None
    blacks = [(r, c) for (r, c, p) in seq if p == 1]
    whites = [(r, c) for (r, c, p) in seq if p == 2]
    if blacks and whites:
        def bbox(ps):
            rs = [x[0] for x in ps]
            cs = [x[1] for x in ps]
            return min(rs), min(cs), max(rs), max(cs)
        b1, b2 = bbox(blacks), bbox(whites)
        ix = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]) + 1)
        iy = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]) + 1)
        inter = ix * iy
        union = ((b1[2] - b1[0] + 1) * (b1[3] - b1[1] + 1) +
                 (b2[2] - b2[0] + 1) * (b2[3] - b2[1] + 1) - inter)
        zone = inter / union if union else None

    parts = [v for v in (contact, proximity, block, zone) if v is not None]
    composite = (sum(parts) / len(parts) * 100.0) if parts else None

    return {
        'n_moves': len(moves),
        'contact': None if contact is None else round(contact, 4),
        'min_dist_median': md,
        'proximity': None if proximity is None else round(proximity, 4),
        'block': None if block is None else round(block, 4),
        'must_block': None if must_block is None else round(must_block, 4),
        'three_block': None if three_block is None else round(three_block, 4),
        'block_chances': tot_n,
        'block_hits': tot_hit,
        'must_chances': must_n,
        'three_chances': three_n,
        'zone': None if zone is None else round(zone, 4),
        'composite': None if composite is None else round(composite, 2),
    }


def aggregate(rows):
    """把多局的指标取平均（只统计有值的局）。"""
    if not rows:
        return {'n': 0}
    out = {'n': len(rows)}
    for k in ('contact', 'proximity', 'block', 'must_block', 'three_block',
              'zone', 'composite'):
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = round(sum(vals) / len(vals), 4) if vals else None
    # 机会数用求和（反映"总共该防多少次、防住多少次"）
    out['block_chances'] = sum(r.get('block_chances') or 0 for r in rows)
    out['block_hits'] = sum(r.get('block_hits') or 0 for r in rows)
    if out['block_chances']:
        out['block_pooled'] = round(out['block_hits'] / out['block_chances'], 4)
    else:
        out['block_pooled'] = None
    return out
