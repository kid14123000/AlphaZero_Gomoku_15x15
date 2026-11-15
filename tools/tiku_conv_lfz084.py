# -*- coding: utf-8 -*-
"""renju(连珠)题库 JSON → 我们的 board 走子序列

坐标：stones 形如 "J10K11I9"  字母 A-O = 列(1..15)，数字 1..15 = 行(A1 在左下)
我们的 move: (14-row)*15 + col  ⇒ A1 = (14,0) = 210，与 Board 的 row-major(顶行=0) 一致
"""
import json, os, re, glob, collections

MODE = {
    0:'封面', 32:'VCF', 64:'VCT', 67:'三手胜', 96:'自由对弈',
    160:'限珠题', 163:'三子五连', 164:'四子五连', 165:'五子五连',
    192:'选点题', 193:'选择题', 194:'找活三点', 195:'找复活三点', 196:'找眠三点',
    197:'找活四点', 198:'找冲四点', 199:'找禁手点', 200:'找三三禁点',
    201:'找四四禁点', 202:'找长连点', 203:'找防冲四抓禁点', 204:'找做V点',
    205:'找做43点', 206:'找做44点', 207:'找VCF防点', 208:'找VCF反防点',
    209:'找狭义双杀点', 210:'找狭义双防点', 211:'找两手四三胜点',
}
TOK = re.compile(r'([A-O])(1[0-5]|[1-9])')
W = H = 15

def pt(tok):
    c, r = TOK.fullmatch(tok).groups()
    return (H - int(r)) * W + (ord(c) - ord('A'))

def five_exists(grid):
    """grid: dict move->player；检查是否已有五连（题目不该有）"""
    for m, p in grid.items():
        h, w = m // W, m % W
        for dh, dw in ((0,1),(1,0),(1,1),(1,-1)):
            n = 1
            for s in (1, -1):
                hh, ww = h + dh*s, w + dw*s
                while 0 <= hh < H and 0 <= ww < W and grid.get(hh*W+ww) == p:
                    n += 1; hh += dh*s; ww += dw*s
            if n >= 5:
                return True
    return False

def norm(p, lib, idx):
    """返回 (moves, side, mode, err)"""
    mode = p.get('mode')
    size = p.get('size') or 15
    side = p.get('side')
    if size != 15:
        return None, 'size!=15'
    if side not in (1, 2):
        return None, 'side 缺失'
    if p.get('stones'):
        toks = TOK.findall(p['stones'])
        seq = [pt(a+b) for a, b in toks]
        if ''.join(a+b for a, b in toks) != p['stones']:
            return None, 'stones 有无法解析的片段'
    elif p.get('blackStones') or p.get('whiteStones'):
        bs = [pt(a+b) for a, b in TOK.findall(p.get('blackStones') or '')]
        ws = [pt(a+b) for a, b in TOK.findall(p.get('whiteStones') or '')]
        # 无手顺：黑白交替重建（黑先）
        seq = []
        for i in range(max(len(bs), len(ws))):
            if i < len(bs): seq.append(bs[i])
            if i < len(ws): seq.append(ws[i])
    else:
        return None, '无 stones'
    if len(set(seq)) != len(seq):
        return None, '有重复点'
    if len(seq) >= 225:
        return None, '棋子过多'
    # 现有五连 → 题面已结束，跳过
    g = {m: (1 if i % 2 == 0 else 2) for i, m in enumerate(seq)}
    if five_exists(g):
        return None, '题面已有五连'
    # 颜色归属：不硬性要求黑先交替，改为「末尾一手 = side 的对手」倒推
    # 题库里大量题面是摆好局面，stones 顺序未必黑先，side 才是权威
    n = len(seq)
    opp = 3 - side
    starts = opp if n % 2 == 1 else side   # 第 0 手颜色
    colors = [starts if i % 2 == 0 else 3 - starts for i in range(n)]
    g2 = {m: c for m, c in zip(seq, colors)}
    if five_exists(g2):
        return None, '按 side 倒推后题面已有五连'
    return (seq, side, mode, starts), None

def main():
    rows, errs = [], collections.Counter()
    for f in sorted(glob.glob('lfz084/*.json')):
        j = json.load(open(f))
        if not isinstance(j, dict): continue
        lib = (j.get('defaultSettings') or {}).get('title') or os.path.basename(f).replace('_puzzle.json','')
        ps = j.get('puzzles') or []
        if isinstance(ps, dict): ps = list(ps.values())
        for i, p in enumerate(ps):
            if not isinstance(p, dict): continue
            res, err = norm(p, lib, i)
            if err:
                errs[err] += 1; continue
            seq, side, mode, starts = res
            rows.append({'lib': lib, 'idx': i, 'mode': mode, 'mode_title': MODE.get(mode, '?%s'%mode),
                         'side': side, 'starts': starts, 'n_stones': len(seq), 'moves': seq,
                         'title': p.get('title'), 'comment': p.get('comment'),
                         'options': p.get('options'), 'sequence': p.get('sequence')})
    os.makedirs('out', exist_ok=True)
    with open('out/tiku_all.jsonl','w') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False)+'\n')
    print('转换成功 %d 题' % len(rows))
    print('跳过原因:', dict(errs))
    c = collections.Counter(r['mode_title'] for r in rows)
    print('\n按题型:')
    for k, v in c.most_common(): print('  %-16s %5d' % (k, v))
    c2 = collections.Counter(r['side'] for r in rows)
    print('\n先手: 黑先 %d / 白先 %d' % (c2[1], c2[2]))
    import statistics
    print('题面棋子数: 中位 %d, 范围 %d~%d' % (
        statistics.median(r['n_stones'] for r in rows),
        min(r['n_stones'] for r in rows), max(r['n_stones'] for r in rows)))
    print('带解答 sequence 的题: %d' % sum(1 for r in rows if r.get('sequence')))

main()
