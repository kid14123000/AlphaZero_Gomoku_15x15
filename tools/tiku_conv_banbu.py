# -*- coding: utf-8 -*-
"""banbu-gomoku(MIT) 题库 → 统一 jsonl

两种格式：
  A) 开宝数组: [["J10,1","G9,2",...], ...]   "col+row,color"，1=黑 2=白（显式颜色）
  B) native 对象: {title,source,license,puzzles:[{blackStones,whiteStones,side,comment,...}]}

输出字段: lib / idx / side(该谁走) / starts(第0手的颜色) / n_stones / moves
🚨 starts 必须导出：moves[0] 不一定是黑子。灌棋盘时要
   board.init_board(start_player=starts-1)，否则轮次会错（实测 419 题踩过）。
"""
import json, glob, os, re, collections

W = H = 15
TOK = re.compile(r'^([A-Za-z]{1,2})(\d{1,2})\s*(?:,\s*([12]))?$')


def pt(coord):
    m = TOK.match(coord.strip())
    if not m:
        return None, None
    c, r, col = m.groups()
    ci = ord(c.upper()) - ord('A')
    ri = int(r)
    if not (0 <= ci < W and 1 <= ri <= H):
        return None, None
    return (H - ri) * W + ci, (int(col) if col else None)


def chunks(v):
    return re.findall(r'[A-Za-z]{1,2}\d{1,2}(?:,[12])?', v) if isinstance(v, str) else []


def finish(seq, cols, side):
    """统一收尾：补颜色、算 side 与 starts、校验"""
    if not seq:
        return None, '空题'
    if len(set(seq)) != len(seq):
        return None, '重复点'
    if any(c is None for c in cols):
        cols = [1 if i % 2 == 0 else 2 for i in range(len(seq))]
    nb, nw = cols.count(1), cols.count(2)
    if abs(nb - nw) > 1:
        return None, '黑白数不合法(%d/%d)' % (nb, nw)
    infer = 1 if nb == nw else (1 if nw > nb else 2)      # 少的一方走
    if side is None:
        side = infer
    elif side != infer:
        return None, 'side 与棋子数矛盾'
    n = len(seq)
    starts = (3 - side) if n % 2 else side                # 第0手颜色
    return (seq, side, starts), None


def norm_side(v):
    if isinstance(v, str):
        return {'black': 1, 'white': 2}.get(v.lower(), None)
    return v if v in (1, 2) else None


def parse_entry(entry):
    if isinstance(entry, dict):
        if entry.get('blackStones') or entry.get('whiteStones'):
            seq, cols = [], []
            for k, col in (('blackStones', 1), ('whiteStones', 2)):
                for ch in chunks(entry.get(k) or ''):
                    m, _ = pt(ch)
                    if m is not None:
                        seq.append(m)
                        cols.append(col)
            return finish(seq, cols, norm_side(entry.get('side')))
        raw = entry.get('stones') or []
        if isinstance(raw, str):
            raw = [raw]
        seq, cols = [], []
        for v in raw:
            for ch in chunks(v):
                m, c = pt(ch)
                if m is not None:
                    seq.append(m)
                    cols.append(c)
        return finish(seq, cols, norm_side(entry.get('side')))

    if isinstance(entry, list):
        seq, cols = [], []
        for v in entry:
            for ch in chunks(v):
                m, c = pt(ch)
                if m is not None:
                    seq.append(m)
                    cols.append(c)
        return finish(seq, cols, None)

    return None, '未知条目类型'


def main():
    rows, errs = [], collections.Counter()
    for f in sorted(glob.glob('banbu/*.json')):
        lib = os.path.basename(f).replace('.json', '')
        j = json.load(open(f))
        entries = j['puzzles'] if isinstance(j, dict) and 'puzzles' in j else j
        if isinstance(entries, dict):
            entries = list(entries.values())
        for i, e in enumerate(entries):
            res, err = parse_entry(e)
            if err:
                errs[err] += 1
                continue
            seq, side, starts = res
            rows.append({'lib': lib, 'idx': i, 'side': side, 'starts': starts,
                         'moves': seq, 'n_stones': len(seq)})
    os.makedirs('out', exist_ok=True)
    with open('out/banbu_all.jsonl', 'w') as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('转换成功 %d 题' % len(rows), ' 跳过:', dict(errs))
    c = collections.Counter(r['lib'] for r in rows)
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print('  %-26s %5d' % (k, v))
    print('先手: 黑 %d / 白 %d' % (sum(1 for r in rows if r['side'] == 1),
                                sum(1 for r in rows if r['side'] == 2)))
    print('首手为白: %d 题' % sum(1 for r in rows if r['starts'] == 2))


main()
