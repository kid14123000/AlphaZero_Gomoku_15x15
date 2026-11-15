#!/usr/bin/env python3
"""VCF / VCT 扫描器 —— 统计黑白双方在对局中"强制攻击链"的使用情况

定义（务实口径，只依赖棋谱，不需要网络/搜索）：
  造四 (FOUR)  ：落下这手后，本方存在 >=1 个成五点 ⇒ 对手被迫应对
  活三 (THREE) ：落下这手后，本方存在某个点，走到那里会形成 >=2 个成五点
                 （即下一步能做成"双点四"，等价于活三/眠三以上的进攻形）
  强制链       ：同一方连续的若干手，每手都是 FOUR 或 THREE（中间被对手应手隔开不算断）
    · VCF 链 = 链内全是 FOUR（连续冲四）
    · VCT 链 = 链内含 THREE 与 FOUR 混合（连续威胁）
    · 三链   = 链内全是 THREE

用法：
  python tools/vcf_vct_scan.py                      # 默认扫 logs/games.jsonl
  python tools/vcf_vct_scan.py --log <path> --limit 200
  python tools/vcf_vct_scan.py --json out.json --top 10
"""
import argparse
import json
import os
import statistics as st

N = 15
LAB = "ABCDEFGHIJKLMNO"
DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def _five_at(board, r, c, p):
    for dr, dc in DIRS:
        for k in range(5):
            r0, c0 = r - k * dr, c - k * dc
            ok = True
            for t in range(5):
                rr, cc = r0 + t * dr, c0 + t * dc
                if not (0 <= rr < N and 0 <= cc < N) or board.get((rr, cc)) != p:
                    ok = False
                    break
            if ok:
                return True
    return False


def win_points(board, p, limit=3, center=None, radius=6):
    """p 的成五点（可限定在 center 附近以提速）；limit 命中即停"""
    out = []
    if center is None:
        pts = [(r, c) for r in range(N) for c in range(N)]
    else:
        r0, c0 = center
        pts = [(r, c) for r in range(max(0, r0 - radius), min(N, r0 + radius + 1))
               for c in range(max(0, c0 - radius), min(N, c0 + radius + 1))]
    for r, c in pts:
        if (r, c) in board:
            continue
        board[(r, c)] = p
        hit = _five_at(board, r, c, p)
        del board[(r, c)]
        if hit:
            out.append((r, c))
            if len(out) >= limit:
                return out
    return out


def is_open_three(board, p, center, radius=5):
    """p 是否存在一手，走到那里后会有 >=2 个成五点（活三/双四前身）"""
    r0, c0 = center
    for r in range(max(0, r0 - radius), min(N, r0 + radius + 1)):
        for c in range(max(0, c0 - radius), min(N, c0 + radius + 1)):
            if (r, c) in board:
                continue
            board[(r, c)] = p
            n = len(win_points(board, p, limit=2, center=(r, c), radius=5))
            del board[(r, c)]
            if n >= 2:
                return True
    return False


def scan_game(g):
    """返回 {'black': {...}, 'white': {...}} 的链信息"""
    mv = g["moves"]
    board = {}
    seq = {1: [], 2: []}          # 每方的着法分类序列
    for i, m in enumerate(mv):
        p = 1 if i % 2 == 0 else 2
        pos = (m[0], m[1])
        board[pos] = p
        fours = win_points(board, p, limit=1, center=pos, radius=6)
        if fours:
            seq[p].append(("FOUR", pos))
        elif is_open_three(board, p, pos):
            seq[p].append(("THREE", pos))
        else:
            seq[p].append(("NORM", pos))

    out = {}
    for p, name in ((1, "black"), (2, "white")):
        best = {"len": 0, "type": "-", "start": None, "end": None}
        cur = []
        for kind, pos in seq[p]:
            if kind in ("FOUR", "THREE"):
                cur.append((kind, pos))
                if len(cur) > best["len"]:
                    kinds = {k for k, _ in cur}
                    typ = "VCF" if kinds == {"FOUR"} else ("三链" if kinds == {"THREE"} else "VCT")
                    best = {"len": len(cur), "type": typ,
                            "start": cur[0][1], "end": cur[-1][1],
                            "fours": sum(1 for k, _ in cur if k == "FOUR"),
                            "threes": sum(1 for k, _ in cur if k == "THREE")}
            else:
                cur = []
        out[name] = best
    return out


def loc(x):
    return "%s%d" % (LAB[x[1]], x[0] + 1) if x else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs/games.jsonl"))
    ap.add_argument("--limit", type=int, default=0, help="只扫最后 N 局（0=全部）")
    ap.add_argument("--json", default="")
    ap.add_argument("--top", type=int, default=5, help="列长链样例条数")
    a = ap.parse_args()

    games = [json.loads(l) for l in open(a.log) if l.strip()]
    games = [g for g in games if g.get("moves")]
    if a.limit:
        games = games[-a.limit:]
    print("扫描 %s：%d 局\n" % (os.path.basename(a.log), len(games)))

    agg = {1: dict(n=0, win=0, chain2=0, chain2_win=0, lens=[], vcf=0, vct=0, three=0),
           2: dict(n=0, win=0, chain2=0, chain2_win=0, lens=[], vcf=0, vct=0, three=0)}
    samples = []
    for g in games:
        w = g["winner"]
        r = scan_game(g)
        for p, key in ((1, "black"), (2, "white")):
            A = agg[p]
            b = r[key]
            A["n"] += 1
            won = (w == p)
            A["win"] += won
            A["lens"].append(b["len"])
            if b["len"] >= 2:
                A["chain2"] += 1
                A["chain2_win"] += won
                if b["type"] == "VCF":
                    A["vcf"] += 1
                elif b["type"] == "VCT":
                    A["vct"] += 1
                else:
                    A["three"] += 1
                samples.append((b["len"], b["type"], g["batch"], "黑" if p == 1 else "白",
                                won, loc(b["start"]), loc(b["end"])))

    print("%-4s %5s %6s %8s %10s %8s %8s %8s" %
          ("执色", "局数", "胜率", "≥2手链", "有链时胜率", "链均长", "最长", "VCF/VCT/三"))
    for p, name in ((1, "黑"), (2, "白")):
        A = agg[p]
        if not A["n"]:
            continue
        print("%-4s %5d %5.0f%% %7.0f%% %11s %8.2f %8d   %d/%d/%d" %
              (name, A["n"], 100 * A["win"] / A["n"],
               100 * A["chain2"] / A["n"],
               ("%.0f%%" % (100 * A["chain2_win"] / A["chain2"])) if A["chain2"] else "-",
               sum(A["lens"]) / len(A["lens"]), max(A["lens"]),
               A["vcf"], A["vct"], A["three"]))

    print("\n最长攻击链样例：")
    for ln, typ, b, c, won, s, e in sorted(samples, reverse=True)[:a.top]:
        print("  %-4s batch %-6s %d 手 %s 链（%s→%s）%s" %
              (c, b, ln, typ, s, e, "胜" if won else "负"))

    if a.json:
        with open(a.json, "w") as f:
            json.dump({("black" if p == 1 else "white"): {
                k: (v if k != "lens" else v) for k, v in agg[p].items()} for p in (1, 2)}, f, ensure_ascii=False, indent=1)
        print("\n明细已写 %s" % a.json)


if __name__ == "__main__":
    main()
