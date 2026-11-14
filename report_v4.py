#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""9x9/5连 训练半小时汇报（在 M4 上运行）

战斗激烈度直接复用项目自带的 web/combat.py（与 web 监控看板同口径）：
  contact 接触率 / proximity 贴身度 / block 防守率 / zone 区域共享
  composite = 四项等权平均 × 100
"""
import glob
import json
import os
import shutil
import statistics as st
import subprocess
import sys
import time

D = os.path.expanduser("~/AlphaZero_Gomoku_15x15_v4")
M = os.path.join(D, "logs/metrics.jsonl")
G = os.path.join(D, "logs/games.jsonl")
N = 15
ROW = 5
GEN = 40             # publish_every = 40 局/代
CENT = range(5, 10)  # 15×15 中心 5x5

sys.path.insert(0, os.path.join(D, "web"))
try:
    import combat
    combat.set_board(N, ROW)     # 🚨 必须先设板宽，否则 9×9 边缘活四误判
    HAVE_COMBAT = True
except Exception as e:           # noqa: BLE001
    HAVE_COMBAT = False
    COMBAT_ERR = repr(e)


def load_metrics():
    rows = []
    for line in open(M):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def load_games(cap=900):
    recs = []
    try:
        with open(G) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("moves"):
                    recs.append(d)
    except FileNotFoundError:
        return []
    return recs[-cap:]


def alive():
    try:
        pid = subprocess.run(["pgrep", "-f", "parallel_train.py"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        if not pid:
            return False, ""
        full = subprocess.run(["ps", "-o", "command=", "-p", pid[0]],
                              capture_output=True, text=True, timeout=10).stdout.strip()
        return True, full
    except Exception:
        return False, ""


def center_ratio(games, cap=400):
    hit = tot = 0
    for g in games[-cap:]:
        for m in (g.get("moves") or [])[:8]:
            r, c = m[0], m[1]
            tot += 1
            if r in CENT and c in CENT:
                hit += 1
    return 100.0 * hit / tot if tot else 0.0


def _has5threat(board, p, N=15):
    """p 方下一手能否成五（即 p 现在有活三/冲四/活四级威胁）"""
    for r in range(N):
        for c in range(N):
            if (r, c) in board:
                continue
            board[(r, c)] = p
            ok = False
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                for k in range(5):
                    r0, c0 = r - k * dr, c - k * dc
                    if all(0 <= r0 + t * dr < N and 0 <= c0 + t * dc < N and
                           board.get((r0 + t * dr, c0 + t * dc)) == p for t in range(5)):
                        ok = True
                        break
                if ok:
                    break
            del board[(r, c)]
            if ok:
                return True
    return False


def white_aggr(games, cap=100):
    """白方"敢不敢下"：主动造威胁 / 单纯堵 占白方手数比例
    "敢下"越高越健康；崩盘期特征是 造威胁↓ + 单纯堵↑"""
    gs = [g for g in games[-cap:] if g.get("moves")]
    wm = atk = blk = 0
    for g in gs:
        b = {}
        for i, m in enumerate(g["moves"]):
            r, c = m
            mover = 1 if i % 2 == 0 else 2
            if mover == 2 and i >= 6:
                wm += 1
                b[(r, c)] = 2
                created = _has5threat(dict(b), 2)
                del b[(r, c)]
                if created:
                    atk += 1
                elif _has5threat(dict(b), 1):
                    blk += 1
            b[(r, c)] = mover
    if not wm:
        return None
    return len(gs), 100.0 * atk / wm, 100.0 * blk / wm


def combat_block(games, cap):
    if not games or not HAVE_COMBAT:
        return None, 0
    rows = [combat.analyze(g["moves"]) for g in games[-cap:]]
    agg = combat.aggregate(rows)
    agg["n_games"] = len(rows)
    return agg, len(rows)


def seg_trend(vals):
    if len(vals) < 6:
        return ""
    k = max(2, len(vals) // 3)
    a, b = st.mean(vals[:k]), st.mean(vals[-k:])
    d = b - a
    arrow = "↑" if d > abs(a) * 0.02 else ("↓" if d < -abs(a) * 0.02 else "→")
    return "%s%+.2f" % (arrow, d)


def main():
    rows = load_metrics()
    games = [r for r in rows if r.get("type") == "game"]
    ups = [r for r in rows if r.get("type") == "update"]
    if not games:
        print("9x9 训练汇报：日志里还没有对局记录")
        return
    batch = games[-1].get("batch")
    gen = batch // GEN
    ok, cmd = alive()
    wk = cmd.split("--workers")[1].split()[0] if (ok and "--workers" in cmd) else "?"
    vt = "q" if "value_target q" in cmd else ("mix" if "value_target mix" in cmd else "±1")

    print("⏱ 15×15 v4 汇报 %s" % time.strftime("%H:%M"))
    print("状态 %s%s | batch %d（第 %d 代）| 本run %.1fh | value目标 %s"
          % ("运行中" if ok else "已停止", (" | %s workers" % wk) if ok else "",
             batch, gen, games[-1]["elapsed"] / 3600.0, vt))

    lo = max(0, batch - 10 * GEN)
    recent = [g for g in games if g.get("batch", 0) >= lo] or games[-200:]
    L = [g["episode_len"] for g in recent]
    n_recent = len(recent)
    bl = sum(1 for g in recent if g.get("winner") == 1)
    wh = sum(1 for g in recent if g.get("winner") == 2)
    dr = n_recent - bl - wh
    print("近10代(%d→%d, %d局) 手数 %.1f（中位 %.0f）| >25手 %.0f%%"
          % (lo, batch, n_recent, st.mean(L), st.median(L),
             100 * sum(1 for x in L if x > 25) / n_recent))
    print("   胜率分布 黑 %.2f / 白 %.2f / 和 %.2f"
          % (bl / n_recent, wh / n_recent, dr / n_recent))
    prev = [g for g in games if lo - GEN <= g.get("batch", 0) < lo]
    if prev:
        npv = len(prev)
        pb = sum(1 for g in prev if g.get("winner") == 1)
        pw = sum(1 for g in prev if g.get("winner") == 2)
        print("   ↳ 上一代：手数 %.1f→%.1f | 黑 %.2f→%.2f | 白 %.2f→%.2f | 和 %.2f→%.2f"
              % (st.mean([g["episode_len"] for g in prev]), st.mean(L),
                 pb / npv, bl / n_recent, pw / npv, wh / n_recent,
                 (npv - pb - pw) / npv, dr / n_recent))

    gs = load_games()
    if gs and HAVE_COMBAT:
        cur, n = combat_block(gs, 300)
        old, _ = combat_block(gs[:-300], 300) if len(gs) > 400 else (None, 0)
        print("战斗激烈度(近%d局) 综合 %.1f | 接触 %.2f 贴身 %.2f 防守 %.2f 区域 %.2f"
              % (n, cur["composite"] or 0, cur["contact"] or 0, cur["proximity"] or 0,
                 cur["block"] or 0, cur["zone"] or 0))
        if old and old.get("composite") is not None and cur.get("composite") is not None:
            print("   ↳ 前300局综合 %.1f → 现在 %.1f（%+.1f）| 必堵点 %d 次/堵中 %d"
                  % (old["composite"], cur["composite"], cur["composite"] - old["composite"],
                     cur.get("block_chances") or 0, cur.get("block_hits") or 0))
        print("中心布局(前8手, 近%d局) 中心5x5 %.0f%%" % (min(400, len(gs)), center_ratio(gs, 400)))
        wa = white_aggr(gs, 100)
        if wa:
            print("白方攻守(近%d局) 主动造威胁 %.1f%% | 单纯堵 %.1f%%   [健康≈21%% / 24%%; 崩盘≈18%% / 32%%]" % wa)
    elif not HAVE_COMBAT:
        print("战斗激烈度：无法 import web/combat.py（%s）" % COMBAT_ERR)

    u = [x for x in ups if x.get("batch", 0) >= lo] or ups[-40:]
    if u:
        losses = [x.get("loss", 0) for x in u]
        evs = [x.get("ev_new", 0) for x in u]
        print("loss %.3f %s | entropy %.2f | KL %.4f | lr %.2f"
              % (st.mean(losses), seg_trend(losses), st.mean([x.get("entropy", 0) for x in u]),
                 st.mean([x.get("kl", 0) for x in u]), u[-1].get("lr_multiplier", 0)))
        print("value(ev_new) %.3f %s | 缓冲 %s"
              % (st.mean(evs), seg_trend(evs), u[-1].get("buffer", "?")))
    seq = []
    for k in range(max(0, gen - 5), gen + 1):
        blk = [x for x in ups if k * GEN <= x.get("batch", 0) < (k + 1) * GEN]
        if blk:
            seq.append("%d代 %.2f/%.2f" % (k, st.mean([x.get("loss", 0) for x in blk]),
                                           st.mean([x.get("ev_new", 0) for x in blk])))
    if seq:
        print("逐代(loss/ev) " + " | ".join(seq))
    # ── 布局/开局方向（只读统计，不碰训练）──
    def _layout(gs, cap=300):
        seg = [g for g in gs[-cap:] if g.get("moves")]
        if len(seg) < 30:
            return None
        firsts = [(g["moves"][0][0], g["moves"][0][1]) for g in seg]
        cf = sum(1 for r, c in firsts if 5 <= r <= 9 and 5 <= c <= 9) / len(firsts)
        # 经验熵（首手分布）
        from collections import Counter
        cnt = Counter(firsts)
        ent = -sum((v / len(firsts)) * __import__("math").log(v / len(firsts)) for v in cnt.values())
        c8 = []
        for g in seg:
            mv = g["moves"][:8]
            if mv:
                c8.append(sum(1 for r, c in mv if 5 <= r <= 9 and 5 <= c <= 9) / len(mv))
        return cf, (st.mean(c8) if c8 else 0), ent, len(cnt)

    lay = _layout(gs, 300)
    if lay:
        cf, c8, ent, npos = lay
        print("布局(近300局) 首手中心5x5 %.0f%% | 前8手中心 %.0f%% | 首手熵 %.2f bit | 首手选点 %d 种"
              % (100 * cf, 100 * c8, ent, npos))

    # ── 近 8 小时 vs 上一个 8 小时 ──
    now = time.time()
    cur_w = [g for g in games if now - g.get("t", 0) <= 8 * 3600]
    prev_w = [g for g in games if 8 * 3600 < now - g.get("t", 0) <= 16 * 3600]
    if len(cur_w) > 20 and len(prev_w) > 20:
        def _st(ws):
            L = [x["episode_len"] for x in ws]
            b = sum(1 for x in ws if x.get("winner") == 1) / len(ws)
            return st.mean(L), b
        pl, pb = _st(prev_w)
        cl, cb = _st(cur_w)
        print("近8h(%d局) vs 上8h(%d局): 手数 %.1f→%.1f (%+.1f) | 黑 %.2f→%.2f (%+.2f)"
              % (len(cur_w), len(prev_w), pl, cl, cl - pl, pb, cb, cb - pb))
        uc = [x for x in ups if now - x.get("t", 0) <= 8 * 3600]
        up = [x for x in ups if 8 * 3600 < now - x.get("t", 0) <= 16 * 3600]
        if uc and up:
            lc = st.mean([x.get("loss", 0) for x in uc]); lp = st.mean([x.get("loss", 0) for x in up])
            ec = st.mean([x.get("entropy", 0) for x in uc]); ep = st.mean([x.get("entropy", 0) for x in up])
            print("   ↳ loss %.2f→%.2f (%+.2f) | entropy %.2f→%.2f (%+.2f)"
                  % (lp, lc, lc - lp, ep, ec, ec - ep))

    # ── 短局占比 ──
    l100 = games[-100:]
    if len(l100) >= 30:
        s20 = sum(1 for x in l100 if x["episode_len"] <= 20) / len(l100)
        s25 = sum(1 for x in l100 if x["episode_len"] <= 25) / len(l100)
        print("短局占比(近%d局) ≤20手 %.0f%% | ≤25手 %.0f%%" % (len(l100), 100 * s20, 100 * s25))

    # ── 检查点 / 磁盘 ──
    cks = glob.glob(os.path.join(D, "checkpoints", "*.model"))
    du = shutil.disk_usage(os.path.expanduser("~"))
    print("检查点 %d 个(~%.1f GB) | 磁盘剩 %.1f GB"
          % (len(cks), len(cks) * 9.6 / 1024, du.free / 1e9))

    print("注：日志只有 total loss，policy/value 未拆分；value 侧用 ev_new 代理")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("9x9 汇报出错: %r" % (e,))
        sys.exit(0)
