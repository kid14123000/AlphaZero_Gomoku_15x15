#!/usr/bin/env python3
"""v4 课程学习控制器（分层采样版）

按白率动态设置【四类样本的目标份额】，写 sync/class_weight.json：
    0 = B_win  黑方走 & 黑赢
    1 = B_lose 黑方走 & 黑输
    2 = W_win  白方走 & 白赢   ← 崩盘时要抬的就是这一类
    3 = W_lose 白方走 & 白输

同时把 sync/handicap.json 固定为 800（不对称搜索预算已证伪，保持中性）。

可随时 kill，不影响训练进程（训练侧每次读文件 mtime，读不到就退回均匀采样）。
"""
import json
import os
import time

D = os.path.expanduser('~/AlphaZero_Gomoku_15x15_v4')
MET = os.path.join(D, 'logs/metrics.jsonl')
CW = os.path.join(D, 'sync/class_weight.json')
HC = os.path.join(D, 'sync/handicap.json')
LOG = os.path.join(D, 'logs/class_weight_log.jsonl')

WINDOW = 100
DEBOUNCE = 60
POLL = 60

# 三档份额（每档和为 1.0）；B_win 任何情况下 ≥ 0.30（黑方不能被饿死）
SHARES = {
    'collapse': {'B_win': 0.35, 'B_lose': 0.10, 'W_win': 0.22, 'W_lose': 0.33},
    'mid':      {'B_win': 0.40, 'B_lose': 0.09, 'W_win': 0.16, 'W_lose': 0.35},
    'healthy':  {'B_win': 0.44, 'B_lose': 0.08, 'W_win': 0.10, 'W_lose': 0.38},
}
LO, HI = 0.20, 0.35


def last_games(n):
    rows = []
    try:
        with open(MET) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:                 # noqa: BLE001
                    continue
                if d.get('type') == 'game':
                    rows.append(d)
    except FileNotFoundError:
        return []
    return rows[-n:]


def load_state():
    if os.path.exists(LOG):
        try:
            with open(LOG) as f:
                lines = [l for l in f if l.strip()]
            if lines:
                return json.loads(lines[-1])
        except Exception:                         # noqa: BLE001
            pass
    return {'tier': 'collapse', 'batch': 0, 'white_rate': None}


def atomic_write(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=0)
    os.replace(tmp, path)


def main():
    # handicap 固定中性（课程学习只用分层采样）
    os.makedirs(os.path.dirname(HC), exist_ok=True)
    atomic_write(HC, {'black_sims': 800, 'batch': 0, 'level': 0, 'note': 'disabled'})

    st = load_state()
    tier = st.get('tier', 'collapse')
    atomic_write(CW, SHARES[tier])
    print('[ctl] 启动 tier=%s shares=%s' % (tier, SHARES[tier]), flush=True)

    while True:
        g = last_games(WINDOW)
        if len(g) >= 50:
            batch = max(int(x.get('batch') or 0) for x in g)
            wr = sum(1 for x in g if x.get('winner') == 2) / len(g)
            want = 'collapse' if wr < LO else ('healthy' if wr > HI else 'mid')
            since = batch - int(st.get('batch') or 0)
            if want != tier and since >= DEBOUNCE:
                tier = want
                st = {'tier': tier, 'batch': batch, 'white_rate': wr,
                      'ts': time.time(), 'shares': SHARES[tier]}
                atomic_write(CW, SHARES[tier])
                with open(LOG, 'a') as f:
                    f.write(json.dumps(st, ensure_ascii=False) + '\n')
                print('[ctl] 白率 %.2f → tier=%s shares=%s' % (wr, tier, SHARES[tier]), flush=True)
            elif wr != st.get('white_rate') and since >= DEBOUNCE:
                st['white_rate'] = wr
                st['batch'] = batch
                with open(LOG, 'a') as f:
                    f.write(json.dumps(st, ensure_ascii=False) + '\n')
        time.sleep(POLL)


if __name__ == '__main__':
    main()
