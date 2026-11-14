# -*- coding: utf-8 -*-
"""
推理引擎：给 web/ 三个可视化页面（对弈 / MCTS 搜索 / 网络前向）共用。

设计原则
--------
* 不复制训练代码里的任何逻辑 —— Board / MCTS / PolicyValueNet 全部直接 import，
  页面看到的就是训练时用的那份实现，不会出现"看板和训练不一致"。
* 网络张量 ↔ 棋盘的坐标换算集中在本文件（`to_grid`），前端只管画。

🚨 坐标口径（上游自带的坑，必须记住）
    game.Board.current_state() 末尾有 `square_state[:, ::-1, :]`
        → 张量第 0 行 = 棋盘第 14 行
    本文件的 `to_grid()` 统一翻回来，所以**返回给前端的 grid 一律是
    grid[r*15 + c]，r=0 在屏幕最上方**，与 policy 的 move 编号 r*15+c 同口径。
"""

import os
import sys
import copy
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from game import Board                                    # noqa: E402
from policy_value_net_pytorch import PolicyValueNet        # noqa: E402
from mcts_alphaZero import MCTS, MCTSPlayer, softmax, TreeNode   # noqa: E402

WIDTH = 15
HEIGHT = 15
N_IN_ROW = 5
BLACK, WHITE = 1, 2


def configure(width=None, height=None, n_in_row=None):
    """按训练日志的 start 记录同步棋盘规格。

    9×9/四连 与 15×15/五连 共用同一套看板：尺寸绝不能写死，
    否则换盘后 /play /mcts /network /replay 全按 15×15 解析，棋位全错。
    """
    global WIDTH, HEIGHT, N_IN_ROW
    if width:
        WIDTH = int(width)
    if height:
        HEIGHT = int(height)
    if n_in_row:
        N_IN_ROW = int(n_in_row)


def spec():
    """当前棋盘规格，给前端渲染用"""
    return {'width': WIDTH, 'height': HEIGHT, 'n_in_row': N_IN_ROW}

_lock = threading.Lock()
_net_cache = {}      # path -> (mtime, PolicyValueNet)


# ─────────────────────────────────────────────────────────────
# 棋盘
# ─────────────────────────────────────────────────────────────
def make_board(stones=None, width=None, height=None, n_in_row=None):
    """从 [{r,c}] 构造棋盘。颜色按落子顺序交替（黑先），与网页摆谱口径一致。

    ⚠️ 必须强制交替：current_state() 的 plane3（执子方常量）用的是
       `len(self.states) % 2 == 0`，若棋子黑白数不匹配，网络输入就自相矛盾。
    """
    width = WIDTH if width is None else int(width)
    height = HEIGHT if height is None else int(height)
    n_in_row = N_IN_ROW if n_in_row is None else int(n_in_row)
    b = Board(width=width, height=height, n_in_row=n_in_row)
    b.init_board(0)
    b.states = {}
    b.availables = list(range(width * height))
    for i, s in enumerate(stones or []):
        r, c = int(s['r']), int(s['c'])
        if not (0 <= r < height and 0 <= c < width):
            continue
        m = r * width + c
        if m not in b.availables:
            continue
        b.states[m] = BLACK if i % 2 == 0 else WHITE
        b.availables.remove(m)
    n = len(b.states)
    b.current_player = BLACK if n % 2 == 0 else WHITE
    b.last_move = _last_move_of(stones) if stones else -1
    return b


def _last_move_of(stones):
    if not stones:
        return -1
    s = stones[-1]
    return int(s['r']) * WIDTH + int(s['c'])


def board_to_stones(board):
    """按手数（落子顺序）返回棋子列表 —— 顺序在 dict 里靠 insertion order 保住。"""
    out = []
    for i, (m, p) in enumerate(board.states.items()):
        out.append({'r': int(m) // board.width, 'c': int(m) % board.width,
                    'p': int(p), 'n': i + 1})
    return out


def to_grid(arr2d):
    """张量平面 → 前端网格（翻回上下，见文件头口径说明）。返回 225 长度的 list。"""
    a = np.asarray(arr2d, dtype=np.float64)
    return [round(float(x), 6) for x in np.flipud(a).ravel()]


# ─────────────────────────────────────────────────────────────
# 权重
# ─────────────────────────────────────────────────────────────
def list_models():
    """列出可用于 15x15 的权重（自动跳过仓库自带 6x6/8x8）。"""
    out = []
    ck = os.path.join(ROOT, 'checkpoints')
    cands = []
    if os.path.isdir(ck):
        for f in os.listdir(ck):
            if f.startswith('ckpt_it') and f.endswith('.model'):
                cands.append(os.path.join(ck, f))
    for f in ('current_policy.model', 'best_policy.model'):
        p = os.path.join(ROOT, f)
        if os.path.exists(p):
            cands.append(p)

    def it_of(path):
        base = os.path.basename(path)
        if base.startswith('ckpt_it'):
            try:
                return int(base[len('ckpt_it'):-len('.model')])
            except ValueError:
                return -1
        return 1 if base == 'best_policy.model' else 10 ** 9  # current 排最后

    for p in sorted(cands, key=it_of):
        base = os.path.basename(p)
        if base.startswith('ckpt_it'):
            label = '第 %d 代' % it_of(p)
        elif base == 'best_policy.model':
            label = 'best（历史最佳）'
        else:
            label = 'current（最新，可能正在写）'
        try:
            st = os.stat(p)
        except OSError:
            continue
        out.append({'id': os.path.relpath(p, ROOT), 'label': label,
                    'size': st.st_size,
                    'mtime': int(st.st_mtime)})
    out.reverse()          # 最新在前
    return out


def default_model():
    ms = list_models()
    for m in ms:
        if m['id'].endswith('ckpt_it') or 'ckpt_it' in m['id']:
            return m['id']
    return ms[0]['id'] if ms else None


def get_net(model_file=None):
    """按 mtime 缓存：训练在跑、权重被覆盖时会自动重载（不用重启 web）。"""
    if not model_file:
        model_file = default_model()
    if not model_file:
        raise RuntimeError('找不到任何 15x15 权重（checkpoints/ 与 current_policy.model 都不存在）')
    path = model_file if os.path.isabs(model_file) else os.path.join(ROOT, model_file)
    if not os.path.exists(path):
        raise RuntimeError('权重不存在: %s' % model_file)
    mtime = os.stat(path).st_mtime
    with _lock:
        hit = _net_cache.get(path)
        if hit and hit[0] == mtime:
            return hit[1]
        net = PolicyValueNet(WIDTH, HEIGHT, model_file=path)
        net.policy_value_net.eval()
        _net_cache[path] = (mtime, net)
        return net


# ─────────────────────────────────────────────────────────────
# 干净前向（value / policy），不算 MCTS
# ─────────────────────────────────────────────────────────────
def clean_eval(net, board):
    """返回 (prob 网格 225, value, 合法 mask 225)。

    value 是「当前该走的一方」的期望胜率 ∈ [-1,1]。
    prob 已按合法落点归一化（上游 policy_value_fn 没有归一化）。
    """
    import torch
    pvn = net.policy_value_net
    state = np.ascontiguousarray(
        board.current_state().reshape(-1, 4, WIDTH, HEIGHT))
    with torch.no_grad():
        logp, v = pvn(torch.from_numpy(state).float())
    probs = np.exp(logp.numpy().ravel())
    legal = np.zeros(WIDTH * HEIGHT, dtype=bool)
    legal[board.availables] = True
    probs = np.where(legal, probs, 0.0)
    s = probs.sum()
    if s > 0:
        probs = probs / s
    return probs, float(v.numpy().ravel()[0]), legal


# ─────────────────────────────────────────────────────────────
# 棋谱逐步数值
# ─────────────────────────────────────────────────────────────
def game_value_series(moves, model_file=None, winner=None):
    """给一局棋谱的每一步算数值，供回放页显示「这一步双方胜率多少」。

    对每个局面在该步落子**之前**做一次干净前向（不走搜索、不加噪声）：
      - value   该走方视角的期望胜率 ∈ [-1,1]
      - prior   网络给这步实际落点的概率
      - net_best 网络自己会走哪一步（用于对比「AI 想下」和「实际下了」）

    黑胜率换算：黑该走时 = 50+50v，白该走时 = 50-50v（v 是执子方视角）。

    🚨 手数口径（n = 盘面上的子数，= 已下完的手数）：
      第 i 个点评估的是「盘面 i 子」的局面，所以 n 必须写 i，**不能写 i+1**。
      旧实现写 i+1，导致曲线上每个点都往右错一位：空盘被标成「第1手」、
      29 子被标成「第30手」、真正的 30 子终局盘面根本没进曲线；
      而回放页光标停在 n 手时盘面是 n 子 ⇒ 卡片拿到的估值总是差一手。
      现在 n == 子数 == 回放页光标 == 网络前向页的「N 手」，三处同口径。

    若传入 winner（1 黑 / 2 白 / -1 和），末尾追加一个 terminal 点：
      n = 总手数，black_pct = 真实结果（0/100），
      net_black_pct = 网络对**终局盘面**的估值 —— 白已连五它还给黑 69% 的那种
      盲区，靠这个字段暴露出来，不再被真实结果盖住。

    V24Net 在 GPU 上每步约 1ms，一局 40 手总共 ~40ms，可以随详情接口一起返回。
    """
    try:
        net = get_net(model_file)
    except Exception:                       # noqa: BLE001  没有权重时优雅降级
        return []
    b = make_board()
    out = []
    for i, mv in enumerate(moves or []):
        try:
            r, c = int(mv[0]), int(mv[1])
        except (TypeError, IndexError, ValueError):
            break
        if not (0 <= r < HEIGHT and 0 <= c < WIDTH):
            break
        m = r * WIDTH + c
        if m not in b.availables:
            break
        cur = b.current_player
        try:
            probs, val, legal = clean_eval(net, b)
        except Exception:                   # noqa: BLE001
            break
        black = 50.0 + 50.0 * val if cur == BLACK else 50.0 - 50.0 * val
        prior = float(probs[m]) if m < len(probs) else None
        best = None
        if legal.any():
            masked = np.where(legal, probs, -1.0)
            bi = int(np.argmax(masked))
            best = [bi // WIDTH, bi % WIDTH]
        out.append({
            'n': i,                 # ← 盘面 i 子（不是 i+1）
            'r': r, 'c': c,
            'player': int(cur),
            'value': round(float(val), 4),
            'black_pct': round(black, 2),
            'white_pct': round(100.0 - black, 2),
            'prior': None if prior is None else round(prior, 4),
            'net_best': best,
            'is_best': bool(best is not None and best[0] == r and best[1] == c),
        })
        b.do_move(m)
    # 终局真相点：n = 总手数（盘面满子），真实结果 + 网络对终局盘面的估值
    if winner in (1, 2, -1) and out:
        bp = 100.0 if winner == 1 else (0.0 if winner == 2 else 50.0)
        net_bp = None
        try:
            cur = b.current_player
            _p, val, _l = clean_eval(net, b)
            net_bp = round(50.0 + 50.0 * val if cur == BLACK else 50.0 - 50.0 * val, 2)
        except Exception:                   # noqa: BLE001
            net_bp = None
        out.append({
            'n': len(out),          # 走完所有手 ⇒ 子数 == len(out)
            'r': None, 'c': None,
            'player': None,
            'value': None,
            'black_pct': bp,
            'white_pct': 100.0 - bp,
            'net_black_pct': net_bp,
            'prior': None,
            'net_best': None,
            'is_best': False,
            'terminal': True,
        })
    return out


def game_q_series(moves, model_file=None, sims=100, c_puct=5.0, winner=None):
    """用 MCTS 搜索价值重算整局逐步 Q —— 和训练时 value 标签同口径。

    与 game_value_series 的区别：
      game_value_series  单次前向（干净、快，~1ms/步）—— 看网络「直觉」
      game_q_series      每步跑 MCTS 再取根 Q（慢，9×9 上 100 sims ≈ 0.25s/步）
                         —— 看「网络 + 搜索」合起来的判断力

    用途：同一局棋谱，换最新权重重算一遍，比较「胜负分野」出现在第几手。
    分野提前 ⇒ 网络更早看出走向；分野推后或消失 ⇒ 变钝了。

    返回结构与 game_value_series 一致，另加 'source': 'mcts'。
    """
    try:
        net = get_net(model_file)
    except Exception:                       # noqa: BLE001
        return []
    b = make_board()
    player = MCTSPlayer(net.policy_value_fn, c_puct=c_puct,
                        n_playout=max(1, int(sims)), is_selfplay=1)
    player.mcts.update_with_move(-1)

    out = []
    for i, mv in enumerate(moves or []):
        try:
            r, c = int(mv[0]), int(mv[1])
        except (TypeError, IndexError, ValueError):
            break
        if not (0 <= r < HEIGHT and 0 <= c < WIDTH):
            break
        m = r * WIDTH + c
        if m not in b.availables:
            break
        cur = b.current_player
        try:
            player.mcts.get_move_probs(b, temp=1.0)
        except Exception:                   # noqa: BLE001
            break
        q = float(player.mcts.last_root_q)   # 已在 get_move_probs 里取负，= 该走方视角
        black = 50.0 + 50.0 * q if cur == BLACK else 50.0 - 50.0 * q
        # 网络给这步的先验概率，方便同屏对照
        try:
            probs, _v, legal = clean_eval(net, b)
            prior = float(probs[m]) if m < len(probs) else None
        except Exception:                   # noqa: BLE001
            prior = None
        out.append({
            'n': i, 'r': r, 'c': c, 'player': int(cur),
            'value': round(q, 4),
            'black_pct': round(black, 2),
            'white_pct': round(100.0 - black, 2),
            'prior': None if prior is None else round(prior, 4),
            'net_best': None, 'is_best': False,
            'source': 'mcts', 'sims': int(sims),
        })
        b.do_move(m)
        player.mcts.update_with_move(m)

    if winner in (1, 2, -1) and out:
        bp = 100.0 if winner == 1 else (0.0 if winner == 2 else 50.0)
        out.append({
            'n': len(out), 'r': None, 'c': None, 'player': None,
            'value': None, 'black_pct': bp, 'white_pct': 100.0 - bp,
            'prior': None, 'net_best': None, 'is_best': False,
            'terminal': True, 'source': 'mcts',
        })
    return out


def find_divide(values, winner, thresh=70.0):
    """胜负分野：胜率首次明确站到最终赢家一侧、且此后不再翻回对面的手数。

    判据（对黑胜）：black_pct ≥ thresh，并且从该手到终局（不含 terminal）
    再没有跌回 100-thresh 以下。
    返回 {'n': 手数 or None, 'pct': 当时胜率, 'side': 'black'|'white'}。

    为什么要求「此后不再翻回」：单点的尖峰不算分野，网络可能只是抖动。
    """
    seq = [v for v in (values or []) if not v.get('terminal') and
           v.get('black_pct') is not None]
    if not seq or winner not in (1, 2):
        return {'n': None, 'pct': None, 'side': None}
    lo, hi = 100.0 - thresh, thresh
    for i, v in enumerate(seq):
        bp = v['black_pct']
        if winner == 1 and bp >= hi:
            if all(x['black_pct'] >= lo for x in seq[i:]):
                return {'n': v['n'], 'pct': bp, 'side': 'black'}
        elif winner == 2 and bp <= lo:
            if all(x['black_pct'] <= hi for x in seq[i:]):
                return {'n': v['n'], 'pct': bp, 'side': 'white'}
    return {'n': None, 'pct': None, 'side': None}


def value_map(net, board):
    """价值热力图：在每个空点替当前方落一子，返回「落子后对走子方的期望胜率」。

    批量一次前向（225 个局面），15x15 小网在 CPU 上 ~0.2s，可以每步都算。
    """
    import torch
    if not board.availables:
        return None
    moves = list(board.availables)
    states = []
    for m in moves:
        b = copy.deepcopy(board)
        b.do_move(m)
        states.append(b.current_state())
    arr = np.ascontiguousarray(np.array(states, dtype=np.float32))
    pvn = net.policy_value_net
    with torch.no_grad():
        _, v = pvn(torch.from_numpy(arr))
    v = v.numpy().ravel()
    # 落子后轮到对手 → 对手视角的 v，取负才是走子方的收益
    vals = -v
    grid = [None] * (WIDTH * HEIGHT)
    for m, x in zip(moves, vals):
        r, c = m // WIDTH, m % WIDTH
        grid[r * WIDTH + c] = round(float(x), 4)
    return grid


# ─────────────────────────────────────────────────────────────
# MCTS 搜索（带路径追踪）
# ─────────────────────────────────────────────────────────────
class TracedMCTS(MCTS):
    """在原版 MCTS 上只加"记录每次模拟走了哪条路"，不改任何搜索逻辑。"""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=100):
        super(TracedMCTS, self).__init__(policy_value_fn, c_puct, n_playout)
        self.trace = []

    def reset(self):
        """清空搜索树与追踪记录（原版靠 update_with_move(-1) 做同样的事）。"""
        self._root = TreeNode(None, 1.0)
        self.trace = []

    def playout_traced(self, state):
        """复制原版 _playout，额外记录路径与被选子节点的 (visits, Q, P)。"""
        node = self._root
        path = []
        while not node.is_leaf():
            action, node = node.select(self._c_puct)
            state.do_move(action)
            path.append({'action': int(action)})

        action_probs, leaf_value = self._policy(state)
        # 🚨 policy_value_fn 返回的是 zip 生成器（只能迭代一次）：
        #    必须先物化成 list，否则 expand 用掉之后就再也拿不到先验了（实测 leaf['legal']=0）
        action_probs = [(int(a), float(p)) for a, p in action_probs]
        leaf_value = float(leaf_value)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
            leaf_prior = sorted(action_probs, key=lambda x: -x[1])
            leaf_prior = [(a, round(p, 6)) for a, p in leaf_prior]
            leaf_info = {'terminal': False,
                         'value': round(leaf_value, 4),
                         'legal': len(leaf_prior),
                         'prior_top': leaf_prior[:5]}
        else:
            leaf_value = 0.0 if winner == -1 else (
                1.0 if winner == state.get_current_player() else -1.0)
            leaf_info = {'terminal': True, 'value': round(leaf_value, 4),
                         'winner': int(winner), 'legal': 0, 'prior_top': []}

        node.update_recursive(-leaf_value)

        # 回溯后沿路径重走一遍，读更新后的访问数/Q（前端要画"这条线被加强了多少"）
        cur = self._root
        for p in path:
            cur = cur._children[p['action']]
            p['visits'] = int(cur._n_visits)
            p['Q'] = round(float(cur._Q), 4)
            p['P'] = round(float(cur._P), 6)
        self.trace.append({'path': path, 'leaf': leaf_info,
                           'depth': len(path),
                           'root_visits': int(self._root._n_visits)})
        return leaf_value


def mcts_viz(board, net, n_playout=100, c_puct=5, temp=1.0, trace_cap=400):
    """跑 n_playout 次模拟，返回树/根节点候选/路径追踪/与网络先验的偏离。"""
    n_playout = max(1, min(int(n_playout), 4000))
    searcher = TracedMCTS(net.policy_value_fn, c_puct=c_puct, n_playout=n_playout)
    root_value = None
    try:
        _, root_value, _ = clean_eval(net, board)
    except Exception:
        pass

    for _ in range(n_playout):
        searcher.playout_traced(copy.deepcopy(board))

    root = searcher._root
    prior_probs, prior_value, legal = clean_eval(net, board)

    children = []
    for a, nd in root._children.items():
        if nd._n_visits <= 0:
            continue          # 根节点展开时会建出全部合法点（先验层），
                              # 但没被访问过的点不该算"候选"——它们会让页面画出一片 0 值噪声
        children.append({'action': int(a),
                         'visits': int(nd._n_visits),
                         'Q': round(float(nd._Q), 4),
                         'P': round(float(nd._P), 6)})
    children.sort(key=lambda x: -x['visits'])
    n_expanded = len(root._children)

    total_visits = sum(c['visits'] for c in children)
    for c in children:
        c['prob'] = round(100.0 * c['visits'] / total_visits, 2) if total_visits else 0.0
        r, cc = c['action'] // WIDTH, c['action'] % WIDTH
        c['r'], c['c'] = r, cc

    # 温度采样概率（与 MCTSPlayer.get_action 同口径：softmax(1/temp · log(visits))）
    visits = np.array([c['visits'] for c in children], dtype=np.float64)
    if len(visits) and visits.sum() > 0:
        act_probs = softmax(1.0 / max(temp, 1e-3) * np.log(visits + 1e-10))
    else:
        act_probs = np.zeros(0)
    for c, p in zip(children, act_probs):
        c['temp_prob'] = round(float(p) * 100, 2)

    # MCTS 访问分布 vs 网络先验的偏离（页面「搜索是否只是照抄 policy」的判据）
    pol_dev = None
    if children and legal.any():
        p_mcts = np.zeros(WIDTH * HEIGHT)
        for c in children:
            p_mcts[c['action']] = c['visits'] / total_visits if total_visits else 0
        p_net = prior_probs
        d = p_mcts - p_net
        nz = prior_probs > 0
        pol_dev = {
            'tv': round(float(0.5 * np.abs(d).sum()), 4),
            'mse': round(float((d ** 2).mean()), 6),
            'mae': round(float(np.abs(d).mean()), 6),
            'cos': round(float((p_mcts * p_net).sum() /
                               ((np.linalg.norm(p_mcts) * np.linalg.norm(p_net)) + 1e-12)), 4),
            'kl': round(float(np.sum(np.where(nz,
                        prior_probs * np.log((prior_probs + 1e-12) / (p_mcts + 1e-12)), 0.0))), 4),
            'argmax_same': bool(np.argmax(p_mcts) == np.argmax(prior_probs)),
            'n_legal': int(legal.sum()),
        }

    # 深度分布：policy/value 配合好 → 搜得深；配合差 → 浅层乱撞
    depth_hist = {}
    for s in searcher.trace:
        d = s['depth'] + 1
        depth_hist[d] = depth_hist.get(d, 0) + 1

    trace = searcher.trace[:trace_cap]
    return {
        'sim_count': n_playout,
        'c_puct': c_puct,
        'temp': temp,
        'player': int(board.current_player),
        'root_value': root_value,
        'root_visits': int(root._n_visits),
        'n_expanded': n_expanded,
        'children': children[:20],
        'best': children[0] if children else None,
        'policy_grid': to_grid(prior_probs.reshape(WIDTH, HEIGHT)),
        'prior_value': prior_value,
        'pol_dev': pol_dev,
        'depth_hist': [{'depth': k, 'count': v}
                       for k, v in sorted(depth_hist.items())],
        'avg_depth': round(sum(depth_hist[k] * k for k in depth_hist) /
                           max(1, sum(depth_hist.values())), 2),
        'max_depth': max(depth_hist) if depth_hist else 0,
        'steps': trace,
        'truncated': len(searcher.trace) > trace_cap,
        'stones': board_to_stones(board),
    }


# ─────────────────────────────────────────────────────────────
# 网络前向：逐层激活
# ─────────────────────────────────────────────────────────────
def _net_layer_spec(pvn):
    """按网络结构自适应给出「要抓哪些层」。返回 [(key, 显示名, op说明, module, kind)]。

    🚨 不能写死 v1 Net 的层名（conv1/conv2/conv3/act_conv1/val_conv1…）：
       15×15 这条线用的是 V24Net（conv_input + N×res_blocks + 双头，层名完全不同），
       照抄 v1 的选择器会 getattr 直接抛
       AttributeError: 'V24Net' object has no attribute 'conv1'，
       整个「网络前向」页从此打不开。
    """
    spec = []

    if hasattr(pvn, 'res_blocks'):
        # ── V24Net：输入卷积 + 残差块（首/中/末抽样）+ 双头
        spec.append(('conv_input', 'conv_input',
                     'Conv2d(4→%d, k3)+BN+ReLU' % pvn.conv_input.out_channels,
                     pvn.conv_input, 'conv'))
        nb = len(pvn.res_blocks)
        for i in sorted(set([0, nb // 2, nb - 1])) if nb else []:
            blk = pvn.res_blocks[i]
            spec.append(('res%d' % i, 'res_blocks[%d]' % i,
                         'ResBlock %d→%d（2×conv+BN+残差）' % (
                             blk.conv1.in_channels, blk.conv2.out_channels),
                         blk, 'conv'))
        spec += [
            ('policy_conv', 'policy_conv',
             'Conv2d(%d→1, k3)+BN+ReLU' % pvn.policy_conv.in_channels,
             pvn.policy_conv, 'conv'),
            ('policy_conv2', 'policy_conv2',
             'Conv2d(%d→1, k1)+BN+ReLU' % pvn.policy_conv2.in_channels,
             pvn.policy_conv2, 'conv'),
            ('policy_fc', 'policy_fc',
             'Linear(%d→%d)+LogSoftmax' % (pvn.policy_fc.in_features,
                                           pvn.policy_fc.out_features),
             pvn.policy_fc, 'vec'),
            ('value_conv', 'value_conv',
             'Conv2d(%d→1, k3)+BN+ReLU' % pvn.value_conv.in_channels,
             pvn.value_conv, 'conv'),
            ('value_fc1', 'value_fc1',
             'Linear(%d→128)+ReLU' % pvn.value_fc1.in_features,
             pvn.value_fc1, 'vec'),
            ('value_fc2', 'value_fc2', 'Linear(128→1)+tanh',
             pvn.value_fc2, 'vec'),
        ]
        return spec

    # ── v1 Net（上游 0.33M）：三层 conv + 双头
    spec += [
        ('conv1', 'conv1', 'Conv2d(4→32, k3)', pvn.conv1, 'conv'),
        ('conv2', 'conv2', 'Conv2d(32→64, k3)', pvn.conv2, 'conv'),
        ('conv3', 'conv3', 'Conv2d(64→128, k3)', pvn.conv3, 'conv'),
        ('act_conv1', 'act_conv1', 'Conv2d(128→4, k1)', pvn.act_conv1, 'conv'),
        ('act_fc1', 'act_fc1', 'Linear+LogSoftmax(225)', pvn.act_fc1, 'vec'),
        ('val_conv1', 'val_conv1', 'Conv2d(128→2, k1)', pvn.val_conv1, 'conv'),
        ('val_fc1', 'val_fc1', 'Linear(→64)', pvn.val_fc1, 'vec'),
        ('val_fc2', 'val_fc2', 'Linear(→1)+tanh', pvn.val_fc2, 'vec'),
    ]
    return spec


def net_viz(board, net):
    """一次前向，抓每层激活。返回逐层热力图 + policy/value 双头。"""
    import torch
    pvn = net.policy_value_net
    pvn.eval()

    state = np.ascontiguousarray(board.current_state().reshape(-1, 4, WIDTH, HEIGHT))
    x = torch.from_numpy(state).float()

    spec = _net_layer_spec(pvn)
    captured = {}
    handles = []

    def hook(name):
        def _h(mod, inp, out):
            captured[name] = out.detach().clone()
        return _h

    for key, _nm, _op, mod, _kind in spec:
        handles.append(mod.register_forward_hook(hook(key)))
    try:
        with torch.no_grad():
            logp, v = pvn(x)
    finally:
        for h in handles:
            h.remove()

    layers = []

    def add_conv(name, op, tensor, module=None, max_channels=8):
        a = tensor[0].numpy()                 # (C,15,15)
        C = a.shape[0]
        mean_map = a.mean(axis=0)
        amax_map = a.max(axis=0)
        flat = a.ravel()
        maps = [{'label': '通道均值', 'grid': to_grid(mean_map)},
                {'label': '通道最大', 'grid': to_grid(amax_map)}]
        for i in range(min(C, max_channels)):
            maps.append({'label': 'ch%d' % i, 'grid': to_grid(a[i])})
        layers.append({
            'name': name, 'op': op, 'shape': list(tensor[0].shape),
            'params': int(sum(p.numel() for p in module.parameters())) if module else 0,
            'stats': {'mean': round(float(flat.mean()), 4),
                      'std': round(float(flat.std()), 4),
                      'max': round(float(flat.max()), 4),
                      'min': round(float(flat.min()), 4),
                      'active': round(float((flat > 0).mean()), 4),
                      'dead_ch': int(sum(1 for i in range(C)
                                         if float(np.abs(a[i]).max()) < 1e-6)),
                      'channels': int(C)},
            'maps': maps,
        })

    def add_vec(name, op, tensor, module=None, grid=None):
        a = tensor[0].numpy().ravel()
        layers.append({
            'name': name, 'op': op, 'shape': list(tensor[0].shape),
            'params': int(sum(p.numel() for p in module.parameters())) if module else 0,
            'stats': {'mean': round(float(a.mean()), 4),
                      'std': round(float(a.std()), 4),
                      'max': round(float(a.max()), 4),
                      'min': round(float(a.min()), 4),
                      'active': round(float((a > 0).mean()), 4),
                      'dead_ch': 0,
                      'channels': int(a.size)},
            'maps': ([{'label': 'policy logits', 'grid': grid}] if grid else []),
        })

    prod = np.exp(logp[0].numpy().ravel())
    legal = np.zeros(WIDTH * HEIGHT, dtype=bool)
    legal[board.availables] = True
    prod = np.where(legal, prod, 0.0)
    if prod.sum() > 0:
        prod = prod / prod.sum()
    pgrid = [round(float(x), 6) for x in prod]

    # 按 spec 顺序逐层落表（结构自适应，v1 Net 与 V24Net 都能跑）
    for key, nm, op, mod, kind in spec:
        t = captured.get(key)
        if t is None:
            continue
        if kind == 'conv':
            add_conv(nm, op, t, mod)
        else:
            add_vec(nm, op, t, mod, pgrid if key.startswith('policy') else None)

    top = sorted([{'r': i // WIDTH, 'c': i % WIDTH, 'prob': round(float(p), 6)}
                  for i, p in enumerate(prod)], key=lambda x: -x['prob'])[:10]

    ent = -float(np.sum(np.where(prod > 0, prod * np.log(prod + 1e-12), 0.0)))
    val = float(v[0].numpy().ravel()[0])

    # 输入 4 平面的可读名（上游口径：ch0 当前方 / ch1 对手 / ch2 上一手 / ch3 执子色常量）
    st = state[0]
    inputs = [{'label': 'ch0 当前方棋子', 'grid': to_grid(st[0])},
              {'label': 'ch1 对手棋子', 'grid': to_grid(st[1])},
              {'label': 'ch2 上一手位置', 'grid': to_grid(st[2])},
              {'label': 'ch3 执子色常量', 'grid': to_grid(st[3])}]

    return {
        'player': int(board.current_player),
        'move_count': len(board.states),
        'inputs': inputs,
        'layers': layers,
        'policy': {'top': top, 'grid': pgrid,
                   'entropy': round(ent, 4),
                   'max_prob': round(float(prod.max()), 4),
                   'legal': int(legal.sum())},
        'value': round(val, 4),
        # 上游 value 是「当前走子方」视角；同时给黑白两个百分数，避免口径混淆
        # 🚨 value_black_pct 是「黑方优势」(-100~100)，不是胜率 ——
        #    前端拿 `100 - value_black_pct` 当白方胜率会算出 136% 这种数。
        #    胜率要走 50+50v 映射，单独给字段，别让调用方自己猜口径。
        'value_black_pct': round(100.0 * (val if board.current_player == BLACK else -val), 2),
        'win_black_pct': round(50.0 + 50.0 * (val if board.current_player == BLACK else -val), 2),
        'win_white_pct': round(50.0 - 50.0 * (val if board.current_player == BLACK else -val), 2),
        'stones': board_to_stones(board),
    }


# ─────────────────────────────────────────────────────────────
# 对弈会话
# ─────────────────────────────────────────────────────────────
class Session(object):
    """一局棋的状态机。人机 / AI 自战都走这里。"""

    def __init__(self):
        # RLock：new() 持锁期间要调 state()（它自己也加锁）——用普通 Lock 会自锁死
        self.lock = threading.RLock()
        self.board = None
        self.moves = []          # [(r,c)]
        self.config = {}
        self.over = False
        self.winner = None       # 1 黑 / 2 白 / -1 和
        self.win_history = []    # [{n, black_pct}]
        self.last_ai = None
        self.value_map = None

    # ── 开局
    def new(self, cfg):
        with self.lock:
            self.config = cfg
            stones = cfg.get('first_moves') or []
            b = make_board(stones)
            self.board = b
            self.moves = [(int(s['r']), int(s['c'])) for s in stones]
            self.over = False
            self.winner = None
            self.last_ai = None
            self.win_history = []
            for i in range(len(self.moves)):
                self._record_wr()
            self._refresh_value_map()
            # 开局就轮到 AI 的话让它走一步
            self._maybe_ai()
            return self.state()

    # ── 人类落子
    def human_move(self, r, c):
        with self.lock:
            if self.board is None:
                return False, '游戏未开始'
            if self.over:
                return False, '游戏已结束'
            r, c = int(r), int(c)
            if not (0 <= r < HEIGHT and 0 <= c < WIDTH):
                return False, '坐标越界'
            m = r * WIDTH + c
            if m not in self.board.availables:
                return False, '该点已有子'
            self.board.do_move(m)
            self.moves.append((r, c))
            self._after_move()
            self._maybe_ai()
            return True, None

    # ── AI 走一步（供前端轮询 AI 自战）
    def ai_step(self):
        with self.lock:
            if self.board is None:
                return False, '游戏未开始'
            if self.over:
                return False, '游戏已结束'
            if not self._cur_is_ai():
                return False, '现在不是 AI 回合'
            self._ai_move()
            return True, None

    # ── 内部
    def _cur_is_ai(self):
        if self.board is None:
            return False
        side = self.board.current_player
        key = 'black_type' if side == BLACK else 'white_type'
        return self.config.get(key) == 'ai'

    def _weight_for(self, side):
        return self.config.get('black_weight' if side == BLACK
                               else 'white_weight')

    def _maybe_ai(self):
        """轮到 AI 就走**一步**（不循环）。

        ⚠️ 不能在这里 while 循环：AI vs AI 时会把整局下完再返回，
        前端就只能干等一次长请求，看不到逐手推进。改成一步一返回，
        由前端轮询 /api/game/ai 推进。
        """
        if self.over or not self._cur_is_ai():
            return
        self._ai_move()

    def _ai_move(self):
        side = self.board.current_player
        wf = self._weight_for(side)
        net = get_net(wf)
        n_playout = int(self.config.get('n_playout', 200))
        temp = float(self.config.get('temp', 0.3))
        c_puct = float(self.config.get('c_puct', 5))

        player = MCTSPlayer(net.policy_value_fn, c_puct=c_puct,
                            n_playout=n_playout, is_selfplay=0)
        move, probs = player.get_action(self.board, temp=temp, return_prob=1)

        # 记下这手的搜索依据（访问分布 top3 + 网络 value）
        order = np.argsort(probs)[::-1][:3]
        _, val, _ = clean_eval(net, self.board)
        self.last_ai = {
            'r': int(move) // WIDTH, 'c': int(move) % WIDTH,
            'side': int(side),
            'n_playout': n_playout,
            'value_before': round(val, 4),
            'top': [{'r': int(i) // WIDTH, 'c': int(i) % WIDTH,
                     'prob': round(float(probs[i]) * 100, 2)} for i in order],
            'weight': wf,
        }
        self.board.do_move(int(move))
        self.moves.append((int(move) // WIDTH, int(move) % WIDTH))
        self._after_move()

    def _after_move(self):
        end, winner = self.board.game_end()
        if end:
            self.over = True
            self.winner = int(winner)
        self._record_wr()
        self._refresh_value_map()

    def _record_wr(self):
        """每步后记一次「黑方胜率」，供页面画胜率走势。"""
        if self.board is None or self.over:
            return
        try:
            net = get_net(self._weight_for(BLACK) or self._weight_for(WHITE))
            _, v, _ = clean_eval(net, self.board)
            black_pct = 100.0 * (v if self.board.current_player == BLACK else -v)
            self.win_history.append({'n': len(self.moves),
                                     'black_pct': round(black_pct, 2)})
        except Exception:
            pass

    def _refresh_value_map(self):
        if self.board is None or self.over or not self.board.availables:
            self.value_map = None
            return
        try:
            net = get_net(self._weight_for(self.board.current_player))
            self.value_map = value_map(net, self.board)
        except Exception:
            self.value_map = None

    # ── 状态快照
    def state(self):
        with self.lock:
            if self.board is None:
                return None
            side = self.board.current_player
            return {
                'stones': board_to_stones(self.board),
                'moves': [{'r': r, 'c': c} for r, c in self.moves],
                'current': int(side),
                'move_count': len(self.moves),
                'over': bool(self.over),
                'winner': self.winner,
                'value_map': self.value_map,
                'win_history': self.win_history,
                'config': self.config,
                'last_ai': self.last_ai,
                'cur_is_ai': self._cur_is_ai(),
                'availables': len(self.board.availables),
            }
