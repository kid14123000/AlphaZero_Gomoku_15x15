# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import random
import argparse
import json
import os
import time
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
try:                                    # Rapfi 未安装时不影响训练
    from rapfi_player import (RapfiPlayer, RapfiError,
                              play_one_game, OPENING_POINTS)
except Exception:                       # noqa: BLE001
    RapfiPlayer = None
    RapfiError = Exception
from policy_value_net_pytorch import PolicyValueNet  # Pytorch（本仓唯一实现）


class MetricsLogger(object):
    """Append-only JSONL metrics log, consumed by web/dashboard.py"""

    def __init__(self, path=None):
        self.path = path
        self.t0 = time.time()
        if path:
            d = os.path.dirname(os.path.abspath(path))
            if d and not os.path.isdir(d):
                os.makedirs(d)

    def log(self, kind, **fields):
        if not self.path:
            return
        rec = {'type': kind, 't': time.time(), 'elapsed': time.time() - self.t0}
        rec.update(fields)
        with open(self.path, 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()


class TrainPipeline():
    def __init__(self, init_model=None,
                 board_width=15, board_height=15, n_in_row=5,
                 n_playout=400, game_batch_num=1500, check_freq=50,
                 play_batch_size=1, epochs=5, batch_size=512,
                 pure_mcts_playout_num=1000, metrics_file=None,
                 batch_offset=0, init_lr_multiplier=1.0, ckpt_dir='checkpoints',
                 warmup_min_buffer=2000, warmup_updates=10,
                 save_games=None, gate_step=200, gate_max=1200,
                 rapfi_bin=None, rapfi_depth=2, rapfi_every=0, rapfi_games=4,
                 rapfi_who='absolute', eval_pure=0, value_target='z',
                 dirichlet_eps=0.25, dirichlet_alpha=0.3,
                 temp_switch=0, temp_after=None,
                 c_puct=None, root_noise=None):
        # params of the board and the game
        self.board_width = board_width
        self.board_height = board_height
        self.n_in_row = n_in_row
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        # 温度调度：手数 < temp_switch 用 temp，之后用 temp_after（None=不调度）
        self.temp_switch = int(temp_switch)
        self.temp_after = temp_after
        self.n_playout = n_playout  # num of simulations for each move
        # 2026-11-14: c_puct 5 → 1.25（v24 同款有效值）。5 = junxiaosong 上游默认，
        # 几乎完全跟着网络先验走，800 次模拟的纠错被浪费；LZ 用 0.5 但公式不同不可比。
        self.c_puct = (float(c_puct) if c_puct is not None
                       else float(os.environ.get('GOMOKU_C_PUCT', '1.25')))
        # 噪声注入位置：True=根先验(AGZ/v24 式) / False=旧行为(只混落子抽样)
        self.root_noise = (bool(int(root_noise)) if root_noise is not None
                           else bool(int(os.environ.get('GOMOKU_ROOT_NOISE', '1'))))
        self.buffer_size = 10000
        self.batch_size = batch_size  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        # ── 课程学习：按 (行动方 × 胜负) 四类分层采样 ──
        # 0=黑走&黑赢 1=黑走&黑输 2=白走&白赢 3=白走&白输
        # 目标份额由 sync/class_weight.json 提供（控制器动态写）；空/缺失=均匀采样
        self.class_weight_file = str(os.environ.get('GOMOKU_CLASS_WEIGHT_FILE', '') or '')
        self.short_white_win_min = int(os.environ.get('GOMOKU_WW_MIN_LEN', '20'))
        self._cw_mtime = None
        self._cw_shares = None
        self._cls_hist = [0, 0, 0, 0]
        self._cls_batch = [0, 0, 0, 0]
        # 稀有类独立储库：只放 class=2（白走&白赢）的增广样本，不受主 buffer 的 FIFO 淘汰
        self.ww_reservoir = deque(maxlen=int(os.environ.get('GOMOKU_WW_RESERVOIR', '24000')))
        self.load_ww_reservoir()
        self.play_batch_size = play_batch_size
        self.epochs = epochs  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = check_freq
        self.game_batch_num = game_batch_num
        self.best_win_ratio = 0.0
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = pure_mcts_playout_num
        # 门控阶梯：赢满 100% 时 +gate_step，最多 gate_max（原版是 +1000 到 5000，太慢）
        self.gate_step = gate_step
        self.gate_max = gate_max
        # Rapfi 外部锚点（每 rapfi_every 局测 rapfi_games 局；0=关闭）
        self.rapfi_bin = rapfi_bin
        self.rapfi_depth = rapfi_depth
        self.rapfi_every = rapfi_every
        self.rapfi_games = rapfi_games
        self.rapfi_who = rapfi_who
        # 对纯 MCTS 评估开关（0=关，默认）。饱和后只在白耗机时。
        self.eval_pure = int(eval_pure)
        # 自我对弈探索噪声：上游 0.25/0.3。9×9 短局建议 0.10（实测探索手 28%→11%）
        self.dirichlet_eps = float(dirichlet_eps)
        self.dirichlet_alpha = float(dirichlet_alpha)
        # value 头的训练目标：z=最终胜负±1（原版），q=MCTS 根节点搜索价值（软标签），
        # mix=两者加权。q 能区分险胜/碾压，但早期网络弱时 Q 可能都接近 0，
        # 所以保留 z 档可随时回退。
        self.value_target = str(value_target)
        if self.value_target not in ('z', 'q', 'mix'):
            raise ValueError('value_target 必须是 z / q / mix，收到 %r' % value_target)
        self.mix_lambda = 0.5        # mix 档：q 占多少
        self.metrics = MetricsLogger(metrics_file)
        self.game_batch_cur = 0
        self.batch_offset = batch_offset
        self.save_games = save_games
        self.ckpt_dir = ckpt_dir
        self.last_winner = -1
        self.metrics.log('start',
                         board='{}x{}'.format(board_width, board_height),
                         n_in_row=n_in_row,
                         n_playout=n_playout,
                         game_batch_num=game_batch_num,
                         check_freq=check_freq,
                         batch_size=batch_size,
                         epochs=epochs,
                         pure_mcts_playout_num=pure_mcts_playout_num,
                         batch_offset=batch_offset,
                         init_model_used=bool(init_model),
                         init_lr_multiplier=init_lr_multiplier,
                         eval_pure=int(eval_pure),
                         c_puct=self.c_puct,
                         root_noise=int(self.root_noise),
                         rapfi_every=rapfi_every,
                         rapfi_depth=rapfi_depth,
                         value_target=self.value_target)
        # restored adaptive learning-rate state (resuming a long run)
        self.lr_multiplier = init_lr_multiplier
        # ── 续训 warmup：① buffer 不够就不学（避免对着几局数据反复过拟合）
        #                ② 前 N 次 update 线性升 LR（首更只给 20%）──
        self.warmup_min_buffer = int(warmup_min_buffer)
        self.warmup_updates = int(warmup_updates)
        self._n_update = 0
        self._warmup_warned = False
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height,
                                                   model_file=init_model)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.board_width,
                                                   self.board_height)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1,
                                      dirichlet_eps=self.dirichlet_eps,
                                      dirichlet_alpha=self.dirichlet_alpha,
                                      root_noise=self.root_noise)
        # q/mix 模式：开跑前先验符号。反了就拒绝启动，别白跑几百局才发现。
        if self.value_target in ('q', 'mix'):
            self._selfcheck_q_sign()

    def _selfcheck_q_sign(self):
        """符号自检：构造「黑已四连、轮到黑、落子即成五」的必赢局面。

        只有一条赢路时，MCTS 根 Q 必须显著为正（≈+1）。
        符号反了会得到 ≈-1 —— 启动阶段就拦下来，代价 1 秒，
        总好过 619 局（46 分钟）跑完才发现 value 标签全是反的。

        局面（8 手，轮到黑）：
            黑 (4,0)(4,1)(4,2)(4,3) 四连，两端只有 (4,4) 能成五
            白 (0,0)(2,0)(4,6)(6,6) 四个散子，不构成威胁
        """
        b = Board(width=self.board_width, height=self.board_height,
                  n_in_row=self.n_in_row)
        b.init_board(0)
        seq = [(4, 0), (0, 0), (4, 1), (2, 0), (4, 2), (4, 6), (4, 3), (6, 6)]
        for r, c in seq:
            m = r * self.board_width + c
            if m not in b.availables:
                raise RuntimeError('符号自检：构造局面时落点冲突 %s' % ((r, c),))
            b.do_move(m)
        if b.get_current_player() != b.players[0]:
            raise RuntimeError('符号自检：局面轮次不对（应轮到黑先行方）')

        probe = MCTSPlayer(self.policy_value_net.policy_value_fn,
                           c_puct=self.c_puct,
                           n_playout=max(200, self.n_playout // 2),
                           is_selfplay=1)
        probe.mcts.update_with_move(-1)
        probe.mcts.get_move_probs(b, temp=1.0)
        q = float(probe.mcts.last_root_q)

        msg = ('符号自检：黑已四连且轮到黑，根 Q 应为正（≈+1），实测 %+.4f' % q)
        if q < 0.0:
            raise RuntimeError(
                msg + '\n→ last_root_q 符号反了（根节点 _Q 需取负），'
                      '拒绝启动。先跑 probe_q_sign.py 确认。')
        if q < 0.5:
            print('[warn] ' + msg + ' —— 偏低，MCTS 可能没搜到成五点，但不阻断启动',
                  flush=True)
        else:
            print('[ok] ' + msg, flush=True)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z[, cls]), ..., ...]
        cls（第 4 元，可选）= 分层采样用的类别标签，原样透传
        """
        extend_data = []
        for item in play_data:
            state, mcts_prob, winner = item[0], item[1], item[2]
            cls = item[3] if len(item) > 3 else -1
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner, cls))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner, cls))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        last_winner = -1
        for i in range(n_games):
            if self.value_target == 'z':
                winner, play_data = self.game.start_self_play(
                    self.mcts_player, temp=self.temp,
                    temp_switch=self.temp_switch, temp_after=self.temp_after)
                q_list = None
            else:
                winner, play_data, q_list = self.game.start_self_play(
                    self.mcts_player, temp=self.temp, collect_q=True,
                    temp_switch=self.temp_switch, temp_after=self.temp_after)
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)

            # ── value 目标：q / mix 档把第三元换成软标签 ──
            # play_data 元素是 (state, mcts_prob, z)，get_equi_data 原样透传第三元，
            # 所以在这里替换即可，下游训练代码不用动。
            if q_list is not None:
                q_arr = np.asarray(q_list, dtype=np.float32)
                if self.value_target == 'q':
                    new_t = q_arr
                else:
                    zmix = np.asarray([d[2] for d in play_data], dtype=np.float32)
                    lam = self.mix_lambda
                    new_t = lam * q_arr + (1.0 - lam) * zmix
                play_data = [(s, p, float(t))
                             for (s, p, _z), t in zip(play_data, new_t)]
                # 诊断：Q 的分布。若均值和标准差都贴近 0，说明搜索价值没信号，
                # 此时 q 当标签等于让 value 头两边不靠，应退回 mix 或 z。
                q_stats = {'q_mean': float(q_arr.mean()),
                           'q_std': float(q_arr.std()),
                           'q_absmean': float(np.abs(q_arr).mean()),
                           't_mean': float(new_t.mean()),
                           't_std': float(new_t.std())}
            else:
                q_stats = None

            # ── 分层采样标签：(行动方 × 胜负) 0=BB赢 1=B输 2=W赢 3=W输 ──
            # 放在 value 目标替换之后打，避免重建元组时丢掉。
            # 反捡漏：短手数(默认 <20)的白胜样本不打成 2（归 3），不进"正样本加权"
            _short_ww = (winner == 2 and self.episode_len < self.short_white_win_min)
            _cl = []
            for _i in range(self.episode_len):
                _mover = 1 if _i % 2 == 0 else 2
                _c = 2 * (1 if _mover == 2 else 0) + (0 if winner == _mover else 1)
                if _short_ww and _c == 2:
                    _c = 3
                _cl.append(_c)
            play_data = [(d[0], d[1], d[2], _cl[_j]) for _j, d in enumerate(play_data)]

            last_winner = winner
            self.last_winner = winner
            self.metrics.log('game',
                             batch=self.game_batch_cur,
                             episode_len=self.episode_len,
                             winner=winner,
                             **(q_stats or {}))
            # 棋谱落盘（默认关闭）：self.board.states 保留整局手顺，键=move(0..224)，值=走子方
            if self.save_games:
                try:
                    mv = [[int(m) // self.board_width, int(m) % self.board_width]
                          for m in self.board.states.keys()]
                    with open(self.save_games, 'a') as gf:
                        gf.write(json.dumps({'batch': int(self.game_batch_cur),
                                             'winner': int(winner),
                                             'episode_len': int(self.episode_len),
                                             'moves': mv}, ensure_ascii=False) + '\n')
                except Exception as e:
                    print('save_games failed:', e, flush=True)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.add_samples(play_data)
        return last_winner

    # ── 白赢储库持久化（内存 deque 重启即丢，白赢样本稀缺，必须落盘）──
    def save_ww_reservoir(self, path=None):
        p = path or os.environ.get('GOMOKU_WW_FILE', '')
        if not p or not self.ww_reservoir:
            return 0
        try:
            arr_s = np.asarray([d[0] for d in self.ww_reservoir], dtype=np.float32)
            arr_p = np.asarray([d[1] for d in self.ww_reservoir], dtype=np.float32)
            arr_z = np.asarray([d[2] for d in self.ww_reservoir], dtype=np.float32)
            tmp = p + '.tmp'
            np.savez(tmp, states=arr_s, probs=arr_p, zs=arr_z)
            os.replace(tmp + '.npz', p)
            return len(arr_z)
        except Exception as e:                    # noqa: BLE001
            print('[ww] 保存失败 %r' % (e,), flush=True)
            return 0

    def load_ww_reservoir(self, path=None):
        p = path or os.environ.get('GOMOKU_WW_FILE', '')
        if not p or not os.path.exists(p):
            return 0
        try:
            with np.load(p) as d:
                st, pb, zs = d['states'], d['probs'], d['zs']
            n = 0
            for i in range(len(zs)):
                self.ww_reservoir.append((st[i], pb[i], float(zs[i]), 2))
                n += 1
            print('[ww] 载入白赢储库 %d 条（%s）' % (n, p), flush=True)
            return n
        except Exception as e:                    # noqa: BLE001
            print('[ww] 载入失败 %r' % (e,), flush=True)
            return 0

    def add_samples(self, play_data_aug):
        """样本统一入口：进主 buffer，同时把【白赢】样本额外存进独立储库"""
        self.data_buffer.extend(play_data_aug)
        for d in play_data_aug:
            if len(d) > 3 and d[3] == 2:
                self.ww_reservoir.append(d)

    def _load_class_shares(self):
        """读 sync/class_weight.json 的目标份额 [B_win, B_lose, W_win, W_lose]；失败=None(均匀)"""
        if not self.class_weight_file:
            return None
        try:
            m = os.path.getmtime(self.class_weight_file)
            if m == self._cw_mtime:
                return self._cw_shares
            self._cw_mtime = m
            with open(self.class_weight_file) as f:
                d = json.load(f)
            sh = [float(d.get(k, 0.25)) for k in ('B_win', 'B_lose', 'W_win', 'W_lose')]
            s = sum(sh)
            self._cw_shares = [x / s for x in sh] if s > 0 else None
        except Exception:                       # noqa: BLE001  文件缺失/半写
            self._cw_shares = None
        return self._cw_shares

    def sample_batch(self):
        """按四类目标份额分层抽样；无配置/退化时退回均匀抽样"""
        shares = self._load_class_shares()
        n = len(self.data_buffer)
        if shares is None or n <= self.batch_size:
            return random.sample(self.data_buffer, self.batch_size)
        pools = {0: [], 1: [], 2: [], 3: []}
        for d in self.data_buffer:
            c = d[3] if len(d) > 3 else -1
            if c in pools:
                pools[c].append(d)
        pool_len = {c: len(pools[c]) for c in range(4)}
        if self.ww_reservoir:                    # 白赢改用独立储库（供给更足、不被淘汰）
            pools[2] = list(self.ww_reservoir)
        idx = pools
        self._cls_hist = [len(idx[c]) for c in range(4)]
        self._cls_hist_buf = [pool_len[c] for c in range(4)]
        cnt = [min(int(round(self.batch_size * shares[c])), len(idx[c])) for c in range(4)]
        # 份额被小类别截断后，用有富余的类别补齐
        need = self.batch_size - sum(cnt)
        order = sorted(range(4), key=lambda c: len(idx[c]) - cnt[c], reverse=True)
        while need > 0:
            moved = False
            for c in order:
                if need <= 0:
                    break
                if len(idx[c]) - cnt[c] > 0:
                    cnt[c] += 1
                    need -= 1
                    moved = True
            if not moved:
                break
        self._cls_batch = list(cnt)          # 抽样后各类实际入批数量（补齐之后，验证用）
        self._cls_batch = list(cnt)          # 抽样后各类实际入批数量（补齐之后，验证用）
        out = []
        for c in range(4):
            if cnt[c] > 0:
                out.extend(idx[c] if cnt[c] == len(idx[c]) else random.sample(idx[c], cnt[c]))
        if len(out) < self.batch_size:
            out.extend(random.sample(self.data_buffer, self.batch_size - len(out)))
        return out

    def policy_update(self):
        """update the policy-value net（含续训 warmup）"""
        # ① buffer 太小就跳过：早期对着 4~6 局数据跑 5 epoch 会把权重推歪
        if len(self.data_buffer) < self.warmup_min_buffer:
            if not self._warmup_warned:
                print('[warmup] buffer %d < %d，暂停学习直到数据够'
                      % (len(self.data_buffer), self.warmup_min_buffer), flush=True)
                self._warmup_warned = True
            return 0
        self._warmup_warned = False
        # ② 前 warmup_updates 次线性升 LR（0.2× → 1.0×）
        if self.warmup_updates > 0 and self._n_update < self.warmup_updates:
            _warm = 0.2 + 0.8 * (self._n_update / float(self.warmup_updates))
        else:
            _warm = 1.0
        self._n_update += 1
        mini_batch = self.sample_batch()
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier*_warm)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        self.metrics.log('update',
                         batch=self.game_batch_cur,
                         kl=float(kl),
                         lr_multiplier=float(self.lr_multiplier),
                         loss=float(loss),
                         entropy=float(entropy),
                         ev_old=float(explained_var_old),
                         ev_new=float(explained_var_new),
                         buffer=len(self.data_buffer),
                         cls_hist=self._cls_hist,
                         cls_batch=getattr(self, '_cls_batch', []),
                         ww_res=len(self.ww_reservoir),
                         cw_shares=self._cw_shares or [])
        return loss, entropy

    def rapfi_anchor(self, n_games=None):
        """Rapfi 外部锚点（阶段性摸底）：不参与门控，失败只记录、绝不打断训练"""
        if RapfiPlayer is None or not self.rapfi_bin:
            return
        n = n_games or self.rapfi_games
        win = lose = tie = 0
        lens = []
        t0 = time.time()
        for i in range(n):
            rap = None
            try:
                net = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                 c_puct=self.c_puct, n_playout=self.n_playout)
                rap = RapfiPlayer(binary=self.rapfi_bin, depth=self.rapfi_depth,
                                  who_mode=self.rapfi_who, size=self.board_width)
                net_black = (i % 2 == 0)
                # 🚨 必须随机开局：argmax 评估是确定性的，同开局=同一盘棋重放
                fm = OPENING_POINTS[random.randrange(len(OPENING_POINTS))]
                w, n_mv = play_one_game(net, rap, net_black=net_black, first_move=fm)
                lens.append(n_mv)
                my = 1 if net_black else 2
                if w == my:
                    win += 1
                elif w == 3 - my:
                    lose += 1
                else:
                    tie += 1
            except Exception as e:                      # noqa: BLE001
                print('[Rapfi锚点] 失败: {}'.format(e), flush=True)
                self.metrics.log('rapfi_fail', batch=self.game_batch_cur,
                                 err=str(e)[:200])
                return
            finally:
                if rap is not None:
                    rap.close()
        avg = sum(lens) / max(len(lens), 1)
        print('[Rapfi锚点] d{}: {}/{} 胜 均{:.0f}手 用时{:.0f}s'.format(
            self.rapfi_depth, win, n, avg, time.time() - t0), flush=True)
        self.metrics.log('rapfi', batch=self.game_batch_cur, win=win, lose=lose,
                         tie=tie, depth=self.rapfi_depth, n_games=n,
                         avg_len=round(avg, 1), secs=round(time.time() - t0, 1))

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            start_player = i % 2
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=start_player,
                                          is_shown=0)
            win_cnt[winner] += 1
            # winner 1 == current_mcts_player (the net)
            self.metrics.log('eval_game',
                             batch=self.game_batch_cur,
                             start_player=start_player,
                             winner=winner)
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        self.metrics.log('eval',
                         batch=self.game_batch_cur,
                         win=win_cnt[1],
                         lose=win_cnt[2],
                         tie=win_cnt[-1],
                         win_ratio=float(win_ratio),
                         pure_playout=self.pure_mcts_playout_num,
                         n_games=n_games)
        return win_ratio

    def run(self):
        """run the training pipeline"""
        n_this_run = max(0, self.game_batch_num - self.batch_offset)
        if self.batch_offset:
            print("resuming: {} games already done, running {} more -> {} total".format(
                self.batch_offset, n_this_run, self.game_batch_num), flush=True)
        try:
            for i in range(n_this_run):
                self.game_batch_cur = self.batch_offset + i + 1
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}, winner:{}".format(
                        self.game_batch_cur, self.episode_len,
                        {-1: 'tie'}.get(self.last_winner,
                                        'player{}'.format(self.last_winner))),
                        flush=True)
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                # NOTE: gate on the GLOBAL counter so that resuming a run
                # keeps the evaluation cadence (every check_freq games overall),
                # not "every check_freq games since this process started".
                # ── 存档（每 check_freq 局）─────────────────────────
                # 先存再评估：评估崩了也不该丢权重。
                if self.game_batch_cur % self.check_freq == 0:
                    self.policy_value_net.save_model('./current_policy.model')
                    # immutable per-iteration archive so a corrupted/overwritten
                    # current_policy.model never costs more than check_freq games
                    if self.ckpt_dir:
                        if not os.path.isdir(self.ckpt_dir):
                            os.makedirs(self.ckpt_dir)
                        self.policy_value_net.save_model(os.path.join(
                            self.ckpt_dir, 'ckpt_it{}.model'.format(self.game_batch_cur)))

                # ── 纯 MCTS 评估（默认关闭）────────────────────────
                # 2026-11: 对纯 MCTS 已长期 100%，指标饱和没有区分度，
                # 而 gate 会把 playout 一路涨到 1200，单次评估 ~6 分钟
                # （占每 50 局墙钟时间的 35%）。改由 Rapfi 做外部锚点。
                if self.eval_pure and self.game_batch_cur % self.check_freq == 0:
                    print("current self-play batch: {}".format(self.game_batch_cur))
                    win_ratio = self.policy_evaluate()
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model('./best_policy.model')
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num + self.gate_step <= self.gate_max):
                            self.pure_mcts_playout_num += self.gate_step
                            self.best_win_ratio = 0.0
                            print("gate escalated: pure MCTS -> {} playouts".format(
                                self.pure_mcts_playout_num))

                # ── Rapfi 外部锚点（独立节奏）──────────────────────
                # ⚠️ 原来它嵌在上面 check_freq 分支里，节奏被 check_freq 卡住：
                #    check_freq=50 且 rapfi_every=120 时实际在 600 局才跑一次（lcm）。
                #    现在提到外层，按自己的 rapfi_every 走。
                if self.rapfi_every and self.game_batch_cur % self.rapfi_every == 0:
                    self.rapfi_anchor()
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlphaZero Gomoku training (default 15x15, 5 in a row)')
    parser.add_argument('--board_width', type=int, default=15)
    parser.add_argument('--board_height', type=int, default=15)
    parser.add_argument('--n_in_row', type=int, default=5)
    parser.add_argument('--n_playout', type=int, default=400)
    parser.add_argument('--dirichlet_eps', type=float, default=0.25,
                        help='自我对弈探索噪声 ε（上游 0.25；9×9 短局建议 0.10）')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.3)
    parser.add_argument('--temp_switch', type=int, default=0,
                        help='前 N 手用 --temp，之后切 --temp_after（0=不调度）')
    parser.add_argument('--temp_after', type=float, default=None)
    parser.add_argument('--game_batch_num', type=int, default=1500)
    parser.add_argument('--check_freq', type=int, default=50)
    parser.add_argument('--play_batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--pure_mcts_playout_num', type=int, default=1000)
    parser.add_argument('--init_model', type=str, default=None)
    parser.add_argument('--batch_offset', type=int, default=0,
                        help='self-play games already finished in earlier runs of the same lineage')
    parser.add_argument('--init_lr_multiplier', type=float, default=1.0,
                        help='restore the adaptive lr multiplier when resuming')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints',
                        help='archive ckpt_itN.model here (empty string disables)')
    parser.add_argument('--metrics_file', type=str, default='logs/metrics.jsonl')
    parser.add_argument('--save-games', dest='save_games', type=str, default='',
                        help='把每局手顺追加写入该 jsonl（默认关闭）')
    parser.add_argument('--gate_step', type=int, default=200,
                        help='门控赢满 100%% 时纯MCTS模拟数增加量')
    parser.add_argument('--gate_max', type=int, default=1200,
                        help='门控纯MCTS模拟数上限')
    parser.add_argument('--rapfi_bin', type=str, default='',
                        help='Rapfi 二进制的绝对路径（空=不跑外部锚点）')
    parser.add_argument('--rapfi_depth', type=int, default=2,
                        help='Rapfi 锚点深度（config 合法范围 2~200）')
    parser.add_argument('--rapfi_every', type=int, default=0,
                        help='每 N 局跑一次 Rapfi 锚点（0=关闭）')
    parser.add_argument('--rapfi_games', type=int, default=4,
                        help='每次锚点打几局（黑白各半）')
    parser.add_argument('--rapfi_who', type=str, default='absolute',
                        choices=['absolute', 'relative'])
    parser.add_argument('--eval_pure', type=int, default=0,
                        help='1=每 check_freq 局对纯 MCTS 打 10 局评估（默认 0 关闭：'
                             '该指标已长期 100%，且门控会把 playout 涨到 1200，'
                             '单次评估约 6 分钟）')
    parser.add_argument('--value_target', type=str, default='z',
                        choices=['z', 'q', 'mix'],
                        help='value 头的训练目标：z=最终胜负±1（原版，默认）；'
                             'q=MCTS 根节点搜索价值（软标签，能区分险胜/碾压）；'
                             'mix=两者各半')
    args = parser.parse_args()
    print("board {}x{}, n_in_row={}, n_playout={}, game_batch_num={}".format(
        args.board_width, args.board_height, args.n_in_row,
        args.n_playout, args.game_batch_num), flush=True)
    training_pipeline = TrainPipeline(init_model=args.init_model,
                                      board_width=args.board_width,
                                      board_height=args.board_height,
                                      n_in_row=args.n_in_row,
                                      n_playout=args.n_playout,
                                      game_batch_num=args.game_batch_num,
                                      check_freq=args.check_freq,
                                      play_batch_size=args.play_batch_size,
                                      epochs=args.epochs,
                                      batch_size=args.batch_size,
                                      pure_mcts_playout_num=args.pure_mcts_playout_num,
                                      metrics_file=args.metrics_file,
                                      save_games=(args.save_games or None),
                                      gate_step=args.gate_step,
                                      gate_max=args.gate_max,
                                      rapfi_bin=(args.rapfi_bin or None),
                                      rapfi_depth=args.rapfi_depth,
                                      rapfi_every=args.rapfi_every,
                                      rapfi_games=args.rapfi_games,
                                      rapfi_who=args.rapfi_who,
                                      eval_pure=args.eval_pure,
                                      value_target=args.value_target,
                                      batch_offset=args.batch_offset,
                                      init_lr_multiplier=args.init_lr_multiplier,
                                      ckpt_dir=args.ckpt_dir,
                                      dirichlet_eps=args.dirichlet_eps,
                                      dirichlet_alpha=args.dirichlet_alpha,
                                      temp_switch=args.temp_switch,
                                      temp_after=args.temp_after)
    training_pipeline.run()
