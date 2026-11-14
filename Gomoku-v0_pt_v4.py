"""
五子棋 AlphaZero 训练 v18（v12 全量复制 + ch2 视角颜色通道修复）

════════════════════════════════════════════════════════════
v6 → v8 唯一改动（2026-08-07，基于色盲/重黑轻白诊断）：

[根因] v6 单通道输入（canonical: 1=当前玩家 -1=对手）导致颜色偏置:
       网络对"黑方威胁"敏感、对"白方威胁"钝感（颜色对称性测试实测误差 0.7-1.1）
       实战表现为: 活四不堵（白方威胁识别弱）、败局识别全红

[改动] 输入升维 1ch → 4ch（AlphaZero 式 feature planes）:
       ch0: 当前玩家棋子 | ch1: 对手棋子 | ch2: 常数1(当前视角) | ch3: 上一步位置
       颜色翻转 = 交换 ch0/ch1 → 颜色对称性结构性满足，消除"重黑轻白"

[保持不变] 网络主干(128ch×8res)/policy头/value头/训练配置/MCTS参数/增强/数据格式 全同 v6
[目录]    gomoku_checkpoints_v8（独立，避免覆盖 v6）

════════════════════════════════════════════════════════════
v1 → v2 关键改动（2026-08-01，基于一周诊断）：

[根因] v1 的 value 头对"连子=胜负"判断退化（活四/五连盲测全反、
       实战不拦截连五、开局 50% 下边角），loss 却持续下降。
       根因 = 数据多样性不足 + 探索参数错误。

[改动]
1. ✅ temperature 调度（v1 完全没有！）
   ⚠️ 2026-09-25 更正: 本节原文描述的是【v2 旧调度】✗ (前15手 T=1 / 中段线性→0.25 / 终局→0)
      现行调度 = 【v35-B 热改】(2026-09-06 用户拍板), 实现在 self_play() 内:
        step 0-3: 1.00 → 0.60   |   step 4-7: 0.60 → 0.15   |   step 8+: 0.15 恒定平台
      理由: v32 线性调度下速杀局 it7 达 42% ✗, 败方末步全是堵点步(手序7-10, 活三/冲四成型期)
            选到搜索 top1 仅中位 67% ✗ → 不是搜索看不见, 是高温采样把它飞了
            → 修后堵点步 top 选中率 67% → 98% ✓
      取舍: 平台 0.15 而非 0.001 (留残局弹性; v28 教训=过早贪心锁死坏线)
            中盘探索交给【搜索内 Dirichlet noise】(论文式 ✓), 温度只管采样锐度
      原版 AlphaZero: 前30手 T=1, 之后 T→0 (我们是更激进的早降温)
2. ✅ Dirichlet alpha 从 1/合法数(≈0.0045) 固定为 0.03
   - 标准 AlphaZero 用 0.03；v1 的 alpha 太小探索几乎无效
3. ✅ 动态模拟量重新设计（v1: 50/200/400/800 每代固定）
   - 前 30 代: 200 模拟（快、多样，快速积累棋谱）
   - 31-60 代: 400 模拟
   - 61+ 代: 800 模拟（精算收敛）
4. ✅ prior_exploration_fraction 保持 0.25（标准值）
5. ✅ c_init=1.25, c_base=19652 保持（标准值）
6. ✅ 经验回放池：保留最近 N=5 代数据混合训练（v1 只训当前代）
   - 增加数据多样性，防止 value 头在单一棋风里自嗨
7. ✅ 训练轮数自适应：数据多则多训（v1 固定 32 batches×512）
8. ✅ 开局强制多样性：前 3 手强制落子在距离天元 4 格内（含噪音）
   - 防止网络永远从边角开局（v1 的 50% 边角开局是病）
9. ✅ MCTS 根节点噪音始终开启（self-play 时 noise=True）

[保留] 网络结构(128ch×8res)、8方向增强、checkpoint 格式、
      工兵并行框架、FP16/CUDA Graph 优化全部保留。
════════════════════════════════════════════════════════════
"""
import sys, os, time, math, json, glob, pickle, itertools, random, hashlib
import logging
import collections
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info('使用设备: {}'.format(device))

import boardgame2
from boardgame2 import BLACK, WHITE

# ─────────────────────────────────────────────
# 版本标识（v1 的存档/恢复不通用，避免误恢复）
# ─────────────────────────────────────────────
VERSION = "v42"

# ── v8 输入升维: 4 feature planes ──
# ch0: 当前玩家棋子 | ch1: 对手棋子 | ch2: 常数1(当前视角) | ch3: 上一步位置
INPUT_CHANNELS = 4

def make_input(canonical, last_move=None, player=None):
    """1ch canonical (15,15) → 4ch (4,15,15) float32 —— **junxiaosong v4 约定**

    🚨 这里必须和训练时的 Board.current_state() 完全一致，否则权重被喂错输入：
       v4 (game.py)          : ch0 当前方 / ch1 对手 / ch2 最后一手 / ch3 执色常量，
                               且 return square_state[:, ::-1, :]   ← 垂直翻转
       本参考版 v44 原约定  : ch0 当前方 / ch1 对手 / ch2 恒 1.0 / ch3 最后一手，无翻转
       两者通道 2/3 互换 + 翻转缺失 ⇒ 网络把「上一步」当成「执色」读，policy 直接散掉
       （实测：空盘/开局 AI 第一手下 D1 角落 → 就是这个原因）

    canonical: 1=当前玩家 -1=对手 0=空
    player:    当前玩家颜色(1=黑/-1=白)，用于 ch3 执色常量；None 视为黑
    """
    c = np.asarray(canonical)
    out = np.zeros((INPUT_CHANNELS,) + c.shape, dtype=np.float32)
    out[0] = (c > 0)
    out[1] = (c < 0)
    if last_move is not None:
        out[2][int(last_move[0]), int(last_move[1])] = 1.0
    if player is None or int(player) == 1:
        out[3][:, :] = 1.0
    # v4 的 current_state 末尾做了 [:, ::-1, :]，这里同样翻转行。
    # 🚨 必须 ascontiguousarray：切片[::-1] 是负步长视图，torch.from_numpy 直接报
    #    "At least one stride ... is negative"。
    return np.ascontiguousarray(out[:, ::-1, :])


def _to_planes(batch_1ch, players=None):
    """(15,15) 或 (N,15,15) 1ch canonical → (N,4,15,15) tensor（诊断探针用）
    players: (N,) 或标量 当前玩家颜色(1=黑/-1=白), 用于 ch2; None 默认黑 (兼容旧调用)"""
    b = np.asarray(batch_1ch)
    single = b.ndim == 2
    if single:
        b = b[None]
    inp = np.zeros((b.shape[0], INPUT_CHANNELS) + b.shape[1:], dtype=np.float32)
    inp[:, 0] = b > 0
    inp[:, 1] = b < 0
    if players is None:
        inp[:, 2] = 1.0  # v24: ch2 恒定 1
    else:
        pl = np.asarray(players)
        if pl.ndim == 0:
            inp[:, 2] = 1.0  # v24: ch2 恒定 1, 无执色信息
        else:
            inp[:, 2] = 1.0  # v24: ch2 恒定 1, 无执色信息
    x = torch.as_tensor(inp, device=device)
    return x[0] if single else x

# ── v2 新参数 ──
# ⚠️ 2026-09-25 清理: TEMP_HIGH_MOVES / TEMP_LOW_MOVES 已【废弃并删除】
#   原因: 它们是 v2 旧调度的残留, v35-B 热改后已不再参与任何计算 ✗
#         (残留常量+过时文档 = 归因地雷: 复盘时会按错误调度分析 ✗)
#   现行调度见本文件 self_play() 内 (step0-3: 1.00→0.60 | step4-7: →0.15 | step8+: 0.15)
# 2026-09-14 随机开局手数 (用户拍板): 前 N 手强制随机落点, 其余交模型决策
#   3 = v36 原行为 (黑1白1黑2 全随机) — 默认值, 不设变量时行为 100% 不变
#   1 = 只黑1随机(9x9), 白1/黑2 由模型决策 → 模型会自然"响应"黑1, 消除"两团分离"开局
# 动机: 实测"两团都紧(<=3)+团间远(>=4)"的随机开局 → 超短局占比 70.6% (vs 其他 10.9%, 6.5倍)
#       两团分离 = 双方各自发展互不接触 → 一方成势必速杀/"乱下堵不堵活三都没关系"
# v37 (2026-09-16 用户拍板): 默认【0 = 开局完全自由】(前3手全部由模型决策)
#   根因(09-16 实测查明): v36 用 3(前三手强制随机) → 模型【从未学过白1/黑2 决策】
#     → 白方第一手乱下 (实测 200 局: 距黑1 中位 14 格 / 64% 落 8 格以外 / 仅 2% 贴身)
#     → 自由开局下黑方 200/200 全胜、Rapfi 对局执白 11 手速败、只黑1随机白崩 1.1% 同一根因
#   铁律(沿用): 黑1/白1/黑2 必须【全随机或全决策】, 混合必崩 → v37 选全决策
#   兼容: GOMOKU_RANDOM_OPEN_MOVES=3 可恢复 v36 行为 (用于对照实验)
RANDOM_OPEN_MOVES = int(os.environ.get('GOMOKU_RANDOM_OPEN_MOVES', '0'))
# 2026-09-14 超短局降权 (用户拍板, 方案B: 纯训练侧, 不动自对弈 → 零风险)
#   背景: 提速后超短局(~40%)占比过高; 但超短局标签"真实"(非污染) → 只需降采样, 不必剔除
#   CUTOFF=12 (用户口径: 12步内白棋也能直接连五) | WEIGHT=0.3 (降权到 30%)
#   WEIGHT=1.0 = 原行为(均匀采样); 0.0 = 完全剔除
SHORT_GAME_CUTOFF = int(os.environ.get('GOMOKU_SHORT_CUTOFF', '12'))
SHORT_GAME_WEIGHT = float(os.environ.get('GOMOKU_SHORT_WEIGHT', '1.0'))
logging.info('超短局降权: CUTOFF={}步 WEIGHT={} (1.0=不降权原行为; 0.0=剔除)'.format(
    SHORT_GAME_CUTOFF, SHORT_GAME_WEIGHT))
# 防"配置断层": 启动时打印, 便于确认环境变量是否生效 (改 env 但未重启 → 行为与预期不符)
# ★ v42 (2026-10-10): 教练色【动态读 swap_state.json】(自动换边真相源), 回退 env
def _banner_coach_color():
    try:
        _p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'gomoku_checkpoints_v44', 'swap_state.json')
        with open(_p) as _f:
            return int(json.load(_f).get('coach_color', 0))
    except Exception:
        return int(os.environ.get('GOMOKU_COACH_COLOR', '0') or 0)
_bc = _banner_coach_color()


def _student_color_for_game(x):
    """v44 双色同训: 学生执色【按局交替 50/50】 (gid 偶=执黑 1 / 奇=执白 -1)。
    参数 x 可为 gid(int) 或工兵局文件路径 (从 文件名 _game_{gid}.pkl 解析 gid)。
    工兵与主进程共用此口径 ⇒ 每代每色各 ~50 局, 与文件名交替规则严格一致 ✓"""
    try:
        _gid = int(x)
    except (TypeError, ValueError):
        _s = os.path.basename(str(x))
        _gid = int(_s.split('_game_')[-1].replace('.pkl', ''))
    return 1 if _gid % 2 == 0 else -1


logging.info('v44 = 双色同训(单网黑白两视角全学) + 双55%门槛(执黑≥55% 且 执白≥55%, 近3代滑动平均); '
             '工兵学生执色按局交替 50/50, 教练按对手执色; COACH_COLOR={}(兼容留字段/不判色)'.format(_bc))
logging.info('随机开局: RANDOM_OPEN_MOVES={} (3=黑1白1黑2全随机 | 1=只黑1随机, 白1/黑2交模型决策)'.format(RANDOM_OPEN_MOVES))
# 2026-09-14 value 标签模式 (用户拍板: 直接上, 看效果)
#   来源: v33 记录 (Kimi 2026-09-05 评审 P0) —— "Q(s_t,a_t) 有结构性 off-by-one /
#         架构-目标失配 (value head 输出 V(s) 却喂 Q(s,a), 除非输入拼动作编码否则应喂 V)"
#   今日实测支撑: value 残差集中在"胜负未定的中间态"(CE 0.32), 战术激烈区反而最小
#     → 因为 Q(s,a) − V(s) = 该动作的 advantage, 中间态 advantage 波动最大
#   'point'       = Q(s,a_t) 单点 (原行为, 默认 → 不启用时行为 100% 不变)
#   'expectation' = V̂ = Σπ(a)Q(s,a) = MCTS root 期望回报 (消除选择偏差 / 降噪)
VALUE_TARGET_MODE = os.environ.get('GOMOKU_VALUE_TARGET', 'point')
logging.info('value 标签模式: {} ({})'.format(
    VALUE_TARGET_MODE,
    'Vhat=sum(pi*Q) root期望' if VALUE_TARGET_MODE == 'expectation' else 'Q(s,a)单点(原行为)'))
DIRICHLET_ALPHA = 0.05     # v32: v26 配置 α0.05 (α×K=11.25)     # v2: 固定0.03 (v1是 1/N ≈ 0.0045)

# ── v37 门控 (2026-09-16 用户拍板) ──────────────────────────────────────
#   规则: 每 GATE_EVERY 代, 候选(weights.pt) vs 冻结 best(weights_best.pt)
#         打 GATE_GAMES 局【配对赛】(同 seed 一局黑一局白; 开局完全自由 = 匹配训练分布)
#         胜率 ≥ GATE_THRESHOLD → 晋升: weights_best.pt := 候选
#         否则 → 方案B 回滚: weights.pt ← best + 主进程内存同步 + 清空回放池
#   ⚠️ 已知风险 (09-16 实测 200 局): 自由开局下黑方胜率 100% → 配对赛双方各执黑一半
#      → 总分恒 ≈50% → 门控【可能死锁】(永远过不了 55%)。已加死锁告警(只报警不改判)。
GATE_ENABLED = os.environ.get('GOMOKU_GATE', '1') == '1'
GATE_EVERY = int(os.environ.get('GOMOKU_GATE_EVERY', '10'))        # 每 N 代门控一次
GATE_GAMES = int(os.environ.get('GOMOKU_GATE_GAMES', '100'))       # 配对赛局数(须偶数)
GATE_THRESHOLD = float(os.environ.get('GOMOKU_GATE_THRESH', '0.55'))
GATE_SHARDS = int(os.environ.get('GOMOKU_GATE_SHARDS', '16'))  # 2026-09-25: 4→16 (并发数才是瓶颈)
# 2026-09-25 用户拍板 A: 门控 sim 800 → 400 (省一半工作量, 门控 133 → ~66 分钟)
#   为什么公平: 门控是【候选 vs best 同条件对比】, 双方同 sim 即可, 不需要绝对棋力
#   先例: v36 的 36对全矩阵 ELO 就是 sim=100 跑的, 结论后来被 sim=800 全部证实 ✓
#   注: 训练仍用 sim=800 (那是训练底线, 不动 ✓)
GATE_SIM = int(os.environ.get('GOMOKU_GATE_SIM', '400'))
GATE_ROLLBACK = os.environ.get('GOMOKU_GATE_ROLLBACK', '1') == '1'  # 方案B: 失败回滚
# 2026-09-17 用户拍板: REJECT 时【普通情况不清空回放池】
#   理由: 弱策略产出的棋局是合法 off-policy 数据(标签正确), 且 100局55%阈值有 ~18% 误判率
#         → 清空会白扔 10 代数据 + 触发重新 warmup 5 代(每次多花 ~3.8h)
#   仅当【标签被污染】(V̂ 类事故) 才清空 —— 判据取自本代健康体检
GATE_PURGE_ON_CORRUPT = os.environ.get('GOMOKU_GATE_PURGE', '1') == '1'
GATE_CORRUPT_V = float(os.environ.get('GOMOKU_GATE_CORRUPT_V', '0.5'))      # |空盘value| 超此值 = 污染
GATE_CORRUPT_ACT = float(os.environ.get('GOMOKU_GATE_CORRUPT_ACT', '1.0'))  # 激活率(%) 低于此值 = 污染
logging.info('v37 门控: {} | 每 {} 代 | {} 局配对 | 阈值 {:.0%} | 回滚 {}'.format(
    'ON' if GATE_ENABLED else 'OFF', GATE_EVERY, GATE_GAMES, GATE_THRESHOLD,
    'ON(方案B)' if GATE_ROLLBACK else 'OFF'))
REPLAY_POOL_SIZE = 5       # 保留最近5代数据混合训练
VALUE_LOSS_WEIGHT = 1.0    # v33 (2026-09-05): value 标签=搜索Q软标签, MSE 天然量级小;
                           #   若 value 仍抢梯度可调 0.3-0.5 (用户口径: 先学每步好坏, value 配角)

# v41 (2026-10-09): 【梯度检查点 GRAD_CKPT】env GOMOKU_GRAD_CKPT
#   背景: COLOR_SYM=2 让 value batch 样本数翻倍 ⇒ 激活显存峰值翻倍 ⇒ 工兵 OOM
#   修法: 残差块 forward 用 torch.utils.checkpoint 包住 ⇒ 前向不保存中间激活,
#         反向重算激活 ⇒ 显存峰值大降, 而【数学完全等价】(损失与梯度不变)
#         ⇒ 不破坏单变量 (只换显存/算力, 不改数学)
#   开关: GOMOKU_GRAD_CKPT=1 启用 (0/未设 = 关, 行为与不加时逐字节一致)
#   仅 train 模式启用; eval/推理时跳过 (避免 no_grad 下 checkpoint 报错)
#   ★ 只包【残差块】; ✗ 不包整个网络 (无效果), ✗ 不动 policy/value 头
# ─────────────────────────────────────────────
import contextlib
import torch.utils.checkpoint as _grad_ckpt
GRAD_CKPT = os.environ.get('GOMOKU_GRAD_CKPT', '0').strip() == '1'
logging.info('梯度检查点 GRAD_CKPT: {} (env GOMOKU_GRAD_CKPT)'.format('开' if GRAD_CKPT else '关'))


def _grad_ckpt_context(block):
    """checkpoint 的 context_fn: 前向无副作用; 反向【重算】结束后立即把 BatchNorm 的
    running_mean/running_var/num_batches_tracked 还原 ⇒ BN running stats 每次 learn
    只更新一次 (否则反向重算会再更新一次 ⇒ 与 CKPT=0 训练数学不等价)。
    只处理 BN buffer, 不影响梯度/损失; 逐位等价 (等价性测试证明 Δ=0)。"""
    @contextlib.contextmanager
    def _restore_bn():
        bns = [m for m in block.modules()
               if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        snap = [(m.running_mean.clone(), m.running_var.clone(),
                 m.num_batches_tracked.clone()) for m in bns]
        try:
            yield
        finally:
            for m, (rm, rv, nb) in zip(bns, snap):
                m.running_mean.copy_(rm)
                m.running_var.copy_(rv)
                m.num_batches_tracked.copy_(nb)
    return (contextlib.nullcontext(), _restore_bn())

# ─────────────────────────────────────────────
# 网络（与 v1 相同）
# ─────────────────────────────────────────────
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class AlphaZeroNet(nn.Module):
    """v6 网络: opt/v1 式极简头 (3×3 感受野直出), 主干 8×ResBlock
    policy: conv(128→1, k3) → bn → log_softmax 直出 (225)
    value:  conv(128→1, k3) → bn → fc(225→1) 单层
    ⚠️ policy 输出 log 概率 (v2 learn 需要), 结构与 v1/opt 相同
    """
    def __init__(self, conv_channels=128, residual_blocks=8, board_size=15):
        super().__init__()
        self.board_size = board_size
        self.conv_input = nn.Conv2d(INPUT_CHANNELS, conv_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(conv_channels)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(conv_channels, conv_channels, conv_channels)
            for _ in range(residual_blocks)
        ])
        # Policy head (v1式): 3×3 直出单通道
        # v20 (2026-08-27, 用户手术 v2): policy 头加隐藏层 — 保留式双分支
        # 旧识别模块(继承): conv(128→1, 3×3, pad=1) → BN(1) → ReLU  [it234 权重完整继承]
        # 新通道(随机):     conv(128→1, 1×1) → BN(1) → ReLU            [新增识别通道]
        # 组合层(随机):     cat 2通道 → flatten(450) → Linear(450→225) → log_softmax
        # 原 v19 极简头直出 (~1.2K 参, 无跨位置组合); 新头 ~103K 参, Linear 提供全局组合
        self.policy_conv = nn.Conv2d(conv_channels, 1, 3, padding=1, bias=False)   # 继承 (识别模块)
        self.policy_bn = nn.BatchNorm2d(1)                                        # 继承
        self.policy_conv2 = nn.Conv2d(conv_channels, 1, 1, padding=0, bias=False) # 随机 (新通道)
        self.policy_bn2 = nn.BatchNorm2d(1)                                       # 随机
        self.policy_fc = nn.Linear(board_size * board_size * 2, board_size * board_size)  # 随机 (组合层)
        # Value head (v19, 2026-08-26): conv(128→1,3×3) + 隐藏层 fc(225→128) → fc(128→1)
        # 主流 AlphaZero/Leela 式带隐藏层 value 头 (v18 为 225→1 单 FC 直出)
        # value_conv/value_bn 与 v18 同形状 → 继承权重直接复用, 仅 fc 层为新结构
        self.value_conv = nn.Conv2d(conv_channels, 1, 3, padding=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            # v41: 梯度检查点 (仅 train 且开关开时); eval/推理走原路径
            if GRAD_CKPT and self.training:
                x = _grad_ckpt.checkpoint(block, x, use_reentrant=False,
                                          context_fn=lambda b=block: _grad_ckpt_context(b))
            else:
                x = block(x)
        # Policy head: 1通道 → (N, 1, 15, 15) → (N, 225) → log_softmax
        # policy (v20 保留式): 旧3×3识别分支 + 新1×1通道 → cat → flatten(450) → Linear(450→225) → log_softmax
        p1 = F.relu(self.policy_bn(self.policy_conv(x)))     # (N,1,15,15) 继承识别器
        p2 = F.relu(self.policy_bn2(self.policy_conv2(x)))   # (N,1,15,15) 新通道
        p = torch.cat([p1, p2], dim=1)                       # (N,2,15,15)
        p = p.reshape(p.shape[0], -1)                        # (N, 450)
        p = F.log_softmax(self.policy_fc(p), dim=1)
        # Value head (v19): conv → flatten(225) → fc1(225→128) → relu → fc2(128→1) → tanh
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.shape[0], -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v


def to_tensor(x):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device).unsqueeze(1)


def to_numpy(x):
    return x.detach().cpu().numpy()


# ─────────────────────────────────────────────
# Agent（v2 新增 temperature / alpha / 开局约束）
# ─────────────────────────────────────────────
class AlphaZeroAgent:
    def __init__(self, env, batches=1, batch_size=4096,
                 kwargs=None, sim_count=200,
                 c_init=1.25, c_base=19652., prior_exploration_fraction=0.25,
                 lr=0.001, disable_compile=False):
        if kwargs is None:
            kwargs = {}
        self.env = env
        self.board = np.zeros_like(env.board)
        self.batches = batches
        self.batch_size = batch_size

        conv = kwargs.get('conv_filters', [128])[0]
        resid = len(kwargs.get('residual_filters', [[128, 128]] * 8))
        raw_net = AlphaZeroNet(conv_channels=conv, residual_blocks=resid).to(device)

        self.raw_net = raw_net
        # v6: 恢复 torch.compile 提速 (速度是 v2 的 3.5-4.5 倍差距主因)
        # 修复: learn 的 scaler 分支已补 zero_grad (原缺失导致 CUDAGraph 梯度冲突)
        if not disable_compile and device.type == 'cuda':
            try:
                self.net = torch.compile(raw_net, mode='reduce-overhead', fullgraph=False)
                logging.info('torch.compile 已启用 (reduce-overhead)')
            except Exception as e:
                logging.warning('torch.compile 失败: {}, 使用原始网络'.format(e))
                self.net = raw_net
        else:
            self.net = raw_net
            logging.info('torch.compile 禁用或不可用，使用原始网络')

        # v12: BN 参数(weight/bias)单独分组, weight_decay=0 (防 BN γ 收缩死亡螺旋);
        #      其余参数保持 v8 原样 weight_decay=1e-4。其余训练配置与 v8 完全一致。
        bn_params = []
        other_params = []
        for name, param in self.raw_net.named_parameters():
            if param.requires_grad:
                if 'bn' in name and ('weight' in name or 'bias' in name):
                    bn_params.append(param)
                else:
                    other_params.append(param)
        self.optimizer = optim.Adam([
            {'params': bn_params, 'weight_decay': 0.0},
            {'params': other_params, 'weight_decay': 1e-4},
        ], lr=lr)
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        # v19 (2026-08-26, Kimi 三步走): 阶段1 冻结主干+Policy, 只训新 Value 头
        self.freeze_policy_only = False
        self.freeze_value_only = False

        self.reset_mcts()
        self.sim_count = sim_count
        self.c_init = c_init
        self.c_base = c_base
        self.prior_exploration_fraction = prior_exploration_fraction
        # v2: 开局中心约束（search 与 decide 共用）
        self.center_only = False
        n = self.board.shape[0]
        center = n // 2
        self.center_mask = np.zeros((n, n), dtype=bool)
        for r in range(n):
            for c in range(n):
                self.center_mask[r, c] = (abs(r - center) + abs(c - center) <= 4)

        # ── v36 攒批推理钩子 (2026-09-13) ──
        # 环境变量 GOMOKU_INFER_SOCK 存在 → predict_inference 走远程攒批服务;
        # 不设 → 行为与原来 100% 一致 (零影响)
        self._remote = None
        _sock = os.environ.get('GOMOKU_INFER_SOCK')
        if _sock:
            try:
                import importlib.util as _ilu
                _here = os.path.dirname(os.path.abspath(__file__))
                _spec = _ilu.spec_from_file_location(
                    'remote_predict_v37',
                    os.path.join(_here, 'scripts', 'remote_predict_v37.py'))
                _rpm = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_rpm)
                self._remote = _rpm.RemotePredictor(_sock)
                logging.info('✅ 已启用远程攒批推理: %s', _sock)
            except Exception as e:
                logging.error('❌ 远程推理初始化失败, 回退本地推理: %s', e)
                self._remote = None

    def reset_mcts(self):
        def zero_board_factory():
            return np.zeros_like(self.board, dtype=float)
        self.q = collections.defaultdict(zero_board_factory)
        self.count = collections.defaultdict(zero_board_factory)
        self.policy = {}
        self.valid = {}
        self.winner = {}

    @torch.inference_mode()
    def predict_inference(self, board_batch, last_moves=None, players=None):
        """board_batch: (N,15,15) canonical; last_moves: (N,2) 或 None → 内部转 4-plane
        players: (N,) 或标量 当前玩家颜色(1=黑/-1=白), 用于 ch2; None 默认黑 (兼容旧调用)
        ⚠️ 推理固定 eval: 只在 training 模式时切一次 (省每次 predict 的 eval() 全模块遍历开销)"""
        # v36 攒批推理: 有远程后端则走 socket (返回形状/语义与本地完全一致)
        if getattr(self, '_remote', None) is not None:
            return self._remote.predict(board_batch, last_moves, players)
        if self.net.training:
            self.net.eval()
        b = np.asarray(board_batch)
        n = b.shape[0]
        bs = b.shape[1]
        inp = np.zeros((n, INPUT_CHANNELS) + b.shape[1:], dtype=np.float32)
        inp[:, 0] = b > 0          # ch0 当前方
        inp[:, 1] = b < 0          # ch1 对手
        # ── 🚨 本权重的四通道是 junxiaosong v4 约定，不是参考版 v24 约定 ──
        #   v4 (game.py Board.current_state):
        #       ch2 = 最后一手位置      ch3 = 执色常量(轮到黑=1.0, 轮到白=0.0)
        #       return square_state[:, ::-1, :]     ← 整块垂直翻转
        #   参考版 v24:
        #       ch2 = 恒 1.0            ch3 = 最后一手位置        不翻转
        #   通道 2/3 互换 + 缺翻转 ⇒ 网络把「上一手」当「执色」读，
        #   value 头直接饱和(实测 value 热力图全 +1、全盘 std=0)，
        #   policy 也散(对弈第一手下角落 D1)。
        if last_moves is not None:
            for i, lm in enumerate(last_moves):
                if lm is not None:
                    inp[i, 2, int(lm[0]), int(lm[1])] = 1.0
        if players is not None:
            pl = np.asarray(players)
            if pl.ndim == 0:
                inp[:, 3] = 1.0 if int(pl) == 1 else 0.0
            else:
                for i in range(min(n, pl.shape[0])):
                    inp[i, 3] = 1.0 if int(pl[i]) == 1 else 0.0
        # 与 v4 的 current_state 一致：整块沿棋盘行方向翻转
        # （ascontiguousarray：负步长视图 torch 不接受）
        inp = np.ascontiguousarray(inp[:, :, ::-1, :])

        tensor = torch.as_tensor(inp, device=device)
        with torch.cuda.amp.autocast():
            p, v = self.net(tensor)
        # policy 输出是在【翻转坐标系】上的 225 维 → 翻回真实坐标，
        # 这样上层 search/decide 里 idx→(r,c) 的既有假设完全不用改。
        pc = to_numpy(p.float())
        pc = np.ascontiguousarray(pc.reshape(n, bs, bs)[:, ::-1, :].reshape(n, bs * bs))
        return pc, to_numpy(v.float())

    def predict(self, board_batch, last_moves=None, players=None):
        return self.predict_inference(board_batch, last_moves, players)

    def _value_target(self, s, location):
        """2026-09-14: value 训练标签 (供 self_play 存 pkl 的 qa 字段)
        point       = Q(s,a_t) 单点 (原行为, VALUE_TARGET_MODE 非 expectation 时)
        expectation = V̂ = Σπ(a)Q(s,a) = MCTS root 期望回报

        动机 (v33 记录 + 2026-09-14 实测):
          value head 输入是【局面 s】(不含动作 a), 用 Q(s,a_t) 当标签存在
          "架构-目标失配": Q(s,a)−V(s) = 该动作的 advantage, 直接进标签。
          实测残差集中在"胜负未定的中间态" → advantage 噪声主源。
        成本: count[s]/q[s] 已在 search() 维护 → 只多一次 225 维加权求和, 无需重搜。
        回退: 随机开局点(未走 MCTS) count 为空 → 自动回退原行为。
        """
        if VALUE_TARGET_MODE != 'expectation':
            return float(self.q[s][location])
        counts = self.count[s]
        tot = float(counts.sum())
        if tot <= 0:
            return float(self.q[s][location])
        pi = counts / tot
        return float((pi * self.q[s]).sum())

    def decide(self, observation, greedy=False, return_prob=False, noise=True,
               temperature=1.0, force_center=False, move_no=0, last_move=None,
               return_q=False, random_first_ring=False, random_open3=False):
        """v2: 支持 temperature 采样 + 开局中心约束
        temperature=1.0: 按概率采样（探索）
        temperature→0:   贪心（接近 argmax）
        force_center: 开局强制落子在天元附近（训练早期防止边角开局）
        last_move: 本局面刚落的子 (r,c)，作为 v8 输入 plane 3（None 则全 0）
        """
        board, player = observation
        canonical_board = player * board
        s = canonical_board.tobytes()
        # ⚠️ 2026-08-18 count污染修复: 每步固定搜 sim_count 次 (论文式, 树可复用但搜索量不打折)
        # 原 while count[s].sum() < sim_count: 当 count 被父节点搜索灌满(>=800)时跳过搜索,
        # 直接用被污染的分布采样 -> 杀点不锁定/训练目标杀点信号弱 (白棋杀棋不精准)
        # ⚠️ 2026-08-29 noise 注入修正(论文式): 根先验加一次 noise, 800 sim 共用
        # 原实现每次 sim 都重新采样 noise (explore 放大 800 倍, value 极端化疑似病根)
        # 做法: noise 在 decide 层生成一次, 写入根先验 policy[s]; search 不再重新采样
        if noise:
            if s not in self.policy:
                # 首次需先推理缓存根节点先验 (search 内部会做, 但这里要拿到 policy 才能混 noise)
                self.search(canonical_board, prior_noise=False, last_move=last_move, player=player)
            alpha = DIRICHLET_ALPHA
            nv = np.random.gamma(alpha, 1., board.shape)
            nv *= self.valid[s]
            nv /= nv.sum()
            self.policy[s] = (1. - self.prior_exploration_fraction) * self.policy[s] \
                             + self.prior_exploration_fraction * nv
        for _ in range(self.sim_count):
            self.search(canonical_board, prior_noise=False, last_move=last_move, player=player)
        prob = self.count[s] / self.count[s].sum()

        # ── v2: 开局中心约束 ──
        if force_center:
            center = board.shape[0] // 2
            prob = prob.copy()
            for r in range(board.shape[0]):
                for c in range(board.shape[1]):
                    if abs(r - center) + abs(c - center) > 4:
                        prob[r, c] = 0.0
            if prob.sum() > 0:
                prob /= prob.sum()

        # ── v2: temperature 采样 ──
        # 2026-08-23 修复: temp<=1e-2 直接 argmax。
        # 原因: temp=1e-3 时 exp(log_p/temp)=exp(log_p*1000) 全下溢→soft全0→0/0=NaN 崩溃
        # 语义: 1e-3 本就是"接近贪心", argmax 等价且数值安全
        if temperature <= 1e-2 or greedy:
            location_index = np.argmax(prob.reshape(-1))
        else:
            # 按 temperature 软化后的分布采样 (temp 较大时 exp 不消失, 安全)
            log_p = np.log(prob + 1e-12)
            soft = np.exp(log_p / temperature)
            soft /= soft.sum()
            location_index = np.random.choice(prob.size, p=soft.reshape(-1))

        location = np.unravel_index(location_index, prob.shape)
        # v35 (2026-09-05): 黑首手随机让一手 — move_no==0 时 50% 概率强制落"距天元5-8子"环带
        # 动机: 与 v34 同(白胜样本供给防数据单边化螺旋), 但 v34 4纯角让手太狠 →
        # 让手局白 96%(黑后手翻盘仅 4%)→ "后手成功防守"经验真空 → 正常局白依然崩(7-10%)。
        # 环带让得较轻: 目标让手局白 ~65-75% → 黑有 25-35% 翻盘面(长局防守反击) →
        # 产出后手成功样本, canonical 翻转 = 正常局白防守样本(治本通道)。
        # 防毒(同 v34): prob列仍=搜索 count 分布(让点未被搜索则 prob≈0, policy 不学让点);
        # qa = 让点搜索Q; 首手样本照常记录(不旁路, 棋谱完整性/胜利检测不受影响)。
        if random_first_ring and move_no == 0 and np.random.rand() < 0.5:
            ring = _handicap_ring(board.shape[0])
            location = ring[np.random.randint(len(ring))]
            while board[location]:  # 保险: 让点被占则换 (首手必空盘, 防未来改动)
                location = ring[np.random.randint(len(ring))]

        # v36 (2026-09-06): 随机开局 — 前3手(黑1白1黑2) 100% 强制落"中心7x7"(距天元<=3子, 49点)未占随机点
        # 动机: v35 只随机黑1(环带让手) 10 代正常局白不爬(it17-18 5%/4%) + PK ring 赛 it16 vs it4
        #   10:10 零增益 = 黑必胜靠 黑1+黑2+黑3 连贯进攻链, 只断黑1 不够 → v36 断前两手(黑1黑2全随机)
        #   + 白1也随机(防白捡"随机后先手"便宜反向失衡); 100%全随机(用户拍板, 先验证机制再谈混合)。
        # 范围=中心7x7 矩形(|dr|<=3 且 |dc|<=3, 49点, 用户口径"中心7x7 49点"; 非 d²<=25 圆 81点)。
        # 防毒(同 v35 ring): prob列仍=搜索 count 分布(随机点未被搜索则 prob≈0, policy 不学随机点;
        #   target=对"含随机前缀局面"的搜索应对 → policy 学泛化应对而非固定开局);
        # qa/记录照常(不旁路, 棋谱完整性/胜利检测不受影响)。
        # v36b (2026-09-07): 黑1 改全棋盘 13x13(距边1格 r,c∈[1,13], 169点)随机 — 白1/黑2 仍中心7x7
        # 动机: 中心7x7 仅 49 点, 黑背熟全部开局模式 → 白胜率下行(it22 27% / it28 22% 破30%失衡线)
        #   → 黑1 扩 169 点(71% 落中心7x7 外) = 黑背不住 + 黑1 常落外围天然弱化先手; 白1黑2 不动(白随机后先手反失衡风险仍在)
        # v36c (2026-09-07): 黑1 13x13 → 9x9(距天元<=4, r,c∈[3,11], 81点) — v36b 过调回调
        #   实测 it34-43 白胜率平台 56-64%(让手列=黑1落环带局白 68-75% 居高) = 黑1 落外围天然弱 → 让手太狠
        # 2026-09-14: 3 → RANDOM_OPEN_MOVES (环境变量控制; 默认3=原行为)
        if random_open3 and move_no < RANDOM_OPEN_MOVES:
            c0 = board.shape[0] // 2
            # 2026-09-14: 黑1 保持 9x9(81点, 距天元<=4) — v36c 口径不变 (用户拍板)
            lim = 4 if move_no == 0 else 3   # 黑1: 9x9; 白1/黑2: 7x7 (RANDOM_OPEN_MOVES=1 时后者不生效)
            pts = [(r, cc) for r in range(board.shape[0]) for cc in range(board.shape[1])
                   if abs(r - c0) <= lim and abs(cc - c0) <= lim and not board[r, cc]]
            location = pts[np.random.randint(len(pts))]
        # v33: return_q=True → 附 value 训练标签 (当前玩家视角, 供 value 训练)
        # 2026-09-14: 改走 _value_target (point=Q(s,a_t) 原行为 / expectation=V̂=ΣπQ)
        if return_prob:
            if return_q:
                return location, prob, self._value_target(s, location)
            return location, prob
        return location

    def set_freeze_policy_only(self, freeze):
        """v20 (2026-08-27, 用户手术): freeze=True → 冻结主干+Value 头, 只训 policy 头参数
        freeze=False → 解冻全网络。只改 requires_grad, 不重建 optimizer
        (optimizer 参数引用固定, requires_grad=False 的参数梯度为 None 自动跳过;
         避免重建导致 GradScaler 状态丢失)。"""
        if freeze == self.freeze_policy_only:
            return
        net = self.raw_net if (hasattr(self, 'raw_net') and self.raw_net is not None) else self.net
        for name, param in net.named_parameters():
            if freeze:
                param.requires_grad = name.startswith('policy_')
            else:
                param.requires_grad = True
        self.freeze_policy_only = freeze
        logging.info('v20 冻结切换: {} (policy_only={})'.format(
            '冻结主干+Value, 只训 Policy 头' if freeze else '解冻全网络', freeze))

    def set_freeze_value_only(self, freeze):
        """v23 (2026-08-29, 重置value头后冻结期): freeze=True → 冻结主干+Policy 头, 只训 value 头参数
        freeze=False → 解冻全网络。只改 requires_grad, 不重建 optimizer
        (optimizer 参数引用固定, requires_grad=False 的参数梯度为 None 自动跳过;
         避免重建导致 GradScaler 状态丢失)。"""
        if freeze == self.freeze_value_only:
            return
        net = self.raw_net if (hasattr(self, 'raw_net') and self.raw_net is not None) else self.net
        for name, param in net.named_parameters():
            if freeze:
                param.requires_grad = name.startswith('value_')
            else:
                param.requires_grad = True
        self.freeze_value_only = freeze
        logging.info('v23 冻结切换: {} (value_only={})'.format(
            '冻结主干+Policy, 只训 Value 头' if freeze else '解冻全网络', freeze))

    def learn(self, dfs):
        self.net.train()
        # v8: 每个 df 补 last_move 列 (从相邻 board diff 推, 不改原数据格式)
        dfs_aug = []
        for d in dfs:
            d = d.reset_index(drop=True)
            last_moves = []
            boards = d['board'].values
            last_moves.append(None)
            for i in range(1, len(d)):
                prev = np.asarray(boards[i - 1])
                cur = np.asarray(boards[i])
                diff = np.argwhere(prev != cur)
                last_moves.append(tuple(diff[0]) if len(diff) else None)
            d['last_move'] = last_moves
            # 2026-09-14: 标记超短局 (供降权采样; 口径见 SHORT_GAME_CUTOFF)
            d['_short'] = (len(d) <= SHORT_GAME_CUTOFF)
            dfs_aug.append(d)
        df = pd.concat(dfs_aug).reset_index(drop=True)
        # ── v44 双色同训【核心】: 不再剔除教练行 ⇒ 黑白两个视角的行【全量学】──────
        #   v42 用 df[df['player'] != _coach_c] 剔掉教练色 ⇒ 网只认识自己那一半,
        #   MCTS 树里一半节点(对方该下)落在分布外 ⇒ 搜索不敢相信叶子 ⇒ 算不出强杀 ✗
        #   v44: 单网双色, 学生每代执黑一半/执白一半, 所有行都进训练 (全量学) ✓
        logging.info('v44 双色同训: 全量学 {} 条样本 (黑白两视角都学, 无剔行)'.format(len(df)))
        # v33: 新池必有 qa 列(该步搜索Q软标签); 旧数据无 qa 时回退全局结果±1(兜底兼容)
        if 'qa' not in df.columns:
            df['qa'] = (df['player'] * df['winner']).astype(np.float32)
        losses = {'policy_loss': [], 'value_loss': [], 'total_loss': []}
        for batch in range(self.batches):
            if SHORT_GAME_WEIGHT < 1.0:
                # 2026-09-14: 超短局降权采样 (权重 0.3 → 相对占比约 15%, 因步数天然更少)
                w = np.where(df['_short'].values, SHORT_GAME_WEIGHT, 1.0).astype(np.float64)
                w /= w.sum()
                indices = np.random.choice(len(df), size=self.batch_size, p=w)
            else:
                indices = np.random.choice(len(df), size=self.batch_size)
            players, boards, probs, qs = (
                np.stack(df.loc[indices, field])
                for field in ['player', 'board', 'prob', 'qa']
            )
            last_moves = df.loc[indices, 'last_move'].values
            canonical_boards = players[:, np.newaxis, np.newaxis] * boards
            # v33: value 标签 = qa 该步搜索Q软标签 (当前玩家视角, 已含方向, 不再乘 players)
            vs = qs[:, np.newaxis]

            # v8: 1ch canonical (N,15,15) → 4-plane input (N,4,15,15)
            inp = np.zeros((len(players), INPUT_CHANNELS) + boards.shape[1:],
                           dtype=np.float32)
            inp[:, 0] = canonical_boards > 0
            inp[:, 1] = canonical_boards < 0
            # v24: ch2 恒定 1, 无执色信息 (撤销 v18 编码)
            inp[:, 2] = 1.0
            for i, lm in enumerate(last_moves):
                if lm is not None:
                    inp[i, 3, int(lm[0]), int(lm[1])] = 1.0

            # ── v39 颜色对调增强 (视角对调) ── GOMOKU_COLOR_SYM
            #   0 = 关闭 (默认, 100% 原行为, 逐字节不变)
            #   1 = 严格规格: 追加 (board取负 / qa取负 / prob不变) 样本, 与原样本同权
            #   2 = 数学修正版: 颜色对调样本【只进 value 损失】, policy 损失只吃原样本
            #   数学依据 (颜色对调 ⇒ canonical 整体反号 ⇒ 网络输入 = -canonical):
            #     · value : V(-c) = -V(c)  ✓ 零和(同一棋盘对面门将) + 颜色对称 ⇒ 取负正确
            #     · policy: π(-c) ≠ π(c)   ✗ 同一棋盘上"该谁走"变了, 最优着法随之变;
            #       实测 L1(π,π_-c)=1.71 ≈ 随机基线 1.93 ⇒ 规格"prob 不变"是错的
            _colorsym = os.environ.get('GOMOKU_COLOR_SYM', '0').strip()
            _is_flip = None
            if _colorsym in ('1', '2'):
                boards2 = -boards                      # 棋子颜色互换 (ch0/ch1 随 >0/<0 自动换位)
                canonical2 = players[:, np.newaxis, np.newaxis] * boards2   # = -canonical_boards
                inp2 = np.zeros_like(inp)
                inp2[:, 0] = canonical2 > 0
                inp2[:, 1] = canonical2 < 0
                inp2[:, 2] = 1.0
                inp2[:, 3] = inp[:, 3]                 # ch3 last_move 位置不变 (颜色对调不改落点)
                vs2 = -vs                              # 从对面看 ⇒ 价值取负
                probs2 = probs                         # =1 复用同一 prob; =2 靠掩码屏蔽策略项
                inp = np.concatenate([inp, inp2])
                probs = np.concatenate([probs, probs2])
                vs = np.concatenate([vs, vs2])
                _is_flip = np.concatenate([np.zeros(len(boards), bool),
                                           np.ones(len(boards), bool)])

            # 8 方向对称增强 (v8 修正: v6 的 fliplr 对 4D 无效, 反射分支实际失效,
            # 这里用 np.flip(axis=-1) 正确实现水平翻转)
            boards_aug = [inp]
            probs_aug = [probs]
            vs_aug = [vs]
            for reflection in [False, True]:
                rot_start = 1 if not reflection else 0
                for rot_k in range(rot_start, 4):
                    b = np.flip(inp, axis=-1) if reflection else inp
                    p = np.flip(probs, axis=-1) if reflection else probs
                    boards_aug.append(np.rot90(b, rot_k, axes=(2, 3)))
                    probs_aug.append(np.rot90(p, rot_k, axes=(1, 2)))
                    vs_aug.append(vs)
            if _is_flip is not None:                   # 掩码随几何增强复制 8 份 (块内序恒为 原+翻转)
                _is_flip = np.tile(_is_flip, len(boards_aug))

            inp = np.concatenate(boards_aug)
            probs = np.concatenate(probs_aug)
            vs = np.concatenate(vs_aug)

            x = torch.as_tensor(inp, device=device)
            with torch.cuda.amp.autocast():
                pred_p, pred_v = self.net(x)
                # pred_p: (batch, 225) → reshape为 (batch, 15, 15) 与 probs 对齐
                bs = pred_p.shape[0]
                board_sz = int(math.sqrt(pred_p.shape[1]))
                pred_p_b = pred_p.reshape(bs, board_sz, board_sz)
                _prow = (torch.as_tensor(probs, device=device) *
                         pred_p_b).sum(dim=(1, 2))
                if _colorsym == '2' and _is_flip is not None:
                    # 修正版: 颜色对调样本无正确 policy 标签 → 策略损失只用原样本
                    _m = ~torch.as_tensor(_is_flip, device=device)
                    policy_loss = -(_prow * _m).sum() / _m.sum().clamp(min=1)
                else:
                    policy_loss = -_prow.mean()
                value_loss = F.mse_loss(pred_v, torch.as_tensor(vs, device=device))
                total_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()   # v6 修复: scaler 分支也要清梯度 (原代码缺失, 梯度跨batch累积)
            else:
                total_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            losses['policy_loss'].append(policy_loss.item())
            losses['value_loss'].append(value_loss.item())
            losses['total_loss'].append(total_loss.item())

        # ── 变更② (2026-10-05, 用户批准): learn() 尾部回收临时张量 + 显存缓存 ──
        #   纯资源回收: del 减引用 + empty_cache 归还未使用的缓存块
        #   不改任何数值/梯度/权重 (empty_cache 只释放"已分配但未使用"的块)
        #   try/except 保护: 任何异常都不中断训练
        try:
            del x, inp, probs, vs, pred_p, pred_v, pred_p_b, canonical_boards
            del players, boards, qs, indices, last_moves
        except Exception:
            pass
        try:
            del boards2, canonical2, inp2, probs2, vs2, _is_flip, _m
        except Exception:
            pass
        try:
            del boards_aug, probs_aug, vs_aug
        except Exception:
            pass
        try:
            import gc as _gc
            _gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        self.reset_mcts()
        result = {}
        if losses['policy_loss']:
            result['policy_loss'] = sum(losses['policy_loss']) / len(losses['policy_loss'])
        if losses['value_loss']:
            result['value_loss'] = sum(losses['value_loss']) / len(losses['value_loss'])
        if losses['total_loss']:
            result['total_loss'] = sum(losses['total_loss']) / len(losses['total_loss'])
        return result

    def search(self, board, prior_noise=False, last_move=None, player=None):
        s = board.tobytes()

        if s not in self.winner:
            # canonical board: 1=当前玩家, -1=对手
            # get_winner((board, BLACK)) 检测"值为1的棋子五连"→返回1 = 当前玩家胜
            # 恰好是"当前玩家视角"(与 value 标签 players*winners 一致) → 无需翻转
            self.winner[s] = self.env.get_winner((board, BLACK))
        if self.winner[s] is not None:
            return self.winner[s]

        if s not in self.policy:
            # v18: 若未传 player (外部直接调 search), 用 canonical 无法推颜色 → 默认黑(与旧行为一致)
            p_color = player if player is not None else 1
            pis, vs = self.predict(
                board[np.newaxis],
                last_moves=np.array([last_move]) if last_move is not None else None,
                players=p_color)
            pi = np.exp(np.asarray(pis[0]))  # ⚠️ 修复: net输出log概率, 转概率 (2026-08-10 发现search用log当概率致先验反转)
            if pi.shape != board.shape:
                pi = pi.reshape(board.shape)
            v = float(np.asarray(vs[0]).reshape(-1)[0])
            valid = (board == 0).astype(np.float32)
            masked_pi = pi * valid
            total_masked_pi = np.sum(masked_pi)
            if total_masked_pi <= 0:
                masked_pi = valid
                total_masked_pi = np.sum(masked_pi)
            self.policy[s] = masked_pi / total_masked_pi
            self.valid[s] = valid
            return v

        count_sum = self.count[s].sum()
        coef = (self.c_init + np.log1p((1 + count_sum) / self.c_base)) * \
               math.sqrt(count_sum) / (1. + self.count[s])
        if prior_noise:
            # v2: alpha 固定 0.03（标准 AlphaZero），v1 是 1/N ≈ 0.0045
            alpha = DIRICHLET_ALPHA
            noise = np.random.gamma(alpha, 1., board.shape)
            noise *= self.valid[s]
            noise /= noise.sum()
            prior = (1. - self.prior_exploration_fraction) * \
                    self.policy[s] + \
                    self.prior_exploration_fraction * noise
        else:
            prior = self.policy[s]
        ub = np.where(self.valid[s], self.q[s] + coef * prior, np.nan)
        location_index = np.nanargmax(ub)
        location = np.unravel_index(location_index, board.shape)

        (next_board, next_player), _, _, _ = self.env.next_step(
            (board, BLACK), np.array(location))
        next_canonical_board = next_player * next_board
        # v18 修复 (2026-08-22): next_player 是 boardgame2 的视角翻转标记(传入BLACK恒=-1),
        # 不是绝对颜色! ch2 需要每层翻转的绝对颜色 → 传 -player (不能传 next_player)
        # 否则黑视角层被当成白, 深层 MCTS 评估全错
        next_v = self.search(next_canonical_board, last_move=location, player=-player)
        # v1 原始写法 (server_sync/scripts/Gomoku-v0_pt_opt.py 一致): 已验证正确
        v = next_player * next_v

        self.count[s][location] += 1
        self.q[s][location] += (v - self.q[s][location]) / \
                               self.count[s][location]
        return v


# ─────────────────────────────────────────────
# 自我对弈（v2: temperature 调度 + 开局约束）
# ─────────────────────────────────────────────
def self_play(env, agent, return_trajectory=False, verbose=False,
              visualize=False, temperature_schedule=True, force_center=False,
              coach_agent=None, coach_color=None):
    # ── v38: 教练学习 (冻结对手) ──────────────────────────────────
    #   coach_agent: 冻结的教练网络对象 (本地推理, 不带 _remote, 不走共享服务)
    #   coach_color: 教练执的颜色 (1=黑 / -1=白); 该色走子由教练决策, 其余归 agent(学生)
    #   ⚠️ 教练的走子行会写进 trajectory, 但【训练时被剔除】(见 learn)
    #      原因: 那些行的 policy/value 目标来自冻结旧网络的搜索 → 保留=自我蒸馏跑偏 ✗
    # ⚠️ 2026-08-15 修复: 每局开始必须 reset_mcts() (标准 AlphaZero 每局独立搜索树)
    # 之前(书版继承)每局不清理 → 空盘 key 跨局串扰: 局2-10 复用局1搜索分布 →
    # 首手定式 + 探索独立性被破坏 + 偶然错误被放大10倍 (v8 it99+ 每代90%同首手根因)
    agent.reset_mcts()
    if coach_agent is not None:
        coach_agent.reset_mcts()
    if return_trajectory:
        trajectory = []
    observation, _ = env.reset()
    max_moves = env.board.size
    last_move = None
    for step in itertools.count():
        if step >= max_moves:
            break
        board, player = observation

        # ── v2: temperature 调度 ──
        # 2026-08-23 用户拍板修正: 旧调度 total_est=max_moves//2=112 远大于实际对局~45手
        # → 终局段(>=102)永远走不到, 中段线性在45手时仍≈0.74 → 实际全局≈1 随机采样
        # → 15x15推荐: 前16手 temp=1 探索, 之后 temp=1e-3 接近贪心 (棋谱质量↑ 信号↑)
        temp = 1.0
        if temperature_schedule:
            # v35-B (2026-09-06 热改, 用户拍板): 前段快速降温保堵点稳定
            # 背景: v32 线性(step7≈0.9)下速杀局(≤11手)it7 达42%; 回测42/42局败方末步
            # (堵点步, 手序7-10中位7, 活三/冲四成型期)选搜索top率仅中位67%(top p中位0.65
            # 100%确定 = 纯采样飞非搜索盲区); 堵点必须低温防抖。
            # B调度: step0-3: 1.0→0.6; step4-7: 0.6→0.15; step7+: 0.15平台
            #   模拟验证: 堵点步 top 选中率 67%→98%(中位), <90%占比 81%→38%
            # 权衡: 0.15平台非0.001(留残局弹性, v28线性换因=过早贪心锁死坏线);
            #   中盘多样性由搜索内 Dirichlet noise 提供, 温度只控采样锐度(不抑制搜索探索)
            if step <= 3:
                temp = 1.0 - 0.4 * step / 3.0           # 0:1.00 1:0.87 2:0.73 3:0.60
            elif step <= 7:
                temp = 0.6 - 0.45 * (step - 3) / 4.0    # 4:0.49 5:0.38 6:0.26 7:0.15
            else:
                temp = 0.15                              # 平台: 堵点/杀防稳定

        # ── v2: 开局中心约束（前3手） ──
        fc = force_center and step < 3

        _mover = agent
        if coach_agent is not None and coach_color is not None and player == coach_color:
            _mover = coach_agent
        action, prob, q_val = _mover.decide(observation, return_prob=True, return_q=True,
                                    temperature=temp, force_center=fc,
                                    move_no=step, last_move=last_move,
                                    random_open3=True)  # v36: 前3手100%随机开局(中心7x7) — 替代v35黑首手50%让手(断黑进攻链治先手红利)
        if verbose:
            logging.info('第 {} 步：玩家 {}, 动作 {}, temp={:.2f}'.format(
                step, player, action, temp))
        observation, reward, terminated, truncated, info = env.step(action)
        last_move = np.array(action, dtype=int)
        winner = env.get_winner(observation)
        done = terminated or truncated
        if return_trajectory:
            trajectory.append((player, board, prob, q_val))
        if done:
            break
    if return_trajectory:
        df_trajectory = pd.DataFrame(trajectory,
                                     columns=['player', 'board', 'prob', 'qa'])
        df_trajectory['winner'] = winner
        return df_trajectory
    else:
        return winner


# ─────────────────────────────────────────────
# 存档/读档（v2 独立目录，避免覆盖 v1）
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'gomoku_checkpoints_v44')
WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, 'weights.pt')
METADATA_PATH = os.path.join(CHECKPOINT_DIR, 'metadata.json')
WORKER_DIR = os.path.join(CHECKPOINT_DIR, 'worker_data')
REPLAY_DIR = os.path.join(CHECKPOINT_DIR, 'replay_pool')  # v2: 经验回放池


# ─────────────────────────────────────────────
# v42 (2026-10-10): 【自动换边调度】 —— v39 教练法 + 自动换边
#   触发: 学生 vs 教练 胜率 ≥ GOMOKU_SWAP_WINRATE 且【连续】GOMOKU_SWAP_STREAK 代
#   防抖: 两次换边间隔 ≥ GOMOKU_SWAP_MIN_GENS 代
#   动作: weights_coach.pt ← 学生 / weights.pt ← 原教练 / 教练色取反 (两网互换角色)
#   真相源: swap_state.json (★ env 改不到【已在跑的工兵】⇒ 必须落文件, 否则换边瞬间
#           工兵用旧教练色 + learn 用新剔行色 ⇒ 样本静默混色污染 ✗)
#   总开关: GOMOKU_SWAP_ENABLE (默认 0 = 完全关闭, 行为与 v39 教练法一致)
# ─────────────────────────────────────────────
SWAP_STATE_PATH = os.path.join(CHECKPOINT_DIR, 'swap_state.json')
SWAP_ENABLE = os.environ.get('GOMOKU_SWAP_ENABLE', '0').strip() == '1'
SWAP_WINRATE = float(os.environ.get('GOMOKU_SWAP_WINRATE', '0.55'))
SWAP_STREAK = int(os.environ.get('GOMOKU_SWAP_STREAK', '2'))
SWAP_MIN_GENS = int(os.environ.get('GOMOKU_SWAP_MIN_GENS', '5'))
COACH_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, 'weights_coach.pt')
LEDGER_PATH = os.path.join(SCRIPT_DIR, 'round_ledger.jsonl')  # 与 round_switch 工具链同文件


def _env_coach_color():
    return int(os.environ.get('GOMOKU_COACH_COLOR', '0') or 0)


def _read_swap_state():
    """读换边真相源; 不存在时用 env 初始化 (SWAP_ENABLE=0 时不会走到这里)"""
    try:
        with open(SWAP_STATE_PATH) as _f:
            _st = json.load(_f)
        if 'coach_color' not in _st:
            _st['coach_color'] = _env_coach_color()
        return _st
    except Exception:
        return {'coach_color': _env_coach_color(), 'swap_index': 0,
                'last_swap_gen': -10 ** 9, 'streak': 0}


def _write_swap_state(_st):
    """原子写 (tmp + os.replace): 防工兵读到写一半的 JSON ✗"""
    _st = dict(_st)
    _st['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    _tmp = SWAP_STATE_PATH + '.tmp'
    with open(_tmp, 'w') as _f:
        json.dump(_st, _f, ensure_ascii=False, indent=2)
    os.replace(_tmp, SWAP_STATE_PATH)


def _current_coach_color():
    """教练色真相源 = swap_state.json 优先, 回退 env
    (工兵载教练 / learn 剔行 / 胜率统计【三处统一走它】⇒ 换边后色一致, 不静默污染)"""
    if not SWAP_ENABLE:
        return _env_coach_color()
    return int(_read_swap_state().get('coach_color', _env_coach_color()) or 0)


def _net_fp(path):
    """权重张量指纹 (口径与 round_ledger 的 fp() 一致: 排除 running*, 保留 nbt)"""
    try:
        _sd = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as _e:
        return 'ERR:%s' % str(_e)[:40]
    if isinstance(_sd, dict) and 'state_dict' in _sd:
        _sd = _sd['state_dict']
    _h = hashlib.md5()
    for _k in sorted(_sd):
        _v = _sd[_k]
        if 'running' in str(_k) or not hasattr(_v, 'numpy'):
            continue
        _h.update(np.asarray(_v.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return _h.hexdigest()[:16]


def _do_swap(agent, iteration, winrate, config, _st, _c):
    """执行换边: 文件对调 + 内存重载 + 色取反 + 记账 (原子: 先存 tmp 再落)"""
    import shutil
    _stu_fp = _net_fp(WEIGHTS_PATH)          # 学生 (刚训练完, 即将冻结为教练)
    _coach_fp = _net_fp(COACH_WEIGHTS_PATH)  # 原教练 (即将成为新学生)
    _tmp_new_stu = COACH_WEIGHTS_PATH + '.swapnew'
    # ① 先把【原教练】存档到 tmp (下一步会被覆盖)
    shutil.copy2(COACH_WEIGHTS_PATH, _tmp_new_stu)
    # ② weights_coach.pt ← 学生
    shutil.copy2(WEIGHTS_PATH, COACH_WEIGHTS_PATH)
    # ③ weights.pt ← 原教练
    shutil.move(_tmp_new_stu, WEIGHTS_PATH)
    # ④ 内存: 主进程 agent 重载为【新学生】(=原教练), 保持只有一份网在内存 (显存不翻倍)
    try:
        _sd = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
        _net = agent.raw_net if (hasattr(agent, 'raw_net') and agent.raw_net is not None) else agent.net
        _net.load_state_dict(_sd, strict=False)
        _net.to(device)
    except Exception as _e:
        logging.error('v42 换边后内存重载失败 (文件已换, 主进程仍持旧网!): {}'.format(_e))
    # ④b 【关键修复 2026-10-12】把【新学生】的权重【立刻】推给共享推理服务
    #     病因: save_checkpoint 只在【每代末】推 update_weights(history_path) ⇒
    #           换边发生在代末【之后】⇒ 下一代产局时服务里【仍是上一代学生的网】✗
    #           实测: 换边后第一代"黑学生"用的其实是【白网的权重】(熵 0.447 vs 0.075 ⇒ 不是同一份)
    #           ⇒ 该代对局不可信(85%伪影 / 均手腰斩 / 两边同网), 且【会计入 streak】✗✗
    #     ⇒ 换边瞬间补推一次, 保证下一代一起跑就是【新学生】的权重 ✓
    try:
        if getattr(agent, '_remote', None) is not None:
            agent._remote.update_weights(WEIGHTS_PATH)
            logging.critical('v42 换边后已推送新学生权重给共享服务: {}'.format(
                os.path.basename(WEIGHTS_PATH)))
    except Exception as _e2:
        logging.error('v42 换边后推送新学生权重给共享服务失败 (服务仍持旧权重, 下一代对局不可信): {}'.format(_e2))
    # ⑤ 教练色取反 + 记账
    _c_new = -_c
    _st['coach_color'] = _c_new
    _st['swap_index'] = int(_st.get('swap_index', 0)) + 1
    _st['streak'] = 0
    _st['last_swap_gen'] = iteration
    _write_swap_state(_st)
    try:
        _row = {'action': 'swap', 'round': 'v42', 'swap_index': _st['swap_index'],
                'from_coach_color': _c, 'to_coach_color': _c_new,
                'student_fp': _stu_fp, 'coach_fp': _coach_fp,
                'trigger_winrate': round(float(winrate), 2), 'trigger_gen': iteration,
                'slot': os.environ.get('GOMOKU_SLOT', 'v42'),
                'ts': time.strftime('%Y-%m-%d %H:%M:%S')}
        with open(LEDGER_PATH, 'a') as _lf:
            _lf.write(json.dumps(_row, ensure_ascii=False) + '\n')
    except Exception as _e:
        logging.warning('v42 round_ledger 追加失败(忽略): {}'.format(_e))
    logging.critical('★★★ v42 自动换边 #{}: 学生({}胜率{:.1f}%)达标 ⇒ 冻结为新教练; '
                     '原教练转正为新学生; 教练色 {} → {} (代{})'.format(
                         _st['swap_index'], {1: '黑', -1: '白'}.get(-_c, '?'), float(winrate),
                         _c, _c_new, iteration))


# v44 双色同训: 执黑/执白胜率【近3代滑动平均】历史 (治噪声; v42 本轮白等 11 代才连上)
_wr_black_hist = []
_wr_white_hist = []


def _maybe_swap(agent, iteration, wr_black, wr_white, config):
    """v44 双色同训·【双门槛】换边: 【执黑胜率】与【执白胜率】近3代滑动平均【同时】≥阈值,
    连续 STREAK 代 + 距上次换边≥MIN_GENS ⇒ 执行换边 (禁止"一边倒补偿"式毕业)。
    返回 True(已换边)/False(未换). ★ 必须在 learn+save_checkpoint 之后调用。"""
    if not SWAP_ENABLE:
        return False
    # v44: 本代双色胜率入历史 (NaN=无有效对局不入), 取近3代均值
    if wr_black is not None and wr_black == wr_black:
        _wr_black_hist.append(wr_black)
    if wr_white is not None and wr_white == wr_white:
        _wr_white_hist.append(wr_white)
    _avg_b = (sum(_wr_black_hist[-3:]) / len(_wr_black_hist[-3:])) if _wr_black_hist else float('nan')
    _avg_w = (sum(_wr_white_hist[-3:]) / len(_wr_white_hist[-3:])) if _wr_white_hist else float('nan')
    if _avg_b != _avg_b or _avg_w != _avg_w:     # 任一为 NaN (无有效对局) 跳过
        return False
    _st = _read_swap_state()
    _c = int(_st.get('coach_color', _env_coach_color()) or 0)
    if _avg_b >= SWAP_WINRATE * 100.0 and _avg_w >= SWAP_WINRATE * 100.0:   # 双门槛同时达标 +1
        _st['streak'] = int(_st.get('streak', 0)) + 1
    else:
        _st['streak'] = 0
    _streak = int(_st.get('streak', 0))
    _last = int(_st.get('last_swap_gen', -10 ** 9))
    _gap_ok = (iteration - _last) >= SWAP_MIN_GENS
    logging.info('v44 换边判据: 执黑近3代 {:.1f}% | 执白近3代 {:.1f}% | 双门槛 {:.0f}% | streak={}/{}'.format(
        _avg_b, _avg_w, SWAP_WINRATE * 100, _streak, SWAP_STREAK))
    if _streak >= SWAP_STREAK and _gap_ok and _c in (1, -1):
        _do_swap(agent, iteration, (_avg_b + _avg_w) / 2.0, config, _st, _c)
        return True
    _write_swap_state(_st)                       # 未触发也落盘 streak (重启可续计)
    return False


try:
    if SWAP_ENABLE and not os.path.exists(SWAP_STATE_PATH):
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        _write_swap_state({'coach_color': _env_coach_color(), 'swap_index': 0,
                           'last_swap_gen': -10 ** 9, 'streak': 0})
        logging.info('v42 自动换边: 已初始化 swap_state.json (coach_color={})'.format(_env_coach_color()))
except Exception as _e:
    logging.warning('v42 swap_state 初始化失败(忽略): {}'.format(_e))
if SWAP_ENABLE:
    logging.info('v42 自动换边: ENABLE=1 阈值={:.2f} 连续={}代 最短间隔={}代 (真相源 swap_state.json)'.format(
        SWAP_WINRATE, SWAP_STREAK, SWAP_MIN_GENS))
else:
    logging.info('v42 自动换边: ENABLE=0 (关闭, 行为与 v39 教练法一致)')


def _ensure_coach_weights():
    """v42 (2026-10-10): 【首次起训初始化首个教练】——用户设计: 首个教练 = 初始权重(与学生同源于随机初始化),
    WARMUP 攒池后学生 vs 这份冻结教练对打 (初期 ≈50%, 涨到阈值 ⇒ 换边)。
    落点: 【weights.pt 首次落盘之后】(必须【先建学生权重再复制】, 否则复制空 ✗)。
    仅在 SWAP_ENABLE=1 且 weights_coach.pt 不存在时执行; swap_state.json 初值由模块加载时已写
    (coach_color=env 初始教练色, swap_index=0)。SWAP_ENABLE=0 时不动 (行为与 v39 一致)。
    """
    if not SWAP_ENABLE:
        return False
    if os.path.exists(COACH_WEIGHTS_PATH):
        return False
    if not os.path.exists(WEIGHTS_PATH):
        return False
    import shutil as _sh
    try:
        _sh.copy2(WEIGHTS_PATH, COACH_WEIGHTS_PATH)   # 学生权重冻结为【首个教练】
        logging.critical('★★★ v42 首个教练初始化: weights_coach.pt ← weights.pt '
                         '(初始学生权重冻结为冻结教练; 学生仍=同权重, 与教练一黑一白对打)')
        return True
    except Exception as _e:
        logging.error('v42 首个教练初始化失败 (将退化为普通自对弈!): {}'.format(_e))
        return False

_HANDICAP_RING_CACHE = {}
def _handicap_ring(size=15):
    """v35: 黑首手让子点集 = 距天元(中心)径向 5-8 子的环带
    欧式平方距离 25<=d^2<=64(径向 5.0-8.0 子) → 排除 4 角(d≈9.9, v34 太狠)与中心区;
    15x15 中心=7, 环带 ~110 点(多对称类, 让手多样性远超 v34 单对称类4角)
    """
    if size not in _HANDICAP_RING_CACHE:
        c = size // 2
        pts = [(r, cc) for r in range(size) for cc in range(size)
               if 25 <= (r - c) ** 2 + (cc - c) ** 2 <= 64]
        _HANDICAP_RING_CACHE[size] = pts
    return _HANDICAP_RING_CACHE[size]


def config_hash(train_iterations, train_episodes_per_iteration, batches, batch_size, sim_count, net_kwargs, **extra):
    raw = json.dumps({
        'train_iterations': train_iterations,
        'train_episodes_per_iteration': train_episodes_per_iteration,
        'batches': batches,
        'batch_size': batch_size,
        'sim_count': sim_count,
        'net_kwargs': net_kwargs,
        'version': VERSION,
        **extra,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def save_checkpoint(agent, iteration, config):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if hasattr(agent, 'raw_net') and agent.raw_net is not None:
        save_net = agent.raw_net
    else:
        save_net = agent.net
    torch.save(save_net.state_dict(), WEIGHTS_PATH)
    # ★ v42: 学生权重【刚落盘】后, 首个教练初始化 (SWAP_ENABLE=1 且 weights_coach.pt 缺失时复制)
    #   —— 必须在 weights.pt 之后调用 (先建学生再复制), 且须早于 metadata.json 落盘 (工兵见此才启动)
    _ensure_coach_weights()
    history_path = os.path.join(CHECKPOINT_DIR, f'weights_it{iteration}_v42.pt')  # v33 标识后缀
    torch.save(save_net.state_dict(), history_path)
    # v36 攒批推理: 必须在写 metadata(工兵发现新代的信号) 之前让 server 加载新权重,
    # 否则工兵会先看到新代、却用旧权重产局
    if getattr(agent, '_remote', None) is not None:
        try:
            agent._remote.update_weights(history_path)
            logging.info('已通知 inference server 热加载: %s', os.path.basename(history_path))
        except Exception as e:
            logging.error('通知 server 更新权重失败: %s', e)
    hash_config = {k: v for k, v in config.items() if k != 'losses'}
    metadata = {
        'iteration': iteration + 1,
        'config_hash': config_hash(**hash_config),
        'config': config,
        'version': VERSION,
    }
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    logging.info('检查点已保存 (iteration {})'.format(iteration))


def load_checkpoint(agent, config):
    if not os.path.exists(WEIGHTS_PATH) or not os.path.exists(METADATA_PATH):
        logging.info('未找到 v2 检查点，从头开始训练')
        return None
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    if metadata.get('version') != VERSION:
        logging.warning('检查点版本不匹配 ({} vs {}), 忽略旧版存档'.format(
            metadata.get('version'), VERSION))
        return None
    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
    # v19 (2026-08-26): 继承 v18 权重时, 旧 value_fc(225→1) 键与新 value_fc1/fc2 结构不匹配
    # pop 旧键 → strict=False 加载 (value_conv/value_bn 同形状继承, fc 新层随机初始化)
    if 'value_fc.weight' in state_dict:
        del state_dict['value_fc.weight']
    if 'value_fc.bias' in state_dict:
        del state_dict['value_fc.bias']
    # v20 (2026-08-27, 保留式 v2): 旧 policy_conv/policy_bn(128→1,3×3) 与新结构同形状 → 完整继承(识别模块)
    # 新 policy_conv2/policy_bn2/policy_fc 旧档无此键 → strict=False 自动随机初始化
    if hasattr(agent, 'raw_net') and agent.raw_net is not None:
        agent.raw_net.load_state_dict(state_dict, strict=False)
        agent.raw_net.to(device)
        agent.net = agent.raw_net
    else:
        agent.net.load_state_dict(state_dict, strict=False)
        agent.net.to(device)
    agent.reset_mcts()
    iteration = metadata.get('iteration', -1)
    logging.info('已从 v2 检查点恢复 (上次训练到 iteration {})'.format(iteration))
    return iteration


# ─────────────────────────────────────────────
# v2: 每代健康体检（价值盲测 + 激活率 + 死神经元）
# 训练完一代立即执行，结果写入 health_log.json
# ─────────────────────────────────────────────
HEALTH_LOG_PATH = os.path.join(CHECKPOINT_DIR, 'health_log.json')


def _probe_board(rows, cols, color):
    b = np.zeros((15, 15), dtype=np.float32)
    for r, c in zip(rows, cols):
        b[r, c] = color
    return b


def _probe_board_alt(rows, cols, color, opp_scatter=6, seed=None):
    """交替局面探针：主线(rows,cols)放 color 棋子，再加 opp_scatter 个对手散子(远离主线)。

    纯色探针(黑活四=棋盘仅4个黑子)是 OOD——训练自对弈从未出现单色棋盘，
    真实对局中黑方形成活四时白方已下过 N 手。本函数构造接近真实的交替局面。
    """
    b = np.zeros((15, 15), dtype=np.float32)
    for r, c in zip(rows, cols):
        b[r, c] = color
    if opp_scatter:
        rng = random.Random(seed)
        main_cells = set(zip(rows, cols))
        placed = 0
        guard = 0
        while placed < opp_scatter and guard < 2000:
            guard += 1
            r, c = rng.randrange(15), rng.randrange(15)
            if (r, c) in main_cells or b[r, c] != 0:
                continue
            if min(abs(r - rr) + abs(c - cc) for rr, cc in main_cells) < 3:
                continue
            b[r, c] = -color
            placed += 1
        if placed < opp_scatter:
            return None  # 无法放置足够散子（极端情况）
    return b


def _policy_to_prob(p_tensor, board_shape):
    """把 policy 头输出转成 (H,W) 概率分布。

    ⚠️ v1/v2 输出语义不同（2026-08-02 实踩，首次结论全错）：
    - v1 original forward 已做全局 225 softmax（输出就是概率）→ 只 reshape
    - v2 policy_fc 输出 logits → reshape(-1) 全局 softmax 再 reshape
    对 reshape 后 15×15 按行(dim=-1) softmax 是错的（每行归一化成 1/15≈0.067 假象）。
    """
    p = p_tensor.reshape(-1)
    # 检测是否已是概率: 和≈1 且全在 [0,1] → v1 已 softmax
    if bool((p >= 0).all()) and abs(float(p.sum()) - 1.0) < 0.05:
        return p.reshape(board_shape).cpu().numpy()
    p = torch.softmax(p, dim=-1)
    return p.reshape(board_shape).cpu().numpy()


def health_check(agent, iteration):
    """对当前网络跑价值盲测 + 激活率检查，返回dict"""
    net = agent.raw_net if (hasattr(agent, 'raw_net') and agent.raw_net is not None) else agent.net
    net.eval()
    result = {'iteration': iteration, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}

    # ── 1. 价值盲测（训练中真实出现的中间局面，非终局状态） ──
    # 注意: 训练数据里从未出现过"已成五连"的棋盘(成五即终局,不进trajectory)
    # 所以探针只用活三/活四/冲四等真实中间状态
    probes = {
        '黑活四': (_probe_board([7]*4, [5, 6, 7, 8], 1), 1.0),
        '白活四': (_probe_board([7]*4, [5, 6, 7, 8], -1), 1.0),
        '黑活三': (_probe_board([7]*3, [5, 6, 7], 1), 0.6),
        '白活三': (_probe_board([7]*3, [5, 6, 7], -1), 0.6),
        '黑冲四': (_probe_board([3, 4, 5, 7], [7, 7, 7, 7], 1), 0.8),  # 断一格的四连
        '白冲四': (_probe_board([3, 4, 5, 7], [7, 7, 7, 7], -1), 0.8),
        # ⚠️ 2026-08-01 补盲: 五连是最基础"已赢"形状, 纯色探针漏测导致"黑冲四准但黑五连判反"假象;
        # 注意: 活四探针(7,5-8)本身就是连续四连, 无需另设四连
        '黑五连': (_probe_board([7]*5, [5, 6, 7, 8, 9], 1), 1.0),  # 已赢, OOD但必须监控
        '白五连': (_probe_board([7]*5, [5, 6, 7, 8, 9], -1), 1.0),
        '空盘': (np.zeros((15, 15), dtype=np.float32), 0.0),
        # ⚠️ 2026-08-16 补盲: 原探针全是横/竖方向, 斜向棋型从未检测（用户发现白棋斜向识别差）
        # 主对角线 (r,r) 斜向变体: 活四(5-8)两端(4,4)/(9,9)空; 冲四(4,4-6,6)+(8,8)断(7,7); 五连(5-9)
        '黑活四斜': (_probe_board([5, 6, 7, 8], [5, 6, 7, 8], 1), 1.0),
        '白活四斜': (_probe_board([5, 6, 7, 8], [5, 6, 7, 8], -1), 1.0),
        '黑活三斜': (_probe_board([5, 6, 7], [5, 6, 7], 1), 0.6),
        '白活三斜': (_probe_board([5, 6, 7], [5, 6, 7], -1), 0.6),
        '黑冲四斜': (_probe_board([4, 5, 6, 8], [4, 5, 6, 8], 1), 0.8),
        '白冲四斜': (_probe_board([4, 5, 6, 8], [4, 5, 6, 8], -1), 0.8),
        '黑五连斜': (_probe_board([5, 6, 7, 8, 9], [5, 6, 7, 8, 9], 1), 1.0),
        '白五连斜': (_probe_board([5, 6, 7, 8, 9], [5, 6, 7, 8, 9], -1), 1.0),
    }
    # v18 修复 (2026-08-22): 探针传 players (黑棋型=1/白棋型=-1) → ch2 正确编码"当前玩家颜色"
    # 期望统一为"当前玩家视角" (持有棋型的一方优 → +) — 与训练标签 vs=players*winners 一致
    def _probe_players(name):
        if name.startswith('黑'):
            return 1
        if name.startswith('白'):
            return -1
        return None  # 空盘
    with torch.no_grad():
        for name, (b, expect) in probes.items():
            # 🚨 2026-08-23 探针视角修复: _to_planes 输入必须是 canonical(当前玩家=+1/对手=-1)
            # 旧代码直接传绝对棋盘(黑=1/白=-1) → 白视角探针 ch0/ch1 反了(白子进"对手"通道)
            # 修复: canonical = player * 绝对棋盘, 且 players 必须传 player (黑=1/白=-1)
            pl = _probe_players(name)
            if pl is None:
                p, v = net(_to_planes(b[None, :, :], players=None))  # 空盘全0, 无视角
            else:
                p, v = net(_to_planes((pl * b)[None, :, :], players=pl))
            pred = float(v.reshape(-1)[0])
            # 空盘: 期望|value|小; 其他: 期望符号正确
            if expect == 0:
                correct = abs(pred) < 0.3
            else:
                correct = (pred > 0) == (expect > 0)
            result['value_' + name] = round(pred, 4)
            result['value_' + name + '_ok'] = bool(correct)

    # ── 1b. 交替局面探针（2026-08-01 新增: 纯色探针 OOD 缺陷实证后）
    # 纯色盘(黑活四=仅4黑子)训练中从未出现 → 假信号; 真实对局黑白交替。
    # 主线 + 6 对手散子 × 10 组随机, 取均值/标准差; 通过 = 方向对 + |均值|>0.3
    ALT_TRIALS = 10
    ALT_SCATTER = 6
    for name, (b, expect) in probes.items():
        if expect == 0:  # 空盘无散子可加
            continue
        rows, cols, colors = [], [], []
        for r in range(15):
            for c in range(15):
                if b[r, c] != 0:
                    rows.append(r); cols.append(c); colors.append(b[r, c])
        if not rows:
            continue
        main_color = colors[0]  # 主线所有棋子同色
        pl = _probe_players(name)  # 视角: 黑=1/白=-1
        vals = []
        for s in range(ALT_TRIALS):
            ba = _probe_board_alt(rows, cols, main_color, opp_scatter=ALT_SCATTER, seed=1000 + s)
            if ba is None:
                continue
            # 🚨 2026-08-23 视角修复: canonical = pl * 绝对棋盘 (同纯色探针)
            _, va = net(_to_planes((pl * ba)[None, :, :], players=pl))
            vals.append(float(va.reshape(-1)[0]))
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
            ok = (mean > 0) == (expect > 0) and abs(mean) > 0.3
            result['alt_' + name] = round(mean, 4)
            result['alt_' + name + '_std'] = round(std, 4)
            result['alt_' + name + '_ok'] = bool(ok)

    # ── 1c. Policy 头探针（2026-08-02 新增: 体检只看 value 头的盲区实锤后）
    # 背景: MCTS 实战里 policy 是"搜索向导"(PUCT 公式 Q + c·P·√N/(1+n) 的 P),
    #        policy 均匀则搜索无方向 → "体检数值高、实战一碰就碎"的技术根源。
    # 方法: 构造"必须应对"的局面, 测 policy 给关键应对点的概率 vs 随机基线(1/225≈0.0044)。
    # 四类: 黑攻(找成活四点)/白攻/黑守(堵白威胁)/白守(堵黑威胁), 各 4 局面 × 纯色/交替。
    # 通过: 关键点概率 > 8×随机基线(0.0355), 每类多数通过才算该类 ok。
    POLICY_THRESH = 8.0 / 225.0  # ≈0.0355
    POLICY_TRIALS = 5
    POLICY_SCATTER = 6
    policy_cats = {
        '黑攻': (1, [  # 黑有活三/冲四, 找成活四点(进攻要点), 主线=黑(+1)
            ([7, 7, 7], [5, 6, 7], [(7, 8)]),          # 活三横→成活四 [直向]
            ([5, 6, 7], [7, 7, 7], [(8, 7)]),          # 活三竖→成活四 [直向]
            ([3, 4, 5, 7], [7, 7, 7, 7], [(6, 7)]),    # 冲四→成五 [直向]
            ([5, 6, 7], [5, 6, 7], [(8, 8)]),          # 活三斜→成活四 [斜向]
            ([5, 5, 5, 5], [5, 6, 7, 8], [(4, 4), (9, 9)]),  # 斜活四→成五(两端) [斜向]
            ([4, 5, 6, 8], [4, 5, 6, 8], [(7, 7)]),          # 斜冲四→成五(补缺口) [斜向]
        ]),
        '白攻': (-1, [  # 白有活三/冲四, 找成活四点, 主线=白(-1)
            ([7, 7, 7], [5, 6, 7], [(7, 8)]),
            ([5, 6, 7], [7, 7, 7], [(8, 7)]),
            ([3, 4, 5, 7], [7, 7, 7, 7], [(6, 7)]),
            ([5, 6, 7], [5, 6, 7], [(8, 8)]),
            ([5, 5, 5, 5], [5, 6, 7, 8], [(4, 4), (9, 9)]),
            ([4, 5, 6, 8], [4, 5, 6, 8], [(7, 7)]),
        ]),
        '黑守': (-1, [  # 白有活三/冲四, 黑找堵点, 主线=白(-1)是威胁方
            ([7, 7, 7], [5, 6, 7], [(7, 4), (7, 8)]),      # 白活三横→堵两端 [直向]
            ([5, 6, 7], [7, 7, 7], [(4, 7), (8, 7)]),      # 白活三竖 [直向]
            ([3, 4, 5, 7], [7, 7, 7, 7], [(6, 7)]),        # 白冲四→堵缺口 [直向]
            ([5, 6, 7], [5, 6, 7], [(4, 4), (8, 8)]),      # 白活三斜 [斜向]
            ([5, 5, 5, 5], [5, 6, 7, 8], [(4, 4), (9, 9)]),      # 白斜活四→堵两端 [斜向]
            ([4, 5, 6, 8], [4, 5, 6, 8], [(7, 7)]),              # 白斜冲四→堵缺口 [斜向]
        ]),
        '白守': (1, [  # 黑有活三/冲四, 白找堵点, 主线=黑(+1)是威胁方
            ([7, 7, 7], [5, 6, 7], [(7, 4), (7, 8)]),
            ([5, 6, 7], [7, 7, 7], [(4, 7), (8, 7)]),
            ([3, 4, 5, 7], [7, 7, 7, 7], [(6, 7)]),
            ([5, 6, 7], [5, 6, 7], [(4, 4), (8, 8)]),
            ([5, 5, 5, 5], [5, 6, 7, 8], [(4, 4), (9, 9)]),
            ([4, 5, 6, 8], [4, 5, 6, 8], [(7, 7)]),
        ]),
    }
    with torch.no_grad():
        for cat_name, (cat_color, cat_probes) in policy_cats.items():
            pass_counts = {'纯色': 0, '交替': 0}
            straight_counts = {'纯色': 0, '交替': 0}   # 直向(横/竖)
            diag_counts = {'纯色': 0, '交替': 0}       # 斜向(对角线)
            key_means = []
            for pi, (rows, cols, key_cells) in enumerate(cat_probes):
                is_diag = pi >= 3  # 索引0-2直向(横活三/竖活三/冲四), 3-5斜向(斜活三/斜活四/斜冲四)
                main_color = cat_color
                # 🚨 2026-08-23 视角修复: 行动方视角 ≠ 主线颜色!
                # 攻类(黑攻/白攻): 行动方=持子方(主线) → player = cat_color
                # 守类(黑守/白守): 行动方=防守方(堵对方威胁) → player = -cat_color
                # 旧代码统一传 cat_color → 黑守(白威胁黑堵)传成了白视角, 白守(黑威胁白堵)传成黑视角 → 全反
                if cat_name in ('黑攻', '白攻'):
                    act_player = cat_color
                else:
                    act_player = -cat_color
                # 纯色版 (v18: 传 canonical=act_player*棋盘 + players=act_player → ch2 正确视角)
                b_p = _probe_board(rows, cols, main_color)
                p_p, _ = net(_to_planes((act_player * b_p)[None, :, :], players=act_player))
                # v2 输出 logits → softmax; 兼容 v1(已是概率, reshape 即可)
                prob_p = _policy_to_prob(p_p, b_p.shape)
                kp_pure = max(prob_p[r, c] for r, c in key_cells if b_p[r, c] == 0) if key_cells else 0.0
                pass_counts['纯色'] += int(kp_pure > POLICY_THRESH)
                (diag_counts if is_diag else straight_counts)['纯色'] += int(kp_pure > POLICY_THRESH)
                key_means.append(kp_pure)
                # 交替版: 加对手散子, 多组取均值
                kps_alt = []
                for s in range(POLICY_TRIALS):
                    b_a = _probe_board_alt(rows, cols, main_color, opp_scatter=POLICY_SCATTER, seed=2000 + pi * 10 + s)
                    if b_a is None:
                        continue
                    p_a, _ = net(_to_planes((act_player * b_a)[None, :, :], players=act_player))
                    prob_a = _policy_to_prob(p_a, b_a.shape)
                    kps_alt.append(max(prob_a[r, c] for r, c in key_cells if b_a[r, c] == 0) if key_cells else 0.0)
                if kps_alt:
                    kp_alt_mean = sum(kps_alt) / len(kps_alt)
                    pass_counts['交替'] += int(kp_alt_mean > POLICY_THRESH)
                    (diag_counts if is_diag else straight_counts)['交替'] += int(kp_alt_mean > POLICY_THRESH)
            n_probes = len(cat_probes)
            result[f'pol_{cat_name}_纯色通过'] = pass_counts['纯色']
            result[f'pol_{cat_name}_交替通过'] = pass_counts['交替']
            result[f'pol_{cat_name}_直向通过'] = straight_counts['交替']  # 交替版直向(0-2)
            result[f'pol_{cat_name}_斜向通过'] = diag_counts['交替']      # 交替版斜向(3-5)
            result[f'pol_{cat_name}_n'] = n_probes
            result[f'pol_{cat_name}_纯色ok'] = pass_counts['纯色'] >= max(2, n_probes // 2)
            result[f'pol_{cat_name}_交替ok'] = pass_counts['交替'] >= max(2, n_probes // 2)
    # 全局 policy 判定: 四类交替版至少 2 类通过 且 白守(防守) 通过
    pol_alt_oks = [result.get(f'pol_{c}_交替ok', False) for c in ['黑攻', '白攻', '黑守', '白守']]
    result['pol_全局ok'] = bool(sum(pol_alt_oks) >= 2 and result.get('pol_白守_交替ok', False))

    # ── 1d. 堵端敏感度探针（2026-08-16 新增: 死四/冲三盲区监控——用户摆局面实测
    #      黑死四判 +0.56、白活三被对手四连压成 -0.999 = 只认连子不认两端被堵, 与 v8 死四盲区同型）
    # 方法: 同一连子棋型 堵一端(对手子) vs 不堵, |value差| 应显著 >0（堵端威胁降级）
    # 判定: 活四堵端差 >0.5 达标; 活三堵端差 >0.2 达标
    def _blocked_probe(rows, cols, color, block_pos):
        b = np.zeros((15, 15), dtype=np.float32)
        for r, c in zip(rows, cols):
            b[r, c] = color
        b[block_pos] = -color
        return b

    block_probes = {
        '黑活四堵端': (([5, 6, 7, 8], [5, 6, 7, 8]), (4, 4), 0.5),  # 黑四连斜, 白堵(4,4)=死四
        '白活四堵端': (([5, 6, 7, 8], [5, 6, 7, 8]), (4, 4), 0.5),  # 白四连斜, 黑堵(4,4)=死四
        '黑活三堵端': (([5, 6, 7], [5, 6, 7]), (4, 4), 0.2),        # 黑三连斜, 白堵=冲三
        '白活三堵端': (([5, 6, 7], [5, 6, 7]), (4, 4), 0.2),        # 白三连斜, 黑堵=冲三
    }
    with torch.no_grad():
        for name, ((rows, cols), bp, thresh) in block_probes.items():
            main_color = 1 if name.startswith('黑') else -1
            b0 = _probe_board(rows, cols, main_color)
            # 🚨 2026-08-23 视角修复: canonical = main_color * 棋盘 (堵端探针持子方=评估视角)
            _, v0 = net(_to_planes((main_color * b0)[None, :, :], players=main_color))
            b1 = _blocked_probe(rows, cols, main_color, bp)
            _, v1 = net(_to_planes((main_color * b1)[None, :, :], players=main_color))
            v0f = float(v0.reshape(-1)[0]); v1f = float(v1.reshape(-1)[0])
            diff = abs(v0f - v1f)
            result[f'block_{name}_未堵'] = round(v0f, 4)
            result[f'block_{name}_堵端'] = round(v1f, 4)
            result[f'block_{name}_差'] = round(diff, 4)
            result[f'block_{name}_ok'] = bool(diff > thresh)

    # ── 1e. 相对价值探针（2026-08-16 新增: 白棋"不会相对价值比较"——用户局面2实测
    #      白先手活三只给 +0.375（该 +0.7 必胜势）= 双方都有威胁时不会按紧迫性排序）
    # 白视角(当前玩家=白) 期望排序: 己活四 > 己活三vs他活三 > 己活三vs他冲四 > 己活三vs他活四
    def _dual_probe(white_cells, black_cells):
        b = np.zeros((15, 15), dtype=np.float32)
        for r, c in white_cells:
            b[r, c] = 1
        for r, c in black_cells:
            b[r, c] = -1
        return b

    rel_probes = {
        '白活三_对黑活三': (_dual_probe([(5,5),(6,6),(7,7)], [(3,10),(4,11),(5,12)]), 0.5),       # 白先手成活四→白优
        '白活三_对黑冲四': (_dual_probe([(5,5),(6,6),(7,7)], [(2,10),(3,11),(4,12),(6,14)]), -0.3),  # 黑冲四更急,白必须堵→黑优
        '白活四_对黑活三': (_dual_probe([(5,5),(6,6),(7,7),(8,8)], [(3,10),(4,11),(5,12)]), 0.8),   # 白活四必胜
        '白活三_对黑活四': (_dual_probe([(5,5),(6,6),(7,7)], [(3,10),(4,11),(5,12),(6,13)]), -0.8),  # 黑活四必胜
    }
    with torch.no_grad():
        for name, (b_r, expect) in rel_probes.items():
            # v18: rel 探针构造是白视角 (白=+1 黑=-1) → 必须传 players=-1, 否则 ch2 默认黑视角方向全反
            _, v = net(_to_planes(b_r[None, :, :], players=-1))
            vf = float(v.reshape(-1)[0])
            ok = (vf > 0) == (expect > 0) and abs(vf) > 0.3
            result[f'rel_{name}'] = round(vf, 4)
            result[f'rel_{name}_ok'] = bool(ok)
    v_h4 = result.get('rel_白活四_对黑活三', 0)
    v_h3h3 = result.get('rel_白活三_对黑活三', 0)
    v_h3f4 = result.get('rel_白活三_对黑冲四', 0)
    v_h3h4 = result.get('rel_白活三_对黑活四', 0)
    result['rel_排序ok'] = bool(v_h4 > v_h3h3 > v_h3f4 > v_h3h4)

    # ── 2. 激活率（4个代表层） ──
    act_layers = {}
    hooks = []
    target_names = ['res_blocks.3.bn2', 'res_blocks.4.bn2',
                    'res_blocks.5.bn1', 'res_blocks.5.bn2']
    for name, module in net.named_modules():
        if name in target_names:
            holder = {}
            def make_hook(h):
                def hook_fn(mod, inp, out):
                    h['act'] = (out.detach() > 0).float().mean().item()
                return hook_fn
            hooks.append(module.register_forward_hook(make_hook(holder)))
            act_layers[name] = holder

    probe_boards = [probes['黑活四'][0], probes['空盘'][0],
                    _probe_board([3, 5, 9], [3, 8, 12], 1)]
    with torch.no_grad():
        for b in probe_boards:
            net(_to_planes(b[None, :, :]))

    act_vals = []
    for name, holder in act_layers.items():
        val = holder.get('act', 0.0)
        act_vals.append(val)
        result['act_' + name.split('.')[-2] + '.' + name.split('.')[-1]] = round(val * 100, 1)
    if act_vals:
        result['act_mean'] = round(np.mean(act_vals) * 100, 1)
    for h in hooks:
        h.remove()

    # ── 3. BN 死神经元 ──
    dead = 0
    total = 0
    for name, module in net.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            g = module.weight.detach().cpu().numpy()
            total += len(g)
            dead += int((g < 0.01).sum())
    result['bn_dead'] = dead
    result['bn_total'] = total

    # ── 写入日志 ──
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # ── 7. v37 新增: "白1距黑1" 探针 (2026-09-16) ─────────────────────────
    #   背景: v36 前3手强制随机 → 模型从未学过白1决策 → 白方第一手乱下
    #         实测 200 局: 白1 距黑1 中位 14 格 / 64% 落 8 格以外 / 仅 2% 贴身
    #   健康方向: 中位降到 2~4 格, 贴身(≤2)占比上升 = 白方开始学会开局应对 ✓
    #   方法: 构造"仅一黑子"的棋盘(4 个常见黑1位置), 白方视角取 policy argmax,
    #         算与黑子的曼哈顿距离; 同时记录 top1 概率 与 top3 平均距离
    try:
        with torch.no_grad():
            dists, top1s, top3s = [], [], []
            for bpos in [(7, 7), (7, 6), (8, 7), (4, 10)]:
                bm = _probe_board([bpos[0]], [bpos[1]], 1)
                p_out, _ = net(_to_planes((-1 * bm)[None, :, :], players=-1))
                prob = _policy_to_prob(p_out, bm.shape).copy()
                prob[bpos[0], bpos[1]] = 0.0          # 排除已占点
                flat = prob.reshape(-1)
                order = np.argsort(flat)[::-1]
                r1, c1 = np.unravel_index(order[0], prob.shape)
                dists.append(abs(int(r1) - bpos[0]) + abs(int(c1) - bpos[1]))
                top1s.append(float(flat[order[0]]))
                d3 = []
                for k in range(3):
                    r3, c3 = np.unravel_index(order[k], prob.shape)
                    d3.append(abs(int(r3) - bpos[0]) + abs(int(c3) - bpos[1]))
                top3s.append(sum(d3) / 3.0)
            result['w1_dist_med'] = round(float(np.median(dists)), 1)
            result['w1_dist_min'] = float(min(dists))
            result['w1_dist_all'] = [float(x) for x in dists]
            result['w1_top1_med'] = round(float(np.median(top1s)), 4)
            result['w1_top3dist_med'] = round(float(np.median(top3s)), 2)
            result['w1_ok'] = bool(np.median(dists) <= 4)   # 中位≤4 算开始学会应对
    except Exception as e:
        logging.warning('白1距黑1 探针失败: {}'.format(e))

    history = []
    if os.path.exists(HEALTH_LOG_PATH):
        try:
            with open(HEALTH_LOG_PATH) as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(result)
    with open(HEALTH_LOG_PATH, 'w') as f:
        json.dump(history, f, indent=1)

    pol_summary = 'policy探针: 黑攻[{}/{} 交替{}] 白攻[{}/{} 交替{}] 黑守[{}/{} 交替{}] 白守[{}/{} 交替{}] 全局{}'.format(
        result.get('pol_黑攻_纯色通过', 0), result.get('pol_黑攻_n', 4),
        '✅' if result.get('pol_黑攻_交替ok') else '❌',
        result.get('pol_白攻_纯色通过', 0), result.get('pol_白攻_n', 4),
        '✅' if result.get('pol_白攻_交替ok') else '❌',
        result.get('pol_黑守_纯色通过', 0), result.get('pol_黑守_n', 4),
        '✅' if result.get('pol_黑守_交替ok') else '❌',
        result.get('pol_白守_纯色通过', 0), result.get('pol_白守_n', 4),
        '✅' if result.get('pol_白守_交替ok') else '❌',
        '✅' if result.get('pol_全局ok') else '❌')
    logging.info('健康体检: 活四[黑={} {} 白={} {}] 活三[黑={} {} 白={} {}] 冲四[黑={} {} 白={} {}] 五连[黑={} {} 白={} {}] 空盘={} | 激活率={}% 死神经元={}/{} | 交替版: 活四[黑={}({}) 白={}({})] 冲四[黑={}({}) 白={}({})] 活三[黑={}({}) 白={}({})] 五连[黑={}({}) 白={}({})] | {}'.format(
        result.get('value_黑活四'), '✅' if result.get('value_黑活四_ok') else '❌',
        result.get('value_白活四'), '✅' if result.get('value_白活四_ok') else '❌',
        result.get('value_黑活三'), '✅' if result.get('value_黑活三_ok') else '❌',
        result.get('value_白活三'), '✅' if result.get('value_白活三_ok') else '❌',
        result.get('value_黑冲四'), '✅' if result.get('value_黑冲四_ok') else '❌',
        result.get('value_白冲四'), '✅' if result.get('value_白冲四_ok') else '❌',
        result.get('value_黑五连'), '✅' if result.get('value_黑五连_ok') else '❌',
        result.get('value_白五连'), '✅' if result.get('value_白五连_ok') else '❌',
        result.get('value_空盘'),
        result.get('act_mean'), dead, total,
        result.get('alt_黑活四'), '✅' if result.get('alt_黑活四_ok') else '❌',
        result.get('alt_白活四'), '✅' if result.get('alt_白活四_ok') else '❌',
        result.get('alt_黑冲四'), '✅' if result.get('alt_黑冲四_ok') else '❌',
        result.get('alt_白冲四'), '✅' if result.get('alt_白冲四_ok') else '❌',
        result.get('alt_黑活三'), '✅' if result.get('alt_黑活三_ok') else '❌',
        result.get('alt_白活三'), '✅' if result.get('alt_白活三_ok') else '❌',
        result.get('alt_黑五连'), '✅' if result.get('alt_黑五连_ok') else '❌',
        result.get('alt_白五连'), '✅' if result.get('alt_白五连_ok') else '❌',
        pol_summary))
    return result


# ─────────────────────────────────────────────
# 主训练
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# v37 门控执行 (2026-09-16 用户拍板 = 方案B: 候选产数据 + 失败回滚)
# ─────────────────────────────────────────────
BEST_WEIGHTS_PATH = os.path.join(CHECKPOINT_DIR, 'weights_best.pt')
GATE_LOG_DIR = os.path.join(SCRIPT_DIR, 'gate_log')
GATE_HISTORY_PATH = os.path.join(CHECKPOINT_DIR, 'gate_history.json')


def _clear_replay_pool():
    """方案B 回滚时清空回放池 (防污染数据继续喂训练)"""
    n = 0
    if os.path.isdir(REPLAY_DIR):
        for fn in os.listdir(REPLAY_DIR):
            p = os.path.join(REPLAY_DIR, fn)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                    n += 1
                except Exception:
                    pass
    return n


def _append_gate_history(rec):
    try:
        hist = json.load(open(GATE_HISTORY_PATH)) if os.path.exists(GATE_HISTORY_PATH) else []
        hist.append(rec)
        json.dump(hist, open(GATE_HISTORY_PATH, 'w'), indent=1, ensure_ascii=False)
    except Exception as e:
        logging.warning('门控历史写入失败: {}'.format(e))


def _run_gate(iteration, agent, config, health=None):
    """v37 门控: 候选(weights.pt) vs 冻结 best(weights_best.pt), 配对 GATE_GAMES 局

    判据(用户拍板): 候选胜率 ≥ GATE_THRESHOLD (严格胜率, 不换指标)
      通过 → weights_best.pt := 候选 (晋升)
      否则 → 方案B 回滚: weights.pt ← best + 内存同步 + 清空回放池 + 通知 server
    ★ 死锁告警: 双方执白均 0 胜 (=黑方全胜, 配对赛总分恒50%) → 红字报警, 不改判
    """
    import glob as _glob
    import shutil as _shutil
    import subprocess as _subprocess
    t0 = time.time()
    tag = 'v37_g{}'.format(iteration)
    rec = {'iteration': iteration, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
           'verdict': 'ERROR', 'rate': None}

    # ── 污染判据 (2026-09-17): 只有【标签被污染】才清空回放池 ──
    #   判据来自 V̂ 事故的实测特征: 空盘 value 爆到 0.8 / 激活率异常
    corrupt, corrupt_why = False, []
    if health:
        kv = health.get('value_空盘')
        if kv is not None and abs(float(kv)) > GATE_CORRUPT_V:
            corrupt = True
            corrupt_why.append('空盘value=%+.3f' % float(kv))
        ac = health.get('act_mean')
        if ac is not None and float(ac) < GATE_CORRUPT_ACT:
            corrupt = True
            corrupt_why.append('激活率=%.1f%%' % float(ac))
    rec['corrupt'] = corrupt

    os.makedirs(GATE_LOG_DIR, exist_ok=True)

    # 首次: best 不存在 → 用【首次学习后的快照】建立 best
    #   ★ 用户 2026-09-17 澄清的设计意图: 第一次门控必须回答
    #     "从开始学习到现在这 N 代训练, 到底有没有变强" →
    #     best 必须锚定在【学习起点】(warmup 攒满池后第一次 learn 的快照 = it{REPLAY_POOL_SIZE}),
    #     而不是"第一次门控触发时的当前权重"(那样会白等一个周期, 且永远问不到这个问题)
    if not os.path.exists(BEST_WEIGHTS_PATH):
        # ⚠️ 2026-09-25 修正 off-by-one: 实测 warmup 在 iteration 3 结束, iteration 4 就首次 learn
        #    (不是 REPLAY_POOL_SIZE=5 → iteration 5)。所以按数码猜锚点是错的 ✗
        #    正解(语义法): 【第一个与随机初始快照字节不同的快照】= 第一个训练过的模型 ✓
        anchor = None
        try:
            _snaps = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'weights_it*_v42.pt')),
                            key=lambda f: int(os.path.basename(f).replace('weights_it', '').split('_')[0]))
            if len(_snaps) >= 2:
                _base = open(_snaps[0], 'rb').read()          # 第一个快照 = 随机初始化
                for _sp in _snaps:
                    if open(_sp, 'rb').read() != _base:
                        anchor = _sp
                        break
        except Exception as _e:
            logging.warning('【门控】锚点探测失败: {}'.format(_e))
        if anchor is None:
            anchor = WEIGHTS_PATH
            logging.warning('【门控】首次: 未找到"第一个训练过的快照"(可能 warmup 未结束) → 退回当前权重')
        _shutil.copy(anchor, BEST_WEIGHTS_PATH)
        logging.info('【门控】首次: weights_best.pt ← 学习起点快照 {} (第一个训练过的模型)'.format(
            os.path.basename(anchor)))
        rec['best_init_from'] = os.path.basename(anchor)
        rec['verdict'] = 'FIRST'
        _append_gate_history(rec)
        return rec

    # 清旧分片
    for f in _glob.glob(os.path.join(GATE_LOG_DIR, tag + '_shard*')):
        try:
            os.remove(f)
        except Exception:
            pass
    logging.info('【门控】代{}: 候选 vs best, {} 局配对 ({} 分片, sim={}) ...'.format(
        iteration, GATE_GAMES, GATE_SHARDS, GATE_SIM))

    # ── 起分片 (并行) ──
    gate_py = os.path.join(SCRIPT_DIR, 'scripts', 'gate.py')
    procs = []
    for i in range(GATE_SHARDS):
        cmd = [sys.executable, '-u', gate_py,
               '--best', BEST_WEIGHTS_PATH, '--cand', WEIGHTS_PATH, '--tag', tag,
               '--games', str(GATE_GAMES), '--sim', str(GATE_SIM),
               '--shard', str(i), '--nshard', str(GATE_SHARDS), '--gpu',
               '--out', GATE_LOG_DIR]
        lf = open(os.path.join(GATE_LOG_DIR, '{}_shard{}.log'.format(tag, i)), 'w')
        # ★ 2026-09-25 重大 BUG 修复: 门控分片必须【关掉共享攒批推理】
        #   病因: 开 USE_BATCH_INFER 时 agent.predict() 走 /tmp/gomoku_infer.sock
        #         而该服务里的网络 = 【训练进程最新权重】(每代 learn 都热加载)
        #         → load_agent 给 best/cand 各自 load_state_dict 被绕过 ✗
        #         → best 与 cand 实际同一个模型 → 门控变"自己打自己" → 恒 50.0% ✗
        #   铁证: 同 seed 两局长度差 0/50 (镜像) | 关掉后 6/6 非零 + 候选 12:0 ✓
        #   附带收益: 省掉 socket 往返 → 门控快 2.6 倍 (12 局/分 vs 4.6) ✓✓
        _env = os.environ.copy()
        _env.pop('USE_BATCH_INFER', None)
        procs.append(_subprocess.Popen(cmd, cwd=SCRIPT_DIR, stdout=lf,
                                       stderr=_subprocess.STDOUT, env=_env))
    rc = [p.wait() for p in procs]
    logging.info('【门控】分片结束 rc={} 用时 {:.1f} 分钟'.format(rc, (time.time() - t0) / 60))

    # ── 判决 ──
    rep = _subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, 'scripts', 'gate_report.py'),
                           '--tag', tag, '--games', str(GATE_GAMES), '--out', GATE_LOG_DIR],
                          cwd=SCRIPT_DIR, capture_output=True, text=True)
    out = (rep.stdout or '') + (rep.stderr or '')
    logging.info('【门控】判决输出:\n{}'.format(out.strip()))

    # 解析 gate_summary.tsv 末行
    sumfile = None
    for _cand_path in [os.path.join(GATE_LOG_DIR, 'gate_summary.tsv'),
                       os.path.join(SCRIPT_DIR, 'gate_log', 'gate_summary.tsv')]:
        if os.path.exists(_cand_path):
            sumfile = _cand_path
            break
    if sumfile is None:
        sumfile = os.path.join(GATE_LOG_DIR, 'gate_summary.tsv')
    verdict, rate, cw, cb = None, None, None, None
    try:
        lines = [l for l in open(sumfile).read().strip().split('\n') if l.strip()]
        f = lines[-1].split('\t')
        if f[0] == tag:
            n_games, best_w, cand_w, draws = int(f[1]), int(f[2]), int(f[3]), int(f[4])
            rate = float(f[5])
            cb = f[8]
            cw = f[9]
            verdict = f[11].strip()
            rec.update({'n_games': n_games, 'best_wins': best_w, 'cand_wins': cand_w,
                        'draws': draws, 'rate': rate, 'cand_as_black': cb, 'cand_as_white': cw})
    except Exception as e:
        logging.warning('【门控】解析汇总失败: {}'.format(e))

    if verdict is None:
        logging.error('【门控】无法解析判决 → 保守处理: 不晋升不回滚')
        rec['verdict'] = 'ERROR'
        _append_gate_history(rec)
        return rec

    # ── 死锁告警 (只报警不改判) ──
    try:
        cand_w_white = int(cw.split('/')[0]) if cw else -1
        best_w_white = int(cb.split('/')[0]) if cb else -1
        if verdict == 'REJECT' and cand_w_white == 0 and best_w_white == 0:
            logging.critical('🚨🚨 【门控死锁告警】双方执白均 0 胜 (黑方 100% 胜) '
                             '→ 配对赛总分恒≈50%, 55% 阈值永远达不到! '
                             '本次已按规则处理, 但建议立即人工介入: '
                             '切杀速判据 / 改用平衡开局协议 / 改规则')
            rec['deadlock_warning'] = True
    except Exception:
        pass

    # ── 2026-09-25 新增: 杀速统计 (只报告, 不改判) ──
    #   背景: 在"黑必胜均衡"下胜率判据恒≈50%失效 ✗ → 用户早前定的方案是改比【手数】
    #   读数: 候选执黑时手数【越短】= 候选杀得越快 ✓; best执黑时手数【越长】= 候选防守越好 ✓
    #   ⚠️ 前提: 必须【每局独立 seed】(旧"同seed配对"下每对两局长度相同 → 两组均值恒等 ✗)
    try:
        _rs = []
        for _f in sorted(glob.glob(os.path.join(GATE_LOG_DIR, tag + '_shard*.jsonl'))):
            for _l in open(_f):
                _l = _l.strip()
                if _l:
                    try:
                        _rs.append(json.loads(_l))
                    except Exception:
                        pass
        if _rs:
            import statistics as _stat
            _cbk = [_r['steps'] for _r in _rs if _r.get('black') == 'cand']
            _bbk = [_r['steps'] for _r in _rs if _r.get('black') == 'best']
            _mc = int(_stat.median(_cbk)) if _cbk else None
            _mb = int(_stat.median(_bbk)) if _bbk else None
            rec['kill_speed'] = {'cand_as_black_median': _mc, 'best_as_black_median': _mb,
                                 'cand_as_black_n': len(_cbk), 'best_as_black_n': len(_bbk)}
            _s1 = '候选杀得快 ✓' if (_mc is not None and _mb is not None and _mc < _mb) else \
                  ('best杀得快 ✗' if (_mc is not None and _mb is not None and _mb < _mc) else '持平')
            _s2 = '候选防守好 ✓' if (_mc is not None and _mb is not None and _mb > _mc) else \
                  ('候选防守差 ✗' if (_mc is not None and _mb is not None and _mb < _mc) else '持平')
            logging.info('【门控】杀速统计(报告用, 不改判): 候选执黑中位手数 {} (n={}) [{}] | '
                         'best执黑中位手数 {} (n={}) [{}]'.format(
                             _mc, len(_cbk), _s1, _mb, len(_bbk), _s2))
    except Exception as _e:
        logging.warning('【门控】杀速统计失败: {}'.format(_e))

    # ── 晋升 / 回滚 ──
    if verdict == 'PASS':
        _shutil.copy(WEIGHTS_PATH, BEST_WEIGHTS_PATH)
        rec['verdict'] = 'PASS'
        logging.info('【门控】✅ 通过 (候选胜率 {:.1%} ≥ {:.0%}) → 晋升 weights_best.pt'.format(
            rate or 0, GATE_THRESHOLD))
    else:
        rec['verdict'] = 'REJECT'
        logging.info('【门控】❌ 不通过 (候选胜率 {:.1%} < {:.0%})'.format(rate or 0, GATE_THRESHOLD))
        # ── 2026-09-25 用户拍板 (门控 BUG 修复后): REJECT → 【回滚到 best, 拒绝用候选继续产局】✓ ──
        #   为什么现在改成回滚 (此前"不回滚"的结论已作废 ✗):
        #     那个"不回滚"是在门控【带病(共享推理=自己打自己, 恒50%)】的假数据下拍的 ✗
        #     门控已修真 → REJECT 是【真信号】(候选确实没变强) → 应该拒绝使用 ✓
        #   不回滚的风险: 候选继续训练会持续漂移, 与 best 越走越远,
        #     而门控每次都拒 → 【无界漂移】✗✗ (训练跑到 best 之外去, 还不自知)
        #   回滚 = AGZ 原设计 (每轮从 best 出发 ✓)
        #   保留两项: ① 回放池【保留】✓ (弱数据≠毒数据; 5 代滑窗自然淘汰)
        #             ② 检测到标签污染(V̂类) → 回滚 + 清空池 (原逻辑 ✓)
        #   注: 回滚后迭代号继续 (it30 被拒 → 从 it31 起用 best 权重重新爬)
        #       语义上等价于"退回 best 重训 10 代", 只是编号继续, 下次门控落在 it40 ✓
        logging.warning('【门控】执行回滚: weights.pt ← best (拒绝用候选继续产局) ✗')
        if GATE_ROLLBACK:
            try:
                sd = torch.load(BEST_WEIGHTS_PATH, map_location=device)
                if hasattr(agent, 'raw_net') and agent.raw_net is not None:
                    agent.raw_net.load_state_dict(sd, strict=False)
                    agent.raw_net.to(device)
                    agent.net = agent.raw_net
                else:
                    agent.net.load_state_dict(sd, strict=False)
                    agent.net.to(device)
                agent.reset_mcts()
                _shutil.copy(BEST_WEIGHTS_PATH, WEIGHTS_PATH)
                if getattr(agent, '_remote', None) is not None:
                    try:
                        agent._remote.update_weights(BEST_WEIGHTS_PATH)
                    except Exception as e:
                        logging.warning('【门控】通知 server 回滚权重失败: {}'.format(e))
                # 2026-09-17: 普通 REJECT 不清空回放池 (弱数据=合法 off-policy, 且阈值有误判率)
                #   只有检测到标签污染(V̂类)才清空 → 也不会触发重新 warmup ✓
                purged = 0
                if GATE_PURGE_ON_CORRUPT and corrupt:
                    purged = _clear_replay_pool()
                    logging.warning('【门控】🚨 检测到【标签污染】({}) → 清空回放池 {} 个文件'.format(
                        '/'.join(corrupt_why), purged))
                    rec['purged'] = purged
                    rec['purge_reason'] = corrupt_why
                else:
                    logging.info('【门控】回放池【保留】✓ (无标签污染迹象; 弱数据是合法 off-policy 数据, '
                                 '会被 5 代滑窗自然淘汰; 且不触发重新 warmup)')
                logging.info('【门控】方案B 回滚完成: weights.pt←best, 内存权重已还原, '
                             '污染清除 {} 个文件 (候选快照仍在 weights_it{}_v42.pt 可事后分析)'.format(
                                 purged, iteration))
                rec['rolled_back'] = True
            except Exception as e:
                logging.error('【门控】回滚失败 (严重, 需人工确认): {}'.format(e))
                rec['rollback_error'] = str(e)

    _append_gate_history(rec)
    return rec


# ══════════ v38: 熔断器参数与状态 ══════════
#   连续3代上升 且 空盘>0.25 → 回滚到最近健康点 (<0.15) ✓
#   关闭方法: GOMOKU_SPIRAL_RISE=0 (行为与 v37 完全一致 ✓)
_SPIRAL_RISE = int(os.environ.get('GOMOKU_SPIRAL_RISE', '3'))
_SPIRAL_THR = float(os.environ.get('GOMOKU_SPIRAL_THR', '0.25'))
_SPIRAL_HEALTHY = float(os.environ.get('GOMOKU_SPIRAL_HEALTHY', '0.15'))
_SPIRAL_MAXTRIP = int(os.environ.get('GOMOKU_SPIRAL_MAXTRIP', '2'))
# ★ 2026-09-26 用户拍板: 新增 MODE 开关 (默认 full = 原行为不变 ✓)
#   full = 触发即回滚 + 30代内2次停机 (v38 原行为)
#   warn = 【只告警】不回滚/不停机 (角色镜像版 v39 用; 见 _spiral_guard 内注释)
_SPIRAL_MODE = os.environ.get('GOMOKU_SPIRAL_MODE', 'full').strip().lower()
_spiral_hist = []
_spiral_trips = []
# ★ 2026-09-26 (用户拍板"动态取学生执色"): 胜率口径不再硬编码白方 ──
#   教练执黑(1)  → 学生 = 白(-1)
#   教练执白(-1) → 学生 = 黑(+1)   ← v39 镜像版
#   无教练(0, 普通自对弈) → 退回旧口径(白胜), 保持历史脚本行为不变
_COACH_C = _current_coach_color()          # v42: 动态读 swap_state (换边真相源), 回退 env
_STUDENT_C = (-_COACH_C) if _COACH_C in (1, -1) else -1
_STUDENT_NAME = {1: '黑', -1: '白'}.get(_STUDENT_C, '白')


def _refresh_student_color():
    """v42 (自动换边): 每代从 swap_state 重读教练/学生色, 刷新模块级 _COACH_C/_STUDENT_C/_STUDENT_NAME
    (原为模块级只读一次 ⇒ 换边后熔断器/胜率统计仍用旧色 ✗)"""
    global _COACH_C, _STUDENT_C, _STUDENT_NAME
    _COACH_C = _current_coach_color()
    _STUDENT_C = (-_COACH_C) if _COACH_C in (1, -1) else -1
    _STUDENT_NAME = {1: '黑', -1: '白'}.get(_STUDENT_C, '白')
    return _STUDENT_C
# ★ v38 (2026-09-25): 学生胜率历史 — 供熔断器区分"真螺旋(胜率同步下滑) vs 教练侧副作用"
_winrate_hist = []


def _rebuild_winrate_hist():
    """重启后从回放池重建白胜率历史 (否则熔断器判据因历史为空而失效 ✗)"""
    try:
        import glob as _g, pickle as _p, re as _re
        _files = []
        for _f in _g.glob(os.path.join(REPLAY_DIR, 'iter*.pkl')):
            _m = _re.search(r'iter(-?\d+)\.pkl$', _f)
            if _m and int(_m.group(1)) > 0:
                _files.append((int(_m.group(1)), _f))
        _files.sort()
        for _it, _f in _files[-8:]:
            try:
                _dd = _p.load(open(_f, 'rb'))
                # ⚠️ 回放池文件是"列表"[逐局 DataFrame], 不是单个 df (2026-09-25 踩过 ✗)
                if isinstance(_dd, list) and len(_dd) > 0:
                    _w = sum(1 for _d in _dd if int(_d['winner'].iloc[0]) == _STUDENT_C)
                    _winrate_hist.append(100.0 * _w / len(_dd))
            except Exception:
                pass
        if _winrate_hist:
            logging.info('学生({})胜率历史(重启重建, {}代): {}'.format(
                _STUDENT_NAME, len(_winrate_hist),
                ' '.join('%.0f%%' % v for v in _winrate_hist)))
        else:
            logging.warning('学生({})胜率历史重建为空 (回放池无数据) → 熔断器将退回旧规则'.format(
                _STUDENT_NAME))
    except Exception as _e:
        logging.warning('学生胜率历史重建失败(忽略): {}'.format(_e))


_rebuild_winrate_hist()
_last_healthy = [None, None]     # [iteration, 空盘value]
if _SPIRAL_RISE > 0:
    if _SPIRAL_MODE == 'warn':
        logging.info('v38 熔断器: 连{}代上升且>{:.2f} → 【仅告警】不回滚/不停机 (GOMOKU_SPIRAL_MODE=warn)'.format(
            _SPIRAL_RISE, _SPIRAL_THR))
    else:
        logging.info('v38 熔断器: 连{}代上升且>{:.2f} → 回滚到最近健康点(<{:.2f}); 30代内{trip}次即停'.format(
            _SPIRAL_RISE, _SPIRAL_THR, _SPIRAL_HEALTHY, trip=_SPIRAL_MAXTRIP))
else:
    logging.info('v38 熔断器: OFF (GOMOKU_SPIRAL_RISE=0)')


import shutil as _shutil          # v38: 模块级导入 (门控那处是函数内导入, 熔断器看不到 ✗)


def _spiral_guard(result, iteration, agent):
    """v38: 空盘value 螺旋熔断器 (2026-09-25 立)

    判据(实测依据): 单代尖峰【会自愈】(v37 it14 +0.612 → it15 +0.074 ✓)
                    单调爬升【不自愈】(it31→it37 连升 7 代 ✗)
      → 所以用"连续 _SPIRAL_RISE 代上升 且 超过 _SPIRAL_THR"
    动作: 回滚 weights.pt ← 最近健康检查点 + 内存同步 + 通知 server + 写标记文件
    返回: 'skip'(关闭/无体检) / 'ok'(未触发) / 'ok-sideeffect'(判定教练侧副作用) /
          'warned'(MODE=warn 仅告警) / 'tripped'(已回滚) / 'stopped'(超次数停机) / 'error'
    ⚠️ 本函数只做原地变更 (_spiral_hist.clear() 等), 绝不对这些名字重新赋值
       —— 否则 Python 会把它们当局部变量, 整个函数访问不到 (2026-09-25 真实踩过 ✗)
    """
    if result is None or _SPIRAL_RISE <= 0:
        return 'skip'
    try:
        _ev = float(result.get('value_空盘', 0.0))
        if abs(_ev) < _SPIRAL_HEALTHY:
            _last_healthy[0] = iteration
            _last_healthy[1] = _ev
        _spiral_hist.append(_ev)
        if len(_spiral_hist) < _SPIRAL_RISE + 1:
            return 'ok'
        _win = _spiral_hist[-(_SPIRAL_RISE + 1):]
        if not all(_win[i + 1] > _win[i] for i in range(len(_win) - 1)):
            return 'ok'
        if _ev <= _SPIRAL_THR:
            return 'ok'
        # ★★★ 加"学生胜率是否在下滑"判据 (执色动态: _STUDENT_C)
        #   依据: v37 真螺旋 = 空盘↑ 且 学生胜率↓ (跌到4~12%) ✗
        #         v38/v39 教练法 = 空盘↑ 但 学生胜率↑ ✓ = 教练侧局面的副作用, 非螺旋
        #   ⚠️ 2026-09-26 修: 原来硬编码按白方统计 → 镜像版(v39 教练执白)方向反了 ✗
        #   规则: 近3代均值 > 基线(最初3代)均值×0.8 → 学生在进步/持稳 → 判定副作用, 不触发
        if len(_winrate_hist) >= 6:
            _rec = sum(_winrate_hist[-3:]) / 3.0
            _base = sum(_winrate_hist[:3]) / 3.0
            if _rec > _base * 0.8:
                logging.warning(
                    '【熔断】空盘上升但学生({})胜率未下滑 (近3代 {:.1f}% vs 基线 {:.1f}%) '
                    '→ 判定教练侧副作用, 不触发 ✓'.format(_STUDENT_NAME, _rec, _base))
                _spiral_hist.clear()
                return 'ok-sideeffect'
        # ★★ 2026-09-26 用户拍板: GOMOKU_SPIRAL_MODE=warn → 【只告警】不回滚/不停机 ✓
        #   理由: v39 角色镜像(学生执黑 = 先手方)后空盘value语义随之改变;
        #         THR=0.25 / RISE=3 是 v38(学生执白 = 后手)时代标定的, 镜像后未重标定 ✗
        #   实测: 空盘 0.2931 → 0.3668 与 黑胜率 50% → 60% 【同向】 → 更像"学生把自己的
        #         先手优势学出来"的伴随现象, 而非 value 自举螺旋(有冻结教练当外部锚点 ✓)
        #   本分支: 只写日志 + spiral_warn.log 留证; 不复制权重/不通知 server/不写 aborted_spiral.flag
        #          /不计入 _spiral_trips (因此永不 SystemExit 停机 ✓)
        if _SPIRAL_MODE == 'warn':
            logging.critical('【熔断·仅告警】空盘value螺旋警报: 连{}代上升 {} (阈值 {}) → MODE=warn: 不回滚/不停机 ✗'.format(
                _SPIRAL_RISE, ['%+.4f' % v for v in _win], _SPIRAL_THR))
            try:
                with open(os.path.join(CHECKPOINT_DIR, 'spiral_warn.log'), 'a') as _fw:
                    _fw.write('{} iter={} seq={} thr={} (MODE=warn: 无动作)\n'.format(
                        time.strftime('%Y-%m-%d %H:%M:%S'), iteration,
                        ','.join('%+.4f' % v for v in _win), _SPIRAL_THR))
            except Exception as _e3:
                logging.warning('【熔断】写 spiral_warn.log 失败(忽略): {}'.format(_e3))
            _spiral_hist.clear()
            return 'warned'
        logging.critical('【熔断】空盘value螺旋警报: 连{}代上升 {} (阈值 {})'.format(
            _SPIRAL_RISE, ['%+.4f' % v for v in _win], _SPIRAL_THR))
        _tgt = None
        if _last_healthy[0] is not None:
            _cand_f = os.path.join(CHECKPOINT_DIR, 'weights_it{}_v42.pt'.format(_last_healthy[0]))
            if os.path.exists(_cand_f):
                _tgt = _cand_f
        if _tgt is None and os.path.exists(BEST_WEIGHTS_PATH):
            _tgt = BEST_WEIGHTS_PATH
        if _tgt is None:
            logging.critical('【熔断】找不到可回滚的检查点 → 不动作, 仅告警 ✗')
            return 'ok'
        _shutil.copy(_tgt, WEIGHTS_PATH)
        _sd = torch.load(_tgt, map_location=device)
        _net = agent.raw_net if (hasattr(agent, 'raw_net') and agent.raw_net is not None) else agent.net
        _net.load_state_dict(_sd, strict=False)
        if getattr(agent, '_remote', None) is not None:
            try:
                agent._remote.update_weights(WEIGHTS_PATH)
            except Exception as _e2:
                logging.warning('【熔断】通知 server 回滚权重失败: {}'.format(_e2))
        open(os.path.join(CHECKPOINT_DIR, 'aborted_spiral.flag'), 'a').write(
            'iter={} seq={} rollback_to={}\n'.format(iteration, _win, os.path.basename(_tgt)))
        logging.critical('【熔断】已回滚 weights.pt ← {} (内存已还原, 标记文件已写) ✓'.format(
            os.path.basename(_tgt)))
        _spiral_trips.append(iteration)
        _spiral_hist.clear()          # ★ 原位清空 (曾写成 = [] → UnboundLocalError ✗)
        _recent = [t for t in _spiral_trips if iteration - t < 30]
        if len(_recent) >= _SPIRAL_MAXTRIP:
            logging.critical('【熔断】30代内触发 {} 次 → 停止训练 (需人工确认) ✗'.format(len(_recent)))
            raise SystemExit(0)
        return 'tripped'
    except SystemExit:
        raise
    except Exception as _e:
        logging.warning('【熔断】检查异常(忽略): {}'.format(_e))
        return 'error'


# ══════════ 2026-09-26 用户拍板: 【晚到局结转】 ══════════
# 问题 (实测 it30~34): learn 时只 glob「此刻已完成」的工兵局 → 主进程收满 100 局即开训,
#   排在后面的【长局】在 learn 之后才落盘, 仍带旧 iterN 标签, 后代 glob 不再回头取
#   → 永久丢弃且完全静默: 丢 86/588 = 14.6%, 被丢局均 39.8 步 (= 池内 26.5 的 1.50 倍),
#     ≥31步占比 45% vs 池内 21% ✗ → 训练侧的中后盘/长局样本被腰斩
# 修法 (用户方案): 每代循环开头, 把【上一代晚到的】工兵局去重后并入 replay_pool/iter{N-1}.pkl
#   · 晚到局 = 【已下完的整局 + 完整 qa 序列】(worker 是下完才 pickle.dump) → 直接可训 ✓
#   · 零机时成本; 滞后性可忽略 —— 池里本来就有 4 代前的老数据, 这些只滞后 1 代 ✓
# 开关: GOMOKU_LATE_MERGE=1 启用 (代码默认 0 = 行为不变 ✓); 关掉时同样记日志(可见化)
_LATE_MERGE = os.environ.get('GOMOKU_LATE_MERGE', '0') == '1'


def _game_fp(d):
    """局指纹 (步数, 胜方, 终局棋盘字节) — 去重用"""
    try:
        return (len(d), int(d['winner'].iloc[0]), d['board'].values[-1].tobytes())
    except Exception:
        return None


def _merge_late_games(prev_it):
    """把 prev_it 代【晚到】的工兵局并入 replay_pool/iter{prev_it}.pkl (去重 + 原子写)"""
    if prev_it < 0:
        return
    pool_f = os.path.join(REPLAY_DIR, 'iter{}.pkl'.format(prev_it))
    if not os.path.exists(pool_f):
        return
    try:
        with open(pool_f, 'rb') as f:
            pool = pickle.load(f)
        if not isinstance(pool, list):
            pool = [pool]
        seen = set(_game_fp(g) for g in pool)
        late, late_len = [], []
        for f in sorted(glob.glob(os.path.join(WORKER_DIR, 'iter{}_*_game_*.pkl'.format(prev_it)))):
            try:
                with open(f, 'rb') as fh:
                    g = pickle.load(fh)
            except Exception:
                continue
            k = _game_fp(g)
            if k is None or k in seen:
                continue
            seen.add(k); late.append(g); late_len.append(len(g))
        if not late:
            return
        _n31 = sum(1 for x in late_len if x >= 31)
        logging.warning('【晚到局】it{}: {} 局未进池 (均{:.1f}步, 中位{:.0f}, ≥31步 {}局) → {}'.format(
            prev_it, len(late), sum(late_len) / len(late_len),
            sorted(late_len)[len(late_len) // 2], _n31,
            'MODE=on: 已并入 (池 {}→{}局) ✓'.format(len(pool), len(pool) + len(late))
            if _LATE_MERGE else 'MODE=off: 丢弃 (GOMOKU_LATE_MERGE=0) ✗'))
        if not _LATE_MERGE:
            return
        pool.extend(late)
        _tmp = pool_f + '.tmp'
        with open(_tmp, 'wb') as f:
            pickle.dump(pool, f)
        os.replace(_tmp, pool_f)
        with open(os.path.join(CHECKPOINT_DIR, 'late_merge.log'), 'a') as f:
            f.write('{} it{} +{}局 (均{:.1f}步, ≥31步{}局) → 池 {}局\n'.format(
                time.strftime('%Y-%m-%d %H:%M:%S'), prev_it, len(late),
                sum(late_len) / len(late_len), _n31, len(pool)))
    except Exception as _e:
        logging.warning('【晚到局】it{} 结转失败(忽略): {}'.format(prev_it, _e))


def train(config=None):
    if config is None:
        config = {
            'train_iterations': 500,   # 2026-08-11 续跑: 100→500 (判断用代码默认值, 不是metadata)
            # 2026-09-13 攒批版: 主进程只管调度+训练, 产局全交工兵(20工兵×5局=100局)
            'train_episodes_per_iteration': 0,
            'batches': 32,
            'batch_size': 512,
            'sim_count': 800,       # v6: 固定800模拟，全程不变
            'net_kwargs': {
                'conv_filters': [128],
                'residual_filters': [[128, 128]] * 8,
                'policy_filters': [128],
            },
            'lr': 0.001,
            'worker_count': 20,          # 2026-10-10: 10->20 提速(每代 ~70min→~45min)
            'worker_games_per_worker': 6,   # 2026-10-10: 11->6 (20×6=120 期望, 与 v41 一致)
            # 超额原因: 主进程等满 100 局就开训, 但若某工兵抽到长局(80+步, 10-18分钟)
            #   则恰好卡在 99/100 死等。多产 1 局后 19 个快工兵先凑够 → 完全不等长局。
            'worker_expected_games': 100,   # 主进程实际等待的目标局数 (20工兵×5 = 100)
            'value_target': 'search_q',  # v33 (2026-09-05): value标签=自对弈搜索Q软标签(用户方案)
            #   治±1全局标签信用分配: 输局好步也吃-1 / 空盘被单边数据推极端 → 崩白;
            #   Q=该步800sim搜索值, terminal命中给真实信号, 无死锁(用户拍板纯Q)
            # v19 (2026-08-26, Kimi 三步走): 阶段1 前 N 代冻结主干+Policy 只训 Value 头,
            # N 代后解冻全网络。value 头学会读取主干特征后再让主干适应新监督。
            # 08-26 用户拍板: 冻结 5 → 10 代 (新 value 头容量小, 多给几代稳定学特征读取)
            'freeze_policy_iters': 0,   # v24: 无冻结, 标准一起训 (ch2 恒1 已灭捷径, 无需防抢跑手术)
            'freeze_value_iters': 0,    # v24 (2026-08-30, 从零+ch2恒1): 不冻结, 全网络一起训
        }

    env = gym.make('Gomuku-v0')
    agent = AlphaZeroAgent(env=env, kwargs=config['net_kwargs'],
                           batches=config['batches'],
                           batch_size=config['batch_size'],
                           sim_count=config['sim_count'],
                           lr=config.get('lr', 0.001),
                           disable_compile=True)
    logging.info('v2 训练启动, device={}'.format(device))

    train_iterations = config['train_iterations']
    train_episodes_per_iteration = config['train_episodes_per_iteration']

    resumed = load_checkpoint(agent, config)
    start_iteration = 0
    if resumed is not None:
        start_iteration = resumed

    # ★ v42: 续跑兜底 —— 已有学生权重但教练权重缺失 (手工删过 / 老目录升级) 时补建首个教练;
    #   从零首启时 weights.pt 尚不存在, 此处 no-op, 由 save_checkpoint 首次落盘后完成 ✓
    _ensure_coach_weights()

    if start_iteration >= train_iterations:
        logging.info('训练已完成, 如需重跑请删除 {} 目录'.format(CHECKPOINT_DIR))
        return

    GAME_DATA_DIR = os.path.join(CHECKPOINT_DIR, 'game_data')
    os.makedirs(GAME_DATA_DIR, exist_ok=True)
    os.makedirs(REPLAY_DIR, exist_ok=True)

    for iteration in range(start_iteration, train_iterations):
        # ── v42: 每代开头刷新学生/教练色 (自动换边真相源 swap_state) ──
        _refresh_student_color()

        # ── 2026-09-26 用户拍板: 【晚到局】结转 (长局不再被静默丢弃) ──
        #   实测 it30~34 丢 86/588 局 (14.6%), 且全是长局 (均 39.8 步 vs 池内 26.5) ✗
        #   覆盖【池窗口内所有代】而非只 iteration-1 → 任何一代的晚到局都能救回 (幂等, 无副作用)
        for _mi in range(max(0, iteration - REPLAY_POOL_SIZE), iteration):
            _merge_late_games(_mi)
        # ── v24 (2026-08-30, 从零+ch2恒1): 无冻结, 标准 AlphaZero 全网络一起训 ──
        # (ch2 恒1 已从输入层灭空盘捷径, value 无捷径可抢跑, 冻结/λ 等手术刀不再需要)
        # ── v2: 模拟量固定 400（全程不变，对比实验） ──
        # ── v6: 固定 800 模拟（全程不变） ──
        target = config.get('sim_count', 800)
        if agent.sim_count != target:
            agent.sim_count = target
            logging.info('v8 固定sim: 设置至 {} (代{})'.format(target, iteration))

        # ── 开局约束：不强制，让网络自由学习开局（v2 不启用） ──
        force_center = False

        dfs_trajectory = []
        saved_games = sorted(glob.glob(
            os.path.join(GAME_DATA_DIR, f'iter{iteration}_game_*.pkl')))
        if saved_games:
            logging.info('检测到 {} 场已保存的对局数据，正在恢复...'.format(len(saved_games)))
            for gf in saved_games:
                with open(gf, 'rb') as f:
                    dfs_trajectory.append(pickle.load(f))
            completed = len(dfs_trajectory)
        else:
            completed = 0
        iter_start = time.time()

        # ── 主进程自对弈 ──
        for episode in range(completed, train_episodes_per_iteration):
            df_trajectory = self_play(env, agent,
                                      return_trajectory=True, verbose=False,
                                      force_center=force_center)
            winner = df_trajectory['winner'].iloc[0]
            winner_str = '黑胜' if winner == 1 else '白胜' if winner == -1 else '平局'
            elapsed = time.time() - iter_start
            remaining = elapsed / (episode + 1) * (
                train_episodes_per_iteration - episode - 1)
            logging.info('代{} 局{}/{}: {} 经验{}条 | sim={} | {:.0f}s/局 | 已过{:.0f}s 预计剩余{:.0f}s'.format(
                iteration, episode + 1, train_episodes_per_iteration,
                winner_str, len(df_trajectory), agent.sim_count,
                elapsed / (episode + 1), elapsed, remaining))
            dfs_trajectory.append(df_trajectory)
            game_path = os.path.join(GAME_DATA_DIR,
                                     f'iter{iteration}_game_{episode}.pkl')
            with open(game_path, 'wb') as f:
                pickle.dump(df_trajectory, f)
        main_games = len(dfs_trajectory)
        logging.info('自对弈完成: {}局 (主进程)'.format(len(dfs_trajectory)))

        # ── 收集工兵经验 ──
        worker_games = 0
        os.makedirs(WORKER_DIR, exist_ok=True)
        worker_game_pattern = os.path.join(
            WORKER_DIR, 'iter{}_*_game_*.pkl'.format(iteration))
        # 2026-09-13: 与工兵"超额产局"解耦 —— 工兵多产作缓冲, 主进程只等这个目标数
        expected = config.get('worker_expected_games',
                              config.get('worker_count', 9) * config.get('worker_games_per_worker', 10))
        # v2 fix: 首次迭代无已存权重时跳过等待工兵（工兵需 weights.pt 才能启动）
        if os.path.exists(WEIGHTS_PATH):
            # 2026-09-13: 停滞超时保护 —— 防"某工兵长局/崩溃导致缺 1 局, 主进程 while True 死等"
            stall_timeout = 600   # 10 分钟无新局到齐 → 用现有局开训
            last_count = -1
            last_change = time.time()
            while True:
                done = sorted(glob.glob(worker_game_pattern))
                if len(done) >= expected:
                    break
                if len(done) != last_count:
                    last_count = len(done)
                    last_change = time.time()
                elif time.time() - last_change > stall_timeout:
                    logging.warning('⚠️ 等待工兵停滞超时({}s 无新增, 仍 {}/{}局) → 用现有局开训'.format(
                        stall_timeout, len(done), expected))
                    break
                logging.info('等待工兵: {}/{}局已到'.format(len(done), expected))
                time.sleep(10)
        else:
            logging.info('首次迭代：无已存权重，跳过等待工兵，仅用主进程数据')
        worker_files = sorted(glob.glob(worker_game_pattern))
        _dfs_stu_c = []   # v44: 逐局【学生执色】(口径与工兵一致: 文件名 gid 偶=黑/奇=白)
        for gf in worker_files:
            _dfs_stu_c.append(_student_color_for_game(gf))
            with open(gf, 'rb') as f:
                dfs_trajectory.append(pickle.load(f))
        # ★ v44 双色同训: 本代学生【执黑胜率】与【执白胜率】分开统计 (各 ~50 局) → 双门槛 + 熔断器
        _refresh_student_color()   # 保留: 熔断器(仅告警)仍需 _STUDENT_C/_STUDENT_NAME
        _wr_black = _wr_white = float('nan')
        try:
            _n_b = _n_w = _wins_b = _wins_w = 0
            for _i, _d in enumerate(dfs_trajectory):
                # 学生执色: 跳过主进程无教练局(计 0, 不计分色)
                _sc = (_dfs_stu_c[_i - main_games]
                       if main_games <= _i < main_games + len(_dfs_stu_c) else 0)
                _win = int(_d['winner'].iloc[0])
                if _sc == 1:
                    _n_b += 1; _wins_b += (1 if _win == 1 else 0)
                elif _sc == -1:
                    _n_w += 1; _wins_w += (1 if _win == -1 else 0)
            if _n_b + _n_w == 0:
                # 空集(0局) 绝不可记为 0% ✗ (否则污染 _winrate_hist + 把换边 streak 清零): 保留 NaN
                logging.warning('本代学生胜率: 无有效对局 (收集 0 局, 未计入历史/换边计数)')
            else:
                _wr_black = 100.0 * _wins_b / _n_b if _n_b else float('nan')
                _wr_white = 100.0 * _wins_w / _n_w if _n_w else float('nan')
                _winrate_hist.append(100.0 * (_wins_b + _wins_w) / (_n_b + _n_w))   # 熔断器用(整体)
                logging.info('v44 本代学生双色胜率: 执黑 {:.1f}% ({}/{}局) | 执白 {:.1f}% ({}/{}局)'.format(
                    _wr_black, _wins_b, _n_b, _wr_white, _wins_w, _n_w))
        except Exception as _e:
            logging.warning('双色胜率统计失败(忽略): {}'.format(_e))
            worker_games += 1
        logging.info('收集工兵: {}局, 总计 {}局'.format(worker_games, len(dfs_trajectory)))

        # ── 杀工兵释放GPU (★ 2026-09-26 修: ① 名字改回 v39 ② LATE_MERGE=1 时不杀) ──
        #   原两个 bug (互为掩盖): 这里 pkill 的是 'Gomoku-worker_v38.py' ✗, 而下面"重启工兵"
        #   那处拉起的也是 v38 工兵 ✗ → 整个第一轮每代都在"拉起 v38 工兵 / 杀 v38 工兵",
        #   而那些 v38 工兵抱着【v38 槽】的 metadata 反复产局 →
        #   ★ 这才是 2026-09-26 "it64 数据被污染"的真正来源 ✗✗ (不是 LATE_MERGE)
        import subprocess
        _lm = bool(globals().get('_LATE_MERGE', False))
        if _lm:
            logging.info('工兵【未杀】(GOMOKU_LATE_MERGE=1): 让在下的局下完 → 下代结转进池 ✓')
        else:
            subprocess.run(['pkill', '-9', '-f', 'Gomoku-worker_v44.py'], capture_output=True)
        time.sleep(5)
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info('GPU 缓存已释放 (工兵{}杀)'.format('未' if _lm else '已'))

        # ── v2: 经验回放池（保留最近 N 代数据） ──
        # 将当前代数据写入回放池
        pool_file = os.path.join(REPLAY_DIR, f'iter{iteration}.pkl')
        with open(pool_file, 'wb') as f:
            pickle.dump(dfs_trajectory, f)
        # 收集最近 N 代（⚠️ 必须按迭代号数值排序！glob 返回字典序，iter10 < iter2，
        # sorted() 会永远取到 iter5-9 而非最新代——2026-08-01 实踩的致命 bug）
        pool_files = sorted(glob.glob(os.path.join(REPLAY_DIR, 'iter*.pkl')),
                            key=lambda p: int(os.path.basename(p)[4:-4]))[-REPLAY_POOL_SIZE:]
        replay_dfs = []
        pool_detail = []
        import time as _t
        for pf in pool_files:
            with open(pf, 'rb') as f:
                games = pickle.load(f)
            mtime = _t.strftime('%m-%d %H:%M', _t.localtime(os.path.getmtime(pf)))
            pool_detail.append('{} ({}局, {})'.format(os.path.basename(pf), len(games), mtime))
            replay_dfs.extend(games)
        # 清晰输出回放池明细：每代文件名/局数/写入时间（便于发现新旧数据混入）
        logging.info('经验回放池: 取最近 {} 代 = [{}], 共 {} 局'.format(
            len(pool_files), ', '.join(pool_detail), len(replay_dfs)))

        # ── 学习 ──
        # v32 warmup (2026-09-05 用户拍板): 池攒满 REPLAY_POOL_SIZE 代前只自对弈积累,
        # 不学习 (防 value 用 1-2 代单边数据就开训 → 空盘 +0.54 过拟合)
        total_exp = len(replay_dfs)                      # 局数
        total_steps = sum(len(g) for g in replay_dfs)     # 真实样本条数 (每步 1 条)
        warmup = len(pool_files) < REPLAY_POOL_SIZE
        if warmup:
            logging.info('⏳ WARMUP 攒池 {}/{} 代: 本轮只产数据不学习 (权重保持)'.format(
                len(pool_files), REPLAY_POOL_SIZE))
            policy_loss = value_loss = total_loss = float('nan')
        else:
            logging.info('训练 {}: 回放池 {}局 / {}条样本, batches={}, batch_size={}'.format(
                iteration, total_exp, total_steps, agent.batches, agent.batch_size))
            loss_info = agent.learn(replay_dfs)
            policy_loss = loss_info.get('policy_loss', 0)
            value_loss = loss_info.get('value_loss', 0)
            total_loss = loss_info.get('total_loss', 0)
            logging.info('训练 {}: 学习完成  policy_loss={:.4f}  value_loss={:.4f}  total={:.4f}'.format(
                iteration, policy_loss, value_loss, total_loss))

        # 存档 (warmup 也存档推进 iteration, 权重未变)
        config['losses'] = config.get('losses', {})
        if not warmup:
            config['losses'][str(iteration)] = {
                'policy_loss': round(policy_loss, 4),
                'value_loss': round(value_loss, 4),
                'total_loss': round(total_loss, 4),
            }
        save_checkpoint(agent, iteration, config)

        # ── v44 双色同训·双门槛换边调度 (★ 必须在 learn + save_checkpoint 之后:
        #    保证本代【存档】用【换边前】的学生网; 换边后 swap_state / 权重 立即翻转,
        #    供【下一代】工兵读取) ──
        try:
            _maybe_swap(agent, iteration, _wr_black, _wr_white, config)
        except Exception as _e:
            logging.warning('v44 自动换边调度异常(忽略, 本代不换边): {}'.format(_e))

        # ── v2: 每代健康体检 (warmup 跳过: 权重未变, 结果重复) ──
        result = None
        if not warmup:
            try:
                result = health_check(agent, iteration)
            except Exception as e:
                logging.warning('健康体检失败: {}'.format(e))


        # ── v38: 空盘value 螺旋熔断器 (已抽成 _spiral_guard, 见上方定义) ──
        _spiral_guard(result, iteration, agent)



        # ── v37 门控 (每 GATE_EVERY 代; warmup 跳过) ──
        #   用户拍板: 严格按胜率 55% 判决; 失败则方案B 回滚
        #   ⚠️ 注意: 健康体检在本行之前执行, 若门控触发回滚, 本代 health_log 记录的是
        #      【被丢弃的候选】的体检值 (分析时需对照 gate_history.json 剔除这些代)
        if GATE_ENABLED and not warmup and iteration > 0 and iteration % GATE_EVERY == 0:
            try:
                _gate = _run_gate(iteration, agent, config, health=result)
                logging.info('【门控】代{} 结果: {}'.format(iteration, _gate.get('verdict')))
            except Exception as e:
                logging.error('【门控】异常 (保守处理: 不晋升不回滚): {}'.format(e))

        # ── TensorBoard 权重监控（2026-08-01 新增：可视化权重更新; warmup 跳过） ──
        if not warmup:
            try:
                import torch.utils.tensorboard as _tb
                _net = agent.raw_net if (hasattr(agent, 'raw_net') and agent.raw_net is not None) else agent.net
                _tb_dir = os.path.join(CHECKPOINT_DIR, 'tb_runs')
                _writer = _tb.SummaryWriter(log_dir=_tb_dir)
                # loss 标量
                _writer.add_scalar('loss/policy', policy_loss, iteration)
                _writer.add_scalar('loss/value', value_loss, iteration)
                _writer.add_scalar('loss/total', total_loss, iteration)
                # 权重统计（每层 weight 的 mean/std + 全网络 L2 范数）
                total_l2 = 0.0
                for name, module in _net.named_modules():
                    if hasattr(module, 'weight') and module.weight is not None:
                        w = module.weight.detach().float()
                        total_l2 += float((w ** 2).sum())
                        _writer.add_scalar('weight_mean/{}'.format(name), float(w.mean()), iteration)
                        _writer.add_scalar('weight_std/{}'.format(name), float(w.std()), iteration)
                    if hasattr(module, 'bias') and module.bias is not None:
                        b = module.bias.detach().float()
                        _writer.add_scalar('bias_mean/{}'.format(name), float(b.mean()), iteration)
                _writer.add_scalar('weight/l2_total', total_l2 ** 0.5, iteration)
                # BN gamma 统计（检测 gamma 萎缩）
                bn_gammas = []
                for name, module in _net.named_modules():
                    if isinstance(module, nn.BatchNorm2d):
                        bn_gammas.append(float(module.weight.detach().mean()))
                if bn_gammas:
                    _writer.add_scalar('bn/gamma_mean', sum(bn_gammas) / len(bn_gammas), iteration)
                # 激活率（若体检已算）
                if result is not None and result.get('act_mean') is not None:
                    _writer.add_scalar('health/activation_rate', result['act_mean'] / 100.0, iteration)
                if result is not None and result.get('bn_dead') is not None:
                    _writer.add_scalar('health/dead_neurons', result['bn_dead'], iteration)
                _writer.close()
                logging.info('TensorBoard 已写入: {} (代{})'.format(_tb_dir, iteration))
            except Exception as e:
                logging.warning('TensorBoard 写入失败: {}'.format(e))

        # ── 重启工兵 (★ 2026-09-26 修: ① 脚本名 v38→v39 ✗✗ ② 改成"仅补缺") ──
        import subprocess as _sp
        _need = config.get('worker_count', 9)
        try:
            _alive = int(_sp.run(['pgrep', '-fc', '[G]omoku-worker_v44.py'],
                                 capture_output=True, text=True).stdout.strip() or 0)
        except Exception:
            _alive = 0
        if _alive >= _need:
            logging.info('工兵巡检: 存活 {} 个 (需 {}), 跳过重启 ✓'.format(_alive, _need))
        else:
            for wid in range(_need):
                # 用 >> 追加日志，避免覆盖旧日志（历史教训: > 会清空旧日志）
                cmd = 'screen -dmS gomoku_w{} bash -c "{} -u {} {} >> {} 2>&1"'.format(
                    wid, sys.executable,
                    os.path.join(SCRIPT_DIR, 'Gomoku-worker_v44.py'),
                    wid, os.path.join(SCRIPT_DIR, 'worker_v44_{}.log'.format(wid)))
                os.system(cmd)
            logging.info('工兵巡检: 存活 {} 个 → 补起至 {} 个'.format(_alive, _need))


if __name__ == '__main__':
    import hashlib
    train()
