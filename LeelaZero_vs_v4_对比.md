# Leela Zero 训练策略 vs **正在跑的 v4**（parallel_train.py 管线）

> 上一版本文比错了脚本：当时对的是 `Gomoku-v0_pt_v4.py`（web/对战用的推理参考脚本，那个是 v42/v44 双色同训血缘）。
> **正在跑的**是 `parallel_train.py + train.py + worker_selfplay.py + mcts_alphaZero.py + game.py + net_v24.py`
> 这条 junxiaosong 血缘（`logs/train_v4.log`）。本文全部以实际进程 argv + 代码行数为准。

## 0. 实测运行态（`ps` 取的真实 argv + `logs/train_v4.log`）

```
parallel_train.py --workers 8 --board_width 15 --board_height 15 --n_in_row 5 \
  --n_playout 800 --game_batch_num 100000 --check_freq 50 --publish_every 40 \
  --batch_size 512 --epochs 5 --value_target z \
  --dirichlet_eps 0.25 --dirichlet_alpha 0.03 \
  --init_model checkpoints/ckpt_it2656.model --batch_offset 2656 --init_lr_multiplier 0.0878
worker_selfplay.py --n_playout 800 --temp 1.0 --dirichlet_eps 0.25 --dirichlet_alpha 0.03 \
  --temp_switch 0 --device cpu
```

| 项 | 实测值 |
|---|---|
| 已跑 | 13418 局 / 2 天 14 小时（≈2.9 局/分），版本 252（每 40 局发一次权重） |
| loss / entropy | 2.64~2.72 / 2.25~2.49（ln225=5.416 为均匀基线） |
| explained_var old/new | 0.38~0.51 / 0.58~0.68 |
| kl / lr_multiplier | 0.03~0.06 / **0.088**（有效 lr = 2e-3×0.088 = **1.76e-4**） |
| 形态（最近 100 局） | 均手 20.7 / 中位 20 / **黑 75% : 白 25%** / ≤11 手速杀 19% |
| 开局多样性 | 最近 100 局前 4 手 **100/100 全不重复**（多样性不缺） |

关键代码事实：
- `train.py:80` `c_puct = 5`；`train.py:81` `buffer_size = 10000`；`train.py:73` `learn_rate = 2e-3`；`train.py:86` `kl_targ = 0.02`
- `game.py:213-216` 落子温度 **前 4 手 T=1.0，之后 `exp(-(n-4)/8)`，地板 0.05**（第 8 手 0.61、12 手 0.37、16 手 0.22、≥28 手 0.05）。worker 传的 `--temp 1.0` 在这条路径里**没被用到**。
- `mcts_alphaZero.py:175` 训练标签 = **去温度的访问数归一化**（`last_label`）→ 这点与 LZ 一致。
- `mcts_alphaZero.py:230` 自对弈噪声 = **只加到「落子抽样分布」上**：`move = choice(acts, (1-ε)·probs + ε·dirichlet)`，**树里的根先验没有加噪**。
- `parallel_train.py:129-141` `value_target='z'`（±1 硬标签）；`get_equi_data` 8 倍增强后直接进 buffer。
- `parallel_train.py:47-54` `publish()` **无条件发布**：每 40 局把当前权重直接发给 8 个工兵，**没有任何晋升/对局校验**。

## 1. 逐项对比

| 环节 | Leela Zero | 正在跑的 v4 |
|---|---|---|
| 每手搜索 | 3200 playouts | **800 sim** |
| PUCT c | **0.5**（另 logpuct 0.015 / logconst 1.7 / FPU 0.25·√Σ先验） | **c_puct = 5**（无 FPU 项） |
| 根噪声 | ε=0.25, α=0.03×361/N，**加在根先验上** | ε=0.25, α=0.03，**加在落子抽样上** |
| 落子温度 | **前 30 手 T=1**，之后取访问数最大 | 前 4 手 1.0 → exp 衰减 → **≥28 手 0.05** |
| policy 标签 | 访问数分布 | 访问数分布（同） |
| value 标签 | 胜负 ±1 | 胜负 ±1（同） |
| 优化器 | Momentum SGD 0.05 + l2 1e-4，常数 | **Adam 2e-3 × KL 自适应倍率**（kl_targ 0.02）+ l2 1e-4 |
| batch / 步数 | 512，训练流式持续跑 | 512 × **epochs 5**，每收一批数据更新一次 |
| 增强 | 每次遍历**随机取 8 对称之一**（在线变体） | 入库时**固定展开 8 份**（同一副本反复用） |
| 回放窗口 | 服务端滑窗 + 1M 条 shuffle buffer | **10000 条（8 倍增强后）≈1250 原始步 ≈ 60 局** |
| 权重平均 | **SWA：最近 16 快照指数加权 + 重算 BN** | **无** |
| 晋升机制 | **SPRT**：95% 置信、>55%(≈35Elo) 才换 best | **无**，每 40 局直接替换 |
| 对手 | 永远当前最强（纯自对弈） | 纯自对弈（8 工兵打同一份权重） |
| 网络 | 18 plane / 40×256 res / ≈48M | 4 plane / **8×128 res** / ≈2.5M |
| 单局算力 | 3200×约200手 ≈ 64 万次推理 | 800×约20.5手 ≈ **1.6 万次**（差 ~39 倍） |

## 2. 五个真差异（按影响排序）

**① 没有晋升门 —— 这是与 LZ 最本质的差别。**
LZ 每一代候选都要过 SPRT，过不了就不发；v4 是「训完就发」。所以**没有任何机制阻止权重变差**，只能靠事后 PK/Rapfi 才看得出。
自家记录「loss 低≠强、权重 40 代漂 10%」正是这个结构的必然结果：loss 只保证拟合当前 buffer，不保证棋力。

**② Dirichlet 噪声加错位置（可验证的结构性差异）。**
LZ/AGZ 把噪声加在**根先验**上，搜索因此会去访问那些点，访问分布本身变宽 → 标签和落子一致。
v4 把噪声只加在**落子抽样**上：树里搜索完全不探索，访问分布依然很尖，而实际落子可能被噪声拽走 ——
于是**同一局面「标签说走 A、实际走了 B」**。B 局面的标签再学，A 的信息就浪费了，而且策略头的目标比搜索实际相信的更自信。
这是一行改动量级（把噪声搬进 `MCTS._playout` 的根先验），但会改变全部数据分布，必须单独一代 A/B。

**③ c_puct=5 vs 0.5，差 10 倍。**
我们的常数是 junxiaosong 教程默认值，LZ 有专门的调参器把它压到 0.5（更信 Q、少信先验）。
800 sim 的网先验本身噪声不小，c=5 意味着搜索高度跟着先验走 → 「网说哪就搜哪」，800 次模拟的纠错能力被浪费。
这是**纯推理侧参数**，不动训练、不改数据，可以直接 PK 验证。

**④ 温度衰减太快，和我们看到的形态矛盾。**
≥28 手固定 0.05（近 argmax），而中位局长只有 20 手 → **整局后半段基本确定性**。
但实测最近 100 局前 4 手 100/100 不重复、≤11 手速杀占 19%：
说明多样性来自前 4 手高温 + 噪声，**中盘没有多样性**，速杀局一旦开局走歪就直接走完。LZ 到 30 手还在 T=1。

**⑤ 回放窗口太窄。**
10000 条是**增强后**的条数，除以 8 ≈ 1250 原始步 ≈ **60 局**、约 20 分钟的数据量。
Net 每 512×5 步就从这 60 局里反复抽 → 严重过拟合当前局面分布（explained_var 0.6 而棋力不一定涨）。
LZ 的窗口是几十万局量级。我们样本效率低，但 60 局仍然偏窄。

## 3. 可抄的（按成本排序）

| # | 动作 | 成本 | 说明 |
|---|---|---|---|
| 1 | **c_puct 0.5 / 1 / 5 三档 PK**（现成 `pk_vs_rapfi.py` 改双网） | 零训练机时 | 纯推理参数，最便宜的一条；LZ 唯一公开调过的搜索参数 |
| 2 | **SWA**：最近 8~16 个 `checkpoints/ckpt_it*.model` 指数加权，用 `logs/games.jsonl` 回放局面重算 BN | 零训练机时（1~2h 推理） | 治「权重漂移」，LZ 官方长期开着 |
| 3 | **晋升门（SPRT）**：新权重 vs 当前 best，α=β=0.05、elo0=0/elo1=35，30~60 局判定 | 占少量机时（可停 1~2 工兵） | 把「训完就发」改成「过门才发」，是 LZ 闭环的核心 |
| 4 | **噪声搬到根先验** | 一行 + 一整代数据 | 改动会污染全局数据分布，必须单独跑一代对比（别和别的改动同时上） |
| 5 | **回放窗口 10000 → 40000~80000 条**（≈240~480 局） | 改 1 个常数，显存/内存无关 | 直接治「60 局反复抽」；但 buffer 变大后要看 loss 是否还降得动 |

**不要抄**：3200 playouts（CPU 上 4 倍机时买不到对应收益，先做上面 1~3）、±1→软标签（v4 已经是 ±1，且在 800 sim 下与 LZ 同口径，暂不动）、40×256 网络（算力不够）。

---

## 4. 附：v24 的噪声位置核对（结论：v24 是对的）

源码：`~/Downloads/alpha-zero-gomoku_code_2026-10-13/legacy/versions/Gomoku-v0_pt_v24.py`（`VERSION="v24"`）

```python
# 326-337 行 decide()：噪声加在【根先验】，然后才跑 800 次搜索
if noise:
    if s not in self.policy: self.search(..., prior_noise=False, ...)
    nv = np.random.gamma(DIRICHLET_ALPHA, 1., board.shape)
    nv *= self.valid[s]; nv /= nv.sum()
    self.policy[s] = (1. - 0.25) * self.policy[s] + 0.25 * nv
for _ in range(self.sim_count):           # sim_count=800
    self.search(..., prior_noise=False, ...)
prob = self.count[s] / self.count[s].sum()   # 标签 = 访问数（已含噪声影响的探索）
```

⇒ **v24 = AGZ 论文式：α=0.03、ε=0.25、每手注入一次到根先验，之后 800 次共享**。标签由含噪声探索出的访问数给出，搜索与落子一致。
「噪声加错位置」只发生在**正在跑的 parallel_train 线**（`mcts_alphaZero.py:230`，junxiaosong 上游写法）。

**但由此得到一个必须修正的判断**：v24 的落子温度就是 `step<4 → 1.0，之后 exp(-(step-4)/8)，地板 0.05`（v24 第 588-592 行），
与正在跑的 v4（`game.py:213-216`）**完全一致** —— 所以：
- 上文的第 ④ 条（温度衰减过快）**v24 同样存在，而 v24 it73 是做题天花板（73.3%）**，不能把 v4 的慢进步归因到温度上；
- v4 与 v24 在"搜索探索"上的真正差别只有两处：**噪声位置**（根先验 vs 落子抽样）和 **c_puct**（v24 = 1.25 / 正在跑的 v4 = 5）。
- 即：正在跑的 v4 **抄了 v24 的温度调度，但没抄它的噪声位置和 c_puct**。
