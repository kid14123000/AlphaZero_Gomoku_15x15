# 参考引擎调研：KataGomo / AlphaGomoku（本地克隆 + 源码阅读，2026-11-14）

> 起因：Gomocup 2026 上 AZ 系引擎已进世界前三（详见对话记录与官网结果页），而我们的 v4 线仍在 13% 白胜率挣扎。
> 本文只记**从源码里读出来的、可直接对照我们项目的事实**，不做方案决策。
> 本地位置：`~/ref_engines/KataGomo`（分支 `gomoku`）、`~/ref_engines/AlphaGomoku`（`master`）。两者都在本仓库之外，未入库。

## 0. 克隆信息

| 仓库 | 本地 HEAD | 分支 | 文件数 | 说明 |
|---|---|---|---|---|
| `github.com/hzyhhzy/KataGomo` | `250a15f update readme.md` | `gomoku` | 223 | `master` 只有文档（8 个文件）；**代码在各游戏分支里**，gomoku 在 `gomoku` 分支 |
| `github.com/MaciejKozarzewski/AlphaGomoku` | `1f81840 Changed default network paths.` | `master` | 269 | 完整 C++ 源码，37k 行 |

复现：

```bash
mkdir -p ~/ref_engines && cd ~/ref_engines
git clone --depth 1 https://github.com/hzyhhzy/KataGomo.git
git clone --depth 1 https://github.com/MaciejKozarzewski/AlphaGomoku.git
cd KataGomo && git fetch --depth 1 origin gomoku && git checkout -B gomoku origin/gomoku
```

## 1. 两个引擎是什么

| | KataGomo | AlphaGomoku |
|---|---|---|
| 血统 | **KataGo**（AlphaZero-like，Go）改造成多棋种 | 自研，正统 "AlphaZero for Gomoku" |
| 语言 | C++ 引擎 + Python(TensorFlow) 训练 | 纯 C++（含自研 `ml` 后端） |
| 作者 | Zhiyang Hang（同时是 Rapfi 作者之一） | Maciej Kozarzewski |
| 支持的规则 | Freestyle / Standard / Renju / Caro（分支或规则参数） | Freestyle / Standard / Renju / Caro5 / Caro6 |
| 训练成本自述 | README：*"can reach the top-level AIs with very low cost (probably <50 dollars)"* | 2023 版：*"New networks trained on 5 million games"* |

## 2. 代码结构（读之前先看这张）

**KataGomo `cpp/`**：`search/`（search.cpp + LCB/fpu/VCF 参数）、`game/`、**`forbiddenPoint/ForbiddenPointFinder.cpp`（禁手判定，独立模块）**、`neuralnet/`、`vcfsolver/`、`distributed/`、`dataio/`；`python/` 是训练侧（`model.py`、`train.py`、`modelconfigs.py`、`data.py`），配置 `configs/selfplay.cfg`。

**AlphaGomoku `src/`**：`search/monte_carlo/`（Edge/Node/Tree/EdgeSelector/NNEvaluator）、`search/alpha_beta/`（含 MoveGenerator/ThreatGenerator/ThreatSpaceSearch）、**`networks/`（AGNetwork、NNUE、MovesLeftNetwork、NNInputFeatures）**、**`selfplay/`（GameGenerator、OpeningGenerator、TrainingManager）**、`dataset/`、`patterns/`（DefensiveMoveTable 等）、`protocols/`。

## 3. 五个可以直接拿来做对照的发现

### 3.1 ★ 他们的自对弈开局是「随机 + 按估值筛平衡」的

`src/selfplay/OpeningGenerator.cpp`：

```cpp
const float balance = std::fabs(iter->task.getValue().getExpectation() - 0.5f);
if (balance < (0.1f + 0.01f * trials)) {          // 只收 |V-0.5| < 0.1 的开局
    completed_openings.push_back(iter->moves);
    trials = 0;
} else trials++;                                   // 否则丢弃重抽（阈值随 trials 缓慢放宽）
```

开局本体在 `src/utils/misc.cpp::prepareOpening`：**随机走 0~15 手**（`randInt(6)+randInt(6)+randInt(6)`，均值约 7.5），落子在中心权重的分布上采样；`randInt(1000)==0` 时有 1‰ 概率给 **0 手开局**（即完整交给模型）。

用法：`GameGenerator.cpp` 里 `OpeningGenerator(gameOptions, 8)`（目标平均 8 手），每局 `game.loadOpening(...)`。

**这条和我们直接相关**：v36 我们做的是「前 3 手全随机」，**没有任何平衡性筛选** ⇒ 网络学成"抓坏开局直接速杀"，正常局依然不行。AlphaGomoku 的做法是随机开局 **再用 value 判均衡**，把"先手必胜"的开局直接扔掉。这是同一族手段的**带反馈版本**。

### 3.2 ★ playout cap randomization（KataGomo `configs/selfplay.cfg`）

```
cheapSearchProb = 0.75        # 75% 的局面用便宜搜索
cheapSearchVisits = 50
maxVisits = 250               # 注意：训练时最多只有 250 次访问
reducedVisitsMin = 50
rootNumSymmetriesToSample = 4 # 根节点采样 4 个对称变换
useLcbForSelection = true;  lcbStdevs = 5.0;  minVisitPropForLCB = 0.15
cpuctExploration = 1.1;  fpuReductionMax = 0.2;  valueWeightExponent = 0.5
rootDirichletNoiseTotalConcentration = 10.83;  rootDirichletNoiseWeight = 0.25
chosenMoveTemperatureEarly = 0.75;  chosenMoveTemperature = 0.15;  halflife = 19
```

同一局面在 50 / 250 两种预算下各产生一次价值标签 ⇒ **value 目标不再"同一个局面只有一个确定答案"**。
对照我们：`n_playout=800` 固定、`c_puct=1.25`、无 LCB、无 fpu reduction。而我们自己诊断出的白崩机制之一正是「劣势局 value 学成确定值 → 毒化 MCTS」。

### 3.3 ★ 辅助目标：我们只有一个 value

- KataGomo `train.py` 同时训练 13 项指标：`p0loss / p1loss / vloss / tdvloss / smloss(scoremean) / leadloss / vtimeloss / sbpdfloss+sbcdfloss(score belief) / oloss(ownership) / sloss(scoring) / fploss(futurepos) / reg`。
- AlphaGomoku 除了 policy+value，还有 **`MovesLeftNetwork`（剩余手数头）**。

对照我们：只有 policy(CE) + value(MSE)。**"这局还要走多少手 / 局面有多复杂"这类信号我们完全没有**，而它恰好是分层采样想手工补上的东西（按手数、按胜负切段）。

### 3.4 禁手是独立模块，不是塞进网络

`cpp/forbiddenPoint/ForbiddenPointFinder.{h,cpp}` —— renju 双三/双四/长连的判定完全在规则层。`configs/selfplay.cfg` 里 `useVCF=true; maxVCFnodes=300000` 说明他们还在搜索层挂了 VCF 求解器。

⇒ 说明"规则层修正 + 神经网络"是**分开的两层**；他们的 freestyle 引擎不靠禁手也能到世界第二，禁手只是另一个规则集。

### 3.5 我们的搜索是他们的极小版本

| 项 | 他们 | 我们 v4 |
|---|---|---|
| 训练时访问数 | 50~250（cap randomization） | 800 固定 |
| 选子 | LCB(5σ) + fpuReduction 0.2 + root policy temperature | PUCT + c_puct 1.25 |
| 对称 | 根节点采样 4 个对称 | 无（训练侧做 8 重增广） |
| 求解器 | VCF（30 万节点上限） | 无 |
| 网络 | KataGo b6c96 ~ b40c384 | V24Net 2.5M（128ch×8 残差） |

## 4. 一句话总结这次阅读

**它们并不是"没有人工干预"，而是把干预放在了三个正确的位置**：
① 开局（随机 + 估值筛平衡）、② 价值标签（多预算、辅助目标）、③ 规则（禁手作为独立模块，用于 renju/standard，不用于 freestyle）。

我们过去几次尝试基本都撞在①上（随机开局/让手），但**少了"筛选"这一步**；②我们完全没做。

## 5. 可以最小成本试的三件事（均未实施，等拍板）

1. **给随机开局加平衡过滤**：自对弈前 N 手随机后，用当前网络 value 判 `|v| < 阈值` 才开局，否则重抽。改动只在 `worker_selfplay.py` 的开局段，不动训练与采样。（对应 §3.1）
2. **加"剩余手数"辅助头**：给 `value` 头旁边挂一个 `moves_left` 回归头（target = `episode_len - 当前手数`），权重小。它能直接替掉我们手工设计的"短手白胜反捡漏"启发式。（对应 §3.3）
3. **playout cap randomization**：按 75% / 25% 概率用 200 / 800 sims 产标签。**注意这会让"局重复/熵低"的诊断口径失效**，必须先想清楚怎么判读再上。（对应 §3.2）

## 6. 不建议照抄的

- **搬 KataGo 全套并不现实**：它是围绕 Go 设计的（ownership/score 有明确语义），五子棋没有"目数"，`scoremean`/`scorebelief` 这套要用五子棋的等价值（如"剩余手数差"）重定义。
- **不要把禁手当"解决白崩"的手段**：Gomocup 里 AZ 引擎在 freestyle 也能拿第二；禁手只是让另一个规则集变得可玩。
- **不要用它们的绝对 Elo 当我们目标**：那是 300 秒/手 + 云端 + 数千万局训练出来的，我们是 M4 CPU + 800 sims + 每代几百局。
