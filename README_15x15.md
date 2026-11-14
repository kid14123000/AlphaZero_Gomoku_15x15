# 15x15 本地运行说明（junxiaosong/AlphaZero_Gomoku 改造版）

上游：https://github.com/junxiaosong/AlphaZero_Gomoku （MIT）
本机克隆源：gitcode 镜像（GitHub 直连被墙）
目录：`~/AlphaZero_Gomoku`，venv：`.venv`（Python 3.12 + torch 2.15，CPU）

## 相对上游的改动

| 文件 | 改动 | 原因 |
|---|---|---|
| `train.py` | 改用 `policy_value_net_pytorch`；棋盘默认 15x15/五连；全部参数改成 argparse | 上游默认 Theano（已死）+ 硬编码 6x6 四连 |
| `policy_value_net_pytorch.py` | `loss.data[0]`→`loss.item()`；`log_softmax` 显式 `dim=1`；`torch.FloatTensor(list)`→`torch.from_numpy(np.asarray(...,float32))` | 现代 PyTorch 兼容 + 训练集张量化提速 |
| `human_play_15.py`（新增） | 人机对局，紧凑棋盘，可加载 `.model`，可退回纯 MCTS | 15x15 用上游 graphic 一屏放不下；上游 human_play 写死 8x8 且只读 numpy 权重 |
| `run_15x15.sh`（新增） | 15x15 训练一键启动（尾部可追加任意 train.py 参数） | 免手敲参数 |
| `resume.sh`（新增） | 断点续训一条命令，自动推导 `batch_offset` / `init_lr_multiplier` / 最新检查点 | 49 小时长跑，重启/崩溃后必须能无痛接上 |
| `train.py` 续训三件套 | `--batch_offset`（全局局数编号）、`--init_lr_multiplier`（恢复自适应 lr）、`--ckpt_dir`（`checkpoints/ckpt_itN.model` 归档） | 见下「断点续训」 |

## 跑训练

```bash
cd ~/AlphaZero_Gomoku
./run_15x15.sh 2>&1 | tee train_15x15.log
```

实测速度（M4，CPU，n_playout=400）：**一局自对弈 15–17 s**（50–70 手）。
耗时实测（2026-11，200 局那一轮）：**200 局 = 51 分钟**，其中含 3 次评估。

⚠️ 评估（`policy_evaluate`，对纯 MCTS 打 10 局）实测 **84 秒/次**（100 playouts 的纯 MCTS，对局很短）。
`check_freq=50` → 每约 13 分钟训练插一次 84 秒评估，占比 ~10%，可接受。
想彻底关掉评估就 `--check_freq 100000`（但那样就没有对纯 MCTS 的胜率曲线了）。

看 `kl / loss / entropy`：entropy 起点 ≈ ln(225)=5.42（均匀）。
**实测 200 局：loss 5.96 → 3.87，entropy 5.39 → 3.61，ev_new −0.57 → +0.73**（确实在学）。

### ⚠️ 对纯 MCTS 评估已关闭（`--eval_pure 0`，默认）

该指标从 800 局起长期 **100%**，且门控（赢满就加 playout）会把它一路推到 1200，
**单次评估实测 345s（5.75 分钟），占每 50 局墙钟时间的 37%** —— 纯白烧机时。
2026-11 已关闭，棋力改看 **Rapfi 外部锚点**。要重新打开：`--eval_pure 1`。

关闭后**检查点照常每 `check_freq` 局保存**（`current_policy.model` + `checkpoints/ckpt_itN.model`），
只是不再跑那 10 盘、也不再更新 `best_policy.model`。
看板对应位置会显示「已停用 / 改看 Rapfi」，不会把历史 100% 当成当前棋力。

### ETA 别写死单局耗时
单局耗时随手数变化很大：**45 手/局 ≈ 12s，14 手/局 ≈ 5s**。
`resume.sh` 现在用「最近 40 局实测中位数」算 ETA，不写常数。

`./web/start.sh --port 8770` 默认只画 **第 1000 局之后**的曲线（1–999 局是随机棋，会把纵轴压扁）。
- `?from=0` 看全部；`?from=2000` 自定义起点（页脚有链接，并显示当前窗口）
- x 轴标题会写成「代数（= 自对弈局数，只显示 ≥ 1000）」，避免误以为数据丢了

## 断点续训（49 小时长跑必备）

```bash
./resume.sh --dry-run     # 先看它推导出的参数对不对
./resume.sh               # 从最新检查点继续，目标沿用上次的 10000 局
./resume.sh --total 20000 # 加大目标
```

`resume.sh` 自动推导三件不能手填的东西：

| 量 | 来源 | 填错会怎样 |
|---|---|---|
| `batch_offset` | 最新 `checkpoints/ckpt_itN.model` 的 N | 曲线 x 轴重头开始，前后半段没法比 |
| `init_lr_multiplier` | `logs/metrics.jsonl` 最后一条 `update.lr_multiplier` | **重置成 1.0 = 学习率瞬间涨 5 倍**，KL 爆掉 |
| `init_model` | `ckpt_itN.model`（不是 `current_policy.model`） | current 可能正被写坏，归档件才是安全的 |

同时它带 `caffeinate -i` 启动，防止 Mac 空闲休眠把训练冻住。

### 为什么 `check_freq` 用「全局局数」而不是「本次进程局数」
`train.py` 里判据是 `self.game_batch_cur % self.check_freq == 0`，`game_batch_cur = batch_offset + i + 1`。
若用 `i+1`，续训后评估会从 0 重新数，曲线上的评估点全错位。

## 训练看板（本地 Web）

```bash
cd ~/AlphaZero_Gomoku
./web/start.sh --port 8770      # 必须走 .venv（engine.py 要 torch）
# 浏览器打开 http://localhost:8770
```

> ⚠️ `web/start.sh` 用 `.venv/bin/python`。系统 python3 (3.9) 没有 torch，
> 用错解释器会在调用 `/api/viz/*` 时报 `No module named 'torch'`。

四个页面（顶部导航互通）：

| 页面 | 路径 | 用途 |
|---|---|---|
| 📊 训练监控 | `/dashboard` | 曲线看板（见下） |
| 🗂 棋谱回放 | `/replay` | 训练自对弈对局逐步复盘（选局/步进/自动播放/五连标注） |
| ♟ 对弈 | `/play` | 人机 / AI 自战，落子即显示网络价值 + AI 搜索依据 |
| 🔍 MCTS 搜索 | `/mcts` | 摆谱 → 跑搜索 → 候选/访问/Q/先验 → 逐次模拟回放 |
| 🧠 网络前向 | `/network` | 一次干净前向，逐层热力图 + policy/value 双头 |

前端照 `alpha-zero-gomoku_code_2026-10-13/gomoku_web` 那套驾驶舱的风格做：
暗色卡片 + Chart.js 4.4.0（**本地 `web/static/chart.min.js`，不依赖 CDN/npm**）+ 右上角状态徽章 + 手动刷新按钮，5 秒自动刷新。

### 后端结构

```
web/server.py        路由 + 训练指标聚合 + 局面存档（stdlib http.server）
web/engine.py        推理引擎：棋盘/权重/MCTS追踪/逐层前向/对弈会话
web/replay_store.py  棋谱库（logs/games.jsonl 的索引/分页/统计）
web/static/gomoku.js   共享棋盘绘制 + 摆谱编辑器 + 权重下拉 + 导航
web/static/gomoku.css  共享暗色样式
```

**engine 不复制训练代码**：`Board` / `MCTS` / `PolicyValueNet` 全部直接 import，
页面看到的就是训练时那份实现。

🚨 **坐标口径**：上游 `Board.current_state()` 末尾有 `square_state[:, ::-1, :]`，
张量第 0 行 = 棋盘第 14 行。`engine.to_grid()` 统一翻回来，
所以传给前端的 grid 一律 `grid[r*15+c]`，**r=0 在屏幕最上方**，与 policy 的 move 编号同口径。

### ⚔ 战斗激烈度指标（`web/combat.py`）

起因：对局里出现过「黑一路连成五子、白在旁边 5 格外交替落子」的棋 —— 双方各打各的，
互相不干扰。胜率和手数都看不出这个毛病（手数短只说明有一方赢了），
所以必须直接量化**有没有在跟对手较劲**。

只依赖落子手顺 `[[r,c],...]`，不读网络、不依赖训练日志。四个子指标（0~1）：

| 指标 | 定义 | 越大越激烈 |
|---|---|---|
| `contact` 接触率 | 落点在敌方棋子切比雪夫距离 ≤1 的比例 | ✓ |
| `proximity` 贴身度 | `1 - median(每手到最近敌方子的距离)/4`，夹到 0~1 | ✓ |
| `block` 防守率 | 对手有「下一手成五」或「下一手成活四」时，本方去应对的比例 | ✓ |
| `zone` 区域共享 | 双方棋子包围盒 交集面积/并集面积 | ✓ |

`composite = 四者等权平均 × 100`。**等权是有意的**：不做加权调参，
避免"指标好看"变成拟合出来的结果。

防守率分两级统计：
- **必堵点**（对手下一手成五，不堵立刻输）
- **活三点**（对手下一手成活四，不堵两步后被双四收）

实测（1449 局，第 313–1739 代）：**该防 7765 次，防住 486 次 = 6.3%**。
即 94% 该防的手都没防。这就是"互相不理对方"的量化结果。

🚨 **必须排除的一种误算**：本方自己下一手就能成五时，不该被要求去防守（直接赢就是了）。
不排除会把"赢家的制胜一手"算成成功防守 —— 实测这类局面每局都有一次（就是最后一手），
排除后该防次数从 8459 降到 7765。
⚠️ 反过来也要知道：本指标只排除"立刻赢"，不区分"主动抢速度更快"和"单纯没看见"。
所以 6.3% 是「没去应对」的上界口径，不等于「全都是漏防」。

趋势（每 100 代一档）：

| 代数 | 激烈度 | 接触率 | 防守率 | 手数中位 |
|---|---|---|---|---|
| 300–399 | 43.5 | 45.2% | 3.5% | 44 |
| 800–899 | 27.4 | 32.4% | 4.6% | 19 |
| 1000–1099 | 23.2 | 25.9% | 5.1% | 15 |
| 1400–1499 | 29.3 | 32.8% | 7.6% | 15 |
| 1500–1599 | 27.8 | 27.9% | **14.7%** | 14.5 |
| 1700–1739 | 30.4 | 31.7% | 14.4% | 14 |

读法：激烈度随"学得快"一路掉（43→23，因为速胜不需要缠斗），
1400 代后开始回升（23→31），**防守率从 3~5% 抬到 14%+** —— 白方开始理黑方了。

⚠️ 性能：约 4.8ms/局。全量算 10000 局要 ~48s，所以做了 **sidecar 增量缓存**
（`logs/combat_cache.json`，按 idx 存）。文件是追加写、idx 稳定，只补新局，
稳态下每次请求只算 1~2 局（~10ms）。冷启动可先手动预热：

```bash
cd web && PYTHONPATH=. ../.venv/bin/python -c "
from replay_store import GameStore
s=GameStore('../logs/games.jsonl', combat_cache='../logs/combat_cache.json')
s.combat_series(bucket=100); s.save_combat_cache()"
```

### 🗂 棋谱回放页要点

数据源是 `train.py --save-games logs/games.jsonl`（每局一行，含完整手顺）。
`resume.sh` 已经带这个参数 —— **丢了它白棋开局面板会没数据**，回放页也会空。

- **主键用行号 `idx`，不是 batch**：断点续训会重跑一小段，同一 batch 可能有两条不同对局。
- 列表只传摘要（代数/胜方/手数），点开某一局才读 `moves`，10000 局也不会卡。
- 支持按胜方 / 代数区间筛选，按代数或手数排序，40 局一页。
- 「最短局在前」是排查速胜/速败最快的入口（现在最短 9 手就结束）。
- 获胜五连在**末手**才用红圈标出（中途标会剧透）。
- 「📋 送到 MCTS / 前向页」把当前手数的局面存进 `positions.json`，另外两页可直接载入拆解。

### API

| 端点 | 说明 |
|---|---|
| `GET /api/metrics` | 训练曲线聚合 |
| `GET /api/models` | 可用权重列表（自动跳过仓库自带 6x6/8x8） |
| `POST /api/game/new` | 开局 `{black_type,white_type,black_weight,white_weight,n_playout,temp,first_moves}` |
| `POST /api/game/move` | 人类落子 `{row,col}` |
| `POST /api/game/ai` | AI 走一步（AI vs AI 时前端轮询这个） |
| `GET /api/game/state` | 当前局面 |
| `POST /api/viz/mcts` | `{stones?,model,n_playout,c_puct,temp}` → 候选/追踪/偏离/深度分布 |
| `POST /api/viz/network` | `{stones?,model}` → 逐层激活 + policy/value |
| `POST /api/positions/{save,load,delete}` `GET /api/positions/list` | 局面存档（`web/positions.json`） |
| `GET /api/games/list` | 棋谱列表 `?offset&limit&winner&min_batch&max_batch&order=batch_desc\|batch_asc\|len_asc\|len_desc` |
| `GET /api/games/detail?idx=` | 单局完整手顺 |
| `GET /api/games/stats?bucket=100` | 手数直方图 + 按代黑白胜率 |
| `GET /api/combat?bucket=100` | 战斗激烈度趋势（4 子指标 + 指数，按代分桶） |

`stones` 省略 = 用当前对弈局面；传了就按摆谱局面分析（颜色按放置顺序交替，黑先）。

### 判读口径

**对弈页**
- 价值热力图 = 在每个空点替**当前方**落一子后的期望胜率（已取负号转成走子方视角），红=好。
- AI 面板里的 Top3 是**按温度 τ 调整后的选点概率**，不是原始访问次数；τ=0.05 时基本就是 argmax。
- AI vs AI 是**一步一请求**递增，不是一次算完整局（否则前端只能干等）。

**MCTS 页**
- 候选只列**被访问过**的点（根节点展开时会建出全部合法点的先验层，那些 0 访问点不算候选）。
- `TV` = 0.5·Σ|MCTS 分布 − 网络先验|，即"搜索把多少概率质量挪走了"，比 MSE 直观。
- `argmax 一致` = 搜索的最终选点和网络先验的第一名是否相同；不一致说明搜索改主意了。
- 深度分布大量停在 L1~L2 = 先验分散、在浅层乱撞；配合好会往 L4+ 走。

**网络前向页**
- value 是**当前走子方**视角，页面同时换算成黑白两个百分数，避免看错方向。
- `激活率` = 该层输出 > 0 的比例（ReLU 后）；`死通道` = 整层全 0 的通道数。
- 输入 4 平面口径：ch0 当前方棋子 / ch1 对手棋子 / ch2 上一手位置 / ch3 执子色常量。

看板内容：6 个 KPI（代数进度 / 黑白胜率 / 均手数 / Loss / 熵 / 对纯 MCTS 胜率）
+ 7 张曲线（步长趋势、黑白胜率滚动、Loss、熵、KL+lr、解释方差、评估胜率）+ 评估明细表。

**看板判读口径**
- 黑 = 先手（player1，自对弈里始终先手），白 = 后手；和棋不进胜率分母。
- 滚动胜率用 20 局窗口；**不足 5 局时服务端返回 null 不画点**（1 局就是 0% 或 100%，纯噪声）。
- 均手数用 20 局滚动均值看趋势，单局灰点看离散度。
- 熵只降不涨 = 落点变集中；长期钉在 5.4 = policy 没在学。
- `ev_new ≤ 0` 长时间不抬 = value 头没学到东西。
- ETA = 剩余局数 × 单局中位耗时 + 剩余评估次数 × 单次评估耗时（评估很贵，见下）。

## 人机对局

```bash
cd ~/AlphaZero_Gomoku
./.venv/bin/python human_play_15.py --model current_policy.model --n_playout 400
```
- 输入 `row,col`，例如 `7,7` 天元；`q` 退出。
- 没有 `.model` 时自动退回纯 MCTS：`--pure --n_playout 400`。
- AI 先手：`--human_first 0`。

## 产物

- `current_policy.model`：每次 `check_freq` 覆盖保存（可能被写坏，别长期依赖）
- `best_policy.model`：闯过历史最佳胜率才保存（初始胜率 0 的模型不会存，属上游行为）
- `checkpoints/ckpt_it{N}.model`：每次 `check_freq` 归档一份，**不可变**，续训/回溯用这个
- `logs/metrics.jsonl`：全谱系追加（多次 start 记录共存，看板取最后一条当配置）
- `logs/train_10000.log`：训练 stdout

## 版本历史（15×15 v4 线，追加式）

### 2026-11-14 12:19 搜索结构改动：噪声进根先验 + c_puct 5→1.25

| 项 | 改前 | 改后 |
|---|---|---|
| Dirichlet 噪声位置 | 只混进**落子抽样** `(1-ε)p+ε·Dir`，树里不加噪 | 注入**根子节点先验** `P←(1-ε)P+ε·Dir`，每手一次、N 次模拟共享（AGZ/v24 口径） |
| `c_puct` | 5（junxiaosong 上游默认） | **1.25**（v24 同款有效值；LZ 用 0.5，公式不同不可直接比） |
| 落子抽样 | 噪声 + 温度采样 | 只按温度采样 `probs`（不叠噪声，避免双重加噪） |

- 开关：`--root_noise 1|0`、`--c_puct N`（也可用 env `GOMOKU_ROOT_NOISE` / `GOMOKU_C_PUCT`），旧行为可 0 回退。
- 代码：`mcts_alphaZero.py`（`MCTS._add_root_dirichlet` / `get_move_probs(root_noise=)` / `MCTSPlayer(root_noise=)`）、`worker_selfplay.py`、`parallel_train.py`、`train.py`。
- 证据（同权重同开局同种子，200 sim × 6 手）：被访问根子节点数 **66.8 → 106.0（只换噪声）→ 113.0（只降 c）→ 113.3（组合）**；根先验 ΣP 保持 ≈1。
- 线上续跑：`ckpt_it13467.model` + `batch_offset 13467`，lr_multiplier 0.0878 保持。
- 观测（重启后前 7 局）：均手 **20.2→28.0**、中位 **19→29**、≤11 手速杀 **18%→0%**、单局 194~330s（改前均 133s）
  ⇒ 对局明显变长、速杀消失；**样本产出速率反而下降约 35%**（局/分 3.6→1.7，但每局手数 +39%），
  原因是对局变长 + `Board.deepcopy` 随历史长度变贵。属预期副作用，**不是效率 bug**；判据待 ≥3 代再定。

- 审计：`logs/metrics.jsonl` 的 `start` 记录新增 `c_puct` / `root_noise` 字段。
