"""改完 train.py 后的离线自测：Rapfi 锚点 + 纯MCTS 门控各跑 2 局
不写 metrics.jsonl（metrics_file=None），不影响正在跑的训练
"""
import os
from train import TrainPipeline

BIN = os.path.expanduser('~/rapfi_run/pbrain-rapfi')

p = TrainPipeline(init_model='current_policy.model', n_playout=400,
                  pure_mcts_playout_num=400, metrics_file=None,
                  rapfi_bin=BIN, rapfi_depth=2, rapfi_every=200,
                  rapfi_games=2, rapfi_who='absolute')

print('\n--- 测试1: Rapfi 锚点（2 局，d2）---', flush=True)
p.rapfi_anchor()

print('\n--- 测试2: 纯MCTS 门控（2 局，400 po）---', flush=True)
p.pure_mcts_playout_num = 400
ratio = p.policy_evaluate(n_games=2)
print('gate win_ratio =', ratio, flush=True)
