"""v24 同构网络（用于速度对比 / 后续 v2 实验）

结构照 Gomoku-v0_pt_v24.py:
  conv_input 4→128 k3 (no bias) + BN
  8 × ResidualBlock(128,128,128): conv3x3+BN+ReLU → conv3x3+BN → add → ReLU
  policy: conv3x3 128→1 + BN + ReLU ; conv1x1 128→1 + BN + ReLU ; cat(2) → FC(450→225) → log_softmax
  value : conv3x3 128→1 + BN + ReLU ; flatten(225) → FC(225→128) → ReLU → FC(128→1) → tanh
参数量 ≈ 2.50M（本仓库默认网络 0.33M）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_CHANNELS = 4


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class V24Net(nn.Module):
    def __init__(self, board_size=15, conv_channels=128, residual_blocks=8):
        super().__init__()
        self.board_size = board_size
        n = board_size
        self.conv_input = nn.Conv2d(INPUT_CHANNELS, conv_channels, 3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(conv_channels)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(conv_channels, conv_channels, conv_channels)
            for _ in range(residual_blocks)])
        self.policy_conv = nn.Conv2d(conv_channels, 1, 3, padding=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(1)
        self.policy_conv2 = nn.Conv2d(conv_channels, 1, 1, padding=0, bias=False)
        self.policy_bn2 = nn.BatchNorm2d(1)
        self.policy_fc = nn.Linear(n * n * 2, n * n)
        self.value_conv = nn.Conv2d(conv_channels, 1, 3, padding=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(n * n, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_input(self.conv_input(x)))
        for blk in self.res_blocks:
            x = blk(x)
        a1 = F.relu(self.policy_bn(self.policy_conv(x)))
        a2 = F.relu(self.policy_bn2(self.policy_conv2(x)))
        a = torch.cat([a1, a2], dim=1).flatten(1)
        logp = F.log_softmax(self.policy_fc(a), dim=1)
        v = F.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = F.relu(self.value_fc1(v))
        return logp, torch.tanh(self.value_fc2(v))


if __name__ == '__main__':
    net = V24Net()
    n = sum(p.numel() for p in net.parameters())
    print(f'V24Net 参数量 {n:,}')
