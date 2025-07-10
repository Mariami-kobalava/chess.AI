import torch
from torch import nn
import torch.nn.functional as F
# from mcts2 import POLICY_SIZE

POLICY_SIZE = 4672


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_planes=26, num_blocks=10):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(128) for _ in range(num_blocks)]
        )

        # --- Policy Head ---
        self.policy_conv = nn.Conv2d(128, 73, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(73)
        self.policy_fc = nn.Linear(73 * 8 * 8, POLICY_SIZE)

        # --- Value Head ---
        self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)

        # --- Policy Head ---
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = torch.relu(p)
        p = p.view(p.size(0), -1)
        log_p = torch.log_softmax(self.policy_fc(p), dim=1)

        # --- Value Head ---
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = torch.relu(v)
        v = v.view(v.size(0), -1)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return log_p, v

# class CNN(nn.Module):
#     def __init__(self, in_planes=26, num_blocks=10):
#         super().__init__()
#         self.stem = nn.Sequential(
#             nn.Conv2d(in_planes, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True)
#         )
#         self.res_blocks = nn.Sequential(*[ResidualBlock(128) for _ in range(num_blocks)])
#         # Policy head
#         self.policy_conv = nn.Conv2d(128, 73, kernel_size=1)
#         self.policy_bn = nn.BatchNorm2d(73)
#         self.policy_fc = nn.Linear(73*8*8, POLICY_SIZE)
#         # Value head
#         self.value_conv = nn.Conv2d(128, 1, kernel_size=1)
#         self.value_bn = nn.BatchNorm2d(1)
#         self.value_fc1 = nn.Linear(1*8*8, 128)
#         self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch,26,8,8)
        x = self.stem(x)
        x = self.res_blocks(x)

        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))  # (batch,146,8,8)
        p = p.view(p.size(0), -1)                        # flatten to (batch,146*8*8)
        p = self.policy_fc(p)                            # (batch,4672)
        p = F.log_softmax(p, dim=1)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))    # (batch,1,8,8)
        v = v.view(v.size(0), -1)                        # (batch,64)
        v = F.relu(self.value_fc1(v))                    # (batch,256)
        v = torch.tanh(self.value_fc2(v)).view(-1)       # (batch,)

        return p, v