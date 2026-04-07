import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x = x + residual
        x = F.relu(x)

        return x


class CNNNet(nn.Module):
    def __init__(self, in_channels=20, channels=128, num_blocks=10):
        super().__init__()

        # Initial convolution
        self.conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # POLICY HEAD
        self.policy_conv = nn.Conv2d(channels, 73, kernel_size=1)

        # VALUE HEAD
        self.value_conv = nn.Conv2d(channels, 16, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: (batch, 20, 8, 8)
        x = F.relu(self.bn(self.conv(x)))

        for block in self.res_blocks:
            x = block(x)

        # POLICY
        p = self.policy_conv(x)          # Shape: (batch, 73, 8, 8)
        p = p.permute(0, 2, 3, 1)        # Shape: (batch, 8, 8, 73)
        p = p.reshape(p.size(0), -1)     # Shape: (batch, 4672)

        # VALUE
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v