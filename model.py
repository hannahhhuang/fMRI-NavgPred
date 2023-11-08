import torch
import torch.nn as nn
from torch import Tensor
from typing import List



class DualConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_res: bool = False) -> None:
        super(DualConv, self).__init__()
        self.use_res = use_res

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

        if self.use_res:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm3d(out_channels),
            ) if in_channels != out_channels else nn.Identity() #只有channel变化的时候需要res

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x) if self.use_res else None
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        if self.use_res: #每2个conv之后res
            x += residual
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, feats: List[int], use_res: bool) -> None:
        super(ResNet, self).__init__()
        self.use_res = use_res

        # Encoder
        self.encoder = nn.ModuleList([
            DualConv(feats[i], feats[i + 1], use_res)
            for i in range(len(feats) - 1)
        ])
        self.maxpool = nn.MaxPool3d(2)

        # Output
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(feats[-1], 6)

        self.softmax=nn.Softmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        # Encoder
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i < len(self.encoder) - 1: 
                x = self.maxpool(x)

        # Output
        x = self.avgpool(x)  # (B, C, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        x = self.fc(x)

        return x
