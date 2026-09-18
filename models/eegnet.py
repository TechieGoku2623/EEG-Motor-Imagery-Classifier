"""Small EEGNet (Lawhern 2018). Input is (batch, channels, time) or (batch, 1, channels, time)."""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet. f1/d are a bit smaller than the paper so LOSO is runnable on CPU."""

    def __init__(
        self,
        n_channels: int = 64,
        n_samples: int = 641,
        n_classes: int = 5,
        f1: int = 8,
        d: int = 2,
        dropout: float = 0.5,
        kernel_length: int = 64,
    ) -> None:
        super().__init__()
        f2 = f1 * d
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f2, f2, (1, 16), padding=(0, 8), groups=f2, bias=False),
            nn.Conv2d(f2, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classifier = nn.LazyLinear(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) or (B, 1, C, T)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
