
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_groupnorm_groups(channels: int, max_groups: int = 4) -> int:
    max_groups = min(max_groups, channels)
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


class ChannelAttention(nn.Module):

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(), 
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.mlp(self.avg_pool(x))
        return x * w


class DynamicKernelGenerator(nn.Module):

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        hidden_channels: int | None = None,
        weight_softmax: bool = True,
        num_groups: int = 1,  
    ):
        super().__init__()
        self.feat_channels = int(feat_channels)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.weight_softmax = bool(weight_softmax)
        self.num_groups = int(max(1, num_groups))
        if self.feat_channels % self.num_groups != 0:
            raise ValueError(
                f"feat_channels ({self.feat_channels}) must be divisible by num_groups ({self.num_groups})"
            )
        self.channels_per_group = self.feat_channels // self.num_groups

        hidden = hidden_channels or max(32, in_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

        branches = []
        for k in self.kernel_sizes:
            out_dim = self.num_groups * k * k
            branches.append(
                nn.Sequential(
                    nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                    nn.SiLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                    nn.Linear(hidden, out_dim),
                )
            )
        self.branches = nn.ModuleList(branches)

    def _expand_group_kernels(self, w: torch.Tensor, k: int) -> torch.Tensor:
        b = w.shape[0]
        w = w.view(b, self.num_groups, k * k)
        if self.weight_softmax:
            w = F.softmax(w, dim=-1)
        w = w.view(b, self.num_groups, k, k)
        w = w.repeat_interleave(self.channels_per_group, dim=1)
        return w

    def forward(self, fused_feat: torch.Tensor) -> List[torch.Tensor]:
        h = self.stem(fused_feat)
        kernels: List[torch.Tensor] = []
        for k, branch in zip(self.kernel_sizes, self.branches):
            w = branch(h)  
            kernels.append(self._expand_group_kernels(w, k))
        return kernels


def dynamic_depthwise_conv2d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    padding: int,
) -> torch.Tensor:
    b, c, h, w = x.shape
    kb, kc, kk, _ = kernel.shape
    assert b == kb and c == kc, (x.shape, kernel.shape)

    x_ = x.reshape(1, b * c, h, w)
    w_ = kernel.reshape(b * c, 1, kk, kk)
    y = F.conv2d(x_, w_, bias=None, stride=1, padding=padding, groups=b * c)
    return y.reshape(b, c, h, w)


class FeatureAlignmentBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        cam_reduction: int = 2,
        use_groupnorm: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.residual = bool(residual)

        self.fuse = nn.Conv2d(channels * len(self.kernel_sizes), channels, 1)
        if use_groupnorm:
            groups = _valid_groupnorm_groups(channels)
            self.norm = nn.GroupNorm(groups, channels)
        else:
            self.norm = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)
        self.cam = ChannelAttention(channels, reduction=cam_reduction)

    def forward(self, x_ir: torch.Tensor, kernels: Sequence[torch.Tensor]) -> torch.Tensor:
        assert len(kernels) == len(self.kernel_sizes)
        ys = []
        for ksz, ker in zip(self.kernel_sizes, kernels):
            ys.append(dynamic_depthwise_conv2d(x_ir, ker, padding=ksz // 2))
        y = torch.cat(ys, dim=1)
        y = self.fuse(y)
        y = self.act(self.norm(y))
        y = self.cam(y)
        return x_ir + y if self.residual else y


class FDAMPreAlign(nn.Module):

    def __init__(
        self,
        feat_channels: int, 
        kernel_sizes: Sequence[int] = (3, 5, 7),
        num_fab: int = 2,
        dkg_hidden_channels: int | None = None,
        weight_softmax: bool = True,
        use_groupnorm: bool = True,
        num_groups: int = 1,
        share_dkg: bool = False,
    ):
        super().__init__()
        self.feat_channels = int(feat_channels)
        self.kernel_sizes = tuple(int(k) for k in kernel_sizes)
        self.num_fab = int(num_fab)
        self.num_groups = int(max(1, num_groups))
        self.share_dkg = bool(share_dkg)

        num_dkg_modules = 1 if self.share_dkg else self.num_fab
        self.dkgs = nn.ModuleList(
            [
                DynamicKernelGenerator(
                    in_channels=2 * self.feat_channels,
                    feat_channels=self.feat_channels,
                    kernel_sizes=self.kernel_sizes,
                    hidden_channels=dkg_hidden_channels,
                    weight_softmax=weight_softmax,
                    num_groups=num_groups,
                )
                for _ in range(num_dkg_modules)
            ]
        )
        self.fabs = nn.ModuleList(
            [
                FeatureAlignmentBlock(
                    channels=self.feat_channels,
                    kernel_sizes=self.kernel_sizes,
                    use_groupnorm=use_groupnorm,
                )
                for _ in range(self.num_fab)
            ]
        )
        
        print(
            f"[FDAMPreAlign] initialized | "
            f"C={self.feat_channels}, "
            f"kernels={self.kernel_sizes}, "
            f"num_fab={self.num_fab}, "
            f"num_groups={self.num_groups}, "
            f"share_dkg={self.share_dkg}"
        )
        self._printed = False

    def forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor) -> torch.Tensor:

        assert rgb_feat.shape == ir_feat.shape, (rgb_feat.shape, ir_feat.shape)
        x = ir_feat
        for idx, fab in enumerate(self.fabs):
            dkg = self.dkgs[0] if self.share_dkg else self.dkgs[idx]
            fused = torch.cat([rgb_feat, x], dim=1)
            kernels = dkg(fused)
            x = fab(x, kernels)
        if not self._printed:
            print(
                f"[FDAMPreAlign] running | "
                f"rgb={tuple(rgb_feat.shape)}, "
                f"ir={tuple(ir_feat.shape)}"
            )
            self._printed = True
        return x
