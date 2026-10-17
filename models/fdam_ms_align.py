
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.fdam_pre import FDAMPreAlign, _valid_groupnorm_groups


class ReducedFDAMAlign(nn.Module):

    def __init__(
        self,
        channels: int,
        reduced_channels: int = 64,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        num_fab: int = 1,
        dkg_hidden_channels: int | None = None,
        weight_softmax: bool = True,
        use_groupnorm: bool = True,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = int(channels)
        self.reduced_channels = int(min(reduced_channels, channels))
        self.use_checkpoint = bool(use_checkpoint)

        
        self.rgb_reduce = nn.Conv2d(self.channels, self.reduced_channels, kernel_size=1)
        self.ir_reduce = nn.Conv2d(self.channels, self.reduced_channels, kernel_size=1)

        
        self.align = FDAMPreAlign(
            feat_channels=self.reduced_channels,
            kernel_sizes=tuple(int(k) for k in kernel_sizes),
            num_fab=int(num_fab),
            dkg_hidden_channels=dkg_hidden_channels,
            weight_softmax=bool(weight_softmax),
            use_groupnorm=bool(use_groupnorm),
        )

        
        self.out_proj = nn.Conv2d(self.reduced_channels, self.channels, kernel_size=1)
        if use_groupnorm:
            g = _valid_groupnorm_groups(self.channels, max_groups=32)
            self.out_norm = nn.GroupNorm(g, self.channels)
        else:
            self.out_norm = nn.Identity()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor) -> torch.Tensor:
        
        
        if self.use_checkpoint and (rgb_feat.requires_grad or ir_feat.requires_grad):
            return checkpoint(self._forward, rgb_feat, ir_feat)
        return self._forward(rgb_feat, ir_feat)

    def _forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor) -> torch.Tensor:
        rgb_r = self.rgb_reduce(rgb_feat)
        ir_r = self.ir_reduce(ir_feat)

        ir_r_aligned = self.align(rgb_feat=rgb_r, ir_feat=ir_r)

        delta = self.out_norm(self.out_proj(ir_r_aligned))
        return ir_feat + torch.tanh(self.gate) * delta


class FullFDAMAlign(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        num_fab: int = 1,
        dkg_hidden_channels: int | None = None,
        weight_softmax: bool = True,
        use_groupnorm: bool = True,
        use_checkpoint: bool = False,
        num_groups: int = 1,
        share_dkg: bool = True,
    ):
        super().__init__()
        self.channels = int(channels)
        self.use_checkpoint = bool(use_checkpoint)

        self.align = FDAMPreAlign(
            feat_channels=self.channels,
            kernel_sizes=tuple(int(k) for k in kernel_sizes),
            num_fab=int(num_fab),
            dkg_hidden_channels=dkg_hidden_channels,
            weight_softmax=bool(weight_softmax),
            use_groupnorm=bool(use_groupnorm),
            num_groups=int(num_groups),
            share_dkg=bool(share_dkg),
        )

    def forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and (rgb_feat.requires_grad or ir_feat.requires_grad):
            return checkpoint(self._forward, rgb_feat, ir_feat)
        return self._forward(rgb_feat, ir_feat)

    def _forward(self, rgb_feat: torch.Tensor, ir_feat: torch.Tensor) -> torch.Tensor:
        return self.align(rgb_feat=rgb_feat, ir_feat=ir_feat)
