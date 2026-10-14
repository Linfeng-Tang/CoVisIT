import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.unet_utils import conv_nd, normalization, zero_module, ExtractionBlock2


class CrossModalModulation(nn.Module):
    def __init__(self, channels, dims=2, use_checkpoint=False, zero_init_affine=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.rgb_pre = conv_nd(dims, channels, channels, 3, padding=1)
        self.ir_pre = conv_nd(dims, channels, channels, 3, padding=1)
        self.fuse = nn.Sequential(
            conv_nd(dims, channels * 2, channels * 2, 3, padding=1),
            normalization(channels * 2),
            nn.SiLU(),
            conv_nd(dims, channels * 2, channels * 2, 3, padding=1),
            normalization(channels * 2),
            nn.SiLU(),
        )
        rgb_affine = conv_nd(dims, channels, channels * 2, 3, padding=1)
        ir_affine = conv_nd(dims, channels, channels * 2, 3, padding=1)
        self.rgb_affine = zero_module(rgb_affine) if zero_init_affine else rgb_affine
        self.ir_affine = zero_module(ir_affine) if zero_init_affine else ir_affine

    def forward(self, rgb, ir):
        if self.use_checkpoint and (rgb.requires_grad or ir.requires_grad):
            return checkpoint(self._forward, rgb, ir, use_reentrant=False)
        return self._forward(rgb, ir)

    def _forward(self, rgb, ir):
        rgb_base = self.rgb_pre(rgb)
        ir_base = self.ir_pre(ir)
        fused = self.fuse(torch.cat([rgb_base, ir_base], dim=1))
        rgb_mix, ir_mix = torch.chunk(fused, 2, dim=1)

        rgb_gamma, rgb_beta = torch.chunk(self.rgb_affine(rgb_mix), 2, dim=1)
        ir_gamma, ir_beta = torch.chunk(self.ir_affine(ir_mix), 2, dim=1)

        rgb_cm = rgb_base * (1.0 + rgb_gamma) + rgb_beta
        ir_cm = ir_base * (1.0 + ir_gamma) + ir_beta
        return rgb_cm, ir_cm


class CrossModalModulationDownsample(nn.Module):

    def __init__(self, in_channels, out_channels, dims=2, use_checkpoint=False, zero_init_affine=True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.rgb_down = conv_nd(dims, in_channels, out_channels, 3, stride=2, padding=1)
        self.ir_down = conv_nd(dims, in_channels, out_channels, 3, stride=2, padding=1)
        self.fuse = nn.Sequential(
            conv_nd(dims, out_channels * 2, out_channels * 2, 3, padding=1),
            normalization(out_channels * 2),
            nn.SiLU(),
            conv_nd(dims, out_channels * 2, out_channels * 2, 3, padding=1),
            normalization(out_channels * 2),
            nn.SiLU(),
        )
        rgb_affine = conv_nd(dims, out_channels, out_channels * 2, 3, padding=1)
        ir_affine = conv_nd(dims, out_channels, out_channels * 2, 3, padding=1)
        self.rgb_affine = zero_module(rgb_affine) if zero_init_affine else rgb_affine
        self.ir_affine = zero_module(ir_affine) if zero_init_affine else ir_affine

    def forward(self, rgb, ir):
        if self.use_checkpoint and (rgb.requires_grad or ir.requires_grad):
            return checkpoint(self._forward, rgb, ir, use_reentrant=False)
        return self._forward(rgb, ir)

    def _forward(self, rgb, ir):
        rgb_base = self.rgb_down(rgb)
        ir_base = self.ir_down(ir)
        fused = self.fuse(torch.cat([rgb_base, ir_base], dim=1))
        rgb_mix, ir_mix = torch.chunk(fused, 2, dim=1)

        rgb_gamma, rgb_beta = torch.chunk(self.rgb_affine(rgb_mix), 2, dim=1)
        ir_gamma, ir_beta = torch.chunk(self.ir_affine(ir_mix), 2, dim=1)

        rgb_cm = rgb_base * (1.0 + rgb_gamma) + rgb_beta
        ir_cm = ir_base * (1.0 + ir_gamma) + ir_beta
        return rgb_cm, ir_cm


class MultiScaleCrossModalEncoder(nn.Module):

    def __init__(self, in_channels, ch_ds1, ch_ds2, ch_ds4, dims=2, use_checkpoint=False, zero_init_affine=True):
        super().__init__()
        self.rgb_stem = conv_nd(dims, in_channels, ch_ds1, 3, padding=1)
        self.ir_stem = conv_nd(dims, in_channels, ch_ds1, 3, padding=1)

        self.rgb_stage1 = ExtractionBlock2(ch_ds1, dims=dims, use_checkpoint=use_checkpoint)
        self.ir_stage1 = ExtractionBlock2(ch_ds1, dims=dims, use_checkpoint=use_checkpoint)
        self.cm_down_12 = CrossModalModulationDownsample(
            ch_ds1, ch_ds2, dims=dims, use_checkpoint=use_checkpoint, zero_init_affine=zero_init_affine
        )

        self.rgb_stage2 = ExtractionBlock2(ch_ds2, dims=dims, use_checkpoint=use_checkpoint)
        self.ir_stage2 = ExtractionBlock2(ch_ds2, dims=dims, use_checkpoint=use_checkpoint)
        self.cm_down_24 = CrossModalModulationDownsample(
            ch_ds2, ch_ds4, dims=dims, use_checkpoint=use_checkpoint, zero_init_affine=zero_init_affine
        )

    def forward(self, rgb, ir):
        rgb_feats = {}
        ir_feats = {}

        rgb_h = self.rgb_stem(rgb)
        ir_h = self.ir_stem(ir)

        rgb_h = self.rgb_stage1(rgb_h)
        ir_h = self.ir_stage1(ir_h)

        rgb_h, ir_h = self.cm_down_12(rgb_h, ir_h)
        rgb_feats[2] = rgb_h
        ir_feats[2] = ir_h

        rgb_h = self.rgb_stage2(rgb_h)
        ir_h = self.ir_stage2(ir_h)

        rgb_h, ir_h = self.cm_down_24(rgb_h, ir_h)
        rgb_feats[4] = rgb_h
        ir_feats[4] = ir_h

        return rgb_feats, ir_feats
