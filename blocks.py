from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def Normalize(channels, num_groups=32):
    return nn.GroupNorm(
        num_groups=min(num_groups, channels),
        num_channels=channels,
        eps=1e-6,
        affine=True,
    )


class FactoredConv3d(nn.Module):
    """Spatial (H,W) conv + Depth (D) conv, factored.

    In 2D mode (depth_mode="2d"), the depth conv is skipped, so the model
    can process single slices (D=1) without artifacts.
    """

    def __init__(self, in_ch, out_ch, spatial_kernel=3, depth_kernel=3, stride=1, depth_stride=1):
        super().__init__()
        sp = spatial_kernel // 2
        dp = depth_kernel // 2

        self.spatial_conv = nn.Conv3d(
            in_ch, out_ch,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            stride=(1, stride, stride),
            padding=(0, sp, sp),
        )
        self.depth_conv = nn.Conv3d(
            out_ch, out_ch,
            kernel_size=(depth_kernel, 1, 1),
            stride=(depth_stride, 1, 1),
            padding=(dp, 1, 1) if depth_stride > 1 else (dp, 0, 0),
        )
        # Initialize depth conv as identity-like for smooth 2D->3D transition
        nn.init.dirac_(self.depth_conv.weight)
        if self.depth_conv.bias is not None:
            nn.init.zeros_(self.depth_conv.bias)

    def forward(self, x, depth_mode="3d"):
        x = self.spatial_conv(x)
        if depth_mode == "3d":
            x = self.depth_conv(x)
        return x


class FactoredResBlock(nn.Module):
    """ResBlock built on factored spatial + depth convolutions."""

    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.norm1 = Normalize(in_ch)
        self.conv1 = FactoredConv3d(in_ch, out_ch)
        self.norm2 = Normalize(out_ch)
        self.conv2 = FactoredConv3d(out_ch, out_ch)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, depth_mode="3d"):
        h = self.conv1(F.silu(self.norm1(x)), depth_mode)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))), depth_mode)
        return h + self.skip(x)


# ──────────────────────────────────────────────────────
# Multi-Axis Attention
# ──────────────────────────────────────────────────────

class AxisAttention(nn.Module):
    """Self-attention along a single axis.

    axis="d": each (H,W) position attends across D
    axis="h": each (D,W) position attends across H
    axis="w": each (D,H) position attends across W
    """

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.norm = Normalize(channels)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x, axis="d"):
        B, C, D, H, W = x.shape
        h = self.norm(x)

        if axis == "d":
            h = rearrange(h, "b c d h w -> (b h w) d c")
        elif axis == "h":
            h = rearrange(h, "b c d h w -> (b d w) h c")
        elif axis == "w":
            h = rearrange(h, "b c d h w -> (b d h) w c")

        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)

        head_dim = C // self.num_heads
        q = rearrange(q, "b s (nh hd) -> b nh s hd", nh=self.num_heads, hd=head_dim)
        k = rearrange(k, "b s (nh hd) -> b nh s hd", nh=self.num_heads, hd=head_dim)
        v = rearrange(v, "b s (nh hd) -> b nh s hd", nh=self.num_heads, hd=head_dim)

        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "b nh s hd -> b s (nh hd)")
        out = self.proj(out)

        if axis == "d":
            out = rearrange(out, "(b h w) d c -> b c d h w", b=B, h=H, w=W)
        elif axis == "h":
            out = rearrange(out, "(b d w) h c -> b c d h w", b=B, d=D, w=W)
        elif axis == "w":
            out = rearrange(out, "(b d h) w c -> b c d h w", b=B, d=D, h=H)

        return x + out


class MultiAxisAttention(nn.Module):
    """Sequential attention along all three axes (D, H, W).

    In 2D mode (depth_mode="2d"), the depth-axis attention is skipped.
    Total complexity: O(DHW * (D + H + W))
    """

    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.attn_d = AxisAttention(channels, num_heads)
        self.attn_h = AxisAttention(channels, num_heads)
        self.attn_w = AxisAttention(channels, num_heads)

    def forward(self, x, depth_mode="3d"):
        if depth_mode == "3d":
            x = self.attn_d(x, axis="d")
        x = self.attn_h(x, axis="h")
        x = self.attn_w(x, axis="w")
        return x

class Downsample3D(nn.Module):
    """Spatial 2x downsample + optional depth 2x downsample."""

    def __init__(self, channels, downsample_depth=True):
        super().__init__()
        self.downsample_depth = downsample_depth
        if downsample_depth:
            self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=1)
        else:
            self.conv = nn.Conv3d(channels, channels, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x, depth_mode="3d"):
        if depth_mode == "3d" and self.downsample_depth:
            return self.conv(x)
        elif not self.downsample_depth:
            return self.conv(x)
        else:
            # 2D mode but layer has depth downsample: only downsample spatial
            return F.interpolate(x, scale_factor=(1, 0.5, 0.5), mode="trilinear", align_corners=False)


class Upsample3D(nn.Module):
    """Spatial 2x upsample + optional depth 2x upsample."""

    def __init__(self, channels, upsample_depth=True):
        super().__init__()
        self.upsample_depth = upsample_depth
        self.conv = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x, depth_mode="3d"):
        if depth_mode == "3d" and self.upsample_depth:
            x = F.interpolate(x, scale_factor=2, mode="trilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        return self.conv(x)


# ──────────────────────────────────────────────────────
# Foreground-weighted reconstruction loss
# ──────────────────────────────────────────────────────

def weighted_recon_loss(recon, target, mode="l1", fg_weight=5.0, bg_threshold=-0.9):
    """L1 / L2 reconstruction loss with brain-foreground weighting.

    Brain voxels (> bg_threshold) get `fg_weight` times more weight than
    background voxels. This prevents the model from being lazy on brain
    tissue details by only learning to predict -1 for background.
    """
    if mode == "l1":
        pixel_loss = (recon - target).abs()
    elif mode == "mse":
        pixel_loss = (recon - target) ** 2
    else:
        raise ValueError(f"Unknown recon loss mode: {mode}")

    with torch.no_grad():
        fg_mask = (target > bg_threshold).float()
        weight = 1.0 + (fg_weight - 1.0) * fg_mask
        weight = weight / weight.mean()

    return (pixel_loss * weight).mean()


# ──────────────────────────────────────────────────────
# Slice-based Perceptual Loss (VGG features on 2D slices)
# ──────────────────────────────────────────────────────

class SlicePerceptualLoss(nn.Module):
    """Perceptual loss via pretrained VGG16 on 2D slices extracted from 3D volumes.

    Strategy: sample a few axial/coronal/sagittal slices from recon & target,
    expand to 3-channel (VGG expects RGB), extract multi-layer features, compare.
    Lightweight: only runs VGG on a handful of 2D slices.
    """

    def __init__(self, n_slices_per_axis: int = 3):
        super().__init__()
        self.n_slices = n_slices_per_axis

        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
            vgg[16:23],# relu4_3
        ])
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.autocast("cuda", enabled=False)
    @torch.autocast("cpu", enabled=False)
    def _extract_features(self, x):
        if self.blocks[0][0].weight.device != x.device:
            self.blocks = self.blocks.to(x.device)

        x = x.float()
        x = x.repeat(1, 3, 1, 1)
        x = (x + 1.0) / 2.0
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        x = (x - mean) / std

        feats = []
        h = x
        for block in self.blocks:
            h = block(h)
            feats.append(h)
        return feats

    def _sample_slices(self, vol):
        B, C, D, H, W = vol.shape
        target_size = (224, 224)
        slices = []
        for frac in np.linspace(0.25, 0.75, self.n_slices):
            idx = int(D * frac)
            s = vol[:, :, idx, :, :]
            s = F.interpolate(s, size=target_size, mode="bilinear", align_corners=False)
            slices.append(s)
        for frac in np.linspace(0.25, 0.75, self.n_slices):
            idx = int(H * frac)
            s = vol[:, :, :, idx, :]
            s = F.interpolate(s, size=target_size, mode="bilinear", align_corners=False)
            slices.append(s)
        for frac in np.linspace(0.25, 0.75, self.n_slices):
            idx = int(W * frac)
            s = vol[:, :, :, :, idx]
            s = F.interpolate(s, size=target_size, mode="bilinear", align_corners=False)
            slices.append(s)
        return torch.cat(slices, dim=0)

    def forward(self, recon, target):
        if recon.shape[2] == 1:
            recon_slices = recon[:, :, 0, :, :]
            target_slices = target[:, :, 0, :, :]
        else:
            recon_slices = self._sample_slices(recon)
            target_slices = self._sample_slices(target)

        recon_feats = self._extract_features(recon_slices)
        target_feats = self._extract_features(target_slices)

        loss = 0.0
        for rf, tf in zip(recon_feats, target_feats):
            loss = loss + F.l1_loss(rf, tf)
        return loss
