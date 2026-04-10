from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .blocks import (
    Normalize,
    FactoredConv3d,
    FactoredResBlock,
    MultiAxisAttention,
    Downsample3D,
    Upsample3D,
    SlicePerceptualLoss,
    weighted_recon_loss,
)
from .quantizer import VectorQuantizer


class NeuroQuantEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        anat_channels: int = 64,
        mod_channels: int = 32,
        dropout: float = 0.0,
        attention_levels: tuple = (2,),
        num_heads: int = 8,
    ):
        super().__init__()
        self.gradient_checkpointing = False

        self.conv_in = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        channels = [base_channels * m for m in channel_multipliers]
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            block = nn.ModuleDict()
            res_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                res_blocks.append(FactoredResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            block["res"] = res_blocks
            if i in attention_levels:
                block["attn"] = MultiAxisAttention(out_ch, num_heads)
            block["down"] = Downsample3D(out_ch, downsample_depth=True)
            self.down_blocks.append(block)

        # Mid block (global context)
        self.mid_res1 = FactoredResBlock(channels[-1], channels[-1], dropout)
        self.mid_attn = MultiAxisAttention(channels[-1], num_heads)
        self.mid_res2 = FactoredResBlock(channels[-1], channels[-1], dropout)

        self.norm_out = Normalize(channels[-1])

        # ── Dual-stream heads ──
        # Both heads see the same shared feature map F^(4); they differ only
        # in the lightweight conv head + downstream usage (VQ vs FiLM).
        self.head_anat = nn.Conv3d(channels[-1], anat_channels, 1)
        self.head_mod = nn.Conv3d(channels[-1], mod_channels, 1)

        self.anat_channels = anat_channels
        self.mod_channels = mod_channels

    def _run_stage(self, block, h, depth_mode):
        for res in block["res"]:
            h = res(h, depth_mode)
        if "attn" in block:
            h = block["attn"](h, depth_mode)
        h = block["down"](h, depth_mode)
        return h

    def _run_mid(self, h, depth_mode):
        h = self.mid_res1(h, depth_mode)
        h = self.mid_attn(h, depth_mode)
        h = self.mid_res2(h, depth_mode)
        return h

    def forward(self, x, depth_mode="3d"):
        h = self.conv_in(x)

        for block in self.down_blocks:
            if self.gradient_checkpointing and self.training:
                h = torch_checkpoint(self._run_stage, block, h, depth_mode, use_reentrant=False)
            else:
                h = self._run_stage(block, h, depth_mode)

        if self.gradient_checkpointing and self.training:
            h = torch_checkpoint(self._run_mid, h, depth_mode, use_reentrant=False)
        else:
            h = self._run_mid(h, depth_mode)

        h = F.silu(self.norm_out(h))                # shared F^(t)
        z_anat = self.head_anat(h)                  # (B, C_a, D', H', W')
        z_mod = self.head_mod(h)                    # (B, C_m, D', H', W')
        return z_anat, z_mod


# ─────────────────────────────────────────────────────────────────────────
# FiLM decoder
# ─────────────────────────────────────────────────────────────────────────

class FiLMGenerator(nn.Module):
    """MLP that predicts (gamma_l, beta_l) for each decoder layer.

    Input  : u_m = concat(GAP(z_mod), s_m)
    Output : a list of (gamma, beta) tensors, one per layer, sized to that
             layer's channel count.
    """

    def __init__(self, in_dim: int, layer_channels: list, hidden_dim: int = 256):
        super().__init__()
        self.layer_channels = layer_channels
        total = sum(2 * c for c in layer_channels)  # gamma + beta per layer
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, total),
        )
        # Initialize last layer so initial FiLM ≈ identity (gamma=1, beta=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, u: torch.Tensor):
        raw = self.mlp(u)                             # (B, total)
        params = []
        offset = 0
        for c in self.layer_channels:
            gamma = 1.0 + raw[:, offset:offset + c]   # identity init
            offset += c
            beta = raw[:, offset:offset + c]
            offset += c
            params.append((gamma, beta))
        return params


def film_apply(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Channel-wise affine: h' = gamma * h + beta. h is (B, C, D, H, W)."""
    g = gamma.view(gamma.size(0), gamma.size(1), 1, 1, 1)
    b = beta.view(beta.size(0), beta.size(1), 1, 1, 1)
    return g * h + b


class FiLMDecoder3D(nn.Module):

    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        anat_channels: int = 64,
        dropout: float = 0.0,
        attention_levels: tuple = (2,),
        num_heads: int = 8,
    ):
        super().__init__()
        self.gradient_checkpointing = False
        channels = [base_channels * m for m in channel_multipliers]

        self.conv_in = nn.Conv3d(anat_channels, channels[-1], 3, padding=1)

        # Mid block
        self.mid_res1 = FactoredResBlock(channels[-1], channels[-1], dropout)
        self.mid_attn = MultiAxisAttention(channels[-1], num_heads)
        self.mid_res2 = FactoredResBlock(channels[-1], channels[-1], dropout)

        # Up blocks (reverse order)
        self.up_blocks = nn.ModuleList()
        in_ch = channels[-1]
        rev_channels = list(reversed(channels))
        rev_attn_levels = [len(channels) - 1 - l for l in attention_levels]
        for i, out_ch in enumerate(rev_channels):
            block = nn.ModuleDict()
            block["up"] = Upsample3D(in_ch, upsample_depth=True)
            res_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                res_blocks.append(FactoredResBlock(in_ch, out_ch, dropout))
                in_ch = out_ch
            block["res"] = res_blocks
            if i in rev_attn_levels:
                block["attn"] = MultiAxisAttention(out_ch, num_heads)
            self.up_blocks.append(block)

        self.norm_out = Normalize(channels[0])
        self.conv_out = nn.Conv3d(channels[0], out_channels, 3, padding=1)

        # Channel sequence used by the FiLM generator: 1 entry per modulated stage.
        # Order: [mid (channels[-1]), up_block_0_out, up_block_1_out, ..., final_out]
        self.film_layer_channels = [channels[-1]] + list(rev_channels)

    def _run_mid(self, h, depth_mode):
        h = self.mid_res1(h, depth_mode)
        h = self.mid_attn(h, depth_mode)
        h = self.mid_res2(h, depth_mode)
        return h

    def _run_stage(self, block, h, depth_mode):
        h = block["up"](h, depth_mode)
        for res in block["res"]:
            h = res(h, depth_mode)
        if "attn" in block:
            h = block["attn"](h, depth_mode)
        return h

    def forward(self, z_q: torch.Tensor, film_params: list, depth_mode: str = "3d"):
        """
        Args:
            z_q:         quantized anatomical latent (B, C_a, D', H', W')
            film_params: list of (gamma, beta), len == len(self.film_layer_channels)
            depth_mode:  "3d" or "2d"
        """
        assert len(film_params) == len(self.film_layer_channels), (
            f"FiLM expects {len(self.film_layer_channels)} (gamma,beta) pairs, "
            f"got {len(film_params)}"
        )

        h = self.conv_in(z_q)

        # Mid block + FiLM_0
        if self.gradient_checkpointing and self.training:
            h = torch_checkpoint(self._run_mid, h, depth_mode, use_reentrant=False)
        else:
            h = self._run_mid(h, depth_mode)
        gamma, beta = film_params[0]
        h = film_apply(h, gamma, beta)

        # Up blocks + FiLM_{l>=1}
        for i, block in enumerate(self.up_blocks):
            if self.gradient_checkpointing and self.training:
                h = torch_checkpoint(self._run_stage, block, h, depth_mode, use_reentrant=False)
            else:
                h = self._run_stage(block, h, depth_mode)
            gamma, beta = film_params[i + 1]
            h = film_apply(h, gamma, beta)

        h = self.conv_out(F.silu(self.norm_out(h)))
        return torch.tanh(h)


# ─────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer + modality classifier
# ─────────────────────────────────────────────────────────────────────────

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, alpha)


class ModalityAdversary(nn.Module):
    def __init__(self, in_channels: int, num_modalities: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden, 1),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_modalities),
        )

    def forward(self, z_anat: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        z_rev = grad_reverse(z_anat, alpha)
        return self.net(z_rev)


# ─────────────────────────────────────────────────────────────────────────
# Full NeuroQuant model
# ─────────────────────────────────────────────────────────────────────────

class NeuroQuant(nn.Module):
    """Dual-stream 3D VQ-VAE.

    Forward returns a dict with everything needed by the training loop:
      - recon                : reconstructed volume
      - z_anat               : continuous anatomical latent (pre-VQ)
      - z_anat_q             : quantized anatomical latent
      - z_mod                : modality latent
      - vq_loss              : codebook + commitment loss
      - perplexity           : codebook usage
      - indices              : (B, D', H', W') chosen code indices
      - film_params          : list of (gamma_l, beta_l)
      - mod_logits           : modality classifier logits (for adversarial loss)

    For 2D-mode inputs (D=1), depth_mode="2d" should be passed; the
    underlying axis-attention / factored conv blocks already handle this.
    """

    NUM_MODALITIES = 2  # T1w, T2w

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: tuple = (1, 2, 4),
        num_res_blocks: int = 2,
        anat_channels: int = 64,
        mod_channels: int = 32,
        codebook_size: int = 1024,
        commitment_beta: float = 0.25,
        modality_embed_dim: int = 32,
        film_hidden: int = 256,
        dropout: float = 0.0,
        attention_levels: tuple = (2,),
        num_heads: int = 8,
        adv_alpha: float = 1.0,
    ):
        super().__init__()
        self.adv_alpha = adv_alpha

        self.encoder = NeuroQuantEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            anat_channels=anat_channels,
            mod_channels=mod_channels,
            dropout=dropout,
            attention_levels=attention_levels,
            num_heads=num_heads,
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=anat_channels,
            beta=commitment_beta,
        )

        self.decoder = FiLMDecoder3D(
            out_channels=in_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
            anat_channels=anat_channels,
            dropout=dropout,
            attention_levels=attention_levels,
            num_heads=num_heads,
        )

        # Per-modality learned embedding s_m
        self.modality_embedding = nn.Embedding(self.NUM_MODALITIES, modality_embed_dim)

        self.film_generator = FiLMGenerator(
            in_dim=mod_channels + modality_embed_dim,
            layer_channels=self.decoder.film_layer_channels,
            hidden_dim=film_hidden,
        )

        self.modality_adversary = ModalityAdversary(
            in_channels=anat_channels,
            num_modalities=self.NUM_MODALITIES,
        )

    # ── Convenience ──────────────────────────────────────────────────────

    def enable_gradient_checkpointing(self):
        self.encoder.gradient_checkpointing = True
        self.decoder.gradient_checkpointing = True
        print("[NeuroQuant] Gradient checkpointing enabled")

    def disable_gradient_checkpointing(self):
        self.encoder.gradient_checkpointing = False
        self.decoder.gradient_checkpointing = False

    def encode(self, x, depth_mode="3d"):
        return self.encoder(x, depth_mode)

    def quantize(self, z_anat):
        return self.quantizer(z_anat)

    def compute_film(self, z_mod: torch.Tensor, modality: torch.Tensor):
        """Build u_m = concat(GAP(z_mod), s_m) and predict FiLM params."""
        gap = F.adaptive_avg_pool3d(z_mod, 1).flatten(1)        # (B, C_m)
        s_m = self.modality_embedding(modality)                  # (B, C_s)
        u_m = torch.cat([gap, s_m], dim=1)
        return self.film_generator(u_m)

    def decode(self, z_q, film_params, depth_mode="3d"):
        return self.decoder(z_q, film_params, depth_mode)

    # ── Full forward ─────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        modality: torch.Tensor,
        depth_mode: str = "3d",
        run_adversary: bool = True,
    ):
        z_anat, z_mod = self.encoder(x, depth_mode)
        z_anat_q, vq_loss, indices, perplexity = self.quantizer(z_anat)

        film_params = self.compute_film(z_mod, modality)
        recon = self.decoder(z_anat_q, film_params, depth_mode)

        mod_logits = None
        if run_adversary:
            mod_logits = self.modality_adversary(z_anat, alpha=self.adv_alpha)

        return {
            "recon": recon,
            "z_anat": z_anat,
            "z_mod": z_mod,
            "z_anat_q": z_anat_q,
            "indices": indices,
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "film_params": film_params,
            "mod_logits": mod_logits,
        }


# ─────────────────────────────────────────────────────────────────────────
# Losses: 3D SSIM + container
# ─────────────────────────────────────────────────────────────────────────

def _gaussian_kernel_3d(window_size: int = 7, sigma: float = 1.5, device=None, dtype=None):
    coords = torch.arange(window_size, dtype=dtype or torch.float32, device=device)
    coords = coords - (window_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    k3 = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return k3.view(1, 1, window_size, window_size, window_size)


def ssim3d(x: torch.Tensor, y: torch.Tensor, window_size: int = 7, data_range: float = 2.0) -> torch.Tensor:
    """Differentiable 3D SSIM (mean over the volume).

    Works on (B, 1, D, H, W). For 2D-mode inputs (D == 1), falls back to a
    same-window approach but with a depth window of 1.
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    if x.shape[2] == 1:
        # 2D fallback: collapse depth to keep computation cheap
        x2 = x[:, :, 0]
        y2 = y[:, :, 0]
        # Use a 2D Gaussian
        coords = torch.arange(window_size, dtype=x.dtype, device=x.device)
        coords = coords - (window_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2.0 * 1.5 ** 2))
        g = g / g.sum()
        k = (g[:, None] * g[None, :]).view(1, 1, window_size, window_size)
        pad = window_size // 2
        mu_x = F.conv2d(x2, k, padding=pad)
        mu_y = F.conv2d(y2, k, padding=pad)
        mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sigma_x2 = F.conv2d(x2 * x2, k, padding=pad) - mu_x2
        sigma_y2 = F.conv2d(y2 * y2, k, padding=pad) - mu_y2
        sigma_xy = F.conv2d(x2 * y2, k, padding=pad) - mu_xy
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        )
        return ssim_map.mean()

    k = _gaussian_kernel_3d(window_size, 1.5, device=x.device, dtype=x.dtype)
    pad = window_size // 2
    mu_x = F.conv3d(x, k, padding=pad)
    mu_y = F.conv3d(y, k, padding=pad)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv3d(x * x, k, padding=pad) - mu_x2
    sigma_y2 = F.conv3d(y * y, k, padding=pad) - mu_y2
    sigma_xy = F.conv3d(x * y, k, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return ssim_map.mean()


class NeuroQuantLoss(nn.Module):
    """Per-step loss container.

    Combines:
      - L_rec  = ||x - x_hat||_1 + lambda_ssim * (1 - SSIM3D(x, x_hat))
        (with optional foreground weighting via blocks.weighted_recon_loss)
      - L_VQ
      - optional perceptual loss (slice-wise VGG, reused unchanged)

    Cross-modal and adversarial losses are NOT in here — they need
    paired data and are computed in the training loop directly.
    """

    def __init__(
        self,
        ssim_weight: float = 0.5,
        vq_weight: float = 1.0,
        perceptual_weight: float = 0.0,
        fg_weight: float = 5.0,
        bg_threshold: float = -0.9,
    ):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.vq_weight = vq_weight
        self.perceptual_weight = perceptual_weight
        self.fg_weight = fg_weight
        self.bg_threshold = bg_threshold

        self.percep_loss = (
            SlicePerceptualLoss(n_slices_per_axis=3) if perceptual_weight > 0 else None
        )

    def forward(self, recon, target, vq_loss):
        l1 = weighted_recon_loss(recon, target, "l1", self.fg_weight, self.bg_threshold)

        if self.ssim_weight > 0:
            ssim_val = ssim3d(recon.float(), target.float())
            ssim_term = 1.0 - ssim_val
        else:
            ssim_term = torch.tensor(0.0, device=recon.device)

        rec = l1 + self.ssim_weight * ssim_term
        total = rec + self.vq_weight * vq_loss

        p_loss = torch.tensor(0.0, device=recon.device)
        if self.percep_loss is not None:
            p_loss = self.percep_loss(recon, target)
            total = total + self.perceptual_weight * p_loss

        return {
            "loss": total,
            "recon_loss": rec,
            "l1_loss": l1,
            "ssim_term": ssim_term.detach() if torch.is_tensor(ssim_term) else ssim_term,
            "vq_loss": vq_loss.detach(),
            "percep_loss": p_loss,
        }


if __name__ == "__main__":
    model = NeuroQuant(
        base_channels=32,
        channel_multipliers=(1, 2, 4),
        anat_channels=32,
        mod_channels=16,
        codebook_size=512,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"NeuroQuant params: {n_params:.1f}M")

    x = torch.randn(2, 1, 32, 32, 32)
    m = torch.tensor([0, 1])
    with torch.no_grad():
        out = model(x, m, depth_mode="3d")
    print("recon:", out["recon"].shape)
    print("z_anat:", out["z_anat"].shape, "z_mod:", out["z_mod"].shape)
    print("vq_loss:", out["vq_loss"].item(), "perplexity:", out["perplexity"].item())
    print("mod_logits:", out["mod_logits"].shape)

    # 2D path
    x2 = torch.randn(2, 1, 1, 64, 64)
    with torch.no_grad():
        out2 = model(x2, m, depth_mode="2d")
    print("[2D] recon:", out2["recon"].shape)
