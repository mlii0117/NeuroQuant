from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        revive_dead: bool = True,
        revive_threshold: float = 0.1,
    ):
        super().__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.revive_dead = revive_dead
        self.revive_threshold = revive_threshold

        # The codebook is a buffer (NOT a parameter): updated by EMA, not by grad.
        embed = torch.empty(num_embeddings, embedding_dim)
        embed.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.register_buffer("embedding", embed)

        # EMA accumulators
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", embed.clone())
        self.register_buffer("initialized", torch.tensor(False))

        # Reservoir for cold-start k-means init: lets us accumulate encoder
        # vectors across multiple forward calls when a single batch has fewer
        # tokens than K. We keep at most K vectors; once full, init fires.
        self.register_buffer("init_reservoir", torch.empty(0, embedding_dim))

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, encodings: torch.Tensor):
        """Update codebook in place via EMA.

        Args:
            flat:      (N, D) encoder vectors used in this step.
            encodings: (N, K) one-hot assignments.
        """
        # 1) update cluster sizes
        cluster_size_new = encodings.sum(dim=0)                       # (K,)
        self.ema_cluster_size.mul_(self.decay).add_(
            cluster_size_new, alpha=1.0 - self.decay
        )

        # 2) update sums of assigned vectors
        dw = encodings.t() @ flat                                     # (K, D)
        self.ema_w.mul_(self.decay).add_(dw, alpha=1.0 - self.decay)

        # 3) Laplace smoothing
        n = self.ema_cluster_size.sum()
        smoothed = (self.ema_cluster_size + self.eps) / (n + self.K * self.eps) * n

        self.embedding.copy_(self.ema_w / smoothed.unsqueeze(1))

        if self.revive_dead:
            avg = self.ema_cluster_size.mean()
            threshold = max(self.revive_threshold * avg, torch.tensor(1e-3, device=avg.device))
            dead = self.ema_cluster_size < threshold
            n_dead = int(dead.sum().item())
            if n_dead > 0 and flat.size(0) > 0:
                rand_idx = torch.randint(0, flat.size(0), (n_dead,), device=flat.device)
                self.embedding[dead] = flat[rand_idx]
                # reset their EMA stats so they get a fresh start
                self.ema_cluster_size[dead] = self.revive_threshold
                self.ema_w[dead] = flat[rand_idx]

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: continuous anatomical latent (B, C, D, H, W) with C == embedding_dim.

        Returns:
            z_q:        quantized latent, same shape as z, with straight-through grad
            vq_loss:    scalar commitment loss (codebook loss is implicit via EMA)
            indices:    (B, D, H, W) long tensor of selected codebook indices
            perplexity: scalar codebook usage indicator
        """
        assert z.shape[1] == self.D, (
            f"VectorQuantizer expects channel dim {self.D}, got {z.shape[1]}"
        )

        # (B, C, D, H, W) -> (B, D, H, W, C) -> (N, C)
        z_perm = z.permute(0, 2, 3, 4, 1).contiguous()
        flat = z_perm.view(-1, self.D)

        if self.training and not bool(self.initialized.item()):
            with torch.no_grad():
                self.init_reservoir = torch.cat(
                    [self.init_reservoir.to(flat.device, dtype=flat.dtype), flat.detach()],
                    dim=0,
                )
                # Cap reservoir at 2*K to keep memory bounded if init is slow.
                if self.init_reservoir.size(0) > 2 * self.K:
                    perm = torch.randperm(self.init_reservoir.size(0), device=flat.device)[: 2 * self.K]
                    self.init_reservoir = self.init_reservoir[perm]

                if self.init_reservoir.size(0) >= self.K:
                    idx = torch.randperm(self.init_reservoir.size(0), device=flat.device)[: self.K]
                    self.embedding.copy_(self.init_reservoir[idx])
                    self.ema_w.copy_(self.embedding)
                    self.ema_cluster_size.fill_(1.0)
                    self.initialized.fill_(True)
                    # Free reservoir
                    self.init_reservoir = torch.empty(0, self.D, device=flat.device, dtype=flat.dtype)

        # Squared L2 distances: ||z||^2 + ||e||^2 - 2 z·e
        dist = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.pow(2).sum(dim=1)
            - 2.0 * flat @ self.embedding.t()
        )

        indices_flat = dist.argmin(dim=1)                              # (N,)
        encodings = F.one_hot(indices_flat, num_classes=self.K).type(flat.dtype)
        z_q_flat = encodings @ self.embedding                          # (N, D)
        z_q = z_q_flat.view(z_perm.shape)                              # (B, D, H, W, C)

        # EMA codebook update (only in training)
        if self.training:
            self._ema_update(flat.detach(), encodings.detach())

        # Commitment loss only — codebook is updated by EMA, not by grad.
        commitment_loss = F.mse_loss(z_perm, z_q.detach())
        vq_loss = self.beta * commitment_loss

        # Straight-through estimator
        z_q = z_perm + (z_q - z_perm).detach()
        z_q = z_q.permute(0, 4, 1, 2, 3).contiguous()                  # (B, C, D, H, W)

        # Perplexity (codebook usage)
        with torch.no_grad():
            avg_probs = encodings.mean(dim=0)
            perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        indices = indices_flat.view(z_perm.shape[:-1])                 # (B, D, H, W)
        return z_q, vq_loss, indices, perplexity
