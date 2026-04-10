"""
Training script for NeuroQuant: Dual-stream 3D VQ-VAE for multimodal brain MRI.

Reuses the existing 2D/3D joint training scheme: with probability `prob_2d`
each step, an extra 2D-slice mini-batch is processed and added to the loss.
The 2D path uses the same paired T1w/T2w slices, so the cross-modal loss
also runs in 2D.

Per-step losses:
    L_total = L_rec(x, x_hat)
            + L_VQ
            + lambda_cross * L_cross
            + lambda_adv   * L_adv_encoder       (encoder side, via GRL)
            + lambda_perc  * L_perceptual

The modality-classifier head is trained in the SAME backward pass thanks
to the gradient reversal layer (it is updated in the encoder direction
to MAXIMIZE the loss; the classifier itself MINIMIZES it).

Usage:
    accelerate launch --num_processes 1 -m NeuroQuant.train --config NeuroQuant/config.yaml
    accelerate launch --num_processes 4 --multi_gpu -m NeuroQuant.train --config NeuroQuant/config.yaml
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from NeuroQuant.dataset import PairedBrainMRI3DDataset
from NeuroQuant.model import NeuroQuant, NeuroQuantLoss
from NeuroQuant.metrics import psnr, ssim


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(self.decay).add_(m.data, alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


def make_grid(volume, recon, caption_prefix=""):
    import wandb
    vol = volume[0].float().cpu().numpy()
    rec = recon[0].float().cpu().numpy()
    images = []
    D, H, W = vol.shape
    for axis_name, axis, idx_func in [
        ("axial", 0, lambda s: int(s * 0.5)),
        ("coronal", 1, lambda s: int(s * 0.5)),
        ("sagittal", 2, lambda s: int(s * 0.5)),
    ]:
        if axis == 0:
            i = idx_func(D); o = vol[i]; r = rec[i]
        elif axis == 1:
            i = idx_func(H); o = vol[:, i]; r = rec[:, i]
        else:
            i = idx_func(W); o = vol[:, :, i]; r = rec[:, :, i]
        o = (o - o.min()) / (o.max() - o.min() + 1e-8)
        r = (r - r.min()) / (r.max() - r.min() + 1e-8)
        images.append(wandb.Image(np.concatenate([o, r], axis=1),
                                  caption=f"{caption_prefix}{axis_name} (orig | recon)"))
    return images


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def grl_alpha_schedule(global_step, total_steps, max_alpha=1.0, start_step=0):
    if global_step < start_step:
        return 0.0
    p = min(1.0, (global_step - start_step) / max(1, total_steps - start_step))
    return float(max_alpha * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0))


def neuroquant_step(
    model,
    loss_fn: NeuroQuantLoss,
    batch: dict,
    *,
    cross_weight: float,
    adv_weight: float,
    depth_mode: str,
    accelerator: Accelerator,
):
    """Run one paired forward pass and return a loss dict.

    Implements:
      - Per-modality recon + VQ + perceptual                  (L_rec, L_VQ)
      - Cross-modal recon: swap FiLM(T2) onto z_anat^T1, etc. (L_cross)
      - Adversarial modality classification on z_anat         (L_adv)
    """
    x_t1 = batch["T1w"]
    x_t2 = batch["T2w"]
    B = x_t1.size(0)
    device = x_t1.device

    mod_t1 = torch.full((B,), 0, dtype=torch.long, device=device)
    mod_t2 = torch.full((B,), 1, dtype=torch.long, device=device)

    with accelerator.autocast():
        out_t1 = model(x_t1, mod_t1, depth_mode=depth_mode, run_adversary=True)
        out_t2 = model(x_t2, mod_t2, depth_mode=depth_mode, run_adversary=True)

        loss_t1 = loss_fn(out_t1["recon"], x_t1, out_t1["vq_loss"])
        loss_t2 = loss_fn(out_t2["recon"], x_t2, out_t2["vq_loss"])

        recon_total = loss_t1["loss"] + loss_t2["loss"]

        # ── Cross-modal swap ──
        # Use the *quantized* anatomical code from one modality with the
        # FiLM parameters predicted from the other modality. The decoder
        # then must reconstruct the OTHER modality.
        m = accelerator.unwrap_model(model)
        # T2 reconstructed from T1's anatomy + T2's contrast
        recon_t2_from_t1 = m.decode(out_t1["z_anat_q"], out_t2["film_params"], depth_mode)
        # T1 reconstructed from T2's anatomy + T1's contrast
        recon_t1_from_t2 = m.decode(out_t2["z_anat_q"], out_t1["film_params"], depth_mode)

        l_cross = (
            (recon_t2_from_t1 - x_t2).abs().mean()
            + (recon_t1_from_t2 - x_t1).abs().mean()
        )

        # ── Adversarial: classifier on z_anat ──
        # The GRL inside the model already negates the encoder gradient,
        # so we minimize CE here for both directions.
        mod_logits = torch.cat([out_t1["mod_logits"], out_t2["mod_logits"]], dim=0)
        mod_targets = torch.cat([mod_t1, mod_t2], dim=0)
        l_adv = F.cross_entropy(mod_logits, mod_targets)

        total = recon_total + cross_weight * l_cross + adv_weight * l_adv

    return {
        "loss": total,
        "rec_t1": loss_t1["recon_loss"].detach(),
        "rec_t2": loss_t2["recon_loss"].detach(),
        "vq_t1": out_t1["vq_loss"].detach(),
        "vq_t2": out_t2["vq_loss"].detach(),
        "cross": l_cross.detach(),
        "adv": l_adv.detach(),
        "perplexity": 0.5 * (out_t1["perplexity"].detach() + out_t2["perplexity"].detach()),
        "out_t1": out_t1,
        "out_t2": out_t2,
    }


@torch.no_grad()
def validate(accelerator, model, val_loader, loss_fn, global_step, cfg_train):
    import wandb
    model.eval()
    sums = {"loss": 0, "rec_t1": 0, "rec_t2": 0, "cross": 0, "adv": 0,
            "psnr_t1": 0, "psnr_t2": 0, "ssim_t1": 0, "ssim_t2": 0,
            "perplexity": 0}
    n_val = 0
    vis_images = []

    for batch in val_loader:
        out = neuroquant_step(
            model, loss_fn, batch,
            cross_weight=cfg_train.get("cross_weight", 0.5),
            adv_weight=cfg_train.get("adv_weight", 0.001),
            depth_mode="3d",
            accelerator=accelerator,
        )

        rec_t1 = out["out_t1"]["recon"].float()
        rec_t2 = out["out_t2"]["recon"].float()
        x_t1 = batch["T1w"].float()
        x_t2 = batch["T2w"].float()
        sums["psnr_t1"] += psnr(rec_t1, x_t1)["psnr_fg"].item()
        sums["psnr_t2"] += psnr(rec_t2, x_t2)["psnr_fg"].item()
        sums["ssim_t1"] += ssim(rec_t1, x_t1)["ssim_fg"].item()
        sums["ssim_t2"] += ssim(rec_t2, x_t2)["ssim_fg"].item()

        for k in ("loss", "rec_t1", "rec_t2", "cross", "adv", "perplexity"):
            sums[k] += float(out[k])

        n_val += 1
        if accelerator.is_main_process and len(vis_images) < 12:
            vis_images.extend(make_grid(batch["T1w"][0], rec_t1[0], "[T1w] "))
            vis_images.extend(make_grid(batch["T2w"][0], rec_t2[0], "[T2w] "))

    avg = {k: v / max(n_val, 1) for k, v in sums.items()}

    if accelerator.is_main_process:
        log = {f"val/{k}": v for k, v in avg.items()}
        if vis_images:
            log["val/reconstructions"] = vis_images
        wandb.log(log, step=global_step)

    accelerator.print(
        f"\n[VAL @ {global_step}] loss={avg['loss']:.4f}  "
        f"recT1={avg['rec_t1']:.4f}  recT2={avg['rec_t2']:.4f}  "
        f"cross={avg['cross']:.4f}  adv={avg['adv']:.4f}  "
        f"PSNR(T1/T2)={avg['psnr_t1']:.2f}/{avg['psnr_t2']:.2f}  "
        f"SSIM(T1/T2)={avg['ssim_t1']:.4f}/{avg['ssim_t2']:.4f}  "
        f"perp={avg['perplexity']:.1f}\n"
    )

    model.train()
    return avg["loss"]


def train(config):
    cfg_data = config["data"]
    cfg_model = config["model"]
    cfg_train = config["training"]
    cfg_wandb = config["wandb"]
    cfg_ckpt = config["checkpoint"]

    set_seed(cfg_train["seed"])
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg_train["gradient_accumulation_steps"],
        mixed_precision="bf16" if cfg_train.get("mixed_precision", True) else "no",
    )
    accelerator.print(f"Using {accelerator.num_processes} GPU(s)")

    if accelerator.is_main_process:
        import wandb
        run_name = cfg_wandb.get("name") or (
            f"neuroquant_K{cfg_model['codebook_size']}_anat{cfg_model['anat_channels']}"
        )
        run_name += f"_{accelerator.num_processes}gpu"
        wandb.init(
            project=cfg_wandb["project"],
            entity=cfg_wandb.get("entity"),
            name=run_name,
            config=config,
        )

    # ── Data ──
    ds_kwargs = dict(
        data_root=cfg_data["data_root"],
        datasets=cfg_data.get("datasets"),
        target_shape=tuple(cfg_data["target_shape"]),
        val_per_dataset=cfg_data.get("val_per_dataset", 5),
        seed=cfg_train["seed"],
        lower_pct=cfg_data.get("lower_pct", 0.5),
        upper_pct=cfg_data.get("upper_pct", 99.5),
    )

    train_ds = PairedBrainMRI3DDataset(**ds_kwargs, mode="3d", split="train")
    val_ds = PairedBrainMRI3DDataset(**ds_kwargs, mode="3d", split="val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg_train["batch_size"],
        shuffle=True, num_workers=cfg_data["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=cfg_data["num_workers"], pin_memory=True,
    )

    # 2D joint training (paired slices)
    joint_2d3d = cfg_train.get("joint_2d3d", True)
    train_loader_2d = None
    if joint_2d3d:
        train_ds_2d = PairedBrainMRI3DDataset(
            **ds_kwargs, mode="2d", split="train",
            slices_per_volume=cfg_train.get("slices_per_volume", 8),
        )
        train_loader_2d = DataLoader(
            train_ds_2d, batch_size=cfg_train.get("batch_size_2d", 16),
            shuffle=True, num_workers=cfg_data["num_workers"],
            pin_memory=True, drop_last=True,
        )

    # ── Model ──
    model = NeuroQuant(
        in_channels=cfg_model["in_channels"],
        base_channels=cfg_model["base_channels"],
        channel_multipliers=tuple(cfg_model["channel_multipliers"]),
        num_res_blocks=cfg_model["num_res_blocks"],
        anat_channels=cfg_model["anat_channels"],
        mod_channels=cfg_model["mod_channels"],
        codebook_size=cfg_model["codebook_size"],
        commitment_beta=cfg_model.get("commitment_beta", 0.25),
        modality_embed_dim=cfg_model.get("modality_embed_dim", 32),
        film_hidden=cfg_model.get("film_hidden", 256),
        dropout=cfg_model.get("dropout", 0.0),
        attention_levels=tuple(cfg_model.get("attention_levels", [2])),
        num_heads=cfg_model.get("num_heads", 8),
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    accelerator.print(f"NeuroQuant params: {n_params:.1f}M")

    if cfg_train.get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()

    # ── Loss ──
    loss_fn = NeuroQuantLoss(
        ssim_weight=cfg_train.get("ssim_weight", 0.5),
        vq_weight=cfg_train.get("vq_weight", 1.0),
        perceptual_weight=cfg_train.get("perceptual_weight", 0.0),
        fg_weight=cfg_train.get("fg_weight", 5.0),
        bg_threshold=cfg_train.get("bg_threshold", -0.9),
    )

    # ── Optim ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg_train["learning_rate"],
        weight_decay=cfg_train.get("weight_decay", 0.0),
    )
    steps_per_epoch = math.ceil(len(train_loader) / cfg_train["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * cfg_train["max_epochs"]
    scheduler = cosine_with_warmup(optimizer, cfg_train["warmup_steps"], total_steps)

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    if train_loader_2d is not None:
        train_loader_2d = accelerator.prepare(train_loader_2d)

    unwrapped_model = accelerator.unwrap_model(model)
    ema = EMA(unwrapped_model, decay=cfg_train.get("ema_decay", 0.999))

    # ── Resume ──
    start_epoch, global_step, best_val = 0, 0, float("inf")
    resume_path = cfg_ckpt.get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        unwrapped_model.load_state_dict(ckpt["model"])
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"])
        if cfg_ckpt.get("resume_mode") == "full":
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                scheduler.load_state_dict(ckpt["scheduler"])
                start_epoch = ckpt.get("epoch", 0) + 1
                global_step = ckpt.get("global_step", 0)
                best_val = ckpt.get("best_val_loss", float("inf"))
            except Exception as e:
                accelerator.print(f"[resume] optimizer load failed: {e}")
        accelerator.print(f"Resumed from {resume_path}")
        del ckpt
        torch.cuda.empty_cache()

    if accelerator.is_main_process:
        os.makedirs(cfg_ckpt["output_dir"], exist_ok=True)

    save_every = cfg_train["save_every_n_steps"]
    log_every = cfg_train.get("log_every_n_steps", 10)
    prob_2d = cfg_train.get("prob_2d", 0.3)
    cross_max = cfg_train["cross_weight"]
    adv_max = cfg_train["adv_weight"]

    cross_start = cfg_train.get("cross_start_step", 2000)
    adv_start = cfg_train.get("adv_start_step", 5000)
    cross_ramp = cfg_train.get("cross_ramp_steps", 2000)

    accelerator.print(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}")
    accelerator.print(f"Effective batch: "
                      f"{cfg_train['batch_size'] * cfg_train['gradient_accumulation_steps'] * accelerator.num_processes}")

    # Initial validation
    accelerator.print("[step 0] sanity validation...")
    val_loss = validate(accelerator, model, val_loader, loss_fn, global_step, cfg_train)
    best_val = val_loss

    iter_2d = None
    step_t0 = time.time()
    for epoch in range(start_epoch, cfg_train["max_epochs"]):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            # ── Warmup schedules ──
            # cross-modal: 0 until cross_start, then linear ramp to cross_max
            if global_step < cross_start:
                cross_w = 0.0
            else:
                ramp = min(1.0, (global_step - cross_start) / max(1, cross_ramp))
                cross_w = cross_max * ramp

            # adversarial: 0 until adv_start, then GRL alpha sigmoid ramp.
            # adv_weight is intentionally tiny (e.g. 0.001), and GRL alpha
            # additionally scales the encoder-side gradient.
            if global_step < adv_start:
                alpha = 0.0
                adv_w = 0.0
            else:
                alpha = grl_alpha_schedule(global_step, total_steps,
                                           max_alpha=1.0, start_step=adv_start)
                adv_w = adv_max
            unwrapped_model.adv_alpha = alpha

            with accelerator.accumulate(model):
                out = neuroquant_step(
                    model, loss_fn, batch,
                    cross_weight=cross_w, adv_weight=adv_w,
                    depth_mode="3d", accelerator=accelerator,
                )
                total_loss = out["loss"]

                # 2D joint training (paired slices)
                if joint_2d3d and np.random.rand() < prob_2d:
                    if iter_2d is None:
                        iter_2d = iter(train_loader_2d)
                    try:
                        batch_2d = next(iter_2d)
                    except StopIteration:
                        iter_2d = iter(train_loader_2d)
                        batch_2d = next(iter_2d)
                    out_2d = neuroquant_step(
                        model, loss_fn, batch_2d,
                        cross_weight=cross_w, adv_weight=adv_w,
                        depth_mode="2d", accelerator=accelerator,
                    )
                    total_loss = total_loss + cfg_train.get("loss_weight_2d", 0.5) * out_2d["loss"]

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    ema.update(unwrapped_model)
                global_step += 1

                step_dt = time.time() - step_t0
                lr_now = scheduler.get_last_lr()[0]
                accelerator.print(
                    f"[ep {epoch} | step {global_step}/{total_steps}] "
                    f"loss={float(total_loss):.4f}  "
                    f"recT1={float(out['rec_t1']):.4f} recT2={float(out['rec_t2']):.4f}  "
                    f"vq={float(out['vq_t1'] + out['vq_t2'])/2:.4f}  "
                    f"cross={float(out['cross']):.4f} adv={float(out['adv']):.4f}  "
                    f"perp={float(out['perplexity']):.1f} alpha={alpha:.3f}  "
                    f"lr={lr_now:.2e} ({step_dt:.2f}s)"
                )
                step_t0 = time.time()

                if accelerator.is_main_process and global_step % log_every == 0:
                    import wandb
                    wandb.log({
                        "train/loss": float(total_loss),
                        "train/rec_t1": float(out["rec_t1"]),
                        "train/rec_t2": float(out["rec_t2"]),
                        "train/vq": float(out["vq_t1"] + out["vq_t2"]) / 2,
                        "train/cross": float(out["cross"]),
                        "train/adv": float(out["adv"]),
                        "train/perplexity": float(out["perplexity"]),
                        "train/grl_alpha": alpha,
                        "train/lr": lr_now,
                        "train/epoch": epoch,
                    }, step=global_step)

                if global_step % save_every == 0:
                    val_loss = validate(accelerator, model, val_loader, loss_fn, global_step, cfg_train)
                    if accelerator.is_main_process:
                        save = {
                            "epoch": epoch, "global_step": global_step,
                            "model": unwrapped_model.state_dict(),
                            "ema": ema.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "config": config, "best_val_loss": best_val,
                        }
                        path = Path(cfg_ckpt["output_dir"]) / f"neuroquant_step{global_step:07d}.pt"
                        torch.save(save, path)
                        accelerator.print(f"  saved {path}")
                        if val_loss < best_val:
                            best_val = val_loss
                            torch.save({
                                "model": unwrapped_model.state_dict(),
                                "ema": ema.state_dict(),
                                "global_step": global_step,
                                "config": config, "val_loss": val_loss,
                            }, Path(cfg_ckpt["output_dir"]) / "neuroquant_best.pt")
                            accelerator.print(f"  new best val={val_loss:.4f}")
                    accelerator.wait_for_everyone()

        accelerator.print(f"--- epoch {epoch} done (step {global_step}) ---")

    if accelerator.is_main_process:
        torch.save({
            "global_step": global_step,
            "model": unwrapped_model.state_dict(),
            "ema": ema.state_dict(),
            "config": config,
        }, Path(cfg_ckpt["output_dir"]) / "neuroquant_final.pt")
        import wandb
        wandb.finish()

    accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="NeuroQuant/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.resume:
        config["checkpoint"]["resume"] = args.resume
    train(config)


if __name__ == "__main__":
    main()
