import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────
# PSNR
# ──────────────────────────────────────────────────────

def psnr(recon: torch.Tensor, target: torch.Tensor,
         data_range: float = 2.0, bg_threshold: float = -0.9) -> dict:
    
    mse_all = F.mse_loss(recon, target, reduction="none").mean(dim=[1, 2, 3, 4])
    psnr_all = 10.0 * torch.log10(data_range ** 2 / (mse_all + 1e-10))

    fg_mask = (target > bg_threshold).float()
    diff_sq = (recon - target) ** 2
    fg_count = fg_mask.sum(dim=[1, 2, 3, 4]).clamp(min=1.0)
    mse_fg = (diff_sq * fg_mask).sum(dim=[1, 2, 3, 4]) / fg_count
    psnr_fg = 10.0 * torch.log10(data_range ** 2 / (mse_fg + 1e-10))

    return {
        "psnr": psnr_all.mean(),
        "psnr_fg": psnr_fg.mean(),
    }

def _gaussian_kernel_3d(kernel_size: int = 11, sigma: float = 1.5,
                        device: torch.device = None) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = g / g.sum()
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    return kernel_3d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K, K)


def ssim(
    recon: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,
    k1: float = 0.01,
    k2: float = 0.03,
    bg_threshold: float = -0.9,
) -> dict:
    """Compute SSIM: whole-volume and foreground-only."""
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    kernel = _gaussian_kernel_3d(kernel_size, sigma, device=recon.device)
    pad = kernel_size // 2

    mu_x = F.conv3d(recon, kernel, padding=pad)
    mu_y = F.conv3d(target, kernel, padding=pad)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv3d(recon ** 2, kernel, padding=pad) - mu_x_sq
    sigma_y_sq = F.conv3d(target ** 2, kernel, padding=pad) - mu_y_sq
    sigma_xy = F.conv3d(recon * target, kernel, padding=pad) - mu_xy

    sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
    sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / (denominator + 1e-10)

    ssim_all = ssim_map.mean(dim=[1, 2, 3, 4]).mean()

    fg_mask = (target > bg_threshold).float()
    fg_count = fg_mask.sum(dim=[1, 2, 3, 4]).clamp(min=1.0)
    ssim_fg = (ssim_map * fg_mask).sum(dim=[1, 2, 3, 4]) / fg_count
    ssim_fg = ssim_fg.mean()

    return {
        "ssim": ssim_all,
        "ssim_fg": ssim_fg,
    }
