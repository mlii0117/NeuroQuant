from __future__ import annotations

import glob
from collections import defaultdict
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset-specific glob patterns for finding T1w/T2w warped files
DATASET_PATTERNS = {
    "ABCD": {
        "T1w": "ABCD/processed/Structural/registration/Unzip/*/ses-*/anat/T1w_mni_warped_n4.nii.gz",
        "T2w": "ABCD/processed/Structural/registration/Unzip/*/ses-*/anat/T2w_mni_warped_n4.nii.gz",
    },
    "NCANDA": {
        "T1w": "NCANDA/NCANDA_S*/*/t1_brain.nii.gz",
        "T2w": "NCANDA/NCANDA_S*/*/t2_brain.nii.gz",
    }
}

# Default server data root
DEFAULT_DATA_ROOT = "/users/lmj695/pm"


def discover_files(data_root: str, datasets: Optional[list] = None) -> list:
    """Discover all valid NIfTI files across datasets."""
    data_root = Path(data_root)
    if datasets is None:
        datasets = list(DATASET_PATTERNS.keys())

    samples = []
    for ds_name in datasets:
        if ds_name not in DATASET_PATTERNS:
            print(f"Warning: unknown dataset '{ds_name}', skipping.")
            continue
        for modality, pattern in DATASET_PATTERNS[ds_name].items():
            full_pattern = str(data_root / pattern)
            files = sorted(glob.glob(full_pattern))
            for f in files:
                samples.append({"path": f, "modality": modality, "dataset": ds_name})
    return samples


def center_crop_3d(vol: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resize a 3D volume to target_shape via center crop / pad.

    - Axes larger than target: center crop
    - Axes smaller than target: pad with the volume's minimum value (centered)
    """
    pad_val = float(vol.min())
    result = np.full(target_shape, pad_val, dtype=vol.dtype)
    slices_src = []
    slices_dst = []
    for s, t in zip(vol.shape, target_shape):
        if s >= t:
            start = (s - t) // 2
            slices_src.append(slice(start, start + t))
            slices_dst.append(slice(0, t))
        else:
            offset = (t - s) // 2
            slices_src.append(slice(0, s))
            slices_dst.append(slice(offset, offset + s))
    result[slices_dst[0], slices_dst[1], slices_dst[2]] = vol[slices_src[0], slices_src[1], slices_src[2]]
    return result


def normalize_brain_volume(vol: np.ndarray, lower_pct: float = 0.5, upper_pct: float = 99.5) -> np.ndarray:
    brain_mask = vol > 0
    if brain_mask.sum() < 100:
        vmin, vmax = vol.min(), vol.max()
        if vmax - vmin < 1e-8:
            return np.zeros_like(vol)
        return 2.0 * (vol - vmin) / (vmax - vmin) - 1.0

    brain_voxels = vol[brain_mask]
    lower = np.percentile(brain_voxels, lower_pct)
    upper = np.percentile(brain_voxels, upper_pct)

    vol = np.clip(vol, lower, upper)
    vol = 2.0 * (vol - lower) / (upper - lower + 1e-8) - 1.0
    return vol


MODALITY_MAP = {"T1w": 0, "T2w": 1}


def _build_pairs(samples: list) -> list:
    by_parent = defaultdict(dict)  # parent -> {modality: path}
    for s in samples:
        parent = str(Path(s["path"]).parent)
        by_parent[parent][s["modality"]] = (s["path"], s["dataset"])

    pairs = []
    for parent, mods in by_parent.items():
        if "T1w" in mods and "T2w" in mods:
            t1_path, ds = mods["T1w"]
            t2_path, _ = mods["T2w"]
            pairs.append({
                "subject": parent,
                "dataset": ds,
                "T1w": t1_path,
                "T2w": t2_path,
            })
    return pairs


class PairedBrainMRI3DDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        datasets: Optional[list] = None,
        target_shape: tuple = (192, 224, 192),
        mode: str = "3d",
        slices_per_volume: int = 8,
        split: str = "train",
        val_per_dataset: int = 5,
        seed: int = 42,
        lower_pct: float = 0.5,
        upper_pct: float = 99.5,
    ):
        super().__init__()
        self.target_shape = target_shape
        self.mode = mode
        self.slices_per_volume = slices_per_volume
        self.split = split
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

        all_samples = discover_files(data_root, datasets)
        if len(all_samples) == 0:
            raise RuntimeError(f"No files found under {data_root}")

        all_pairs = _build_pairs(all_samples)
        if len(all_pairs) == 0:
            raise RuntimeError(
                "No T1w/T2w paired subjects found. "
                "NeuroQuant requires paired data for the cross-modal loss."
            )

        # Per-dataset deterministic split
        rng = np.random.RandomState(seed)
        by_ds = defaultdict(list)
        for i, p in enumerate(all_pairs):
            by_ds[p["dataset"]].append(i)

        val_idx = set()
        for ds, idxs in by_ds.items():
            perm = rng.permutation(idxs)
            n_val = min(val_per_dataset, len(perm))
            val_idx.update(perm[:n_val].tolist())

        if split == "val":
            self.pairs = [all_pairs[i] for i in sorted(val_idx)]
        else:
            self.pairs = [all_pairs[i] for i in range(len(all_pairs)) if i not in val_idx]

        print(f"\n[PAIRED {split.upper()}/{mode}] {len(self.pairs)} subjects with both T1w+T2w")
        ds_count = defaultdict(int)
        for p in self.pairs:
            ds_count[p["dataset"]] += 1
        for ds, c in sorted(ds_count.items()):
            print(f"  {ds}: {c}")

    def __len__(self):
        if self.mode == "2d":
            return len(self.pairs) * self.slices_per_volume
        return len(self.pairs)

    def _load(self, path: str) -> np.ndarray:
        vol = nib.load(path).get_fdata(dtype=np.float32)
        vol = center_crop_3d(vol, self.target_shape)
        vol = normalize_brain_volume(vol, self.lower_pct, self.upper_pct)
        return vol

    def __getitem__(self, idx):
        pair_idx = idx // self.slices_per_volume if self.mode == "2d" else idx
        info = self.pairs[pair_idx]

        t1 = self._load(info["T1w"])
        t2 = self._load(info["T2w"])

        if self.mode == "2d":
            d = t1.shape[0]
            margin = int(d * 0.1)
            slice_idx = np.random.randint(margin, d - margin)
            t1_s = torch.from_numpy(t1[slice_idx]).unsqueeze(0).unsqueeze(0)
            t2_s = torch.from_numpy(t2[slice_idx]).unsqueeze(0).unsqueeze(0)
            return {
                "T1w": t1_s,
                "T2w": t2_s,
                "modality_T1w": MODALITY_MAP["T1w"],
                "modality_T2w": MODALITY_MAP["T2w"],
                "dataset": info["dataset"],
                "depth_mode": "2d",
            }

        t1_t = torch.from_numpy(t1).unsqueeze(0)  # (1, D, H, W)
        t2_t = torch.from_numpy(t2).unsqueeze(0)
        return {
            "T1w": t1_t,
            "T2w": t2_t,
            "modality_T1w": MODALITY_MAP["T1w"],
            "modality_T2w": MODALITY_MAP["T2w"],
            "dataset": info["dataset"],
            "depth_mode": "3d",
        }
