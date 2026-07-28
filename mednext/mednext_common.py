"""
mednext_common.py — Shared utilities for MedNeXt run_7 (BraTS-PEDs 2026)

Used by both:
  train_mednext_run7_stage1_k3.py         (seed phase, kernel=3)
  train_mednext_run7_stage2_k5_upkern.py  (main phase, kernel=5 via UpKern)

Mirrors train_swinunetr_run9.py's structure (same augmentation recipe, same
dataset oversampling, same fast/full-val gate, same raw-nifti data source —
NOT nnUNet's preprocessed .npy, MedNeXt reads raw nifti directly just like
Swin does) so the three-model ensemble (nnUNet + SwinUNETR + MedNeXt) stays
directly comparable.

Labels: 0=BG 1=ET 2=NET 3=CC 4=ED
nnUNet imagesTr naming: {case}_0000.nii.gz to {case}_0003.nii.gz
"""

import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from torch.utils.data import Dataset


# ── LOGGING ──────────────────────────────────────────────────────────────────

def setup_logging(fold, run_dir, tag):
    """
    fold_dir is run_dir/fold_X/tag — each stage gets its own subfolder so
    stage 2 can never overwrite stage 1's checkpoint_best.pth (which stage 2
    needs to read for UpKern initialization).
    """
    fold_dir = Path(run_dir) / f"fold_{fold}" / tag
    fold_dir.mkdir(parents=True, exist_ok=True)
    log_file = fold_dir / f"train_{tag}_fold{fold}.log"
    root = logging.getLogger()
    root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__), fold_dir


def append_csv_log(fold_dir, tag, row, header):
    """
    Append one epoch's metrics as a CSV row (header written once on first call).
    This is in addition to the per-epoch text log line — gives an easy-to-parse
    record of every single epoch's metrics, not just the ones that produced a
    full val / snapshot, which was the first reason you wanted the original
    logging in place.
    """
    csv_path = Path(fold_dir) / f"epoch_metrics_{tag}.csv"
    is_new = not csv_path.exists()
    with open(csv_path, "a") as f:
        if is_new:
            f.write(",".join(header) + "\n")
        f.write(",".join(str(v) for v in row) + "\n")


# ── AUGMENTATION (same recipe as run_9_swin) ─────────────────────────────────

def augment(image, label, patch_size):
    """
    Strong augmentation for MRI multi-modal data.
    image: (4, H, W, D) float32
    label: (H, W, D) float32/int
    patch_size: (ph, pw, pd) — used to re-crop after zoom

    Applied: flips, 90-deg rotations, zoom, gamma, intensity scale/shift,
    gaussian noise, gaussian blur. Identical to run_9_swin for ensemble
    consistency (same augmentation family across all three architectures).
    """
    for axis in range(3):
        if np.random.random() > 0.5:
            image = np.flip(image, axis=axis + 1).copy()
            label = np.flip(label, axis=axis).copy()

    if np.random.random() < 0.3:
        k = np.random.randint(1, 4)
        image = np.rot90(image, k=k, axes=(1, 2)).copy()
        label = np.rot90(label, k=k, axes=(0, 1)).copy()

    if np.random.random() < 0.3:
        k = np.random.randint(1, 4)
        image = np.rot90(image, k=k, axes=(1, 3)).copy()
        label = np.rot90(label, k=k, axes=(0, 2)).copy()

    if np.random.random() < 0.3:
        scale = np.random.uniform(0.85, 1.15)
        C, H, W, D = image.shape
        nH, nW, nD = int(round(H * scale)), int(round(W * scale)), int(round(D * scale))

        def adjust(arr, target, axis):
            current = arr.shape[axis]
            if current > target:
                start = (current - target) // 2
                slices = [slice(None)] * arr.ndim
                slices[axis] = slice(start, start + target)
                return arr[tuple(slices)]
            elif current < target:
                pad = [(0, 0)] * arr.ndim
                pad[axis] = (0, target - current)
                return np.pad(arr, pad)
            return arr

        # map new-grid indices back to old-grid (correct nearest-neighbour zoom)
        hi = np.clip((np.arange(nH) * H / nH).astype(int), 0, H - 1)
        wi = np.clip((np.arange(nW) * W / nW).astype(int), 0, W - 1)
        di = np.clip((np.arange(nD) * D / nD).astype(int), 0, D - 1)
        image = image[:, hi[:, None, None], wi[None, :, None], di[None, None, :]]
        label = label[hi[:, None, None], wi[None, :, None], di[None, None, :]]

        ph, pw, pd = patch_size
        for ax, target in enumerate([ph, pw, pd]):
            image = adjust(image, target, ax + 1)
            label = adjust(label, target, ax)

    for c in range(image.shape[0]):
        if np.random.random() < 0.3:
            gamma = np.random.uniform(0.7, 1.5)
            ch = image[c]
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max > ch_min:
                ch_norm = (ch - ch_min) / (ch_max - ch_min + 1e-8)
                image[c] = np.power(ch_norm, gamma) * (ch_max - ch_min) + ch_min

    for c in range(image.shape[0]):
        if np.random.random() < 0.5:
            image[c] *= np.random.uniform(0.9, 1.1)
        if np.random.random() < 0.5:
            image[c] += np.random.uniform(-0.1, 0.1)

    for c in range(image.shape[0]):
        if np.random.random() < 0.2:
            std = np.random.uniform(0.0, 0.1)
            image[c] += np.random.normal(0, std, image[c].shape).astype(np.float32)

    if np.random.random() < 0.15:
        sigma = np.random.uniform(0.5, 1.0)

        def gaussian_kernel_1d(sigma, radius=2):
            x = np.arange(-radius, radius + 1)
            k = np.exp(-0.5 * (x / sigma) ** 2)
            return (k / k.sum()).astype(np.float32)

        k = gaussian_kernel_1d(sigma)
        r = len(k) // 2
        for c in range(image.shape[0]):
            ch = np.pad(image[c], ((r, r), (0, 0), (0, 0)), mode='reflect')
            ch = sum(ch[i:i + image.shape[1]] * k[i] for i in range(len(k)))
            ch = np.pad(ch, ((0, 0), (r, r), (0, 0)), mode='reflect')
            ch = sum(ch[:, i:i + image.shape[2]] * k[i] for i in range(len(k)))
            ch = np.pad(ch, ((0, 0), (0, 0), (r, r)), mode='reflect')
            ch = sum(ch[:, :, i:i + image.shape[3]] * k[i] for i in range(len(k)))
            image[c] = ch.astype(np.float32)

    # Final guarantee: force output to exactly patch_size regardless of which
    # augmentations fired. rot90 can swap axes (e.g. H=128 <-> D=112) and if
    # zoom didn't trigger (30% chance), the swapped shape reaches the collator
    # and torch.stack fails. This crop/pad is the single source of truth.
    ph, pw, pd = patch_size
    _, H, W, D = image.shape

    def crop_or_pad(arr, target, axis):
        current = arr.shape[axis]
        if current > target:
            start = (current - target) // 2
            slices = [slice(None)] * arr.ndim
            slices[axis] = slice(start, start + target)
            return arr[tuple(slices)]
        elif current < target:
            pad = [(0, 0)] * arr.ndim
            pad[axis] = (0, target - current)
            return np.pad(arr, pad)
        return arr

    image = crop_or_pad(image, ph, 1)
    image = crop_or_pad(image, pw, 2)
    image = crop_or_pad(image, pd, 3)
    label = crop_or_pad(label, ph, 0)
    label = crop_or_pad(label, pw, 1)
    label = crop_or_pad(label, pd, 2)

    return image, label


# ── DATASET (same oversampling as run_9_swin) ────────────────────────────────

class BraTSDataset(Dataset):
    """
    Oversampling (patch-level, training only):
      67% of patches forced to foreground.
      Within foreground: 50% center on ED (label=4), next 30% on CC (label=3),
      remaining 20% any foreground voxel. No explicit ET oversampling.
    Kept identical to run_9_swin for ensemble consistency.
    """

    def __init__(self, cases, images_dir, labels_dir, patch_size, is_train=True, oversample=True):
        self.cases = cases
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.patch_size = patch_size
        self.is_train = is_train
        self.oversample = oversample

    def __len__(self):
        return len(self.cases)

    def _normalize(self, image):
        out = np.zeros_like(image)
        for c in range(image.shape[0]):
            ch = image[c]
            mask = ch > 0
            if mask.sum() > 0:
                out[c] = (ch - ch[mask].mean()) / (ch[mask].std() + 1e-8)
        return out

    def _crop(self, image, label):
        C, H, W, D = image.shape
        ph, pw, pd = self.patch_size

        if H < ph or W < pw or D < pd:
            image = np.pad(image, [(0, 0), (0, max(0, ph - H)),
                                    (0, max(0, pw - W)), (0, max(0, pd - D))])
            label = np.pad(label, [(0, max(0, ph - H)),
                                    (0, max(0, pw - W)), (0, max(0, pd - D))])
            _, H, W, D = image.shape

        use_fg = self.oversample and np.random.random() < 0.67
        chosen = None

        if use_fg:
            ed_vx = np.argwhere(label == 4)
            cc_vx = np.argwhere(label == 3)
            fg_vx = np.argwhere(label > 0)
            r = np.random.random()
            if len(ed_vx) > 0 and r < 0.50:
                chosen = ed_vx[np.random.randint(len(ed_vx))]
            elif len(cc_vx) > 0 and r < 0.80:
                chosen = cc_vx[np.random.randint(len(cc_vx))]
            elif len(fg_vx) > 0:
                chosen = fg_vx[np.random.randint(len(fg_vx))]

        if chosen is not None:
            h = int(np.clip(chosen[0] - ph // 2, 0, H - ph))
            w = int(np.clip(chosen[1] - pw // 2, 0, W - pw))
            d = int(np.clip(chosen[2] - pd // 2, 0, D - pd))
        else:
            h = np.random.randint(0, max(1, H - ph + 1))
            w = np.random.randint(0, max(1, W - pw + 1))
            d = np.random.randint(0, max(1, D - pd + 1))

        return image[:, h:h+ph, w:w+pw, d:d+pd], label[h:h+ph, w:w+pw, d:d+pd]

    def __getitem__(self, idx):
        case = self.cases[idx]
        mods = [
            nib.load(self.images_dir / f"{case}_000{i}.nii.gz").get_fdata(dtype=np.float32)
            for i in range(4)
        ]
        image = self._normalize(np.stack(mods, axis=0))
        label = nib.load(self.labels_dir / f"{case}.nii.gz").get_fdata(dtype=np.float32)

        if self.is_train:
            image, label = self._crop(image, label)
            image, label = augment(image, label, self.patch_size)

        return {
            "image": torch.from_numpy(image.copy()).float(),
            "label": torch.from_numpy(label.copy()).long(),
        }


# ── METRICS ──────────────────────────────────────────────────────────────────

def dice_score(pred, target, num_classes=5):
    """Per-class Dice [ET, NET, CC, ED]. Returns 1.0 when class absent in both."""
    scores = []
    for c in range(1, num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        denom = p.sum() + t.sum()
        scores.append((2 * inter / (denom + 1e-6)).item() if denom > 0 else 1.0)
    return scores


# ── DEEP SUPERVISION LOSS (standard nnU-Net weighting) ───────────────────────

class DeepSupervisionLoss(nn.Module):
    """
    Standard nnU-Net-style deep supervision weighting:
      weight_i = 1 / 2^i   for i = 0 (full res) ... N-1 (lowest res)
      lowest-resolution output's weight zeroed out, then renormalized to sum 1.
    This is the well-documented default scheme nnU-Net itself uses, not a
    custom invention — chosen because you asked to use what's standard.

    Each auxiliary output's loss term downsamples the label to that output's
    spatial shape via nearest-neighbour interpolation (label is categorical —
    must NOT use linear/trilinear, which would invent fractional class ids).

    Works identically with deep_supervision=False: pass num_outputs=1 and it
    reduces to plain DiceCE on a single tensor — so DS can be switched off by
    flipping the DEEP_SUPERVISION flag in the training script without touching
    this class.
    """

    def __init__(self, num_outputs):
        super().__init__()
        self.base_loss = DiceFocalLoss(to_onehot_y=True, softmax=True, gamma=2.0)
        w = np.array([1.0 / (2 ** i) for i in range(num_outputs)])
        if num_outputs > 1:
            w[-1] = 0.0
        w = w / w.sum()
        self.weights = w.tolist()

    def forward(self, outputs, target):
        """
        outputs: list of tensors (B, C, h_i, w_i, d_i), index 0 = full res
                 (or a single tensor if deep supervision is off)
        target:  (B, H, W, D) long label at full res
        """
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        total = 0.0
        t_full = target.unsqueeze(1).float()
        for w, out in zip(self.weights, outputs):
            if w == 0.0:
                continue
            if out.shape[2:] != target.shape[1:]:
                t_ds = F.interpolate(t_full, size=out.shape[2:], mode='nearest').long()
            else:
                t_ds = t_full.long()
            total = total + w * self.base_loss(out, t_ds)
        return total


# ── SNAPSHOT STATE (decoupled multi-metric, redesigned for density) ──────────

class SnapshotState:
    """
    Redesigned vs run_9_swin's logic. The problem you flagged: et_ref/cc_ref
    used to reset to whatever ET/CC happened to be at the moment a new
    full_best was found — so one early ET spike (e.g. ET=0.78 at the same
    epoch full_mean improved) permanently raised the bar for every future
    snapshot, and nothing else ever qualified.

    Fix: et_ref / cc_ref / ed_ref now ratchet up ONLY when their own metric
    actually drives a snapshot — they are no longer reset by full_best
    improving. That decouples the three trigger channels from each other,
    so a ceiling on one axis doesn't block the other two, and a small `tol`
    means near-ties still count instead of requiring strict float improvement.
    ED is now a first-class trigger too (nnUNet structurally can't detect ED
    well; this architecture might, so it shouldn't be excluded the way it was
    for Swin).

    Target: 30-60 snapshots across both stages combined — event-driven on
    "did this epoch actually learn something good", not a fixed interval.
    """

    def __init__(self):
        self.fast_best = -1.0
        self.full_best = -1.0
        self.et_ref = -1.0
        self.cc_ref = -1.0
        self.ed_ref = -1.0
        self.snapshot_count = 0
        self.last_full_val_epoch = -9999
        # No mean_ok gate on snapshots — CV scores don't translate linearly to
        # leaderboard scores so we want the widest net possible. Any strict
        # improvement on ET, CC, or ED (even by 0.0001) fires a snapshot.
        # The pool is scanned empirically at submission time.

    def state_dict(self):
        return {
            "fast_best": self.fast_best, "full_best": self.full_best,
            "et_ref": self.et_ref, "cc_ref": self.cc_ref, "ed_ref": self.ed_ref,
            "snapshot_count": self.snapshot_count,
            "last_full_val_epoch": self.last_full_val_epoch,
        }

    def load_state_dict(self, d):
        self.fast_best = d.get("fast_best", -1.0)
        self.full_best = d.get("full_best", -1.0)
        self.et_ref = d.get("et_ref", -1.0)
        self.cc_ref = d.get("cc_ref", -1.0)
        self.ed_ref = d.get("ed_ref", -1.0)
        self.snapshot_count = d.get("snapshot_count", 0)
        self.last_full_val_epoch = d.get("last_full_val_epoch", -9999)

    def check_snapshot(self, et, cc, ed, full_mean):
        """
        Returns (should_snapshot: bool, reasons: list[str]).
        Any strict improvement on ET, CC, or ED fires a snapshot — even 0.0001.
        No mean_ok gate: CV scores don't translate linearly to leaderboard so
        we collect as many checkpoints as possible and scan them empirically.
        Each ref ratchets up only when its own axis drives a snapshot.
        """
        et_better = et > self.et_ref
        cc_better = cc > self.cc_ref
        ed_better = ed > self.ed_ref

        if et_better or cc_better or ed_better:
            reasons = []
            if et_better:
                reasons.append(f"ET={et:.4f}(ref={self.et_ref:.4f})")
                self.et_ref = et
            if cc_better:
                reasons.append(f"CC={cc:.4f}(ref={self.cc_ref:.4f})")
                self.cc_ref = cc
            if ed_better:
                reasons.append(f"ED={ed:.4f}(ref={self.ed_ref:.4f})")
                self.ed_ref = ed
            self.snapshot_count += 1
            return True, reasons
        return False, []


# ── VALIDATION ───────────────────────────────────────────────────────────────

def _final_head(out):
    """
    Deep supervision returns a list; index 0 is the full-resolution head
    (confirmed on this server: output shapes were 128x160x112 down to
    8x10x7, in that order). Validation/inference always score against this
    head only, so Dice numbers stay comparable to nnUNet/SwinUNETR, which
    only ever produce one output.
    """
    return out[0] if isinstance(out, (list, tuple)) else out


def run_fast_val(model, val_loader, device, num_passes=1):
    model.eval()
    pass_means = []
    with torch.no_grad():
        for _ in range(num_passes):
            all_dice = []
            for batch in val_loader:
                imgs = batch["image"].to(device)
                lbls = batch["label"].to(device).squeeze()
                out = _final_head(model(imgs))
                pred = torch.argmax(out, dim=1).squeeze()
                all_dice.append(dice_score(pred, lbls))
            pass_means.append(np.mean(all_dice, axis=0))
    model.train()
    return np.mean(pass_means, axis=0)


def run_full_val(model, val_cases, images_dir, labels_dir, patch_size, device):
    model.eval()
    all_dice = []

    def predictor(x):
        return _final_head(model(x))

    with torch.no_grad():
        for case in val_cases:
            mods = [
                nib.load(Path(images_dir) / f"{case}_000{i}.nii.gz").get_fdata(dtype=np.float32)
                for i in range(4)
            ]
            image = np.stack(mods, axis=0)
            for c in range(4):
                ch = image[c]
                mask = ch > 0
                if mask.sum() > 0:
                    image[c] = (ch - ch[mask].mean()) / (ch[mask].std() + 1e-8)
            label = nib.load(Path(labels_dir) / f"{case}.nii.gz").get_fdata(dtype=np.float32)
            imgs = torch.from_numpy(image).float().unsqueeze(0).to(device)
            lbls = torch.from_numpy(label).long().to(device)
            out = sliding_window_inference(imgs, patch_size, 4, predictor, overlap=0.25)
            pred = torch.argmax(out, dim=1).squeeze()
            all_dice.append(dice_score(pred, lbls))
    model.train()
    return np.mean(all_dice, axis=0)


# ── CHECKPOINT / SPLITS ────────────────────────────────────────────────────

def save_checkpoint(path, epoch, model, optimizer, scheduler, meta):
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        **meta,
    }, path)


def load_splits(splits_file, fold):
    splits = json.load(open(splits_file))
    return splits[fold]["train"], splits[fold]["val"]
