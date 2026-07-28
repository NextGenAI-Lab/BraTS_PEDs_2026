"""
train_mednext_run7_stage1_k3.py — MedNeXt-M, kernel=3, BraTS-PEDs 2026, Run 7 Stage 1

Purpose: train a stable kernel=3 MedNeXt as the seed for UpKern initialization
of the kernel=5 model in stage 2. Per MedNeXt's own published ablation, large
kernels (k=5) trained from scratch are indistinguishable from k=3 — UpKern is
the validated path to make k=5 actually pay off, and that matters more here
(294 cases) than on the big public benchmarks the architecture was validated
on, where overfitting risk is lower.

This stage's own snapshots are not thrown away — if any land good ET/CC/ED
individually, they're valid bonus ensemble members alongside the stage-2 k=5
snapshots (a second MedNeXt "flavor": standard-receptive-field conv vs the
wide-receptive-field k=5 net).

Config locked from planning conversation:
  Model:      MedNeXt-M, kernel=3, deep_supervision=True
  Patch:      128 x 160 x 112  (BraTS-Pediatrics MedNeXt paper's validated best for ET/WT)
  Loss:       Deep-supervision-weighted DiceCE (standard nnU-Net weighting), no class weights
  Optimizer:  AdamW + CosineAnnealing
  Augment:    Same family as run_9_swin (flips, rot90, zoom, gamma, noise, blur)
  Oversample: ED(50%) CC(30%) fg(20%), same as run_9_swin
  Snapshot:   Decoupled ET/CC/ED reference ratchets, target 30-60 snapshots total
              across both stages — not best-only, not fixed-interval.
  Data:       Raw nifti (run_2/nnunet_raw), same source Swin reads — no nnUNet
              preprocessing pipeline involved.

Labels: 0=BG 1=ET 2=NET 3=CC 4=ED
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from nnunet_mednext import create_mednext_v1
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mednext_common import (
    BraTSDataset, DeepSupervisionLoss, SnapshotState,
    append_csv_log, dice_score, load_splits, run_fast_val, run_full_val,
    save_checkpoint, setup_logging,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────
RUN_DIR     = "/workspace/BRATS/run_7"
IMAGES_DIR  = "/workspace/BRATS/run_2/nnunet_raw/Dataset001_BraTSPEDs/imagesTr"
LABELS_DIR  = "/workspace/BRATS/run_2/nnunet_raw/Dataset001_BraTSPEDs/labelsTr"
SPLITS_FILE = "/workspace/BRATS/scripts/analysis/splits_final_stratified.json"

TAG = "stage1_k3"   # checkpoints/logs go to run_7/fold_X/stage1_k3/

NUM_EPOCHS   = 200
LR           = 1e-3
                              # validated for MedNeXt specifically like the other knobs
                              # were. Watch the loss curve for the first ~10-20 epochs;
                              # if it's unstable / NaNs, drop to 3e-4 and restart.
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 8             # matches run_9_swin's proven-stable batch at a similar
                              # patch voxel count, on a model with 4x fewer params
                              # (17.5M vs Swin's 72.7M) — likely room to push higher,
                              # see the GPU memory log line after epoch 0 below.
NUM_WORKERS  = 64            # this server has 256 cores / 16GB free shm — pushed up
                              # from Swin's 16 to use more of both, per your instruction.
PREFETCH     = 4
PATCH_SIZE   = (128, 160, 112)
IN_CHANNELS  = 4
OUT_CHANNELS = 5
MODEL_ID     = 'M'
KERNEL_SIZE  = 3
DEEP_SUPERVISION = True      # set False here to fall back to a single-output net —
                              # DeepSupervisionLoss and validation both already handle
                              # that case, nothing else needs to change.

FAST_VAL_MARGIN   = -0.05   # wider than Swin's -0.03 — fast val is noisy, don't
                              # skip full val just because one crop pass dipped slightly
FULL_VAL_COOLDOWN = 3
FULL_VAL_INTERVAL = 9999

SNAPSHOT_START    = 50      # skip noisiest early epochs before tracking snapshots
# No SNAPSHOT_MEAN_GAP or SNAPSHOT_TOL — snapshot fires on any strict ET/CC/ED
# improvement, no matter how small. Pool is scanned at submission time.
# ────────────────────────────────────────────────────────────────────────────


def build_model(device):
    return create_mednext_v1(
        IN_CHANNELS, OUT_CHANNELS, MODEL_ID,
        kernel_size=KERNEL_SIZE, deep_supervision=DEEP_SUPERVISION,
    ).to(device)


def train(args):
    fold = args.fold
    log, fold_dir = setup_logging(fold, RUN_DIR, TAG)

    log.info("=" * 70)
    log.info(f"run_7_mednext | Stage 1 (kernel=3, seed for UpKern) | Fold {fold}")
    log.info(f"Epochs={NUM_EPOCHS} LR={LR} Batch={BATCH_SIZE} Patch={PATCH_SIZE}")
    log.info(f"Model={MODEL_ID} kernel={KERNEL_SIZE} deep_supervision={DEEP_SUPERVISION}")
    log.info(f"Oversample: ED(50%) CC(30%) fg(20%) — same as run_9_swin")
    log.info(f"Snapshot start: ep{SNAPSHOT_START} | fast_val_margin={FAST_VAL_MARGIN}")
    log.info("=" * 70)

    train_cases, val_cases = load_splits(SPLITS_FILE, fold)
    log.info(f"Fold {fold}: Train={len(train_cases)} Val={len(val_cases)}")

    train_ds = BraTSDataset(train_cases, IMAGES_DIR, LABELS_DIR, PATCH_SIZE, is_train=True, oversample=True)
    val_ds   = BraTSDataset(val_cases,   IMAGES_DIR, LABELS_DIR, PATCH_SIZE, is_train=True, oversample=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               persistent_workers=True, prefetch_factor=PREFETCH)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model = build_model(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    # Auto-detect deep supervision output count rather than assuming — avoids
    # a silent mismatch if the architecture's output structure ever changes.
    with torch.no_grad():
        dummy = torch.zeros(1, IN_CHANNELS, *PATCH_SIZE, device=device)
        dummy_out = model(dummy)
    num_outputs = len(dummy_out) if isinstance(dummy_out, (list, tuple)) else 1
    log.info(f"Deep supervision outputs detected: {num_outputs}")
    del dummy, dummy_out
    torch.cuda.empty_cache()

    loss_fn = DeepSupervisionLoss(num_outputs)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024)

    snap = SnapshotState()
    start_epoch = 0

    if args.resume:
        ckpt = fold_dir / "checkpoint_latest.pth"
        if ckpt.exists():
            c = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(c["model"])
            optimizer.load_state_dict(c["optimizer"])
            scheduler.load_state_dict(c["scheduler"])
            start_epoch = c["epoch"] + 1
            snap.load_state_dict(c)
            if "scaler" in c:
                scaler.load_state_dict(c["scaler"])
            log.info(f"Resumed from ep={start_epoch} | {snap.state_dict()}")

    csv_header = ["epoch", "loss", "lr",
                  "train_et", "train_net", "train_cc", "train_ed",
                  "fast_et", "fast_net", "fast_cc", "fast_ed", "fast_mean",
                  "full_done", "full_et", "full_net", "full_cc", "full_ed", "full_mean",
                  "snapshot_fired"]

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        train_dice = []

        for batch in train_loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                out = model(imgs)
                loss = loss_fn(out, lbls)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            with torch.no_grad():
                final = out[0] if isinstance(out, (list, tuple)) else out
                pred = torch.argmax(final.detach(), dim=1)
                train_dice.append(dice_score(pred.squeeze(), lbls.squeeze()))

        if epoch == start_epoch:
            log.info(
                f"  GPU mem after first training epoch: "
                f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB "
                f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
                f"max_allocated={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
                f"(40GB total — if this is well under ~30GB, BATCH_SIZE can likely go higher)"
            )

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        lr_now = scheduler.get_last_lr()[0]
        td = np.nanmean(train_dice, axis=0)

        fd = run_fast_val(model, val_loader, device)
        fmean = float(np.mean(fd))
        fet, fnet, fcc, fed = fd

        log.info(
            f"Epoch {epoch:04d} | loss={avg_loss:.4f} | LR={lr_now:.6f} | "
            f"Train ET={td[0]:.3f} NET={td[1]:.3f} CC={td[2]:.3f} ED={td[3]:.3f} | "
            f"FastVal ET={fet:.3f} NET={fnet:.3f} CC={fcc:.3f} ED={fed:.3f} "
            f"Mean={fmean:.3f} [best={snap.fast_best:.3f}]"
        )

        epochs_since_full = epoch - snap.last_full_val_epoch
        improved_trigger = (fmean > snap.fast_best + FAST_VAL_MARGIN) and (epochs_since_full >= FULL_VAL_COOLDOWN)
        periodic_trigger = epochs_since_full >= FULL_VAL_INTERVAL

        full_done, full_et, full_net, full_cc, full_ed, full_mean, snapshot_fired = False, "", "", "", "", "", False

        if improved_trigger or periodic_trigger:
            if improved_trigger:
                log.info("  >> FastVal improved -> FullVal")
            else:
                log.info(f"  >> Periodic FullVal ({epochs_since_full} epochs since last)")

            snap.last_full_val_epoch = epoch
            mpc = run_full_val(model, val_cases, IMAGES_DIR, LABELS_DIR, PATCH_SIZE, device)
            full_mean_v = float(np.mean(mpc))
            et, net, cc, ed = mpc
            full_done, full_et, full_net, full_cc, full_ed, full_mean = True, et, net, cc, ed, full_mean_v

            log.info(
                f"  >> FULL VAL | ET={et:.4f} NET={net:.4f} CC={cc:.4f} ED={ed:.4f} "
                f"Mean={full_mean_v:.4f} [best={snap.full_best:.4f}]"
            )

            if full_mean_v > snap.full_best:
                snap.full_best = full_mean_v
                snap.fast_best = max(snap.fast_best, fmean)
                save_checkpoint(
                    fold_dir / "checkpoint_best.pth", epoch, model, optimizer, scheduler,
                    {**snap.state_dict(), "full_mean": full_mean_v, "per_class": mpc.tolist()},
                )
                log.info(f"  >> NEW BEST | Mean={full_mean_v:.4f} ET={et:.4f} CC={cc:.4f} ED={ed:.4f}")
            else:
                log.info(f"  >> No improvement ({full_mean_v:.4f} <= full_best={snap.full_best:.4f})")

            # snapshot check moved out of full val block — runs every epoch via fast val

        # ── per-epoch snapshot check (fast val proxy, no full val cost) ──────────
        if epoch >= SNAPSHOT_START:
            fired, reasons = snap.check_snapshot(fet, fcc, fed, 0.0)
            if fired:
                snapshot_fired = True
                # filename uses fast val per-class for ET/CC/ED (what triggered it)
                # and full_mean_v if a full val happened this epoch, else fast mean
                mean_tag = f"{full_mean_v:.4f}" if full_done else f"fv{fmean:.4f}"
                fname = (
                    f"checkpoint_snap{snap.snapshot_count:03d}_ep{epoch:04d}"
                    f"_mean{mean_tag}_et{fet:.4f}_cc{fcc:.4f}_ed{fed:.4f}.pth"
                )
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "fast_et": fet, "fast_cc": fcc, "fast_ed": fed,
                    "fast_mean": fmean,
                    "kernel_size": KERNEL_SIZE,
                }, fold_dir / fname)
                log.info(f"  >> SNAPSHOT #{snap.snapshot_count}: {fname}")
                log.info(f"     Reason: {' | '.join(reasons)}")

        # ── per-epoch snapshot check (fast val proxy, no full val cost) ──────────
        if epoch >= SNAPSHOT_START:
            fired, reasons = snap.check_snapshot(fet, fcc, fed, 0.0)
            if fired:
                snapshot_fired = True
                # filename uses fast val per-class for ET/CC/ED (what triggered it)
                # and full_mean_v if a full val happened this epoch, else fast mean
                mean_tag = f"{full_mean_v:.4f}" if full_done else f"fv{fmean:.4f}"
                fname = (
                    f"checkpoint_snap{snap.snapshot_count:03d}_ep{epoch:04d}"
                    f"_mean{mean_tag}_et{fet:.4f}_cc{fcc:.4f}_ed{fed:.4f}.pth"
                )
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "fast_et": fet, "fast_cc": fcc, "fast_ed": fed,
                    "fast_mean": fmean,
                    "kernel_size": KERNEL_SIZE,
                }, fold_dir / fname)
                log.info(f"  >> SNAPSHOT #{snap.snapshot_count}: {fname}")
                log.info(f"     Reason: {' | '.join(reasons)}")

        append_csv_log(fold_dir, TAG, [
            epoch, f"{avg_loss:.4f}", f"{lr_now:.6f}",
            f"{td[0]:.4f}", f"{td[1]:.4f}", f"{td[2]:.4f}", f"{td[3]:.4f}",
            f"{fet:.4f}", f"{fnet:.4f}", f"{fcc:.4f}", f"{fed:.4f}", f"{fmean:.4f}",
            int(full_done), full_et, full_net, full_cc, full_ed, full_mean,
            int(snapshot_fired),
        ], csv_header)

        save_checkpoint(
            fold_dir / "checkpoint_latest.pth", epoch, model, optimizer, scheduler,
            {**snap.state_dict(), "scaler": scaler.state_dict()},
        )

    log.info("=" * 70)
    log.info(f"Stage 1 complete | FullBest={snap.full_best:.4f} Snapshots={snap.snapshot_count}")
    log.info(f"Seed checkpoint for stage 2 UpKern: {fold_dir / 'checkpoint_best.pth'}")
    log.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
