import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from nnunet_mednext import create_mednext_v1
from nnunet_mednext.run.load_weights import upkern_load_weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from mednext_common import (
    BraTSDataset, DeepSupervisionLoss, SnapshotState,
    append_csv_log, dice_score, load_splits, run_fast_val, run_full_val,
    save_checkpoint, setup_logging,
)

# ── CONFIG ───────────────────────────────────────────────────────────────────

TAG = "stage2_k5_upkern"

NUM_EPOCHS   = 200
LR           = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 8
NUM_WORKERS  = 64
PREFETCH     = 4
PATCH_SIZE   = (128, 160, 112)
IN_CHANNELS  = 4
OUT_CHANNELS = 5
MODEL_ID     = 'M'
KERNEL_SIZE  = 5
DEEP_SUPERVISION = True

SNAP_MARGIN       = 0.05   # snapshot ET/CC gate: fmean > fast_best - 0.05
FULL_VAL_MARGIN   = 0.02   # full val gate: fmean > fast_best - 0.02 (tighter)
FULL_VAL_COOLDOWN = 3      # min epochs between full vals
SNAPSHOT_START    = 0      # warm UpKern start, every epoch is a candidate
# ─────────────────────────────────────────────────────────────────────────────


def build_k3_model(device):
    return create_mednext_v1(
        IN_CHANNELS, OUT_CHANNELS, MODEL_ID,
        kernel_size=3, deep_supervision=DEEP_SUPERVISION,
    ).to(device)


def build_k5_model(device):
    return create_mednext_v1(
        IN_CHANNELS, OUT_CHANNELS, MODEL_ID,
        kernel_size=KERNEL_SIZE, deep_supervision=DEEP_SUPERVISION,
    ).to(device)


def init_from_upkern(fold, run_dir, device, log):
    stage1_dir = Path(run_dir) / f"fold_{fold}" / "stage1_k3"
    seed_path  = stage1_dir / "checkpoint_best.pth"
    if not seed_path.exists():
        raise FileNotFoundError(f"Stage-1 seed not found: {seed_path}")
    log.info(f"Loading stage-1 seed: {seed_path}")
    ckpt = torch.load(seed_path, map_location=device, weights_only=False)
    k3 = build_k3_model(device)
    k3.load_state_dict(ckpt["model"], strict=False)
    log.info("Running UpKern k=3 -> k=5 ...")
    k5 = build_k5_model(device)
    upkern_load_weights(k5, k3)
    del k3
    torch.cuda.empty_cache()
    log.info("UpKern complete.")
    return k5


def train(args):
    fold = args.fold
    log, fold_dir = setup_logging(fold, args.run_dir, TAG)

    log.info("=" * 70)
    log.info(f"run_7_mednext | Stage 2 (kernel=5 UpKern) | Fold {fold}")
    log.info(f"Epochs={NUM_EPOCHS} LR={LR} Batch={BATCH_SIZE} Patch={PATCH_SIZE}")
    log.info(f"SNAP_MARGIN={SNAP_MARGIN} FULL_VAL_MARGIN={FULL_VAL_MARGIN} cooldown={FULL_VAL_COOLDOWN}")
    log.info(f"ED is gating-free — fires on fed>ed_ref regardless of mean")
    log.info("=" * 70)

    train_cases, val_cases = load_splits(args.splits_file, fold)
    log.info(f"Fold {fold}: Train={len(train_cases)} Val={len(val_cases)}")

    train_ds = BraTSDataset(train_cases, args.images_dir, args.labels_dir, PATCH_SIZE,
                             is_train=True, oversample=True)
    val_ds   = BraTSDataset(val_cases,   args.images_dir, args.labels_dir, PATCH_SIZE,
                             is_train=True, oversample=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True,
                               persistent_workers=True, prefetch_factor=PREFETCH)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark       = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    snap        = SnapshotState()
    start_epoch = 0

    if args.resume:
        ckpt_path = fold_dir / "checkpoint_latest.pth"
        if ckpt_path.exists():
            log.info(f"Resuming from {ckpt_path}")
            ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = build_k5_model(device)
            model.load_state_dict(ckpt["model"])
            start_epoch = ckpt["epoch"] + 1
            snap.load_state_dict(ckpt)

            with torch.no_grad():
                dummy = torch.zeros(1, IN_CHANNELS, *PATCH_SIZE, device=device)
                dummy_out = model(dummy)
            num_outputs = len(dummy_out) if isinstance(dummy_out, (list, tuple)) else 1
            del dummy, dummy_out
            torch.cuda.empty_cache()

            loss_fn   = DeepSupervisionLoss(num_outputs)
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            scaler = torch.amp.GradScaler('cuda', init_scale=1024, growth_interval=1000)
            if "scaler" in ckpt:
                scaler.load_state_dict(ckpt["scaler"])
            # LR override — resume with fresh LR, cosine restarts from NUM_EPOCHS
            for pg in optimizer.param_groups:
                pg["lr"] = LR
            log.info(f"LR set to {LR} | ep={start_epoch} snaps={snap.snapshot_count}")
            log.info(f"Refs: et={snap.et_ref:.4f} cc={snap.cc_ref:.4f} ed={snap.ed_ref:.4f}")
        else:
            log.warning("No checkpoint_latest found — starting fresh with UpKern.")
            args.resume = False

    if not args.resume:
        model = init_from_upkern(fold, args.run_dir, device, log)
        with torch.no_grad():
            dummy = torch.zeros(1, IN_CHANNELS, *PATCH_SIZE, device=device)
            dummy_out = model(dummy)
        num_outputs = len(dummy_out) if isinstance(dummy_out, (list, tuple)) else 1
        log.info(f"DS outputs: {num_outputs}")
        del dummy, dummy_out
        torch.cuda.empty_cache()
        loss_fn   = DeepSupervisionLoss(num_outputs)
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
        scaler    = torch.amp.GradScaler('cuda', init_scale=1024, growth_interval=1000)

    log.info(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    csv_header = ["epoch","loss","lr",
                  "tr_et","tr_net","tr_cc","tr_ed",
                  "fv_et","fv_net","fv_cc","fv_ed","fv_mean",
                  "full_done","full_et","full_net","full_cc","full_ed","full_mean",
                  "snapped","total_snaps"]

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss  = 0.0
        train_dice  = []

        for batch in train_loader:
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out  = model(imgs)
                loss = loss_fn(out, lbls)
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.detach()
            with torch.no_grad():
                final = out[0] if isinstance(out, (list, tuple)) else out
                pred  = torch.argmax(final.detach(), dim=1)
                train_dice.append(dice_score(pred.squeeze(), lbls.squeeze()))

        if epoch == start_epoch:
            log.info(
                f"  GPU mem: alloc={torch.cuda.memory_allocated()/1e9:.2f}GB "
                f"max={torch.cuda.max_memory_allocated()/1e9:.2f}GB"
            )

        scheduler.step()
        avg_loss = (epoch_loss / len(train_loader)).item()
        lr_now   = scheduler.get_last_lr()[0]
        td       = np.nanmean(train_dice, axis=0)

        fd    = run_fast_val(model, val_loader, device)
        fmean = float(np.mean(fd))
        fet, fnet, fcc, fed = fd

        log.info(
            f"Epoch {epoch:04d} | loss={avg_loss:.4f} | LR={lr_now:.6f} | "
            f"Train ET={td[0]:.3f} NET={td[1]:.3f} CC={td[2]:.3f} ED={td[3]:.3f} | "
            f"FastVal ET={fet:.3f} NET={fnet:.3f} CC={fcc:.3f} ED={fed:.3f} "
            f"Mean={fmean:.3f} [best={snap.fast_best:.3f}] Snaps={snap.snapshot_count}"
        )

        full_done   = False
        full_mean_v = ""
        full_et = full_net = full_cc = full_ed = ""
        snapshot_fired = False

        # ── SNAPSHOT (every epoch, no cooldown) ──────────────────────────────
        if epoch >= SNAPSHOT_START:
            fired, reasons = snap.check_snapshot(
                fet, fcc, fed, fmean, SNAP_MARGIN, snap.full_best
            )
            if fired:
                snapshot_fired = True
                mean_tag = f"fv{fmean:.4f}"
                fname = (
                    f"checkpoint_snap{snap.snapshot_count:03d}_ep{epoch:04d}"
                    f"_mean{mean_tag}_et{fet:.4f}_cc{fcc:.4f}_ed{fed:.4f}.pth"
                )
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "fast_et": fet, "fast_cc": fcc, "fast_ed": fed,
                    "fast_mean": fmean, "kernel_size": KERNEL_SIZE,
                }, fold_dir / fname)
                log.info(f"  >> SNAPSHOT #{snap.snapshot_count}: {fname}")
                log.info(f"     Reason: {' | '.join(reasons)}")
            else:
                log.info(
                    f"  >> No snap | refs et={snap.et_ref:.4f} cc={snap.cc_ref:.4f} "
                    f"ed={snap.ed_ref:.4f} | fast ET={fet:.4f} CC={fcc:.4f} ED={fed:.4f} "
                    f"mean_ok(ET/CC)={fmean > snap.fast_best - SNAP_MARGIN}"
                )

        # ── FULL VAL (tighter gate, cooldown, updates refs a/b/c) ────────────
        epochs_since_full = epoch - snap.last_full_val_epoch
        full_val_gate = (
            fmean > snap.fast_best - FULL_VAL_MARGIN
            and epochs_since_full >= FULL_VAL_COOLDOWN
        )
        if full_val_gate:
            log.info(f"  >> FullVal gate (fmean={fmean:.4f} > {snap.fast_best - FULL_VAL_MARGIN:.4f})")
            snap.last_full_val_epoch = epoch
            mpc         = run_full_val(model, val_cases, args.images_dir, args.labels_dir,
                                       PATCH_SIZE, device)
            full_mean_v = float(np.mean(mpc))
            et, net, cc, ed = mpc
            full_done   = True
            full_et, full_net, full_cc, full_ed = et, net, cc, ed

            log.info(
                f"  >> FULL VAL | ET={et:.4f} NET={net:.4f} CC={cc:.4f} ED={ed:.4f} "
                f"Mean={full_mean_v:.4f} [best={snap.full_best:.4f}]"
            )

            # ground refs in confirmed full val only — never fast val
            snap.update_refs_from_full_val(et, cc, ed)

            if full_mean_v > snap.full_best:
                snap.full_best = full_mean_v
                snap.fast_best = max(snap.fast_best, fmean)
                save_checkpoint(
                    fold_dir / "checkpoint_best.pth", epoch, model, optimizer, scheduler,
                    {**snap.state_dict(), "full_mean": full_mean_v,
                     "per_class": mpc.tolist(), "kernel_size": KERNEL_SIZE},
                )
                log.info(f"  >> NEW BEST | Mean={full_mean_v:.4f} ET={et:.4f} CC={cc:.4f} ED={ed:.4f}")
            else:
                log.info(f"  >> No improvement ({full_mean_v:.4f} <= {snap.full_best:.4f})")

        append_csv_log(fold_dir, TAG, [
            epoch, f"{avg_loss:.4f}", f"{lr_now:.6f}",
            f"{td[0]:.4f}", f"{td[1]:.4f}", f"{td[2]:.4f}", f"{td[3]:.4f}",
            f"{fet:.4f}", f"{fnet:.4f}", f"{fcc:.4f}", f"{fed:.4f}", f"{fmean:.4f}",
            int(full_done), full_et, full_net, full_cc, full_ed,
            full_mean_v if full_done else "",
            int(snapshot_fired), snap.snapshot_count,
        ], csv_header)

        save_checkpoint(
            fold_dir / "checkpoint_latest.pth", epoch, model, optimizer, scheduler,
            {**snap.state_dict(), "scaler": scaler.state_dict()},
        )

    log.info("=" * 70)
    log.info(f"Stage 2 done | FullBest={snap.full_best:.4f} Snaps={snap.snapshot_count}")
    log.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_dir", required=True, help="Directory to save run logs and checkpoints")
    parser.add_argument("--images_dir", required=True, help="Directory with training images")
    parser.add_argument("--labels_dir", required=True, help="Directory with training labels")
    parser.add_argument("--splits_file", required=True, help="Path to splits_final_stratified.json")
    args = parser.parse_args()
    train(args)
