# inference/model_registry.py

# These are relative references used when constructing full paths dynamically
RUN5 = "nnUNetTrainer_Run5_Snapshot__nnUNetPlans__3d_fullres"
RUN3 = "nnUNetTrainer_C3__nnUNetPlans__3d_fullres"

# 10 inference runs total:
# idx 0-7: nnUNet ensemble (Run5)
# idx 8:   nnUNet ED-only (Run3) — used only for ED intersection, not in WEIGHTS directly
# idx 9:   MedNeXt snap058 — used in ED intersection + slot 8 in WEIGHTS

MODELS = [
    # idx, fold, checkpoint, trainer_path, needs_perm
    (0, 1, "snapshot_ep0406_ema0.7232_et0.8026_cc0.6404.pth", RUN5, True),
    (1, 2, "snapshot_ep0366_ema0.5833_et0.7586_cc0.4251.pth", RUN5, True),
    (2, 4, "snapshot_ep0363_ema0.6722_et0.7248_cc0.6985.pth", RUN5, True),
    (3, 0, "snapshot_ep0489_ema0.5848_et0.7906_cc0.1900.pth", RUN5, True),
    (4, 3, "snapshot_ep0459_ema0.6337_et0.7409_cc0.4222.pth", RUN5, True),
    (5, 2, "snapshot_ep0450_ema0.5864_et0.7353_cc0.3791.pth", RUN5, True),
    (6, 1, "snapshot_ep0335_ema0.7139_et0.7758_cc0.6589.pth", RUN5, True),
    (7, 3, "snapshot_ep0440_ema0.6218_et0.7696_cc0.3990.pth", RUN5, True),
    # idx 8: ED-only nnUNet (Run3) — needs_perm=True (nnUNet output)
    (8, 4, "checkpoint_best.pth", RUN3, True),
]

# (Removed hardcoded MEDNEXT_SNAP058 to be passed dynamically)

# WEIGHTS: 9 columns (idx 0-7 = nnUNet ensemble, idx 8 = ED-intersection result)
# ED-intersection (col 8) = (snap058==4) & (run3==4), gets full weight in ED row only
WEIGHTS = {
    0: (0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.00),  # BG
    1: (0.40, 0.35, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),  # ET
    2: (0.00, 0.00, 0.00, 0.00, 0.00, 0.33, 0.34, 0.33, 0.00),  # NETC
    3: (0.00, 0.00, 0.00, 0.40, 0.35, 0.25, 0.00, 0.00, 0.00),  # CC
    4: (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00),  # ED
}

DATASET_ID   = "001"
PLANS        = "nnUNetPlans"
CONFIG       = "3d_fullres"
CHUNK_SIZE   = 5

# MedNeXt config
MEDNEXT_MODEL_ID  = "M"
MEDNEXT_KERNEL    = 5
MEDNEXT_IN_CH     = 4
MEDNEXT_OUT_CH    = 5
MEDNEXT_PATCH     = (128, 160, 112)
MEDNEXT_OVERLAP   = 0.25


if __name__ == "__main__":
    import os
    import argparse
    parser = argparse.ArgumentParser(description="Verify model configurations.")
    parser.add_argument("--nnunet_results", required=True, help="Path to nnUNet_results directory")
    args = parser.parse_args()
    
    print("=== Verifying model paths ===")
    for idx, fold, ckpt, trainer_path, _ in MODELS:
        actual_trainer_path = os.path.join(args.nnunet_results, f"Dataset{DATASET_ID}_BraTSPEDs", trainer_path)
        p = f"{actual_trainer_path}/fold_{fold}/{ckpt}"
        status = "OK" if os.path.exists(p) else "MISSING"
        print(f"  [{status}] idx={idx} fold={fold} {ckpt}")
    print("=== Done ===")