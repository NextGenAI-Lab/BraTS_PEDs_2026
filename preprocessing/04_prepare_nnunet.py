"""
04_prepare_nnunet.py — Convert BraTS26 PEDs to nnU-Net v2 format

To switch runs: change RUN_DIR only. Everything else follows.

Creates:
  {RUN_DIR}/nnunet_raw/Dataset001_BraTSPEDs/
      imagesTr/   (training images, 4 channels per case)
      labelsTr/   (training segmentations)
      imagesTs/   (validation images, no labels)
      dataset.json

nnU-Net channel convention:
  _0000 = T1N
  _0001 = T1C
  _0002 = T2W
  _0003 = T2F
"""

import os
import json
import shutil
import logging
import numpy as np
import nibabel as nib
from pathlib import Path
from datetime import datetime

# ─── CONFIG — change RUN_DIR to switch runs ───────────────────────────────────
RUN_DIR = "/workspace/BRATS/run_2"

TRAIN_ROOTS = [
    "/workspace/BRATS/data/BraTS26_PED_training",
    "/workspace/BRATS/data/BraTS-PEDs_Batch2_Release",
]
VAL_ROOT = "/workspace/BRATS/data/BraTS26_PED_validation"

NNUNET_RAW   = f"{RUN_DIR}/nnunet_raw"
DATASET_ID   = "001"
DATASET_NAME = f"Dataset{DATASET_ID}_BraTSPEDs"

MODALITY_MAP = {
    "t1n": "0000",
    "t1c": "0001",
    "t2w": "0002",
    "t2f": "0003",
}
SEG_SUFFIX = "seg"

LOG_PATH = f"{RUN_DIR}/logs/prepare_nnunet.log"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def find_file(case_dir: Path, suffix: str):
    matches = list(case_dir.glob(f"*{suffix}.nii.gz"))
    return matches[0] if matches else None


def symlink_or_copy(src: Path, dst: Path, use_symlink=True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(str(src), str(dst))


def process_split(case_dirs, images_dir, labels_dir, split_name):
    images_dir.mkdir(parents=True, exist_ok=True)
    if labels_dir:
        labels_dir.mkdir(parents=True, exist_ok=True)

    ok, skipped = 0, 0

    for case_dir in case_dirs:
        case_id = case_dir.name

        missing = []
        for suf in MODALITY_MAP:
            if find_file(case_dir, suf) is None:
                missing.append(suf)
        if missing:
            log.warning(f"  SKIP {case_id} — missing modalities: {missing}")
            skipped += 1
            continue

        for suf, ch_id in MODALITY_MAP.items():
            src = find_file(case_dir, suf)
            dst = images_dir / f"{case_id}_{ch_id}.nii.gz"
            symlink_or_copy(src, dst)

        if labels_dir is not None:
            seg_src = find_file(case_dir, SEG_SUFFIX)
            if seg_src is None:
                log.warning(f"  SKIP SEG {case_id} — seg not found")
                skipped += 1
                continue
            seg_dst = labels_dir / f"{case_id}.nii.gz"
            symlink_or_copy(seg_src, seg_dst)

        ok += 1

    log.info(f"  {split_name}: {ok} cases linked, {skipped} skipped")
    return ok


def build_dataset_json(dataset_dir: Path, num_training: int):
    ds = {
        "channel_names": {
            "0": "T1N",
            "1": "T1C",
            "2": "T2W",
            "3": "T2F",
        },
        "labels": {
            "background": 0,
            "NET":        1,
            "ED":         2,
            "ET":         3,
            "CC":         4,
        },
        "regions_class_order": [1, 2, 3, 4],
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": "BraTS26_PEDs",
        "description": "BraTS 2026 Task 2 — Pediatric Brain Tumor Segmentation",
        "reference": "https://www.synapse.org/Synapse:syn74837563",
        "licence": "CC-BY 4.0",
        "release": "2026",
    }
    out_path = dataset_dir / "dataset.json"
    with open(out_path, "w") as f:
        json.dump(ds, f, indent=2)
    log.info(f"dataset.json written: {out_path}")
    return ds


def main():
    log.info("=" * 60)
    log.info("BraTS26 PEDs — nnU-Net v2 Dataset Preparation")
    log.info(f"Run dir : {RUN_DIR}")
    log.info(f"Started : {datetime.now()}")
    log.info("=" * 60)

    dataset_dir = Path(NNUNET_RAW) / DATASET_NAME
    images_tr   = dataset_dir / "imagesTr"
    labels_tr   = dataset_dir / "labelsTr"
    images_ts   = dataset_dir / "imagesTs"

    train_cases = []
    for root in TRAIN_ROOTS:
        rp = Path(root)
        if not rp.exists():
            log.warning(f"Training root not found: {root}")
            continue
        cases = sorted([d for d in rp.iterdir() if d.is_dir()])
        log.info(f"Found {len(cases)} cases in {root}")
        train_cases.extend(cases)

    log.info(f"\nTotal training cases: {len(train_cases)}")

    log.info("\nPreparing imagesTr / labelsTr ...")
    num_ok = process_split(train_cases, images_tr, labels_tr, "Training")

    val_path = Path(VAL_ROOT)
    if val_path.exists():
        val_cases = sorted([d for d in val_path.iterdir() if d.is_dir()])
        log.info(f"\nPreparing imagesTs ({len(val_cases)} validation cases)...")
        process_split(val_cases, images_ts, None, "Validation")
    else:
        log.warning(f"Validation root not found: {VAL_ROOT}")

    build_dataset_json(dataset_dir, num_training=num_ok)

    log.info("\n" + "=" * 60)
    log.info("nnU-Net dataset ready.")
    log.info(f"  Dataset path : {dataset_dir}")
    log.info(f"  imagesTr     : {images_tr}")
    log.info(f"  labelsTr     : {labels_tr}")
    log.info(f"  imagesTs     : {images_ts}")
    log.info(f"\nNext: run 05_train_nnunet.sh")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
