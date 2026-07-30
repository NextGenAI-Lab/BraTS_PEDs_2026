"""
04_prepare_nnunet.py — Convert BraTS26 PEDs to nnU-Net v2 format

Usage:
    python 04_prepare_nnunet.py \
        --run_dir /workspace/BRATS/run_2 \
        --train_roots /data/BraTS26_PED_training /data/BraTS-PEDs_Batch2_Release \
        --val_root /data/BraTS26_PED_validation \
        [--symlink]            # optional: symlink instead of copy (faster, needs stable paths)
        [--dataset_id 001]     # optional: default is 001

Creates:
    {run_dir}/nnunet_raw/Dataset{dataset_id}_BraTSPEDs/
        imagesTr/    (training images, 4 channels per case)
        labelsTr/    (training segmentations)
        imagesTs/    (validation images, no labels)
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
import argparse
from pathlib import Path
from datetime import datetime


MODALITY_MAP = {
    "t1n": "0000",
    "t1c": "0001",
    "t2w": "0002",
    "t2f": "0003",
}
SEG_SUFFIX = "seg"


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def find_file(case_dir: Path, suffix: str):
    matches = list(case_dir.glob(f"*{suffix}.nii.gz"))
    return matches[0] if matches else None


def symlink_or_copy(src: Path, dst: Path, use_symlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(str(src), str(dst))


def process_split(case_dirs, images_dir, labels_dir, split_name, use_symlink, log):
    images_dir.mkdir(parents=True, exist_ok=True)
    if labels_dir is not None:
        labels_dir.mkdir(parents=True, exist_ok=True)

    ok, skipped = 0, 0

    for case_dir in case_dirs:
        case_id = case_dir.name

        # Check all modalities present
        missing = [suf for suf in MODALITY_MAP if find_file(case_dir, suf) is None]
        if missing:
            log.warning(f"  SKIP {case_id} — missing modalities: {missing}")
            skipped += 1
            continue

        # Link/copy modality files
        for suf, ch_id in MODALITY_MAP.items():
            src = find_file(case_dir, suf)
            dst = images_dir / f"{case_id}_{ch_id}.nii.gz"
            symlink_or_copy(src, dst, use_symlink)

        # Link/copy segmentation (training only)
        if labels_dir is not None:
            seg_src = find_file(case_dir, SEG_SUFFIX)
            if seg_src is None:
                log.warning(f"  SKIP SEG {case_id} — seg not found")
                skipped += 1
                continue
            seg_dst = labels_dir / f"{case_id}.nii.gz"
            symlink_or_copy(seg_src, seg_dst, use_symlink)

        ok += 1

    log.info(f"  {split_name}: {ok} cases processed, {skipped} skipped")
    return ok


def build_dataset_json(dataset_dir: Path, num_training: int, log):
    ds = {
        "channel_names": {
            "0": "T1N",
            "1": "T1C",
            "2": "T2W",
            "3": "T2F",
        },
        "labels": {
            "background": 0,
            "ET":         1,
            "NET":        2,
            "CC":         3,
            "ED":         4,
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert BraTS26 PEDs data to nnU-Net v2 format."
    )
    parser.add_argument(
        "--run_dir", required=True,
        help="Root run directory. Output goes to {run_dir}/nnunet_raw/."
    )
    parser.add_argument(
        "--train_roots", required=True, nargs="+",
        help="One or more directories containing training case folders."
    )
    parser.add_argument(
        "--val_root", required=True,
        help="Directory containing validation case folders (no labels needed)."
    )
    parser.add_argument(
        "--dataset_id", default="001",
        help="nnU-Net dataset ID, zero-padded to 3 digits (default: 001)."
    )
    parser.add_argument(
        "--symlink", action="store_true",
        help="Symlink files instead of copying. Faster but requires stable source paths."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    run_dir     = Path(args.run_dir)
    dataset_id  = args.dataset_id.zfill(3)
    dataset_name = f"Dataset{dataset_id}_BraTSPEDs"
    dataset_dir  = run_dir / "nnunet_raw" / dataset_name

    log = setup_logging(run_dir / "logs" / "prepare_nnunet.log")

    log.info("=" * 60)
    log.info("BraTS26 PEDs — nnU-Net v2 Dataset Preparation")
    log.info(f"Run dir    : {run_dir}")
    log.info(f"Dataset    : {dataset_name}")
    log.info(f"Mode       : {'symlink' if args.symlink else 'copy'}")
    log.info(f"Started    : {datetime.now()}")
    log.info("=" * 60)

    # Collect training cases from all roots
    train_cases = []
    for root in args.train_roots:
        rp = Path(root)
        if not rp.exists():
            log.warning(f"Training root not found, skipping: {root}")
            continue
        cases = sorted([d for d in rp.iterdir() if d.is_dir()])
        log.info(f"Found {len(cases)} cases in {root}")
        train_cases.extend(cases)

    if not train_cases:
        log.error("No training cases found. Aborting.")
        return

    log.info(f"\nTotal training cases: {len(train_cases)}")

    # Training split
    log.info("\nPreparing imagesTr / labelsTr ...")
    num_ok = process_split(
        train_cases,
        dataset_dir / "imagesTr",
        dataset_dir / "labelsTr",
        "Training",
        args.symlink,
        log,
    )

    # Validation split
    val_path = Path(args.val_root)
    if val_path.exists():
        val_cases = sorted([d for d in val_path.iterdir() if d.is_dir()])
        log.info(f"\nPreparing imagesTs ({len(val_cases)} validation cases)...")
        process_split(
            val_cases,
            dataset_dir / "imagesTs",
            None,
            "Validation",
            args.symlink,
            log,
        )
    else:
        log.warning(f"Validation root not found, skipping: {args.val_root}")

    # dataset.json
    build_dataset_json(dataset_dir, num_training=num_ok, log=log)

    log.info("\n" + "=" * 60)
    log.info("nnU-Net dataset ready.")
    log.info(f"  Dataset path : {dataset_dir}")
    log.info(f"  imagesTr     : {dataset_dir / 'imagesTr'}")
    log.info(f"  labelsTr     : {dataset_dir / 'labelsTr'}")
    log.info(f"  imagesTs     : {dataset_dir / 'imagesTs'}")
    log.info(f"\nNext: run nnUNetv2_plan_and_preprocess, then your training script.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()