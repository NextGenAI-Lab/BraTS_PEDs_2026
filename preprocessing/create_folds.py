"""
03_create_folds.py — Scan NIfTI segmentations → compute label stats → stratified k-fold splits

Outputs:
  splits_final_stratified.json   (nnU-Net splits_final.json format)
  patient_fold_assignment.csv    (patient → fold mapping)
  label_stats.csv                (per-case label statistics)

Usage:
  python stratified_splits.py \
      --data_dirs /path/to/data1 /path/to/data2 \
      --out /path/to/output_dir \
      [--n_folds 5] \
      [--seed 42]

Expected data structure:
  data_dir/
    BraTS-PED-XXXXX-YYY/          <- case directory
      *seg*.nii.gz                 <- segmentation file

Segmentation labels:
  1 = ET (Enhancing Tumour)
  2 = NET (Non-Enhancing Tumour)
  3 = CC (Cystic Component)
  4 = ED (Oedema)

Patient ID and timepoint are parsed from case name: BraTS-PED-{patient_id}-{timepoint}
"""

import json
import argparse
import logging
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

LABELS = {1: "ET", 2: "NET", 3: "CC", 4: "ED"}

GROUP_ORDER = [
    "CC_tiny_small",
    "ET_tiny_small",
    "CC_only",
    "ET_and_CC",
    "ET_only",
    "neither",
]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan segmentations → label stats → stratified k-fold splits."
    )
    parser.add_argument(
        "--data_dirs", nargs="+", required=True,
        help="One or more root directories containing case subdirectories."
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory."
    )
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of folds (default: 5)."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)."
    )
    return parser.parse_args()


# ─── LABEL STATS ──────────────────────────────────────────────────────────────

def parse_case_name(case_name: str):
    """Parse patient_id and timepoint from case directory name.
    Expected format: BraTS-PED-{patient_id}-{timepoint}
    Falls back gracefully if format differs."""
    parts = case_name.split("-")
    try:
        patient_id = parts[2]
        timepoint  = parts[3]
    except IndexError:
        patient_id = case_name
        timepoint  = "000"
    is_post = timepoint != "000"
    return patient_id, timepoint, is_post


def categorize(vox: int) -> str:
    if vox == 0:        return "absent"
    elif vox < 10:      return "tiny(<10)"
    elif vox < 100:     return "small(10-100)"
    elif vox < 1000:    return "medium(100-1k)"
    else:               return "large(>1k)"


def analyze_case(case_dir: Path):
    seg_files = list(case_dir.glob("*seg*.nii.gz"))
    if not seg_files:
        log.warning(f"No seg file found in {case_dir.name}, skipping.")
        return None

    seg     = nib.load(seg_files[0])
    vox_vol = float(np.prod(seg.header.get_zooms()))
    data    = seg.get_fdata().astype(np.uint8)

    patient_id, timepoint, is_post = parse_case_name(case_dir.name)

    result = {
        "case":             case_dir.name,
        "patient_id":       patient_id,
        "timepoint":        timepoint,
        "is_post_treatment": is_post,
        "unique_labels":    str(sorted(np.unique(data).tolist())),
    }

    for lbl, name in LABELS.items():
        mask      = data == lbl
        vox_count = int(mask.sum())
        result[f"{name}_voxels"]   = vox_count
        result[f"{name}_vol_mm3"]  = round(vox_count * vox_vol, 2)
        result[f"{name}_present"]  = vox_count > 0
        result[f"{name}_category"] = categorize(vox_count)

    result["has_ET_and_CC"] = result["ET_present"] and result["CC_present"]
    result["has_ET_only"]   = result["ET_present"] and not result["CC_present"]
    result["has_CC_only"]   = result["CC_present"] and not result["ET_present"]
    result["has_neither"]   = not result["ET_present"] and not result["CC_present"]
    result["has_ED"]        = result["ED_present"]

    return result


def compute_label_stats(data_dirs: list) -> pd.DataFrame:
    all_cases = []
    for d in data_dirs:
        d = Path(d)
        if not d.exists():
            raise FileNotFoundError(f"Data directory not found: {d}")
        cases = sorted([x for x in d.iterdir() if x.is_dir()])
        log.info(f"{d.name}: {len(cases)} cases")
        all_cases.extend(cases)

    log.info(f"Total cases found: {len(all_cases)}")

    rows = []
    for i, case_dir in enumerate(all_cases):
        r = analyze_case(case_dir)
        if r:
            rows.append(r)
        if (i + 1) % 50 == 0:
            log.info(f"  Processed {i+1}/{len(all_cases)}")

    df = pd.DataFrame(rows)
    log.info(f"\nTotal cases analyzed : {len(df)}")
    log.info(f"Unique patients      : {df['patient_id'].nunique()}")
    return df


# ─── STRATIFIED SPLITS ────────────────────────────────────────────────────────

def patient_profile(pid: str, df: pd.DataFrame) -> dict:
    """Aggregate per-case rows into a patient-level profile (worst-case across timepoints)."""
    rows      = df[df["patient_id"] == pid]
    max_idx_et = rows["ET_voxels"].argmax()
    max_idx_cc = rows["CC_voxels"].argmax()
    return {
        "patient_id":         pid,
        "cases":              list(rows["case"]),
        "n_cases":            len(rows),
        "ET_max_vox":         rows["ET_voxels"].max(),
        "CC_max_vox":         rows["CC_voxels"].max(),
        "ED_max_vox":         rows["ED_voxels"].max(),
        "ET_present":         bool(rows["ET_present"].any()),
        "CC_present":         bool(rows["CC_present"].any()),
        "ED_present":         bool(rows["ED_present"].any()),
        "ET_cat":             rows["ET_category"].iloc[max_idx_et],
        "CC_cat":             rows["CC_category"].iloc[max_idx_cc],
        "is_multi_timepoint": len(rows) > 1,
    }


def assign_group(row) -> str:
    """Assign stratification group. Rarest/hardest cases take priority."""
    if row["CC_present"] and row["CC_max_vox"] < 100:  return "CC_tiny_small"
    if row["ET_present"] and row["ET_max_vox"] < 100:  return "ET_tiny_small"
    if row["CC_present"] and not row["ET_present"]:     return "CC_only"
    if row["ET_present"] and row["CC_present"]:         return "ET_and_CC"
    if row["ET_present"] and not row["CC_present"]:     return "ET_only"
    return "neither"


def stratified_kfold(pdf: pd.DataFrame, n_folds: int, seed: int) -> pd.DataFrame:
    """Round-robin assignment within each group, balancing by case count."""
    pdf           = pdf.copy()
    pdf["fold"]   = -1
    rng           = np.random.default_rng(seed)
    fold_counters = [0] * n_folds

    for grp in GROUP_ORDER:
        grp_patients = pdf[pdf["group"] == grp].index.tolist()
        if not grp_patients:
            continue
        rng.shuffle(grp_patients)
        for idx in grp_patients:
            fold = int(np.argmin(fold_counters))
            pdf.at[idx, "fold"] = fold
            fold_counters[fold] += pdf.at[idx, "n_cases"]

    unassigned = (pdf["fold"] == -1).sum()
    if unassigned:
        raise RuntimeError(f"{unassigned} patients were not assigned to any fold.")

    return pdf


def print_fold_summary(pdf: pd.DataFrame, df: pd.DataFrame, n_folds: int):
    log.info("\n=== FOLD SUMMARY ===")
    for f in range(n_folds):
        fold_patients = pdf[pdf["fold"] == f]
        fold_cases    = df[df["patient_id"].isin(fold_patients["patient_id"])]
        log.info(f"\nFold {f}: {len(fold_patients)} patients, {len(fold_cases)} cases")
        for lbl in ["ET", "CC", "ED"]:
            n   = fold_cases[f"{lbl}_present"].sum()
            pct = 100 * n / len(fold_cases) if len(fold_cases) else 0
            log.info(f"  {lbl}: {n}/{len(fold_cases)} ({pct:.1f}%)")
        log.info(f"  Groups: {dict(fold_patients['group'].value_counts())}")


def build_splits(pdf: pd.DataFrame, df: pd.DataFrame, n_folds: int) -> list:
    splits = []
    for f in range(n_folds):
        fold_patient_ids = set(pdf[pdf["fold"] == f]["patient_id"])
        val_cases        = sorted(df[df["patient_id"].isin(fold_patient_ids)]["case"].tolist())
        train_cases      = sorted(df[~df["patient_id"].isin(fold_patient_ids)]["case"].tolist())
        splits.append({"train": train_cases, "val": val_cases})
        log.info(f"Fold {f}: train={len(train_cases)}, val={len(val_cases)}")
    return splits


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Scan data and compute label stats
    df = compute_label_stats(args.data_dirs)

    stats_path = out_dir / "label_stats.csv"
    df.to_csv(stats_path, index=False)
    log.info(f"Saved: {stats_path}")

    # Step 2: Build patient profiles
    patient_ids = sorted(df["patient_id"].unique())
    pdf         = pd.DataFrame([patient_profile(pid, df) for pid in patient_ids])

    log.info(f"\nTotal patients  : {len(pdf)}")
    log.info(f"Multi-timepoint : {pdf['is_multi_timepoint'].sum()}")

    # Step 3: Assign groups
    pdf["group"] = pdf.apply(assign_group, axis=1)
    log.info("\n=== PATIENT GROUPS ===\n" + pdf["group"].value_counts().to_string())

    # Step 4: Stratified k-fold
    pdf = stratified_kfold(pdf, n_folds=args.n_folds, seed=args.seed)

    print_fold_summary(pdf, df, n_folds=args.n_folds)

    # Step 5: Build and save outputs
    splits      = build_splits(pdf, df, n_folds=args.n_folds)
    splits_path = out_dir / "splits_final_stratified.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)
    log.info(f"\nSaved: {splits_path}")

    fold_csv_path = out_dir / "patient_fold_assignment.csv"
    pdf.to_csv(fold_csv_path, index=False)
    log.info(f"Saved: {fold_csv_path}")


if __name__ == "__main__":
    main()