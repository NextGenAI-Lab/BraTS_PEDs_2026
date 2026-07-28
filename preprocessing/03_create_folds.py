import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

CSV = Path("/workspace/BRATS/scripts/analysis/label_stats_v3.csv")
OUT = Path("/workspace/BRATS/scripts/analysis")
df = pd.read_csv(CSV)

# === STEP 1: Build patient-level profile ===
# For each patient, aggregate across all their timepoints
patient_cases = defaultdict(list)
for _, row in df.iterrows():
    patient_cases[row["patient_id"]].append(row["case"])

def patient_profile(pid):
    rows = df[df["patient_id"] == pid]
    # Use worst-case (max) voxels across timepoints for stratification
    return {
        "patient_id": pid,
        "cases": list(rows["case"]),
        "n_cases": len(rows),
        "ET_max_vox": rows["ET_voxels"].max(),
        "CC_max_vox": rows["CC_voxels"].max(),
        "ED_max_vox": rows["ED_voxels"].max(),
        "ET_present": rows["ET_present"].any(),
        "CC_present": rows["CC_present"].any(),
        "ED_present": rows["ED_present"].any(),
        "ET_cat": rows["ET_category"].iloc[rows["ET_voxels"].argmax()],
        "CC_cat": rows["CC_category"].iloc[rows["CC_voxels"].argmax()],
        "is_multi_timepoint": len(rows) > 1,
    }

patients = [patient_profile(pid) for pid in sorted(patient_cases.keys())]
pdf = pd.DataFrame(patients)

print(f"Total patients: {len(pdf)}")
print(f"Multi-timepoint patients: {pdf['is_multi_timepoint'].sum()}")

# === STEP 2: Assign stratification group per patient ===
def assign_group(row):
    et_cat = row["ET_cat"]
    cc_vox = row["CC_max_vox"]
    et_vox = row["ET_max_vox"]

    # Hard/rare cases first
    if row["CC_present"] and cc_vox < 100:      return "CC_tiny_small"
    if row["ET_present"] and et_vox < 100:       return "ET_tiny_small"
    if row["CC_present"] and not row["ET_present"]: return "CC_only"
    if row["ET_present"] and row["CC_present"]:  return "ET_and_CC"
    if row["ET_present"] and not row["CC_present"]: return "ET_only"
    return "neither"

pdf["group"] = pdf.apply(assign_group, axis=1)

print("\n=== PATIENT GROUPS ===")
print(pdf["group"].value_counts().to_string())

# === STEP 3: Stratified assignment — round robin within each group ===
N_FOLDS = 5
pdf["fold"] = -1

# Shuffle within each group with fixed seed for reproducibility
rng = np.random.default_rng(42)

group_order = ["CC_tiny_small", "ET_tiny_small", "CC_only", "ET_and_CC", "ET_only", "neither"]
fold_counters = [0] * N_FOLDS

for grp in group_order:
    grp_patients = pdf[pdf["group"] == grp].index.tolist()
    rng.shuffle(grp_patients)
    # Assign round-robin starting from fold with fewest cases
    for idx in grp_patients:
        fold = int(np.argmin(fold_counters))
        pdf.at[idx, "fold"] = fold
        fold_counters[fold] += pdf.at[idx, "n_cases"]

# === STEP 4: Print fold summary ===
print("\n=== FOLD SUMMARY ===")
for f in range(N_FOLDS):
    fold_patients = pdf[pdf["fold"] == f]
    fold_cases = df[df["patient_id"].isin(fold_patients["patient_id"])]
    print(f"\nFold {f}: {len(fold_patients)} patients, {len(fold_cases)} cases")
    for lbl in ["ET", "CC", "ED"]:
        n = fold_cases[f"{lbl}_present"].sum()
        print(f"  {lbl}: {n}/{len(fold_cases)} ({100*n/len(fold_cases):.1f}%)")
    # Group breakdown
    grp_counts = fold_patients["group"].value_counts()
    print(f"  Groups: {dict(grp_counts)}")

# === STEP 5: Build nnUNet splits_final.json format ===
# For each fold: val = cases in that fold, train = all other cases
splits = []
all_cases = list(df["case"])

for f in range(N_FOLDS):
    fold_patient_ids = set(pdf[pdf["fold"] == f]["patient_id"])
    val_cases = sorted(df[df["patient_id"].isin(fold_patient_ids)]["case"].tolist())
    train_cases = sorted(df[~df["patient_id"].isin(fold_patient_ids)]["case"].tolist())
    splits.append({"train": train_cases, "val": val_cases})
    print(f"\nFold {f}: train={len(train_cases)}, val={len(val_cases)}")

# Save
out_json = OUT / "splits_final_stratified.json"
with open(out_json, "w") as f:
    json.dump(splits, f, indent=2)

# Also save patient-fold mapping for reference
pdf.to_csv(OUT / "patient_fold_assignment.csv", index=False)

print(f"\nSaved: {out_json}")
print(f"Saved: {OUT}/patient_fold_assignment.csv")
