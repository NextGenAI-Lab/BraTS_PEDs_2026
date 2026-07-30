# maskify.py
import os
import numpy as np
import nibabel as nib
from ensemble_accumulator import accumulate
from apply_ed_override import compute_ed_intersection, apply_ed_override

def make_mask(
    case_id,
    model_output_dirs,
    needs_perm,
    snap058_dir,
    snap014_dir,
    run3_dir,
    output_dir,
    ref_nii_path,
):
    # step 1: weighted prob sum from 8 nnUNet models (raw prob space, pre-transpose)
    weighted_sum = accumulate(case_id, model_output_dirs, needs_perm)

    # step 2: argmax → integer mask, then transpose to (H, W, D)
    final_mask = np.argmax(weighted_sum, axis=0).astype(np.uint8)
    final_mask = np.transpose(final_mask, (2, 1, 0))  # now matches .nii.gz space

    # step 3: compute 3-way ED intersection (loads .nii.gz → already in H, W, D)
    ed_intersection = compute_ed_intersection(
        snap058_mask_path = os.path.join(snap058_dir, f"{case_id}.nii.gz"),
        snap014_mask_path = os.path.join(snap014_dir, f"{case_id}.nii.gz"),
        run3_mask_path    = os.path.join(run3_dir,    f"{case_id}.nii.gz"),
    )

    # step 4: override ED on final mask — both now in (H, W, D) space
    if ed_intersection.shape != final_mask.shape:
        raise ValueError(
            f"{case_id}: ED intersection shape {ed_intersection.shape} "
            f"vs final mask shape {final_mask.shape}"
        )
    final_mask = apply_ed_override(final_mask, ed_intersection)

    # step 5: save
    os.makedirs(output_dir, exist_ok=True)
    ref = nib.load(ref_nii_path)
    out_path = os.path.join(output_dir, f"{case_id}.nii.gz")
    nib.save(nib.Nifti1Image(final_mask, ref.affine, ref.header), out_path)
    print(f"Saved: {out_path}")
    return out_path

if __name__ == "__main__":
    print("maskify.py — no standalone test, called from run_pipeline.py")