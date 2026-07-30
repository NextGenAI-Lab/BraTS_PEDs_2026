# apply_ed_override.py
import numpy as np
import nibabel as nib

ED_LABEL = 4

def compute_ed_intersection(snap058_mask_path, snap014_mask_path, run3_mask_path):
    """
    3-way intersection of ED channel across all ED models.
    Returns binary mask (H, W, D).
    """
    s58  = nib.load(snap058_mask_path).get_fdata().astype(np.int16)
    s014 = nib.load(snap014_mask_path).get_fdata().astype(np.int16)
    r3   = nib.load(run3_mask_path).get_fdata().astype(np.int16)

    ed_final = (s58 == ED_LABEL) & (s014 == ED_LABEL) & (r3 == ED_LABEL)
    return ed_final  # bool array (H, W, D)

def apply_ed_override(final_mask, ed_intersection):
    """
    Wipes existing ED from final_mask, injects intersection ED.
    final_mask:     (H, W, D) uint8  — post-argmax, post-transpose
    ed_intersection:(H, W, D) bool   — from compute_ed_intersection
    """
    final_mask[final_mask == ED_LABEL] = 0       # wipe existing ED
    final_mask[ed_intersection] = ED_LABEL        # inject consensus ED
    return final_mask

if __name__ == "__main__":
    print("apply_ed_override.py — no standalone test, used by maskify.py")