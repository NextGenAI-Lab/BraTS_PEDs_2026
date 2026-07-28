# /workspace/BRATS/docker_build/scripts/postprocess.py

import os
import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label


def postprocess_mask(in_path, out_path):
    """
    Connected component filtering on CC channel (label=3) only.
    Removes components with < 3 voxels.
    Saves result to out_path.
    """
    img  = nib.load(in_path)
    mask = img.get_fdata(dtype=np.float32).astype(np.int16)

    binary = (mask == 3).astype(np.uint8)
    if binary.sum() > 0:
        labeled, n = cc_label(binary)
        mask2   = mask.copy()
        removed = 0
        for i in range(1, n + 1):
            if (labeled == i).sum() < 3:
                mask2[labeled == i] = 0
                removed += 1
        if removed > 0:
            mask = mask2

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), out_path)


def postprocess_dir(input_dir, output_dir):
    """
    Run postprocess_mask on all .nii.gz files in input_dir,
    write results to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    cases   = sorted([f for f in os.listdir(input_dir) if f.endswith(".nii.gz")])
    changed = 0

    for fname in cases:
        in_path  = os.path.join(input_dir,  fname)
        out_path = os.path.join(output_dir, fname)
        img      = nib.load(in_path)
        mask     = img.get_fdata(dtype=np.float32).astype(np.int16)

        binary = (mask == 3).astype(np.uint8)
        modified = False
        if binary.sum() > 0:
            labeled, n = cc_label(binary)
            mask2      = mask.copy()
            removed    = 0
            for i in range(1, n + 1):
                if (labeled == i).sum() < 3:
                    mask2[labeled == i] = 0
                    removed += 1
            if removed > 0:
                mask     = mask2
                modified = True
                changed += 1

        nib.save(nib.Nifti1Image(mask, img.affine, img.header), out_path)

    print(f"Done: {len(cases)} cases, {changed} modified")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    postprocess_dir(args.input, args.output)