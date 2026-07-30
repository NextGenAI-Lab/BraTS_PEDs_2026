# ensemble_accumulator.py

import os
import numpy as np

# WEIGHTS: 8 nnUNet cols (0-7), ED handled separately in maskify.py
WEIGHTS = {
    0: (0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125),  # BG
    1: (0.40, 0.35, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00),          # ET
    2: (0.00, 0.00, 0.00, 0.00, 0.00, 0.33, 0.34, 0.33),          # NETC
    3: (0.00, 0.00, 0.00, 0.40, 0.35, 0.25, 0.00, 0.00),          # CC
    4: (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),          # ED — zero, overridden later
}

NUM_CLASSES = 5

def accumulate(case_id, model_output_dirs, needs_perm):
    assert len(model_output_dirs) == 8
    assert len(needs_perm) == 8
    weighted_sum = None
    for col_idx, (out_dir, perm) in enumerate(zip(model_output_dirs, needs_perm)):
        npz_path = os.path.join(out_dir, f"{case_id}.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Missing npz: {npz_path}")
        probs = np.load(npz_path)['probabilities']  # (5, X, Y, Z) — no permute here
        if weighted_sum is None:
            weighted_sum = np.zeros_like(probs, dtype=np.float32)
        for c in range(NUM_CLASSES):
            w = WEIGHTS[c][col_idx]
            if w > 0:
                weighted_sum[c] += w * probs[c]
    return weighted_sum  # caller must: argmax → transpose(2,1,0) → save



if __name__ == "__main__":
    print("=== Weight sums per class (should be 1.0 or 0.0) ===")
    for c, label in enumerate(["BG", "ET", "NETC", "CC", "ED"]):
        s = sum(WEIGHTS[c])
        print(f"  {label}: {s:.4f}")
    print("=== Done ===")