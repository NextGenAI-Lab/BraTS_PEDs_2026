# /workspace/BRATS/docker_build/scripts/run_mednext.py

#!/usr/bin/env python3
import sys
import re
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from nnunet_mednext import create_mednext_v1

from model_registry import (
    MEDNEXT_SNAP058, MEDNEXT_MODEL_ID, MEDNEXT_KERNEL,
    MEDNEXT_IN_CH, MEDNEXT_OUT_CH, MEDNEXT_PATCH, MEDNEXT_OVERLAP
)

SW_BATCH = 4  # reduce if OOM, increase if VRAM allows

CASE_RE = re.compile(r"(BraTS-PED-\d{5}-\d{3})")


def normalize(image):
    out = np.zeros_like(image)
    for c in range(image.shape[0]):
        ch = image[c]
        mask = ch > 0
        if mask.sum() > 0:
            out[c] = (ch - ch[mask].mean()) / (ch[mask].std() + 1e-8)
    return out


def build_model(device):
    return create_mednext_v1(
        MEDNEXT_IN_CH, MEDNEXT_OUT_CH, MEDNEXT_MODEL_ID,
        kernel_size=MEDNEXT_KERNEL, deep_supervision=True,
    ).to(device)


def final_head(out):
    return out[0] if isinstance(out, (list, tuple)) else out


def discover_cases(images_dir):
    cases = set()
    for f in Path(images_dir).glob("*.nii.gz"):
        m = CASE_RE.search(f.name)
        if m:
            cases.add(m.group(1))
    return sorted(cases)


def predict_mednext(ckpt_path, images_dir, output_dir, device=None, sw_batch=SW_BATCH):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir)

    print(f"Loading: {ckpt_path}")
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = build_model(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    def predictor(x):
        return final_head(model(x))

    cases = discover_cases(images_dir)
    print(f"Cases found: {len(cases)}")

    with torch.no_grad():
        for i, cid in enumerate(cases, 1):
            out_path = output_dir / f"{cid}.nii.gz"
            if out_path.exists():
                print(f"  [SKIP] {cid} already done")
                continue

            mods    = []
            ref_img = None
            for m in range(4):
                nii = nib.load(str(images_dir / f"{cid}_{m:04d}.nii.gz"))
                if ref_img is None:
                    ref_img = nii
                mods.append(nii.get_fdata(dtype=np.float32))

            image = normalize(np.stack(mods, axis=0))
            imgs  = torch.from_numpy(image).float().unsqueeze(0).to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = sliding_window_inference(
                    imgs, MEDNEXT_PATCH, sw_batch, predictor, overlap=MEDNEXT_OVERLAP)

            pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            nib.save(
                nib.Nifti1Image(pred, ref_img.affine, ref_img.header),
                str(out_path)
            )
            if i % 10 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)}")

    del model
    torch.cuda.empty_cache()
    print(f"Done -> {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     default=MEDNEXT_SNAP058)
    parser.add_argument("--input",    required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--sw_batch", type=int, default=SW_BATCH)
    parser.add_argument("--device",   default="cuda")
    args = parser.parse_args()

    predict_mednext(
        ckpt_path  = args.ckpt,
        images_dir = args.input,
        output_dir = args.output,
        device     = torch.device(args.device),
        sw_batch   = args.sw_batch,
    )