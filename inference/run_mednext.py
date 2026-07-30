# /workspace/BRATS/docker_build/scripts/run_mednext.py

#!/usr/bin/env python3
import sys
import re
import os
import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from nnunet_mednext import create_mednext_v1

from model_registry import (
    MEDNEXT_MODEL_ID, MEDNEXT_KERNEL,
    MEDNEXT_IN_CH, MEDNEXT_OUT_CH, MEDNEXT_PATCH, MEDNEXT_OVERLAP
)

SW_BATCH = 4  # reduce if OOM, increase if VRAM allows

CASE_RE = re.compile(r"(BraTS-PED-\d{5}-\d{3})")


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


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


def predict_mednext(ckpt_path, images_dir, output_dir, device=None, sw_batch=SW_BATCH, log=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if log: log.info(f"Device: {device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = Path(images_dir)

    if log: log.info(f"Loading: {ckpt_path}")
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
    if log: log.info(f"Cases found: {len(cases)}")

    with torch.no_grad():
        for i, cid in enumerate(cases, 1):
            out_path = output_dir / f"{cid}.nii.gz"
            if out_path.exists():
                if log: log.info(f"  [SKIP] {cid} already done")
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
                if log: log.info(f"  {i}/{len(cases)}")

    del model
    torch.cuda.empty_cache()
    if log: log.info(f"Done -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MedNeXt inference.")
    parser.add_argument("--ckpt",     required=True, help="Path to checkpoint")
    parser.add_argument("--input",    required=True, help="Input directory")
    parser.add_argument("--output",   required=True, help="Output directory")
    parser.add_argument("--sw_batch", type=int, default=SW_BATCH, help="Sliding window batch size")
    parser.add_argument("--device",   default="cuda", help="Device (e.g. cuda, cpu)")
    parser.add_argument("--log_dir",  default="logs", help="Directory to save logs")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log = setup_logging(log_dir / "run_mednext.log")

    predict_mednext(
        ckpt_path  = args.ckpt,
        images_dir = args.input,
        output_dir = args.output,
        device     = torch.device(args.device),
        sw_batch   = args.sw_batch,
        log        = log
    )