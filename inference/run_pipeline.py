# /workspace/BRATS/docker_build/scripts/run_pipeline.py

import os
import sys
import shutil
import argparse

sys.path.insert(0, '/workspace/BRATS/docker_build/scripts')

from stage_input import get_all_cases, get_chunk, stage_chunk
from maskify import make_mask
from postprocess import postprocess_mask
from model_registry import (
    MODELS, MEDNEXT_SNAP058, MEDNEXT_MODEL_ID,
    CHUNK_SIZE, PLANS, CONFIG, DATASET_ID,
    RUN5, RUN3,
)

VENV_BIN     = "/workspace/BRATS/synapse_env/bin"
SCRATCH_BASE = "/tmp/brats_scratch"

# ED models (MedNeXt) — extend this list to add more ED models later
ED_MEDNEXT_CKPTS = [
    "/workspace/BRATS/docker_build/models/mednext/checkpoint_snap058_ep0193_meanfv0.6546_et0.6663_cc0.6490_ed0.5757.pth",
    "/workspace/BRATS/docker_build/models/mednext/checkpoint_snap014_ep0362_fv0.6273_et0.7015_cc0.6410_ed0.3777.pth",
]


def set_nnunet_env():
    os.environ['nnUNet_raw']          = '/tmp/nnunet_raw'
    os.environ['nnUNet_preprocessed'] = '/tmp/nnunet_preprocessed'
    os.environ['nnUNet_results']      = '/workspace/BRATS/docker_build/models/nnunet_results'


def run_nnunet_model(idx, fold, checkpoint, trainer_path, input_dir, output_dir, save_probs=True):
    from run_nnunet import run_nnunet
    trainer = os.path.basename(trainer_path).split('__')[0]
    run_nnunet(
        input_dir   = input_dir,
        output_dir  = output_dir,
        dataset_id  = DATASET_ID,
        trainer     = trainer,
        plans       = PLANS,
        config      = CONFIG,
        fold        = fold,
        checkpoint  = checkpoint,
        device      = "cuda",
        save_probs  = save_probs,
    )


def run_mednext_model(ckpt_path, input_dir, output_dir, sw_batch=4):
    from run_mednext import predict_mednext
    import torch
    predict_mednext(
        ckpt_path  = ckpt_path,
        images_dir = input_dir,
        output_dir = output_dir,
        device     = torch.device("cuda"),
        sw_batch   = sw_batch,
    )


def process_chunk(chunk_cases, input_dir, scratch_dir, output_dir, sw_batch=4):
    os.makedirs(scratch_dir, exist_ok=True)

    # stage input
    print(f"\n[STAGE] Staging {len(chunk_cases)} cases...")
    staged_input = stage_chunk(input_dir, scratch_dir, chunk_cases)

    # run 8 nnUNet ensemble models (with save_probabilities)
    nnunet_out_dirs = []
    for idx, fold, checkpoint, trainer_path, needs_perm in MODELS[:8]:
        out_dir = os.path.join(scratch_dir, f"nnunet_out_idx{idx}")
        print(f"\n[NNUNET] idx={idx} fold={fold} chk={checkpoint}")
        run_nnunet_model(idx, fold, checkpoint, trainer_path, staged_input, out_dir, save_probs=True)
        nnunet_out_dirs.append(out_dir)

    # run run_3 ED-only nnUNet (no save_probabilities)
    run3_cfg = MODELS[8]  # idx=8, fold=4, checkpoint_best.pth, RUN3
    run3_out = os.path.join(scratch_dir, "run3_out")
    print(f"\n[NNUNET-ED] run_3 fold={run3_cfg[1]}")
    run_nnunet_model(
        run3_cfg[0], run3_cfg[1], run3_cfg[2], run3_cfg[3],
        staged_input, run3_out, save_probs=False
    )

    # run MedNeXt ED models
    mednext_out_dirs = []
    for i, ckpt in enumerate(ED_MEDNEXT_CKPTS):
        out_dir = os.path.join(scratch_dir, f"mednext_ed_{i}")
        print(f"\n[MEDNEXT] {os.path.basename(ckpt)}")
        run_mednext_model(ckpt, staged_input, out_dir, sw_batch=sw_batch)
        mednext_out_dirs.append(out_dir)

    # ensemble + ED override + postprocess per case
    print(f"\n[ENSEMBLE] Processing {len(chunk_cases)} cases...")
    needs_perm = [True] * 8
    tmp_masks  = os.path.join(scratch_dir, "final_masks")
    os.makedirs(tmp_masks, exist_ok=True)

    for case_id in chunk_cases:
        ref_nii = os.path.join(nnunet_out_dirs[0], f"{case_id}.nii.gz")

        # make mask (weighted ensemble + ED override)
        mask_path = make_mask(
            case_id           = case_id,
            model_output_dirs = nnunet_out_dirs,
            needs_perm        = needs_perm,
            snap058_dir       = mednext_out_dirs[0],
            snap014_dir       = mednext_out_dirs[1],
            run3_dir          = run3_out,
            output_dir        = tmp_masks,
            ref_nii_path      = ref_nii,
        )

        # postprocess → write directly to /output
        out_path = os.path.join(output_dir, f"{case_id}.nii.gz")
        postprocess_mask(mask_path, out_path)
        print(f"[OUT] {out_path}")

    # wipe scratch
    print(f"\n[WIPE] Cleaning scratch: {scratch_dir}")
    shutil.rmtree(scratch_dir)
    print("[WIPE] Done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    default="/input")
    parser.add_argument("--output",   default="/output")
    parser.add_argument("--sw_batch", type=int, default=4)
    args = parser.parse_args()

    set_nnunet_env()
    os.makedirs(args.output, exist_ok=True)

    all_cases = get_all_cases(args.input)
    print(f"Total cases: {len(all_cases)}")

    n_chunks = (len(all_cases) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"Chunks: {n_chunks} (chunk size: {CHUNK_SIZE})")

    for chunk_idx in range(n_chunks):
        chunk = get_chunk(all_cases, chunk_idx, CHUNK_SIZE)
        print(f"\n{'='*60}")
        print(f"CHUNK {chunk_idx+1}/{n_chunks} — {len(chunk)} cases")
        print(f"{'='*60}")
        scratch = os.path.join(SCRATCH_BASE, f"chunk_{chunk_idx}")
        process_chunk(chunk, args.input, scratch, args.output, sw_batch=args.sw_batch)

    print(f"\n[DONE] All chunks complete. Output: {args.output}")


if __name__ == "__main__":
    main()