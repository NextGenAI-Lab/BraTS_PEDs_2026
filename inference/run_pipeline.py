# inference/run_pipeline.py

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stage_input import get_all_cases, get_chunk, stage_chunk
from maskify import make_mask
from postprocess import postprocess_mask
from model_registry import (
    MODELS, MEDNEXT_SNAP058, MEDNEXT_MODEL_ID,
    CHUNK_SIZE, PLANS, CONFIG, DATASET_ID,
    RUN5, RUN3,
)

def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def set_nnunet_env(nnunet_results: str):
    os.environ['nnUNet_raw']          = '/tmp/nnunet_raw'
    os.environ['nnUNet_preprocessed'] = '/tmp/nnunet_preprocessed'
    os.environ['nnUNet_results']      = nnunet_results


def run_nnunet_model(idx, fold, checkpoint, trainer_path, input_dir, output_dir, save_probs=True, nnunet_bin="nnUNetv2_predict", device="cuda"):
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
        device      = device,
        save_probs  = save_probs,
        nnunet_bin  = nnunet_bin,
    )


def run_mednext_model(ckpt_path, input_dir, output_dir, sw_batch=4, device="cuda"):
    from run_mednext import predict_mednext
    import torch
    predict_mednext(
        ckpt_path  = ckpt_path,
        images_dir = input_dir,
        output_dir = output_dir,
        device     = torch.device(device),
        sw_batch   = sw_batch,
    )


def process_chunk(chunk_cases, input_dir, scratch_dir, output_dir, nnunet_results, ed_mednext_ckpts, sw_batch=4, nnunet_bin="nnUNetv2_predict", device="cuda", log=None):
    os.makedirs(scratch_dir, exist_ok=True)

    # stage input
    if log: log.info(f"\n[STAGE] Staging {len(chunk_cases)} cases...")
    staged_input = stage_chunk(input_dir, scratch_dir, chunk_cases)

    # run 8 nnUNet ensemble models (with save_probabilities)
    nnunet_out_dirs = []
    for idx, fold, checkpoint, trainer_path, needs_perm in MODELS[:8]:
        # Override the trainer path with the provided nnunet_results path
        trainer_rel = os.path.basename(trainer_path)
        actual_trainer_path = os.path.join(nnunet_results, f"Dataset{DATASET_ID}_BraTSPEDs", trainer_rel)
        out_dir = os.path.join(scratch_dir, f"nnunet_out_idx{idx}")
        if log: log.info(f"\n[NNUNET] idx={idx} fold={fold} chk={checkpoint}")
        run_nnunet_model(idx, fold, checkpoint, actual_trainer_path, staged_input, out_dir, save_probs=True, nnunet_bin=nnunet_bin, device=device)
        nnunet_out_dirs.append(out_dir)

    # run run_3 ED-only nnUNet (no save_probabilities)
    run3_cfg = MODELS[8]  # idx=8, fold=4, checkpoint_best.pth, RUN3
    run3_trainer_rel = os.path.basename(run3_cfg[3])
    actual_run3_trainer_path = os.path.join(nnunet_results, f"Dataset{DATASET_ID}_BraTSPEDs", run3_trainer_rel)
    run3_out = os.path.join(scratch_dir, "run3_out")
    if log: log.info(f"\n[NNUNET-ED] run_3 fold={run3_cfg[1]}")
    run_nnunet_model(
        run3_cfg[0], run3_cfg[1], run3_cfg[2], actual_run3_trainer_path,
        staged_input, run3_out, save_probs=False, nnunet_bin=nnunet_bin, device=device
    )

    # run MedNeXt ED models
    mednext_out_dirs = []
    for i, ckpt in enumerate(ed_mednext_ckpts):
        out_dir = os.path.join(scratch_dir, f"mednext_ed_{i}")
        if log: log.info(f"\n[MEDNEXT] {os.path.basename(ckpt)}")
        run_mednext_model(ckpt, staged_input, out_dir, sw_batch=sw_batch, device=device)
        mednext_out_dirs.append(out_dir)

    # ensemble + ED override + postprocess per case
    if log: log.info(f"\n[ENSEMBLE] Processing {len(chunk_cases)} cases...")
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
        if log: log.info(f"[OUT] {out_path}")

    # wipe scratch
    if log: log.info(f"\n[WIPE] Cleaning scratch: {scratch_dir}")
    shutil.rmtree(scratch_dir)
    if log: log.info("[WIPE] Done")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the BraTS 2026 inference pipeline.")
    parser.add_argument("--input", required=True, help="Input directory containing case folders.")
    parser.add_argument("--output", required=True, help="Output directory for final segmentations.")
    parser.add_argument("--nnunet_results", required=True, help="Path to nnUNet_results directory.")
    parser.add_argument("--mednext_ckpts", required=True, nargs="+", help="Paths to MedNeXt checkpoint files.")
    parser.add_argument("--scratch_base", default="/tmp/brats_scratch", help="Base directory for scratch data.")
    parser.add_argument("--sw_batch", type=int, default=4, help="Sliding window batch size for MedNeXt.")
    parser.add_argument("--nnunet_bin", default="nnUNetv2_predict", help="Path or command for nnUNetv2_predict.")
    parser.add_argument("--device", default="cuda", help="Device to use for inference (e.g., cuda, cpu).")
    parser.add_argument("--log_dir", default="logs", help="Directory to save logs.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    log_dir = Path(args.log_dir)
    log = setup_logging(log_dir / "run_pipeline.log")

    set_nnunet_env(args.nnunet_results)
    os.makedirs(args.output, exist_ok=True)

    all_cases = get_all_cases(args.input)
    log.info(f"Total cases: {len(all_cases)}")

    n_chunks = (len(all_cases) + CHUNK_SIZE - 1) // CHUNK_SIZE
    log.info(f"Chunks: {n_chunks} (chunk size: {CHUNK_SIZE})")

    for chunk_idx in range(n_chunks):
        chunk = get_chunk(all_cases, chunk_idx, CHUNK_SIZE)
        log.info(f"\n{'='*60}")
        log.info(f"CHUNK {chunk_idx+1}/{n_chunks} — {len(chunk)} cases")
        log.info(f"{'='*60}")
        scratch = os.path.join(args.scratch_base, f"chunk_{chunk_idx}")
        process_chunk(
            chunk, 
            args.input, 
            scratch, 
            args.output, 
            args.nnunet_results, 
            args.mednext_ckpts, 
            sw_batch=args.sw_batch, 
            nnunet_bin=args.nnunet_bin, 
            device=args.device,
            log=log
        )

    log.info(f"\n[DONE] All chunks complete. Output: {args.output}")


if __name__ == "__main__":
    main()