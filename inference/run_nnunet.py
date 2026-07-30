# inference/run_nnunet.py

import os
import subprocess
import argparse
import logging
from pathlib import Path

def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)

def run_nnunet(
    input_dir,
    output_dir,
    dataset_id,
    trainer,
    plans,
    config,
    fold,
    checkpoint,
    device="cuda",
    save_probs=True,
    nnunet_bin="nnUNetv2_predict",
    log=None,
):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        nnunet_bin,
        "-i", input_dir,
        "-o", output_dir,
        "-d", dataset_id,
        "-tr", trainer,
        "-p", plans,
        "-c", config,
        "-f", str(fold),
        "-chk", checkpoint,
        "--disable_progress_bar",
        "-device", device,
    ]
    if save_probs:
        cmd.append("--save_probabilities")

    if log:
        log.info(f"Running: {' '.join(cmd)}")
    else:
        print(f"Running: {' '.join(cmd)}")
        
    result = subprocess.run(cmd, check=True)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run nnUNetv2 inference.")
    parser.add_argument("--input",      required=True, help="Input directory")
    parser.add_argument("--output",     required=True, help="Output directory")
    parser.add_argument("--dataset_id", default="001", help="Dataset ID")
    parser.add_argument("--trainer",    required=True, help="nnUNet Trainer class")
    parser.add_argument("--plans",      default="nnUNetPlans", help="Plans identifier")
    parser.add_argument("--config",     default="3d_fullres", help="Configuration name")
    parser.add_argument("--fold",       type=int, required=True, help="Fold to run")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file name")
    parser.add_argument("--device",     default="cuda", help="Device (e.g. cuda, cpu)")
    parser.add_argument("--nnunet_bin", default="nnUNetv2_predict", help="Path to nnUNetv2_predict binary")
    args = parser.parse_args()

    run_nnunet(
        input_dir   = args.input,
        output_dir  = args.output,
        dataset_id  = args.dataset_id,
        trainer     = args.trainer,
        plans       = args.plans,
        config      = args.config,
        fold        = args.fold,
        checkpoint  = args.checkpoint,
        device      = args.device,
        nnunet_bin  = args.nnunet_bin,
    )