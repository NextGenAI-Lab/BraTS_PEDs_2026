# /workspace/BRATS/docker_build/scripts/run_nnunet.py

import os
import subprocess
import argparse

PYTHON = "/workspace/BRATS/synapse_env/bin/python"
NNUNET_BIN = "/workspace/BRATS/synapse_env/bin/nnUNetv2_predict"

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
    save_probs=True,        # add this
):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        NNUNET_BIN,
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
        cmd.append("--save_probabilities")   # add this

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--dataset_id", default="001")
    parser.add_argument("--trainer",    required=True)
    parser.add_argument("--plans",      default="nnUNetPlans")
    parser.add_argument("--config",     default="3d_fullres")
    parser.add_argument("--fold",       type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device",     default="cuda")
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
    )