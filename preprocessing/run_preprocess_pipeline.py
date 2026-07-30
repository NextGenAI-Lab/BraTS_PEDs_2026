import os
import shutil
import argparse
import subprocess
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-end nnU-Net preprocessing pipeline entry point.")
    parser.add_argument("--dataset_id", default="001", help="Dataset ID for nnU-Net (e.g. 001).")
    parser.add_argument("--nnunet_raw", required=True, help="Path to nnUNet_raw directory.")
    parser.add_argument("--nnunet_preprocessed", required=True, help="Path to nnUNet_preprocessed directory.")
    parser.add_argument("--nnunet_results", required=True, help="Path to nnUNet_results directory.")
    parser.add_argument("--splits_file", required=True, help="Path to the custom splits_final_stratified.json to inject.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for nnUNet preprocessing.")
    return parser.parse_args()

def main():
    args = parse_args()

    # Set environment variables required by nnU-Net
    os.environ['nnUNet_raw'] = args.nnunet_raw
    os.environ['nnUNet_preprocessed'] = args.nnunet_preprocessed
    os.environ['nnUNet_results'] = args.nnunet_results

    dataset_name = f"Dataset{args.dataset_id.zfill(3)}_BraTSPEDs"
    prep_dir = Path(args.nnunet_preprocessed) / dataset_name

    print("=" * 60)
    print(f"Running nnUNetv2_plan_and_preprocess for {dataset_name}")
    print("=" * 60)

    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(int(args.dataset_id)),
        "-c", "3d_fullres",
        "-np", str(args.num_workers)
    ]
    
    subprocess.run(cmd, check=True)

    print("\n" + "=" * 60)
    print("Injecting Custom Stratified Splits")
    print("=" * 60)
    
    splits_src = Path(args.splits_file)
    if not splits_src.exists():
        raise FileNotFoundError(f"Custom splits file not found: {splits_src}")

    splits_dst = prep_dir / "splits_final.json"
    prep_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(splits_src, splits_dst)
    
    print(f"Successfully copied custom splits to {splits_dst}")
    print("Preprocessing complete. The dataset is ready for training.")

if __name__ == "__main__":
    main()
