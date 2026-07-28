# /workspace/BRATS/docker_build/scripts/stage_input.py

import os
import shutil
import argparse

MODALITY_SUFFIXES = {
    "t1n": "0000",
    "t1c": "0001",
    "t2w": "0002",
    "t2f": "0003",    
}

def get_all_cases(input_dir):
    cases = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])
    return cases

def get_chunk(all_cases, chunk_idx, chunk_size):
    start = chunk_idx * chunk_size
    end   = start + chunk_size
    return all_cases[start:end]

def stage_chunk(input_dir, scratch_dir, cases):
    staged = os.path.join(scratch_dir, "nnunet_input")
    os.makedirs(staged, exist_ok=True)
    for case_id in cases:
        case_dir = os.path.join(input_dir, case_id)
        for modality, channel in MODALITY_SUFFIXES.items():
            src = os.path.join(case_dir, f"{case_id}-{modality}.nii.gz")
            dst = os.path.join(staged, f"{case_id}_{channel}.nii.gz")
            if not os.path.exists(src):
                raise FileNotFoundError(f"Missing: {src}")
            shutil.copy2(src, dst)
    print(f"Staged {len(cases)} cases to {staged}")
    return staged

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True)
    parser.add_argument("--scratch",    required=True)
    parser.add_argument("--chunk_idx",  type=int, required=True)
    parser.add_argument("--chunk_size", type=int, default=100)
    args = parser.parse_args()

    all_cases = get_all_cases(args.input)
    print(f"Total cases found: {len(all_cases)}")
    chunk = get_chunk(all_cases, args.chunk_idx, args.chunk_size)
    print(f"Chunk {args.chunk_idx}: {len(chunk)} cases ({chunk[0]} ... {chunk[-1]})")
    stage_chunk(args.input, args.scratch, chunk)