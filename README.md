<div align="center">
  
# NeuroNinjas: BraTS-PEDs 2026

**Two-Track Ensembling Pipeline for Pediatric Brain Tumor Segmentation**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![nnU-Net](https://img.shields.io/badge/nnU--Net-v2-green.svg)](https://github.com/MIC-DKFZ/nnUNet)

*Team: Nikunj Garg, Gaganpreet Singh, Aditya Kumar, Ankur Gupta | Affiliation: Netaji Subhas University of Technology, New Delhi*
</div>

---

## 📖 Overview

This repository contains the official methodology and implementation for the **NeuroNinjas** submission to the BraTS-PEDs 2026 challenge. Our pipeline focuses on highly accurate, multi-class pediatric brain tumor segmentation targeting the Enhancing Tumor (ET), Non-Enhancing Tumor / Cyst (NET/C), Cystic Component (CC), and Edema (ED) subregions.

Rather than providing a push-button "black box," this repository is designed as a **transparent, reproducible framework**. It provides the complete end-to-end infrastructure required for a researcher to prepare the BraTS dataset, train state-of-the-art segmentation models (nnU-Net and MedNeXt), and orchestrate a sophisticated multi-model ensemble. 

> **Note on Checkpoints:** We do not provide pre-trained weights. This framework is designed so that users train their own models, analyze their own validation logs, select their optimal checkpoints, and construct their own ensemble configuration based on empirical results.

---

## 🧠 Methodology & Pipeline Architecture

Our solution achieves high performance by treating different tumor subregions as distinct optimization problems. The pipeline is broken down into four core phases:

### 1. Stratified Data Preprocessing
Because specific pediatric tumor components (like ET and CC) are underrepresented or entirely absent in many training cases, random cross-validation often creates imbalanced folds. Our preprocessing pipeline offers two major advantages:
*   **Targeted Stratification:** Extracts volume statistics for every raw segmentation and generates a k-fold split based on tumor severity, ensuring balanced representations of rare classes.
*   **Patient-Level Leakage Prevention:** Automatically parses longitudinal scan data (`BraTS-PED-{patient_id}-{timepoint}`), grouping all timepoints for a single patient into a "patient profile." It locks all timepoints for that patient into the exact same fold, completely preventing data leakage across train/validation sets.

### 2. Diverse Model Training
The framework trains two distinct architectures to maximize functional diversity:
*   **Custom nnU-Net Ensembles (`Run5_Snapshot`):** We extended the standard nnU-Net v2 trainer to capture multiple "snapshots" during a single extended training run (500 epochs). Snapshots are triggered dynamically when a model reaches new peaks in EMA validation dice, providing a rich pool of checkpoints without retraining from scratch.
*   **MedNeXt with UpKern (`run7`):** To capture a larger spatial context, we utilize MedNeXt via a strict two-stage protocol: Stage 1 trains a kernel=3 seed, and Stage 2 expands to a kernel=5 architecture using UpKern initialization.

### 3. Generalized Region-Specific Optimization
Model fusion is only as strong as its constituent parts. **This repository is a flexible methodology, not a rigid script.** 
While our default configuration targets ET/CC and ED, users are expected to evaluate their snapshot pools and select checkpoints that peak on *their* target regions of interest (e.g., NET, WT, TC, etc.). You simply register your chosen checkpoints in `model_registry.py` and manually tune the `WEIGHTS` matrix to assign heavier voting power to the models that historically excel at your target subregions.

### 4. Two-Track Inference & Ensembling
Once checkpoints are registered, the inference orchestrator executes a two-track fusion strategy:
*   **Track 1 (Class-Specific Weighted Fusion):** Eight nnU-Net snapshots are run. Their raw softmax probability maps are merged using the user-defined weighting matrix. This allows arbitrary blending strategies depending on user objectives.
*   **Track 2 (ED Override):** As a demonstration of targeted regional intervention, we calculate a strict 3-way intersection across two heavy MedNeXt models and one baseline nnU-Net (`C3`). This consensus mask overrides the ED predictions from Track 1, drastically reducing false positives. Users can adapt this override philosophy for any region.

---

## 📁 Repository Structure

*   **`preprocessing/`**: Scripts to generate patient-stratified folds, parse raw NIfTI data into the strict nnU-Net `_0000` naming convention, and automatically execute the plan-and-preprocess workflow.
*   **`trainer/`**: Custom nnU-Net trainer extensions (`nnUNetTrainer_Run5_Snapshot` and `nnUNetTrainer_C3`).
*   **`mednext/`**: Standalone training scripts for MedNeXt Stage 1 (k=3) and Stage 2 (k=5 UpKern), with custom augmentation and Deep Supervision loss logic.
*   **`inference/`**: The end-to-end inference engine. Contains `run_pipeline.py` (orchestrator), `model_registry.py` (checkpoint configuration map), and the logic for weighted probability fusion and ED intersection override.

---

## ⚙️ Environment Setup

This pipeline was developed on an A100 40GB GPU using Python 3.12. We strongly recommend using a dedicated virtual environment or Docker container.

### 1. Install Base Dependencies
```bash
pip install torch torchvision torchaudio
pip install nibabel numpy scipy scikit-learn monai
```

### 2. Install nnU-Net v2
```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```
*Note: Ensure your `nnUNet_raw`, `nnUNet_preprocessed`, and `nnUNet_results` environment variables are properly exported in your shell profile before proceeding.*

### 3. Install MedNeXt
Our repository relies on MedNeXt for large-kernel feature extraction. You must install the nnU-Net compatible MedNeXt branch:
```bash
git clone https://github.com/MIC-DKFZ/MedNeXt.git
cd MedNeXt
pip install -e .
```
*(For more details, see the [official MedNeXt repository](https://github.com/MIC-DKFZ/MedNeXt)).*

---

## 🛠 Dataset Preparation & Preprocessing

The preprocessing pipeline takes your raw downloaded BraTS 2026 PEDs dataset and completely formats it for training.

### 1. Generate Stratified Folds
Analyzes raw tumor volumes and creates a smart, balanced cross-validation split with patient-level leakage prevention.
```bash
python preprocessing/create_folds.py \
    --data_dirs /path/to/raw/BraTS_PEDs_Training \
    --output_dir /path/to/save/splits \
    --num_folds 5
```

### 2. Prepare nnU-Net Raw Format
Symlinks the raw NIfTI files into nnU-Net's strict channel naming convention (`_0000` to `_0003`) and generates `dataset.json`.
```bash
python preprocessing/prepare_nnunet.py \
    --train_roots /path/to/raw/BraTS_PEDs_Training \
    --run_dir /path/to/workspace \
    --dataset_id 001
```

### 3. Execute Preprocessing & Inject Splits
Runs the heavy `nnUNetv2_plan_and_preprocess` command and automatically injects the stratified splits into the newly created folder.
```bash
python preprocessing/run_preprocess_pipeline.py \
    --dataset_id 001 \
    --nnunet_raw /path/to/workspace/nnUNet_raw \
    --nnunet_preprocessed /path/to/workspace/nnUNet_preprocessed \
    --nnunet_results /path/to/workspace/nnUNet_results \
    --splits_file /path/to/save/splits/splits_final_stratified.json \
    --num_workers 8
```

---

## 🚀 Training the Models

### nnU-Net Training
You must train the models using our custom trainers. To use them, copy the trainer files from `trainer/` into your nnU-Net installation's variants folder (e.g., `nnunetv2/training/nnUNetTrainer/variants/`).

**1. Ensemble Base (`Run5_Snapshot`):** 
Train this for 500 epochs to generate the diverse checkpoint pool for Track 1.
```bash
nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainer_Run5_Snapshot
# Repeat for folds 1, 2, 3, 4
```

**2. ED-Intersection Baseline (`C3`):**
```bash
nnUNetv2_train 001 3d_fullres 4 -tr nnUNetTrainer_C3
```

### MedNeXt Training
MedNeXt training uses raw NIfTI images directly. Provide the `nnUNet_raw` images path generated in preprocessing step 2.

**Stage 1 (Kernel=3 Seed):**
```bash
python mednext/train_mednext_run7_stage1_k3.py \
    --fold 0 \
    --run_dir /workspace/run_7 \
    --images_dir /workspace/nnUNet_raw/Dataset001_BraTSPEDs/imagesTr \
    --labels_dir /workspace/nnUNet_raw/Dataset001_BraTSPEDs/labelsTr \
    --splits_file /path/to/save/splits/splits_final_stratified.json
```

**Stage 2 (Kernel=5 UpKern):**
Initializes from Stage 1 weights.
```bash
python mednext/train_mednext_run7_stage2_k5_upkern.py \
    --fold 0 \
    --run_dir /workspace/run_7 \
    --images_dir /workspace/nnUNet_raw/Dataset001_BraTSPEDs/imagesTr \
    --labels_dir /workspace/nnUNet_raw/Dataset001_BraTSPEDs/labelsTr \
    --splits_file /path/to/save/splits/splits_final_stratified.json
```

---

## 🔍 Model Registry & Checkpoint Selection

This repository does **not** provide pre-trained push-button weights. 

After completing your training runs, you must evaluate your snapshot pools and select the checkpoints that perform best for specific subregions. 
1. Open `inference/model_registry.py`.
2. Update the `MODELS` list with the exact filenames of your chosen nnU-Net checkpoints.
3. Update the `WEIGHTS` matrix if you wish to adjust class-specific voting power based on your validation metrics.

---

## 🎯 Inference Pipeline

Once your `model_registry.py` is configured, you can run the final two-track ensembling pipeline. 

Make sure you pass the exact MedNeXt checkpoints you selected via the `--mednext_ckpts` argument (our internal `maskify` override script expects exactly two MedNeXt checkpoints).

```bash
python inference/run_pipeline.py \
    --input /path/to/unseen/test/images \
    --output /path/to/final_segmentations \
    --nnunet_results /path/to/workspace/nnUNet_results \
    --mednext_ckpts /workspace/run_7/fold_0/stage2_k5_upkern/checkpoint_best.pth /workspace/run_7/fold_1/stage2_k5_upkern/checkpoint_best.pth \
    --scratch_base /tmp/brats_scratch \
    --nnunet_bin nnUNetv2_predict
```

The orchestrator will automatically split cases into chunks, run all nnU-Net models, run the MedNeXt models, fuse the probability maps, override the ED masks via consensus intersection, and output the final `.nii.gz` predictions directly to your output directory.
