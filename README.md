\# NeuroNinjas — BraTS-PEDs 2026



\*\*Team:\*\* Nikunj Garg, Gaganpreet Singh, Aditya Kumar, Ankur Gupta  

\*\*Affiliation:\*\* Netaji Subhas University of Technology, New Delhi



\## Overview

Two-track ensembling pipeline for pediatric brain tumor segmentation

targeting ET, NET/C, CC, and ED subregions.



\## Repository Structure



| Folder | Description |

|---|---|

| `trainer/` | Custom nnU-Net snapshot trainer |

| `preprocessing/` | Stratified fold creation and nnU-Net dataset preparation |

| `mednext/` | MedNeXt training scripts (Stage 1 k=3, Stage 2 k=5 UpKern) |

| `inference/` | Full inference pipeline (Track 1 + Track 2 + post-processing) |



\## Requirements

\- nnU-Net v2

\- MedNeXt (nnU-Net compatible)

\- Python 3.12



\## Inference Pipeline

The full inference pipeline is in `inference/run\_pipeline.py`.

Track 1 (class-specific weighted fusion): `inference/ensemble\_accumulator.py`

Track 2 (ED three-way intersection): `inference/apply\_ed\_override.py`

Post-processing (CC only): `inference/postprocess.py`

