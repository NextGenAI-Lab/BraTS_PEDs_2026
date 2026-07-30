"""
nnUNetTrainer_C3.py — Custom nnU-Net trainer for C3 config.

C3 = Standard baseline trainer for comparison.

Place this file at:
  /workspace/BRATS/synapse_env/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/variants/
"""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_C3(nnUNetTrainer):
    """
    C3: Standard trainer baseline.
    All settings identical to default nnUNetTrainer.
    """

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.print_to_log_file("C3 Trainer active: Standard nnU-Net trainer")
