"""
nnUNetTrainer_Run5_OversampleSnapshot.py
nnUNet default + 500 epochs + LR 0.02 + ET/CC/ED oversampling + fixed snapshot logic.
"""

import os
import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_Run5_OversampleSnapshot(nnUNetTrainer):

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 500
        self.initial_lr = 0.02
        self.oversample_foreground_percent = 0.67

        self._snapshot_start  = 300
        self._snapshot_count  = 0
        self._et_at_best      = -1.0
        self._cc_at_best      = -1.0
        self._old_best_ema    = -1.0

        self.print_to_log_file(
            "Run5_OversampleSnapshot: 500 epochs LR=0.02 ET/CC/ED oversample | "
            "snapshot after ep300 when ET or CC beats ET/CC at best-EMA epoch"
        )

    def get_dataloaders(self):
        tr_loader, val_loader = super().get_dataloaders()
        self._boost_etcced_locations(tr_loader)
        return tr_loader, val_loader

    def _boost_etcced_locations(self, dataloader):
        try:
            if not hasattr(dataloader, 'data') or dataloader.data is None:
                self.print_to_log_file("Oversample: no dataloader.data, skipping")
                return
            modified = 0
            for case_id, case_data in dataloader.data.items():
                if 'class_locations' not in case_data:
                    continue
                for label in [1, 3, 4]:  # ET=1, CC=3, ED=4
                    locs = case_data['class_locations'].get(label, None)
                    if locs is None or len(locs) == 0:
                        continue
                    if len(locs) > 500:
                        locs = locs[np.random.choice(len(locs), 500, replace=False)]
                    case_data['class_locations'][label] = np.tile(locs, (3, 1))
                modified += 1
            self.print_to_log_file(f"Oversample: boosted ET/CC/ED for {modified} cases")
        except Exception as e:
            self.print_to_log_file(f"Oversample ERROR: {e}")
            import traceback
            self.print_to_log_file(traceback.format_exc())

    def _get_smoothed_et_cc(self):
        try:
            all_dice = self.logger.get_value('dice_per_class_or_region', step=None)
            if all_dice is None or len(all_dice) < 1:
                return None, None
            recent = all_dice[-3:]
            et_vals = [float(d[0]) for d in recent if len(d) >= 3]
            cc_vals = [float(d[2]) for d in recent if len(d) >= 3]
            if not et_vals or not cc_vals:
                return None, None
            return float(np.mean(et_vals)), float(np.mean(cc_vals))
        except Exception as e:
            self.print_to_log_file(f"Smoothing error: {e}")
            return None, None

    def on_epoch_end(self):
        super().on_epoch_end()

        try:
            current_ema = self.logger.get_value('ema_fg_dice', step=-1)
            if current_ema is None or self._best_ema is None:
                return

            # Detect new best EMA — record ET/CC at this epoch as reference
            if self._best_ema != self._old_best_ema:
                dice_now = self.logger.get_value('dice_per_class_or_region', step=-1)
                if dice_now is not None and len(dice_now) >= 3:
                    self._et_at_best   = float(dice_now[0])
                    self._cc_at_best   = float(dice_now[2])
                    self._old_best_ema = self._best_ema
                    self.print_to_log_file(
                        f"New best EMA={self._best_ema:.4f} at ep{self.current_epoch} | "
                        f"ET_at_best={self._et_at_best:.4f} CC_at_best={self._cc_at_best:.4f}"
                    )

            if self.current_epoch < self._snapshot_start:
                return

            if self._et_at_best < 0 or self._cc_at_best < 0:
                self.print_to_log_file(
                    f"Snapshot skip ep{self.current_epoch}: no best reference yet"
                )
                return

            smooth_et, smooth_cc = self._get_smoothed_et_cc()
            if smooth_et is None or smooth_cc is None:
                return

            mean_ok   = current_ema >= (self._best_ema - 0.02)
            et_better = smooth_et > self._et_at_best
            cc_better = smooth_cc > self._cc_at_best

            if mean_ok and (et_better or cc_better):
                reason = []
                if et_better:
                    reason.append(f"ET={smooth_et:.4f}>ref={self._et_at_best:.4f}")
                if cc_better:
                    reason.append(f"CC={smooth_cc:.4f}>ref={self._cc_at_best:.4f}")

                self._snapshot_count += 1
                fname = (
                    f"snapshot_ep{self.current_epoch:04d}"
                    f"_ema{current_ema:.4f}"
                    f"_et{smooth_et:.4f}"
                    f"_cc{smooth_cc:.4f}.pth"
                )
                self.save_checkpoint(os.path.join(self.output_folder, fname))
                self.print_to_log_file(
                    f"Snapshot #{self._snapshot_count}: {fname} | "
                    f"reason=[{' + '.join(reason)}] | "
                    f"best_ema={self._best_ema:.4f} current_ema={current_ema:.4f}"
                )
            else:
                if not mean_ok:
                    self.print_to_log_file(
                        f"Snapshot skip ep{self.current_epoch}: "
                        f"ema {current_ema:.4f} < best {self._best_ema:.4f} - 0.02"
                    )
                else:
                    self.print_to_log_file(
                        f"Snapshot skip ep{self.current_epoch}: "
                        f"ET={smooth_et:.4f}(ref={self._et_at_best:.4f}) "
                        f"CC={smooth_cc:.4f}(ref={self._cc_at_best:.4f}) no improvement"
                    )

        except Exception as e:
            self.print_to_log_file(f"Snapshot ERROR ep{self.current_epoch}: {e}")
            import traceback
            self.print_to_log_file(traceback.format_exc())
