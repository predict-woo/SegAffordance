"""Field-model training (2026-09-13): the SF3D trainer's losses, metrics,
loss profiles and checkpoints, with model/field_model.py in place of CRIS
and one extra term — the dense depth-field loss (loss_params.depth_field_weight).
Multi-source data (SF3D + hand video) as train_multi_better.py.

    python train_field_better.py fit  --config config/field_joint4_l2anchor.yaml
    python train_field_better.py test --config <cfg> --ckpt_path <ckpt> ...

Legacy trainers / CRIS are untouched: this file only subclasses.
"""
import torch
import torch.nn.functional as F
from pytorch_lightning.cli import LightningCLI

from datasets.multisource_datamodule import MultiSourceDataModule
from model.field_model import FieldModel
from train_SF3D_better import SF3DTrainingModule


class _Recorder(torch.nn.Module):
    """Uncompiled shell around the (compiled) FieldModel that keeps the last
    ModelOutputs for the extra loss term — a plain-python side effect outside
    the compiled region. Attribute reads fall through to the core; the
    per-batch `dense_trunk_detach` flag is forwarded as a property so the
    parent trainer's `core.dense_trunk_detach = ...` reaches the model."""

    def __init__(self, core: FieldModel):
        super().__init__()
        self.core = core
        self.last = None

    def forward(self, *args, **kwargs):
        out = self.core(*args, **kwargs)
        self.last = out
        return out

    @property
    def dense_trunk_detach(self):
        return self.core.dense_trunk_detach

    @dense_trunk_detach.setter
    def dense_trunk_detach(self, value):
        self.core.dense_trunk_detach = bool(value)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("core"), name)


class FieldTrainingModule(SF3DTrainingModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # swap the model (the parent built and possibly compiled a CRIS; it is dropped here)
        core = FieldModel(self.model_params)
        if getattr(self.model_params, "compile_model", False):
            core.compile()
        self.model = _Recorder(core)

    @staticmethod
    def depth_field_loss(depth_field: torch.Tensor, depth_gt: torch.Tensor):
        """L1 on log-depth vs the depth map pooled to the field resolution
        (valid-aware average; a pooled cell is valid when > half its pixels are).
        Returns (loss, n_valid_cells)."""
        h, w = depth_field.shape[-2:]
        d = depth_gt.float()
        valid = ((d > 0.05) & (d < 20.0) & torch.isfinite(d)).float()
        d_sum = F.adaptive_avg_pool2d(d * valid, (h, w))
        v_frac = F.adaptive_avg_pool2d(valid, (h, w))
        target = d_sum / v_frac.clamp(min=1e-6)
        cell_ok = v_frac > 0.5
        n = int(cell_ok.sum())
        if n == 0:
            return depth_field.sum() * 0.0, 0
        err = (depth_field.float() - target.clamp(min=0.05).log()).abs()
        return err[cell_ok].mean(), n

    def _common_step_impl(self, img, depth, word_str_list, targets, batch_idx, step_type):
        total = super()._common_step_impl(img, depth, word_str_list, targets, batch_idx, step_type)
        w = float(getattr(self.loss_params, "depth_field_weight", 0.0))
        out = self.model.last
        self.model.last = None
        if w > 0 and out is not None and out.depth_field is not None and depth is not None:
            L, n = self.depth_field_loss(out.depth_field, depth.to(out.depth_field.device))
            if n > 0:
                total = total + w * L
            self.log(f"{step_type}/L_depth_field", L, on_step=(step_type == "train"), on_epoch=True,
                     logger=True, sync_dist=True)
        return total


if __name__ == "__main__":
    LightningCLI(FieldTrainingModule, MultiSourceDataModule, save_config_callback=None)
