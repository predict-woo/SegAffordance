import os
import typing
from typing import Any
import warnings

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset

from config.opd_train import Config, LossParams, ModelParams, OptimizerParams
from datasets.opdreal_datamodule import OPDRealDataModule
from model.losses import TwistLoss, build_geometric_loss
from model.losses.geometric import ScrewConsistencyLoss, TrajectoryProjectionLoss
from model.losses.split import axis_direction_loss, origin_canonical_loss
from model.segmenter import CRIS
from model.targets import StepTargets, unpack_batch
from utils.tools import DiceBCELoss, MotionVAELoss, create_composite_visualization, make_gaussian_map

torch.set_float32_matmul_precision("high")

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- PyTorch Lightning Module for OPDReal --- #
class OPDRealTrainingModule(pl.LightningModule):
    def __init__(
        self,
        model_params: ModelParams,
        loss_params: LossParams,
        optimizer_params: OptimizerParams,
        config: Config,
        finetune_from_path: typing.Optional[str] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_params = model_params
        self.loss_params = loss_params
        self.optimizer_params = optimizer_params
        self.config = config
        self.freeze_backbone = freeze_backbone

        # Inputs are fixed-size (256², word_len 77), so autotuned cudnn
        # algorithm selection always pays for itself.
        torch.backends.cudnn.benchmark = True

        self.model = CRIS(model_params)
        if getattr(model_params, "compile_model", False):
            # In-place compile keeps state_dict keys unprefixed (no
            # _orig_mod.), so checkpoints stay loadable everywhere.
            self.model.compile()

        self.mask_loss_fn = DiceBCELoss(
            bce_weight=self.loss_params.bce_weight,
            dice_weight=self.loss_params.dice_weight,
        )
        self.point_map_loss_fn = nn.BCEWithLogitsLoss()
        self.coord_loss_fn = nn.L1Loss()
        self.vae_loss_fn = MotionVAELoss(beta=self.loss_params.vae_beta)
        self.motion_type_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.trajectory_loss_fn = nn.MSELoss()
        # "cross_gt" (default, historical) | "pred_pred" | "none".
        # Each variant no-ops on batches lacking the targets it needs, so OPD
        # runs are unaffected whichever is selected.
        self.geometric_loss = build_geometric_loss(self.loss_params)
        # No-ops unless the model has a twist head AND the batch carries a 3D
        # origin (SF3D), same convention as the geometric losses.
        self.traj_projection_loss = TrajectoryProjectionLoss(
            weight=getattr(self.loss_params, "trajectory_proj_weight", 0.0)
        )
        self.twist_loss = TwistLoss(
            weight=getattr(self.loss_params, "twist_weight", 0.5),
            sign_agnostic=getattr(self.loss_params, "twist_sign_agnostic", True),
            metric_rho=getattr(self.loss_params, "twist_metric_rho", 0.25),
            trajectory_weight=getattr(self.loss_params, "trajectory_weight", 0.5),
            hyp_ce_weight=getattr(self.loss_params, "twist_hyp_ce_weight", 0.1),
        )

        if finetune_from_path:
            self.load_finetune_weights(finetune_from_path)

        self.train_dataset: typing.Optional[Dataset] = None
        self.val_dataset: typing.Optional[Dataset] = None
        self.val_vis_samples: typing.Optional[list] = None

        # Visualization reservoir sampling config/state
        self.vis_num_samples: int = self.config.val_vis_samples
        self.base_seed: int = self.config.manual_seed
        self._vis_buffer: typing.List[typing.Tuple] = []
        self._vis_seen_count: int = 0
        self._vis_rng: typing.Optional[torch.Generator] = None

        # Test accumulators
        self._test_ious: typing.List[float] = []
        self._test_point_errors: typing.List[float] = []
        self._test_axis_errors_matched: typing.List[float] = []
        self._test_num_matched: int = 0
        self._test_correct_axis_predictions: int = 0
        self._test_correct_type_in_matched: int = 0
        self._test_correct_pdet_ma: int = 0
        self._test_correct_pdet_mao: int = 0
        self._test_origin_errors_matched: typing.List[float] = []
        self._test_correct_origin_predictions: int = 0
        self._test_num_rotational_matched: int = 0
        self._test_debug_print_count: int = 0
        self.indices_to_visualize: typing.Optional[set] = None

    def forward(
        self, img, depth, tokenized_word, mask_condition, point_condition, motion_gt,
        motion_type_input=None,
    ):
        return self.model(
            img, depth, tokenized_word, mask_condition, point_condition, motion_gt,
            motion_type_input,
        )

    def _common_step(self, batch, batch_idx, step_type="train"):
        img, depth, word_str_list, targets = unpack_batch(batch)
        mask_gt = targets.mask
        point_gt_norm = targets.point_norm
        motion_gt = targets.motion
        motion_type_gt = targets.motion_type

        tokenized_words = self.model.tokenize(
            list(word_str_list), self.model_params.word_len
        ).to(self.device)

        # Auxiliary type hint with conditioning dropout: during training each
        # sample's GT type is replaced by the NULL token with probability
        # motion_type_input_dropout; at val the hint is withheld entirely
        # (None -> NULL for all), so val metrics and checkpoint selection
        # reflect the no-hint deployment condition.
        motion_type_input = None
        if (
            getattr(self.model, "use_motion_type_input", False)
            and step_type == "train"
            and motion_type_gt is not None
        ):
            types = motion_type_gt.to(self.device).long()
            null_index = self.model.motion_type_embedding.num_embeddings - 1
            drop_p = getattr(self.model_params, "motion_type_input_dropout", 0.5)
            dropped = torch.rand(types.shape, device=types.device) < drop_p
            motion_type_input = torch.where(
                dropped, torch.full_like(types, null_index), types
            )

        outputs = self(
            img, depth, tokenized_words, mask_gt, point_gt_norm, motion_gt,
            motion_type_input,
        )
        mask_pred_logits = outputs.mask_logits
        point_pred_logits = outputs.point_logits
        coords_hat = outputs.coords_hat
        motion_pred = outputs.motion_pred
        motion_type_logits = outputs.motion_type_logits
        mu, log_var = outputs.mu, outputs.log_var

        H_map, W_map = mask_pred_logits.shape[-2:]
        mask_gt_float = mask_gt.float().to(mask_pred_logits.device)
        mask_gt_downsampled = F.interpolate(
            mask_gt_float, size=(H_map, W_map), mode="bilinear", align_corners=False
        )

        L_mask = self.mask_loss_fn(mask_pred_logits, mask_gt_downsampled)

        zero = torch.zeros((), device=mask_pred_logits.device)
        if point_pred_logits is not None:
            point_gt_heatmap = make_gaussian_map(
                point_gt_norm, H_map, W_map,
                sigma=self.loss_params.point_sigma,
                device=point_pred_logits.device,
            )
            L_point_map = self.point_map_loss_fn(point_pred_logits, point_gt_heatmap)
            L_coord = self.coord_loss_fn(
                coords_hat, point_gt_norm.to(coords_hat.device)
            )
        else:
            # 3D point mode: the heatmap channel does not exist; the direct
            # 3D term below replaces both. Zero-valued (not absent) so the
            # CSV logger keeps stable columns.
            L_point_map, L_coord = zero, zero

        if motion_pred is None:
            # Axis head off (use_motion_head: false, twist arms): the twist
            # head carries the axis; no axis loss.
            L_vae, L_recon, L_kld = zero, zero, zero
        elif getattr(self.model, "use_cvae", True):
            L_vae, L_recon, L_kld = self.vae_loss_fn(
                motion_pred, motion_gt.to(motion_pred.device), mu, log_var
            )
        else:
            L_motion = axis_direction_loss(
                motion_pred, motion_gt.to(motion_pred.device),
                sign_agnostic=getattr(self.loss_params, "axis_sign_agnostic", True),
            )
            L_vae, L_recon, L_kld = L_motion, L_motion, zero

        if motion_type_logits is not None:
            L_motion_type = self.motion_type_loss_fn(
                motion_type_logits, motion_type_gt.to(motion_type_logits.device)
            )
        else:
            # Type head off (use_motion_type_head: false)
            L_motion_type = zero

        total_loss = (
            (self.loss_params.mask_weight * L_mask)
            + (self.loss_params.point_map_weight * L_point_map)
            + (self.loss_params.coord_weight * L_coord)
            + (self.loss_params.vae_weight * L_vae)
            + (self.loss_params.motion_type_weight * L_motion_type)
        )
        # Weighted terms collected for the optional per-term gradient-norm
        # diagnostic (Config.log_term_grad_norm_interval_steps).
        grad_terms = {
            "mask": self.loss_params.mask_weight * L_mask,
            "point_map": self.loss_params.point_map_weight * L_point_map,
            "coord": self.loss_params.coord_weight * L_coord,
        }

        # gen-6 split arm: direct 3D interaction point (replaces the 2D
        # point_map+coord pair, gated above) and the canonical q* origin
        # target (revolute rows only). Both log a stable zero when the
        # heads/GT are absent, so CSV columns never change shape.
        L_point_3d = zero
        if outputs.point_3d_pred is not None and targets.trajectory is not None:
            anchor_gt = targets.trajectory.to(outputs.point_3d_pred.device)[:, 0]
            L_point_3d = F.mse_loss(outputs.point_3d_pred, anchor_gt)
        L_origin = zero
        if (
            outputs.origin_pred is not None
            and targets.motion_origin_3d is not None
            and targets.trajectory is not None
        ):
            dev = outputs.origin_pred.device
            L_origin = origin_canonical_loss(
                outputs.origin_pred,
                targets.motion_origin_3d.to(dev).float(),
                motion_gt.to(dev).float(),
                targets.trajectory.to(dev)[:, 0].float(),
                motion_type_gt,
            )
        total_loss = (
            total_loss
            + self.loss_params.point_3d_weight * L_point_3d
            + self.loss_params.origin_weight * L_origin
        )
        grad_terms["point_3d"] = self.loss_params.point_3d_weight * L_point_3d
        grad_terms["origin"] = self.loss_params.origin_weight * L_origin
        self.log(
            f"{step_type}/L_point_3d",
            L_point_3d,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_origin",
            L_origin,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        # 2D hand/contact track (2D pretraining). Made relative to its own
        # first point, matching how the head predicts and how the 3D
        # trajectory is handled below.
        if targets.trajectory_2d is not None and outputs.trajectory_2d_pred is not None:
            track_gt = targets.trajectory_2d.to(outputs.trajectory_2d_pred.device)
            track_gt_relative = track_gt - track_gt[:, 0:1, :]
            if targets.trajectory_2d_valid is not None:
                # Invalid points carry [0, 0] placeholders (off-frame /
                # behind-camera); regressing them would train the head toward
                # fabricated corners. The relative frame also needs a real
                # first point, so its validity gates the whole sample.
                valid = targets.trajectory_2d_valid.to(track_gt.device)
                m = (valid & valid[:, 0:1]).unsqueeze(-1).float()
                sq_err = (outputs.trajectory_2d_pred - track_gt_relative).pow(2) * m
                L_trajectory_2d = sq_err.sum() / (2.0 * m.sum()).clamp(min=1.0)
            else:
                L_trajectory_2d = self.trajectory_loss_fn(
                    outputs.trajectory_2d_pred, track_gt_relative
                )
            total_loss = total_loss + self.loss_params.trajectory_2d_weight * L_trajectory_2d
            self.log(
                f"{step_type}/L_trajectory_2d",
                L_trajectory_2d,
                on_step=(step_type == "train"),
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        # Unified screw-motion supervision (use_twist_head): annealed WTA over
        # the K articulation bundles — when active on this batch it ALSO
        # supervises the 3D trajectory inside its distortion (aux
        # "handled_traj"), so the standalone term below must be skipped to
        # avoid double-counting. Runs first because its winner weights gate
        # the screw consistency terms.
        twist_total, twist_terms, twist_aux = self.twist_loss(outputs, targets)
        total_loss = total_loss + twist_total
        grad_terms["twist_wta"] = twist_total
        for term_name, term_value in twist_terms.items():
            self.log(
                f"{step_type}/{term_name}",
                term_value,
                on_step=(step_type == "train"),
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
        if (
            step_type == "train"
            and twist_aux["winner"] is not None
            and outputs.twist_logits is not None
        ):
            self.log(
                "train/twist_T",
                float(self.twist_loss.temperature),
                on_step=True, on_epoch=False, logger=True,
            )
            win = F.one_hot(
                twist_aux["winner"], outputs.twist_logits.shape[1]
            ).float().mean(dim=0)
            for k in range(win.numel()):
                self.log(
                    f"train/wta_win_{k}", win[k],
                    on_step=False, on_epoch=True, logger=True, sync_dist=True,
                )

        if (
            targets.trajectory is not None
            and outputs.trajectory_pred is not None
            and not twist_aux["handled_traj"]
        ):
            # Convert GT trajectory to relative coordinates (first point at origin);
            # the model predicts relative, so they compare directly.
            trajectory_gt_device = targets.trajectory.to(outputs.trajectory_pred.device)
            trajectory_gt_relative = trajectory_gt_device - trajectory_gt_device[:, 0:1, :]

            L_trajectory = self.trajectory_loss_fn(
                outputs.trajectory_pred, trajectory_gt_relative
            )
            total_loss = total_loss + self.loss_params.trajectory_weight * L_trajectory
            grad_terms["trajectory"] = self.loss_params.trajectory_weight * L_trajectory
            self.log(
                f"{step_type}/L_trajectory",
                L_trajectory,
                on_step=(step_type == "train"),
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        # Geometric consistency between the motion/twist and trajectory heads.
        # Which scheme runs is set by loss_params.geometric_loss; every variant
        # no-ops when the batch lacks the targets it needs, so this is safe to
        # call unconditionally. The input depth map is passed so the screw
        # variant can lift the predicted 2D point to a metric anchor. The
        # screw variant takes the annealed WTA weights to gate its GT term.
        if isinstance(self.geometric_loss, ScrewConsistencyLoss):
            geometric_total, geometric_terms = self.geometric_loss(
                outputs, targets, depth, hyp_weights=twist_aux["weights"]
            )
        else:
            geometric_total, geometric_terms = self.geometric_loss(outputs, targets, depth)
        total_loss = total_loss + geometric_total
        grad_terms["screw"] = geometric_total
        proj_total, proj_terms = self.traj_projection_loss(outputs, targets, depth)
        total_loss = total_loss + proj_total
        geometric_terms = {**geometric_terms, **proj_terms}
        for term_name, term_value in geometric_terms.items():
            self.log(
                f"{step_type}/{term_name}",
                term_value,
                on_step=(step_type == "train"),
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        self.log(
            f"{step_type}/loss_total",
            total_loss,
            on_step=(step_type == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_mask",
            L_mask,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_point_map",
            L_point_map,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_coord",
            L_coord,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_vae_total",
            L_vae,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_motion_recon",
            L_recon,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_kld",
            L_kld,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"{step_type}/L_motion_type",
            L_motion_type,
            on_step=(step_type == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        # Smoke-run diagnostic: per-term gradient norms into the shared
        # backbone, to verify small-magnitude geometric terms are not drowned
        # by the high-floor mask/point_map gradients (WTA spec acceptance
        # check). One extra backward per term per logged step; off by default.
        interval = getattr(self.config, "log_term_grad_norm_interval_steps", 0)
        if (
            step_type == "train"
            and interval > 0
            and (self.global_step % interval) == 0
        ):
            params = [
                p for p in self.model.backbone.parameters() if p.requires_grad
            ]
            for name, term in grad_terms.items():
                if not (torch.is_tensor(term) and term.requires_grad):
                    continue
                grads = torch.autograd.grad(
                    term, params, retain_graph=True, allow_unused=True
                )
                sq = sum(
                    g.float().pow(2).sum() for g in grads if g is not None
                )
                if torch.is_tensor(sq):
                    self.log(
                        f"train/grad_norm/{name}", sq.sqrt(),
                        on_step=True, on_epoch=False, logger=True,
                    )
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.trainer.is_global_zero:
            return

        if (
            self.config.log_image_interval_steps > 0
            and (self.global_step + 1) % self.config.log_image_interval_steps == 0
        ):
            self.log_visualizations()

    def _wta_temperature(self, epoch: int) -> float:
        """Annealing schedule (WTA spec): exponential decay from T0 to 0.01
        over `twist_wta_anneal_frac` of max_epochs, then hard winner (0)."""
        T0 = getattr(self.loss_params, "twist_wta_T0", 10.0)
        frac = getattr(self.loss_params, "twist_wta_anneal_frac", 0.8)
        max_epochs = self.trainer.max_epochs if self.trainer is not None else None
        if not max_epochs or max_epochs <= 0:
            return 0.0
        anneal_epochs = max(1.0, frac * max_epochs)
        if epoch >= anneal_epochs:
            return 0.0
        t_min = 0.01
        return float(T0 * (t_min / T0) ** (epoch / anneal_epochs))

    def on_validation_epoch_start(self):
        # Val/test always score the hard winner (T = 0) so val losses are
        # comparable across epochs regardless of the annealing schedule.
        self.twist_loss.temperature = 0.0
        # Reset reservoir and RNG each epoch for reproducibility and variety
        self._vis_buffer = []
        self._vis_seen_count = 0
        self._vis_rng = torch.Generator(device="cpu").manual_seed(
            int(self.base_seed + int(self.current_epoch))
        )

    def on_test_epoch_start(self):
        self.twist_loss.temperature = 0.0

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, "val")
        self._collect_vis_samples_from_batch(batch)
        return loss

    def _frozen_modules(self):
        """The pretrained encoders only.

        For a ViT backbone the pyramid adapter lives inside self.model.backbone
        too, but it is randomly initialised and must always train — freezing it
        would leave the neck reading noise.
        """
        backbone = self.model.backbone
        if hasattr(backbone, "pretrained_modules"):
            return backbone.pretrained_modules()
        return [backbone]

    def _apply_backbone_freeze(self):
        if self.freeze_backbone:
            print("❄️ Freezing pretrained backbone towers (visual + text encoders).")
            for module in self._frozen_modules():
                for param in module.parameters():
                    param.requires_grad = False

    def on_train_epoch_start(self):
        # WTA annealing: the loss object carries the epoch's temperature
        # (val/test hooks reset it to 0 = hard winner).
        self.twist_loss.temperature = self._wta_temperature(int(self.current_epoch))
        # Keep a frozen backbone truly frozen: eval mode stops BatchNorm/dropout
        # updates that requires_grad=False alone would not prevent.
        if self.freeze_backbone:
            for module in self._frozen_modules():
                module.eval()

    def configure_optimizers(self):
        self._apply_backbone_freeze()
        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.optimizer_params.lr,
            weight_decay=self.optimizer_params.weight_decay,
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=self.optimizer_params.scheduler_milestones,
            gamma=self.optimizer_params.scheduler_gamma,
        )
        return [optimizer], [scheduler]

    # Dataset logic has been moved into a separate LightningDataModule

    def log_visualizations(self):
        if not isinstance(self.logger, WandbLogger):
            return
        if not self.val_vis_samples:
            return

        (
            img_list,
            depth_list,
            word_str_list,
            mask_gt_list,
            _bbox_list,
            point_gt_norm_list,
            motion_gt_list,
            motion_type_gt_list,
            original_image_size_list,
            trajectory_gt_list,
            camera_intrinsic_list,
        ) = zip(*self.val_vis_samples)

        img = torch.stack(img_list).to(self.device)
        depth = torch.stack(depth_list).to(self.device)
        mask_gt = torch.stack(mask_gt_list).to(self.device)
        point_gt_norm = torch.stack(point_gt_norm_list).to(self.device)
        motion_gt = torch.stack(motion_gt_list).to(self.device)

        tokenized_words = self.model.tokenize(
            list(word_str_list), self.model_params.word_len
        ).to(self.device)

        with torch.no_grad():
            outputs = self(img, depth, tokenized_words, mask_gt, point_gt_norm, motion_gt)
        mask_pred_logits = outputs.mask_logits
        point_pred_logits = outputs.point_logits
        coords_hat = outputs.coords_hat
        motion_pred = outputs.motion_pred
        motion_type_logits = outputs.motion_type_logits
        trajectory_pred = outputs.trajectory_pred

        try:
            log_data = {}
            for i in range(len(img)):
                img_sample = img[i]
                mask_gt_sample = mask_gt[i].float()
                point_gt_norm_sample = point_gt_norm[i]
                text_description = word_str_list[i]
                mask_pred_sigmoid_sample = torch.sigmoid(mask_pred_logits[i])
                # Split arm (point_prediction_3d): no 2D point head — skip
                # the heatmap panel, keep the rest of the composite.
                point_pred_sigmoid_sample = (
                    torch.sigmoid(point_pred_logits[i])
                    if point_pred_logits is not None
                    else None
                )

                composite_image = create_composite_visualization(
                    image_tensor=img_sample,
                    text_description=text_description,
                    mask_pred_sigmoid=mask_pred_sigmoid_sample,
                    point_gt_norm=point_gt_norm_sample,
                    point_pred_heatmap=point_pred_sigmoid_sample,
                    motion_gt=motion_gt[i],
                    motion_pred=(motion_pred[i] if motion_pred is not None
                                 else torch.zeros(3, device=img.device)),
                    motion_type_gt=motion_type_gt_list[i],
                    motion_type_pred_logits=(
                        motion_type_logits[i] if motion_type_logits is not None
                        else torch.zeros(2, device=img.device)
                    ),
                    trajectory_gt=trajectory_gt_list[i],
                    trajectory_pred=trajectory_pred[i],
                    camera_intrinsic=camera_intrinsic_list[i],
                    original_image_size=original_image_size_list[i],
                    depth_tensor=depth[i],
                )

                log_data[f"val_sample/{i}_sample"] = wandb.Image(composite_image)

            if isinstance(self.logger, WandbLogger):
                self.logger.experiment.log(log_data, step=self.global_step)

        except ImportError:
            print("wandb is not installed. Skipping image logging.")
        except Exception as e:
            print(f"Error logging images to wandb: {e}")

    def on_validation_epoch_end(self):
        # Only log on the main process
        if not getattr(self, "trainer", None) or not self.trainer.is_global_zero:
            return
        if self._vis_buffer:
            self.val_vis_samples = self._vis_buffer
            self.log_visualizations()
        # Release buffer
        self._vis_buffer = []

    def on_test_start(self):
        vis_indices = getattr(self.config, "test_vis_indices", None)
        if vis_indices and self.trainer.is_global_zero:
            self.indices_to_visualize = set(vis_indices)
        else:
            self.indices_to_visualize = None

    # --- Testing support ---
    def test_step(self, batch, batch_idx):
        (
            img,
            depth,
            word_str_list,
            mask_gt,
            bbox_gt,
            point_gt_norm,
            motion_gt,
            motion_type_gt,
            _img_size,
            category_id,
            composite_key,
        ) = batch[:11]
        
        
        
        # Unpack optional camera parameters if they exist
        camera_params_in_batch = len(batch) > 11
        if camera_params_in_batch:
            intrinsic_matrix, motion_origin_3d_gt = batch[11], batch[12]
        else:
            intrinsic_matrix, motion_origin_3d_gt = None, None

        tokenized_words = self.model.tokenize(
            list(word_str_list), self.model_params.word_len
        ).to(self.device)

        with torch.no_grad():
            outputs = self(img, depth, tokenized_words, None, None, None)
        mask_pred_logits = outputs.mask_logits
        point_pred_logits = outputs.point_logits
        coords_hat = outputs.coords_hat
        motion_pred = outputs.motion_pred
        motion_type_logits = outputs.motion_type_logits

        mask_pred_prob = torch.sigmoid(mask_pred_logits)
        mask_pred_upsampled = F.interpolate(
            mask_pred_prob, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False
        )
        if motion_type_logits is not None:
            pred_types = torch.argmax(motion_type_logits, dim=1)
        else:
            pred_types = torch.zeros(img.size(0), dtype=torch.long, device=img.device)

        batch_size = img.size(0)
        # Optional visualization settings
        do_vis = getattr(self.config, "test_visualize_debug", False)
        vis_dir = getattr(self.config, "test_vis_output_dir", "debug_visualizations")
        vis_max = int(getattr(self.config, "test_vis_max_images", 100))
        dm = self.trainer.datamodule
        origin_norm_diagonals = getattr(dm, "origin_norm_diagonals", None)

        for i in range(batch_size):
            # IoU metric
            if getattr(self.config, "test_match_metric", "mask") == "bbox":
                pred_mask_binary = (mask_pred_upsampled[i].squeeze() > self.config.test_pred_threshold).float()
                pred_bbox = self._mask_to_bbox(pred_mask_binary)
                iou_val = self._bbox_iou(pred_bbox.cpu(), bbox_gt[i].cpu()).item()
            else:
                pred_mask_binary = (mask_pred_upsampled[i] > self.config.test_pred_threshold).float()
                iou_val = self._mask_iou(pred_mask_binary, mask_gt[i]).item()
            self._test_ious.append(iou_val)

            # Point error
            point_err = torch.linalg.norm(coords_hat[i] - point_gt_norm[i]).item()
            self._test_point_errors.append(point_err)
            
            # Match gating
            if iou_val > self.config.test_iou_threshold:
                self._test_num_matched += 1

                # Axis error (degrees, direction-agnostic)
                axis_i = (motion_pred[i] if motion_pred is not None
                          else torch.zeros(3, device=motion_gt.device))
                axis_err = self._axis_error_deg(axis_i, motion_gt[i]).item()
                self._test_axis_errors_matched.append(axis_err)

                is_axis_correct = axis_err <= self.config.test_motion_threshold_deg
                if is_axis_correct:
                    self._test_correct_axis_predictions += 1

                is_type_correct = pred_types[i] == motion_type_gt[i]
                if is_type_correct:
                    self._test_correct_type_in_matched += 1

                is_origin_correct = False
                # For translational motion, origin check is passed by default
                if motion_type_gt[i] == 0:
                    is_origin_correct = True
                # For rotational motion, we must calculate the 3D error
                elif motion_type_gt[i] == 1 and camera_params_in_batch and origin_norm_diagonals is not None:
                    self._test_num_rotational_matched += 1
                    # 1. Get predicted 2D origin and depth map
                    pred_origin_norm = coords_hat[i].detach() # (x, y) in [0, 1]
                    depth_map = depth[i].squeeze() # (H, W)
                    H, W = depth_map.shape

                    # 2. Convert normalized coords to pixel coords
                    u, v = int(pred_origin_norm[0] * W), int(pred_origin_norm[1] * H)
                    
                    # 3. Sample depth value (with a small patch for robustness)
                    patch_size = 5
                    u_start, v_start = max(0, u - patch_size//2), max(0, v - patch_size//2)
                    u_end, v_end = min(W, u + patch_size//2 + 1), min(H, v + patch_size//2 + 1)
                    depth_patch = depth_map[v_start:v_end, u_start:u_end]
                    
                    # Filter out zero depth values which are often invalid
                    valid_depths = depth_patch[depth_patch > 0]
                    z_mm = valid_depths.mean().item() if valid_depths.numel() > 0 else depth_map[v, u].item()

                    # 4. Unproject 2D point + depth to 3D
                    if z_mm > 1e-6 and intrinsic_matrix is not None:
                        K = intrinsic_matrix[i].to(self.device)
                        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                        
                        x_cam_mm = (u - cx) * z_mm / fx
                        y_cam_mm = (v - cy) * z_mm / fy
                        pred_origin_3d_mm = torch.tensor([x_cam_mm, y_cam_mm, z_mm], device=self.device)
                        
                        # Convert prediction to meters
                        pred_origin_3d = pred_origin_3d_mm / 1000.0
                        
                        # 5. Calculate error against 3D GT origin
                        gt_origin_3d = motion_origin_3d_gt[i].to(self.device)
                        origin_error = torch.linalg.norm(pred_origin_3d - gt_origin_3d)

                        # Use 3D diagonal from JSON for normalization, adapting to format
                        json_format = getattr(dm, "origin_norm_json_format", "real")
                        if json_format == 'real':
                            lookup_key = category_id[i].item()
                        else: # 'multi'
                            lookup_key = composite_key[i]
                        
                        
                        diagonal_3d = origin_norm_diagonals.get(lookup_key)

                        if diagonal_3d is not None and diagonal_3d > 1e-6:
                            norm_dist = origin_error / diagonal_3d
                            self._test_origin_errors_matched.append(norm_dist.item())
                            origin_threshold = getattr(self.config, "test_origin_threshold", 0.1)
                            if norm_dist <= origin_threshold:
                                is_origin_correct = True
                                self._test_correct_origin_predictions += 1

                if is_type_correct and is_axis_correct:
                    self._test_correct_pdet_ma += 1
                
                if is_type_correct and is_axis_correct and is_origin_correct:
                    self._test_correct_pdet_mao += 1


            # Debug visualization
            if do_vis and self.trainer.is_global_zero:
                current_sample_index: Any = batch_idx * batch_size + i
                if self.indices_to_visualize is None or current_sample_index in self.indices_to_visualize:
                    vis_dir = getattr(
                        self.config, "test_vis_output_dir", "opdreal_debug_visualizations"
                    )
                    point_pred_prob = torch.sigmoid(point_pred_logits)

                    self._save_opdreal_test_debug_visualizations(
                        image_tensor=img[i].detach().cpu(),
                        point_pred_prob_tensor=point_pred_prob[i].detach().cpu(),
                        mask_pred_prob_tensor=mask_pred_prob[i].detach().cpu(),
                        motion_pred=(
                            motion_pred[i] if motion_pred is not None
                            else torch.zeros(3)
                        ).detach().cpu(),
                        pred_motion_type=int(pred_types[i].item()),
                        gt_point_norm=point_gt_norm[i].detach().cpu(),
                        gt_mask_tensor=mask_gt[i].detach().cpu(),
                        gt_motion=motion_gt[i].detach().cpu(),
                        description=word_str_list[i],
                        output_dir=vis_dir,
                        sample_index=current_sample_index,
                    )

        return {}

    def on_test_epoch_end(self):
        dm = getattr(self.trainer, "datamodule", None) if hasattr(self, "trainer") else None
        test_ds = getattr(dm, "test_dataset", None) if dm is not None else None
        total_predictions = len(test_ds) if test_ds is not None else 0
        mean_iou = float(torch.tensor(self._test_ious).mean().item()) if self._test_ious else 0.0
        mean_point_error = float(torch.tensor(self._test_point_errors).mean().item()) if self._test_point_errors else 0.0
        err_adir = float(torch.tensor(self._test_axis_errors_matched).mean().item()) if self._test_axis_errors_matched else 0.0
        mean_origin_error = float(torch.tensor(self._test_origin_errors_matched).mean().item()) if self._test_origin_errors_matched else 0.0

        if self._test_num_matched > 0:
            pass_rate_axis = 100.0 * self._test_correct_axis_predictions / self._test_num_matched
            pass_rate_type = 100.0 * self._test_correct_type_in_matched / self._test_num_matched
        else:
            pass_rate_axis = 0.0
            pass_rate_type = 0.0
        
        if self._test_num_rotational_matched > 0:
            pass_rate_origin = 100.0 * self._test_correct_origin_predictions / self._test_num_rotational_matched
        else:
            pass_rate_origin = 0.0

        if total_predictions > 0:
            p_det = 100.0 * self._test_num_matched / total_predictions
            pdet_m = 100.0 * self._test_correct_type_in_matched / total_predictions
            pdet_ma = 100.0 * self._test_correct_pdet_ma / total_predictions
            pdet_mao = 100.0 * self._test_correct_pdet_mao / total_predictions
        else:
            p_det = 0.0
            pdet_m = 0.0
            pdet_ma = 0.0
            pdet_mao = 0.0

        # Always log to progress bar/console; optionally to external logger
        self.log("test/mean_iou", mean_iou, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/p_det", p_det, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pdet_m", pdet_m, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pdet_ma", pdet_ma, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pdet_mao", pdet_mao, prog_bar=True, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/mean_point_error", mean_point_error, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/err_adir_deg", err_adir, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pass_rate_axis", pass_rate_axis, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pass_rate_type", pass_rate_type, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/mean_origin_error", mean_origin_error, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        self.log("test/pass_rate_origin", pass_rate_origin, prog_bar=False, logger=self.config.log_test_to_wandb, sync_dist=True)
        

        # Print concise summary
        if self.trainer.is_global_zero:
            print("\n--- Test Results ---")
            print(f"Total Samples: {total_predictions}")
            print(f"Mean IoU: {mean_iou:.4f}")
            print(f"PDet (IoU Pass Rate @ >{self.config.test_iou_threshold:.2f} IoU): {p_det:.2f}%")
            print(f"PDet+M (IoU + Motion Type): {pdet_m:.2f}%")
            print(f"PDet+MA (IoU + Motion Type + Axis): {pdet_ma:.2f}%")
            print(f"PDet+MAO (IoU + Motion Type + Axis + Origin): {pdet_mao:.2f}%")
            print(f"\n--- Detailed Stats ---")
            print(f"Mean Point Error (L2): {mean_point_error:.4f}")
            print(f"ERR_ADir (Mean Axis Error for matched): {err_adir:.2f} degrees")
            print(f"Pass Rate Axis (correct axis for matched): {pass_rate_axis:.2f}%")
            print(f"Pass Rate Type (correct type for matched): {pass_rate_type:.2f}%")
            origin_thresh_val = getattr(self.config, "test_origin_threshold", 0.1)
            print(f"Mean Origin Error (normalized, for matched rotational): {mean_origin_error:.4f}")
            print(f"Pass Rate Origin (correct origin for matched rotational @ <{origin_thresh_val:.2f} error): {pass_rate_origin:.2f}%")


        # Reset accumulators for potential further test runs
        self._test_ious.clear()
        self._test_point_errors.clear()
        self._test_axis_errors_matched.clear()
        self._test_num_matched = 0
        self._test_correct_axis_predictions = 0
        self._test_correct_type_in_matched = 0
        self._test_correct_pdet_ma = 0
        self._test_correct_pdet_mao = 0
        self._test_origin_errors_matched.clear()
        self._test_correct_origin_predictions = 0
        self._test_num_rotational_matched = 0
        self._test_debug_print_count = 0

    # --- Utilities for testing ---
    @staticmethod
    def _mask_to_bbox(mask: torch.Tensor) -> torch.Tensor:
        """mask: (H,W) float or bool tensor -> (x_min,y_min,x_max,y_max)"""
        mask_bool = mask.bool()
        if mask_bool.sum() == 0:
            return torch.zeros(4, device=mask.device)
        rows = torch.any(mask_bool, dim=1)
        cols = torch.any(mask_bool, dim=0)
        y_min, y_max = torch.where(rows)[0][[0, -1]]
        x_min, x_max = torch.where(cols)[0][[0, -1]]
        return torch.tensor([x_min, y_min, x_max, y_max], device=mask.device)

    @staticmethod
    def _bbox_iou(box_pred_xyxy: torch.Tensor, box_gt_xywh: torch.Tensor) -> torch.Tensor:
        box_gt_xyxy = torch.cat([box_gt_xywh[:2], box_gt_xywh[:2] + box_gt_xywh[2:]])
        xA = torch.max(box_pred_xyxy[0], box_gt_xyxy[0])
        yA = torch.max(box_pred_xyxy[1], box_gt_xyxy[1])
        xB = torch.min(box_pred_xyxy[2], box_gt_xyxy[2])
        yB = torch.min(box_pred_xyxy[3], box_gt_xyxy[3])
        interArea = torch.clamp(xB - xA, min=0) * torch.clamp(yB - yA, min=0)
        boxPredArea = (box_pred_xyxy[2] - box_pred_xyxy[0]) * (box_pred_xyxy[3] - box_pred_xyxy[1])
        boxGtArea = box_gt_xywh[2] * box_gt_xywh[3]
        return interArea / (boxPredArea + boxGtArea - interArea + 1e-7)

    @staticmethod
    def _mask_iou(mask_pred: torch.Tensor, mask_gt: torch.Tensor) -> torch.Tensor:
        mask_pred_bool = mask_pred.bool()
        mask_gt_bool = mask_gt.bool()
        intersection = torch.logical_and(mask_pred_bool, mask_gt_bool).sum()
        union = torch.logical_or(mask_pred_bool, mask_gt_bool).sum()
        return (intersection.float() + 1e-7) / (union.float() + 1e-7)

    @staticmethod
    def _axis_error_deg(pred_axis: torch.Tensor, gt_axis: torch.Tensor) -> torch.Tensor:
        pred_axis = pred_axis.squeeze()
        gt_axis = gt_axis.squeeze()
        gt_axis_norm = F.normalize(gt_axis, p=2, dim=-1)
        pred_axis_norm = F.normalize(pred_axis, p=2, dim=-1)
        cos_sim = torch.dot(gt_axis_norm, pred_axis_norm)
        cos_sim = torch.abs(cos_sim)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        axis_error_rad = torch.acos(cos_sim)
        return axis_error_rad * 180.0 / torch.tensor(3.141592653589793, device=pred_axis.device)

    # --- Visualization helper ---
    @staticmethod
    def _save_opdreal_test_debug_visualizations(
        image_tensor: torch.Tensor,
        point_pred_prob_tensor: torch.Tensor,
        mask_pred_prob_tensor: torch.Tensor,
        motion_pred: torch.Tensor,  # 3d vector
        pred_motion_type: int,
        gt_point_norm: torch.Tensor,
        gt_mask_tensor: torch.Tensor,
        gt_motion: torch.Tensor,
        description: str,
        output_dir: str,
        sample_index: int,
    ):
        import numpy as np
        import cv2
        from PIL import Image

        os.makedirs(output_dir, exist_ok=True)

        # 1. De-normalize image tensor
        img_np_norm = image_tensor.numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np_norm + mean
        img_np = np.clip(img_np, 0, 1)
        img_bgr = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        h, w, _ = img_bgr.shape

        # --- Common drawing components ---
        def apply_geo_annotations(vis_image, pred_px, pred_py, motion_pred_np):
            # Draw interaction point
            cv2.circle(
                vis_image,
                (pred_px, pred_py),
                radius=5,
                color=(255, 255, 255),
                thickness=-1,
            )

            # Draw motion arrow
            motion_xy = motion_pred_np[:2]
            motion_xy_norm = motion_xy / (np.linalg.norm(motion_xy) + 1e-8)
            arrow_length = 50
            arrow_end_x = pred_px + int(motion_xy_norm[0] * arrow_length)
            arrow_end_y = pred_py + int(motion_xy_norm[1] * arrow_length)
            cv2.arrowedLine(
                vis_image,
                (pred_px, pred_py),
                (arrow_end_x, arrow_end_y),
                (255, 255, 255),
                2,  # Increased thickness
            )
            return vis_image

        def apply_text_annotations(vis_image, desc, img_h, img_w):
            # Add caption with wrapping
            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            margin = 10
            text = f"Desc: {desc}"
            words = text.split(" ")
            lines = []
            current_line = words[0]
            for word in words[1:]:
                test_line = f"{current_line} {word}"
                (text_width, text_height), _ = cv2.getTextSize(
                    test_line, font, font_scale, font_thickness
                )
                if text_width > img_w - 2 * margin:
                    lines.append(current_line)
                    current_line = word
                else:
                    current_line = test_line
            lines.append(current_line)
            y = img_h - margin - (len(lines) - 1) * (text_height + margin)
            for i, line in enumerate(lines):
                line_y = y + i * (text_height + margin)
                (line_width, text_height), _ = cv2.getTextSize(
                    line, font, font_scale, font_thickness
                )
                cv2.rectangle(
                    vis_image,
                    (margin - 5, line_y - text_height - 5),
                    (margin + line_width + 5, line_y + 5),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    vis_image,
                    line,
                    (margin, line_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA,
                )
            return vis_image

        # --- Visualization 1: Point Heatmap ---
        point_pred_prob_np = point_pred_prob_tensor.float().numpy().squeeze()
        point_heatmap_resized = cv2.resize(
            point_pred_prob_np, (w, h), interpolation=cv2.INTER_LINEAR
        )
        point_heatmap_inverted = 1 - point_heatmap_resized
        point_heatmap_colored = cv2.applyColorMap(
            (point_heatmap_inverted * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        vis_image_point = cv2.addWeighted(img_bgr.copy(), 0.6, point_heatmap_colored, 0.4, 0)

        # Get interaction point from argmax of point heatmap
        (pred_py, pred_px) = np.unravel_index(
            np.argmax(point_heatmap_resized), point_heatmap_resized.shape
        )

        vis_image_point = apply_geo_annotations(
            vis_image_point, pred_px, pred_py, motion_pred.numpy()
        )
        out_path_point = os.path.join(
            output_dir, f"sample_{sample_index:06d}_point.png"
        )
        cv2.imwrite(out_path_point, vis_image_point)

        # --- Visualization 2: Mask Heatmap ---
        mask_prob_np = mask_pred_prob_tensor.float().numpy().squeeze()
        # Apply sigmoid sharpening
        k = 20  # Steepness factor
        sigmoid_mask = 1.0 / (1.0 + np.exp(-k * (mask_prob_np - 0.5)))
        mask_heatmap_resized = cv2.resize(
            sigmoid_mask, (w, h), interpolation=cv2.INTER_LINEAR
        )
        mask_heatmap_inverted = 1 - mask_heatmap_resized
        mask_heatmap_colored = cv2.applyColorMap(
            (mask_heatmap_inverted * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        vis_image_mask = cv2.addWeighted(img_bgr.copy(), 0.6, mask_heatmap_colored, 0.4, 0)
        vis_image_mask = apply_geo_annotations(
            vis_image_mask, pred_px, pred_py, motion_pred.numpy()
        )
        out_path_mask = os.path.join(
            output_dir, f"sample_{sample_index:06d}_mask.png"
        )
        cv2.imwrite(out_path_mask, vis_image_mask)

        # --- Visualization 3: Ground Truth ---
        vis_image_gt = img_bgr.copy()

        # Draw GT mask as an overlay
        gt_mask_np = gt_mask_tensor.float().numpy().squeeze()
        gt_mask_resized = cv2.resize(gt_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
        # Create a green overlay for the mask
        gt_mask_overlay = np.zeros_like(vis_image_gt, dtype=np.uint8)
        gt_mask_overlay[gt_mask_resized > 0.5] = (0, 200, 0)  # BGR green
        vis_image_gt = cv2.addWeighted(vis_image_gt, 1.0, gt_mask_overlay, 0.4, 0)

        # Get GT interaction point in pixel coordinates
        gt_px = int(gt_point_norm[0] * w)
        gt_py = int(gt_point_norm[1] * h)

        # Use the same annotation function for the GT visualization
        vis_image_gt = apply_geo_annotations(
            vis_image_gt, gt_px, gt_py, gt_motion.numpy()
        )
        vis_image_gt = apply_text_annotations(vis_image_gt, description, h, w)
        
        out_path_gt = os.path.join(
            output_dir, f"sample_{sample_index:06d}_gt.png"
        )
        cv2.imwrite(out_path_gt, vis_image_gt)

    def _collect_vis_samples_from_batch(self, batch):
        # Match the unpacking logic in _common_step
        trajectory_gt = None
        camera_intrinsic = None
        img, depth, word_str_list, mask_gt, bbox, point_gt_norm, motion_gt, motion_type_gt, img_size = [None] * 9

        if len(batch) >= 9:
             base_items = batch[:9]
             img, depth, word_str_list, mask_gt, bbox, point_gt_norm, motion_gt, motion_type_gt, img_size = base_items

        if len(batch) >= 13:  # SF3D with trajectory (15 adds the 2D columns)
             camera_intrinsic = batch[11]
             trajectory_gt = batch[12]

        batch_size = img.size(0) if hasattr(img, "size") else 0
        for i in range(batch_size):
            sample_trajectory = trajectory_gt[i].detach().cpu() if trajectory_gt is not None else None
            sample_intrinsic = camera_intrinsic[i].detach().cpu() if camera_intrinsic is not None else None

            sample = (
                img[i].detach().cpu(),
                depth[i].detach().cpu(),
                word_str_list[i],
                mask_gt[i].detach().cpu(),
                bbox[i].detach().cpu(),
                point_gt_norm[i].detach().cpu(),
                motion_gt[i].detach().cpu(),
                motion_type_gt[i].detach().cpu(),
                img_size[i].detach().cpu(),
                sample_trajectory,
                sample_intrinsic,
            )

            # True reservoir sampling for uniform random selection
            if self._vis_rng is None:
                self._vis_rng = torch.Generator(device="cpu").manual_seed(
                    int(self.base_seed + int(self.current_epoch))
                )
            
            if len(self._vis_buffer) < self.vis_num_samples:
                # Fill buffer first
                self._vis_buffer.append(sample)
            else:
                # Reservoir sampling: replace with probability k/n
                # where k = vis_num_samples, n = _vis_seen_count + 1
                j = int(torch.randint(0, self._vis_seen_count + 1, (1,), generator=self._vis_rng).item())
                if j < self.vis_num_samples:
                    self._vis_buffer[j] = sample
            
            self._vis_seen_count += 1

    def load_finetune_weights(self, checkpoint_path: str):
        print(f"🔩 Loading weights for finetuning from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Strip "model." prefix from keys to match the model's state_dict
        if any(key.startswith("model.") for key in state_dict.keys()):
            state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

        model_state_dict = self.model.state_dict()

        # Filter out unnecessary keys and load only matching ones
        pretrained_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_state_dict and v.size() == model_state_dict[k].size()
        }

        if not pretrained_dict:
            print("⚠️ Warning: No matching keys found in the pretrained model.")
            return

        model_state_dict.update(pretrained_dict)
        self.model.load_state_dict(model_state_dict)

        loaded_keys = pretrained_dict.keys()
        model_keys = model_state_dict.keys()

        unloaded_keys = [k for k in model_keys if k not in loaded_keys]

        if unloaded_keys:
            print(f"⚠️ Warning: The following keys were not loaded: {unloaded_keys}")
        else:
            print("✅ All model weights loaded successfully.")


if __name__ == "__main__":
    LightningCLI(OPDRealTrainingModule, OPDRealDataModule, save_config_callback=None)
