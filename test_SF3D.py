import os
import argparse
import typing  # For type hinting
import random

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset

import utils.config as config_loader  # Renamed to avoid conflict with cfg variable
from datasets.scenefun3d import (
    SF3DDataset,
    get_default_transforms,
    split_dataset_by_scene,
)
from utils.dataset import tokenize  # For tokenizing text descriptions
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Import the base module from the training script
from train_SF3D import SF3DTrainingModule as BaseSF3DTrainingModule
from train_SF3D import make_gaussian_map

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To suppress tokenizer warnings if any


# --- Helper Dataset Wrapper --- #
class SplitAwareDataset(Dataset):
    """
    A wrapper for a dataset that adds the split name to each sample.
    This is used to identify the split (e.g., 'train', 'val') in the test_step.
    """

    def __init__(self, dataset: Subset, split_name: str):
        self.dataset = dataset
        self.split_name = split_name

    def __getitem__(self, idx):
        # The base dataset returns a tuple of (img, word, mask, point)
        # We append the split name to this tuple.
        return (*self.dataset[idx], self.split_name)

    def __len__(self):
        return len(self.dataset)


# --- PyTorch Lightning Test Module --- #


class SF3DTestModule(BaseSF3DTrainingModule):
    """
    A dedicated LightningModule for testing, inheriting core logic from the training module.
    This module handles the specifics of the test step, such as collecting samples
    and generating visualizations.
    """

    def __init__(self, cfg: typing.Dict):
        # The base module handles hparams saving and config setup
        super().__init__(cfg)
        # For collecting test samples for visualization
        self.test_samples = []

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        """Collect test samples for visualization"""
        # Unpack the batch, which now includes the split name
        img, word_str_list, mask_gt, point_gt_norm, split_list = batch

        # Move data to the correct device
        img = img.to(self.device)
        mask_gt = mask_gt.to(self.device)
        point_gt_norm = point_gt_norm.to(self.device)

        tokenized_words = tokenize(word_str_list, self.cfg.word_len, truncate=True).to(
            self.device
        )

        # Perform inference
        with torch.no_grad():
            # The forward pass is inherited from the base module
            mask_pred_logits, point_pred_logits, coords_hat = self(
                img, tokenized_words, mask_gt, point_gt_norm
            )

        # Store samples for visualization.
        # This assumes a batch size of 1, as configured in test_dataloader.
        if img.size(0) != 1:
            raise ValueError("The test visualization logic assumes a batch size of 1.")

        # The split name is now part of the batch
        split = split_list[0]

        sample_data = {
            "img": img[0].cpu(),
            "word_str": word_str_list[0],
            "mask_gt": mask_gt[0].cpu(),
            "point_gt_norm": point_gt_norm[0].cpu(),
            "mask_pred_logits": mask_pred_logits[0].cpu(),
            "point_pred_logits": point_pred_logits[0].cpu(),
            "coords_hat": coords_hat[0].cpu(),
            "split": split,
        }
        self.test_samples.append(sample_data)

        # We can still compute and return the loss for logging purposes if desired
        # Note: We need to pass the original batch content to the common step
        original_batch = (img, word_str_list, mask_gt, point_gt_norm)
        return self._common_step(original_batch, batch_idx, "test")

    def test_dataloader(self):
        """
        Creates two test dataloaders, one with 10 random samples from the
        training split and one with 10 from the validation split.
        """
        rgb_transform, mask_transform = get_default_transforms(
            image_size=(self.cfg.input_size[0], self.cfg.input_size[1])
        )

        print(f"✅ Loading full dataset from: {self.cfg.train_data_dir}")
        # 1. Load the full dataset using the new LMDB dataset class
        full_dataset = SF3DDataset(
            lmdb_data_root=self.cfg.train_data_dir,
            rgb_transform=rgb_transform,
            mask_transform=mask_transform,
            image_size_for_mask_reconstruction=(
                self.cfg.input_size[0],
                self.cfg.input_size[1],
            ),
        )
        print(f"Full dataset size: {len(full_dataset)}")

        # 2. Split dataset by scene to get the same train/val split as during training
        print(
            f"✅ Splitting dataset with ratio {self.cfg.val_split_ratio} and seed {self.cfg.manual_seed}"
        )
        train_dataset_split, val_dataset_split = split_dataset_by_scene(
            full_dataset,
            val_split_ratio=self.cfg.val_split_ratio,
            manual_seed=self.cfg.manual_seed,
        )
        print(
            f"Train split size: {len(train_dataset_split)}, Val split size: {len(val_dataset_split)}"
        )

        # 3. Randomly select 10 samples from each dataset split
        train_indices = random.sample(
            range(len(train_dataset_split)), min(10, len(train_dataset_split))
        )
        val_indices = random.sample(
            range(len(val_dataset_split)), min(10, len(val_dataset_split))
        )
        print(
            f"✅ Selected {len(train_indices)} train samples and {len(val_indices)} val samples for testing."
        )

        train_subset = Subset(train_dataset_split, train_indices)
        val_subset = Subset(val_dataset_split, val_indices)

        # 4. Wrap datasets to include split information
        train_dataset_wrapped = SplitAwareDataset(train_subset, "train")
        val_dataset_wrapped = SplitAwareDataset(val_subset, "val")

        # 5. Create separate dataloaders
        train_dataloader = DataLoader(
            train_dataset_wrapped,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        val_dataloader = DataLoader(
            val_dataset_wrapped,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        return [train_dataloader, val_dataloader]

    def on_test_epoch_end(self):
        """Save visualizations of test samples to a folder."""
        if not self.test_samples:
            print("No test samples were collected. Skipping visualization.")
            return

        print(f"Processing {len(self.test_samples)} test samples for visualization...")

        # Create output directory
        output_dir = "test_visualizations"
        os.makedirs(output_dir, exist_ok=True)

        for idx, sample_data in enumerate(self.test_samples):
            # Extract data
            img = sample_data["img"]  # (C, H, W)
            word_str = sample_data["word_str"]
            mask_gt = sample_data["mask_gt"].float()  # (1, H_orig, W_orig)
            point_gt_norm = sample_data["point_gt_norm"].unsqueeze(0)  # (1, 2)
            mask_pred_logits = sample_data["mask_pred_logits"]  # (1, H_map, W_map)
            point_pred_logits = sample_data["point_pred_logits"]  # (1, H_map, W_map)
            coords_hat = sample_data["coords_hat"]
            split = sample_data["split"]

            # Apply sigmoid to predictions
            mask_pred_sigmoid = torch.sigmoid(mask_pred_logits)  # (1, H_map, W_map)
            point_pred_sigmoid = torch.sigmoid(point_pred_logits)  # (1, H_map, W_map)

            H_map, W_map = mask_pred_logits.shape[-2:]

            # Create GT point heatmap
            point_gt_heatmap = make_gaussian_map(
                point_gt_norm,
                H_map,
                W_map,
                sigma=self.cfg.loss_point_sigma,
                device="cpu",
            )[
                0
            ]  # (1, H_map, W_map)

            # Convert tensors to numpy for visualization
            img_np = img.permute(1, 2, 0).numpy()  # (H, W, C)
            # Denormalize from [-1,1] to [0,1]
            if img_np.min() < -0.001:
                img_np = (img_np * 0.5 + 0.5).clip(0, 1)

            mask_gt_np = mask_gt.squeeze(0).numpy()  # (H_orig, W_orig)
            mask_pred_np = (
                F.interpolate(
                    mask_pred_sigmoid.unsqueeze(0),
                    size=(img_np.shape[0], img_np.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
                .numpy()
            )  # Upsample to original size

            point_gt_heatmap_np = point_gt_heatmap.squeeze(0).numpy()  # (H_map, W_map)
            point_pred_heatmap_np = point_pred_sigmoid.squeeze(
                0
            ).numpy()  # (H_map, W_map)

            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(
                f'Sample {idx} - Split: {split} - Phrase: "{word_str}"', fontsize=16
            )

            # Input image
            axes[0, 0].imshow(img_np)
            axes[0, 0].set_title("Input Image")
            axes[0, 0].axis("off")

            # GT mask
            axes[0, 1].imshow(mask_gt_np, cmap="gray")
            axes[0, 1].set_title("Ground Truth Mask")
            axes[0, 1].axis("off")

            # Predicted mask
            axes[0, 2].imshow(mask_pred_np, cmap="gray")
            axes[0, 2].set_title("Predicted Mask (Upsampled)")
            axes[0, 2].axis("off")

            # GT point heatmap
            axes[1, 0].imshow(point_gt_heatmap_np, cmap="hot")
            axes[1, 0].set_title("GT Point Heatmap")
            axes[1, 0].axis("off")

            # Predicted point heatmap
            axes[1, 1].imshow(point_pred_heatmap_np, cmap="hot")
            axes[1, 1].set_title("Predicted Point Heatmap")
            axes[1, 1].axis("off")

            # Overlay predicted coordinates on input image
            axes[1, 2].imshow(img_np)
            # Convert normalized coordinates to pixel coordinates
            pred_x = coords_hat[0].item() * img_np.shape[1]
            pred_y = coords_hat[1].item() * img_np.shape[0]
            gt_x = point_gt_norm[0, 0].item() * img_np.shape[1]
            gt_y = point_gt_norm[0, 1].item() * img_np.shape[0]

            axes[1, 2].plot(pred_x, pred_y, "r*", markersize=15, label="Predicted")
            axes[1, 2].plot(gt_x, gt_y, "g+", markersize=15, label="Ground Truth")
            axes[1, 2].set_title("Predicted vs GT Coordinates")
            axes[1, 2].axis("off")
            axes[1, 2].legend()

            plt.tight_layout(rect=(0, 0.03, 1, 0.95))

            # Save the figure
            safe_word_str = "".join(
                c for c in word_str if c.isalnum() or c in (" ", "_")
            ).rstrip()
            sample_filename = (
                f"{split}_sample_{idx:03d}_{safe_word_str.replace(' ', '_')}.png"
            )
            save_path = os.path.join(output_dir, sample_filename)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"Saved visualization for {split} sample {idx}: {save_path}")

        print(f"\nAll visualizations saved to {output_dir}/")

        # Clear the samples list for the next potential run
        self.test_samples.clear()


# --- Main Testing Script --- #


def get_training_parser():
    parser = argparse.ArgumentParser(
        description="Test CRIS model on SceneFun3D data with PyTorch Lightning"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file (overrides defaults)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint to test",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Override settings in the config",
    )
    return parser


def main():
    parser = get_training_parser()
    args = parser.parse_args()

    default_config_path = "config/default_sf3d_train.yaml"
    if not os.path.exists(default_config_path):
        raise FileNotFoundError(
            f"Default config file not found at {default_config_path}. Please create it."
        )
    cfg = config_loader.load_cfg_from_cfg_file(default_config_path)

    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(
                f"User-specified config file not found at {args.config}"
            )
        user_cfg = config_loader.load_cfg_from_cfg_file(args.config)
        cfg.update(user_cfg)

    if args.opts:
        cfg = config_loader.merge_cfg_from_list(cfg, args.opts)

    if not hasattr(cfg, "train_data_dir") or not cfg.train_data_dir:
        raise ValueError(
            "Configuration error: 'train_data_dir' must be specified in your config file."
        )
    # The val_data_dir is no longer needed as we split from the train data
    # if not hasattr(cfg, "val_data_dir") or not cfg.val_data_dir:
    #     raise ValueError(
    #         "Configuration error: 'val_data_dir' must be specified in your config file."
    #     )
    if (
        not hasattr(cfg, "clip_pretrain")
        or not cfg.clip_pretrain
        or cfg.clip_pretrain == "path/to/clip_model.pth"
    ):
        print(
            f"Warning: 'clip_pretrain' is not properly set in config (current: {cfg.clip_pretrain}). Testing might fail if model requires it."
        )

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint file not found at {args.checkpoint}")

    base_output_folder = os.path.dirname(cfg.output_dir)
    cfg.output_dir = os.path.join(base_output_folder, cfg.exp_name)

    pl.seed_everything(cfg.manual_seed, workers=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Set up trainer for testing
    accelerator_type = "cpu"
    devices_val: typing.Union[typing.List[int], str, int] = 1

    if cfg.gpus > 0:
        if not torch.cuda.is_available():
            print(
                "Warning: CUDA is not available, but cfg.gpus > 0. Running on CPU instead."
            )
            accelerator_type = "cpu"
            devices_val = 1
        else:
            accelerator_type = "gpu"
            if cfg.gpus == 1:
                devices_val = [0]
            else:
                devices_val = cfg.gpus
    else:
        accelerator_type = "cpu"
        devices_val = 1

    trainer = pl.Trainer(
        accelerator=accelerator_type,
        devices=devices_val,
        precision=cfg.precision,
        logger=False,  # Disable logging for testing
        default_root_dir=cfg.output_dir,
        deterministic=False,
    )

    print(f"Loading model from checkpoint: {args.checkpoint}")
    print(
        f"Testing with 10 samples from training set and 10 samples from validation set"
    )
    print(f"Visualizations will be saved to: test_visualizations/")

    # Load model from checkpoint and run test
    # Use strict=False because the test module has `test_samples` which is not in the state_dict
    model = SF3DTestModule.load_from_checkpoint(args.checkpoint, strict=False)
    trainer.test(model)


if __name__ == "__main__":
    main()
