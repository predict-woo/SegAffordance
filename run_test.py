import argparse
import os
import torch
import torch.nn as nn
from loguru import logger

from model import build_segmenter
from utils.dataset import RefDataset
from utils.config import load_cfg_from_cfg_file


def get_args():
    parser = argparse.ArgumentParser(
        description="Test Referring Expression Segmentation"
    )
    parser.add_argument(
        "--config", default="path to xxx.yaml", type=str, help="config file"
    )

    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    return cfg


def main():
    # Get configuration
    args = get_args()
    logger.info("Starting test run...")

    # Set to eval mode
    torch.set_grad_enabled(False)

    # Build model (without loading weights)
    logger.info("Building model...")
    model, _ = build_segmenter(args)
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    # Create dataset with a single example
    logger.info("Loading dataset...")
    test_data = RefDataset(
        lmdb_dir=args.val_lmdb,
        mask_dir=args.mask_root,
        dataset=args.dataset,
        split=args.val_split,
        mode="val",
        input_size=args.input_size,
        word_length=args.word_len,
    )

    # Get a single sample
    if len(test_data) == 0:
        raise ValueError("Dataset is empty, check your dataset paths")

    # Print dataset size
    logger.info(f"Dataset size: {len(test_data)}")

    # Get the first sample
    sample = test_data[0]
    logger.info(f"Sample keys: {sample.keys()}")

    # Prepare batch (add batch dimension)
    batch = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.unsqueeze(0)
            if torch.cuda.is_available():
                batch[k] = batch[k].cuda()

    # Print input shapes
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            logger.info(f"Input '{k}' shape: {v.shape}")

    # Forward pass
    logger.info("Running forward pass...")
    try:
        outputs = model(batch)

        # Print output shapes
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    logger.info(f"Output '{k}' shape: {v.shape}")
        else:
            logger.info(f"Output shape: {outputs.shape}")

        logger.info("✅ Forward pass successful!")
    except Exception as e:
        logger.error(f"❌ Forward pass failed with error: {e}")
        raise

    logger.info("Test complete - all dimensions seem to be working correctly!")


if __name__ == "__main__":
    main()
