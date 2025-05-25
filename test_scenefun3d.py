import torch
from torch.utils.data import DataLoader
import argparse
import os # Added for dummy file creation path handling
import typing

from datasets.scenefun3d import SF3DDataset, get_default_transforms
from model.segmenter import CRIS
import utils.config as config # Assuming utils.config is available
from utils.dataset import tokenize # Import the tokenize function

# DummyConfig class removed

def get_parser():
    parser = argparse.ArgumentParser(description='Test CRIS model with SceneFun3D data')
    parser.add_argument('--config',
                        default='configs/default_test_config.yaml', # Example default path
                        type=str,
                        help='Path to the YAML configuration file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='Override some settings in the config.')
    # Add any other specific arguments needed for testing if different from train config
    return parser

def main():
    args = get_parser().parse_args()

    # Load config from YAML file
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)

    # --- Configuration for Data --- #
    # These could also be in the YAML, but for a test script, some can be hardcoded or taken from cfg
    PROCESSED_DATA_DIR = getattr(cfg, 'processed_data_dir', "/local/home/andrye/dev/SF3D_Proc")
    BATCH_SIZE = getattr(cfg, 'batch_size_val', 2) # Use a validation batch size or a specific test batch size
    
    input_size_cfg = getattr(cfg, 'input_size', (224, 224))
    if isinstance(input_size_cfg, int):
        IMAGE_SIZE = (input_size_cfg, input_size_cfg)
    elif isinstance(input_size_cfg, (list, tuple)) and len(input_size_cfg) == 2:
        IMAGE_SIZE = tuple(input_size_cfg)
    else:
        print(f"Warning: 'input_size' in config is '{input_size_cfg}', which is not an int or a 2-element list/tuple. Using default (224,224).")
        IMAGE_SIZE = (224, 224)
    
    IMAGE_SIZE = typing.cast(typing.Tuple[int, int], IMAGE_SIZE) # Explicit cast for linter

    print(f"Attempting to load data from: {PROCESSED_DATA_DIR}")
    print(f"Using IMAGE_SIZE: {IMAGE_SIZE}")

    # Get default transforms
    # Assuming get_default_transforms uses image_size that matches cfg.input_size
    rgb_transform, mask_transform = get_default_transforms(image_size=IMAGE_SIZE)

    # Create Dataset
    item_dataset = SF3DDataset(
        processed_data_root=PROCESSED_DATA_DIR,
        rgb_transform=rgb_transform,
        mask_transform=mask_transform,
        skip_items_without_motion=True 
    )

    if len(item_dataset) == 0:
        print("No items found. Check PROCESSED_DATA_DIR.")
        return

    # Create DataLoader
    item_dataloader = DataLoader(
        item_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=getattr(cfg, 'workers_val', 0) # Use validation workers or 0 for test
    )

    print(f"Successfully created DataLoader with {len(item_dataloader)} batches.")

    # --- Model Setup --- # 
    # cfg is now loaded from YAML
    
    # Ensure clip_pretrain path from config is handled correctly
    # For testing, if the actual model file isn't crucial and might not exist, create a dummy one.
    clip_pretrain_path = cfg.clip_pretrain
    if not os.path.exists(clip_pretrain_path):
        print(f"Warning: CLIP pretrain path {clip_pretrain_path} from config does not exist. Creating a dummy file.")
        os.makedirs(os.path.dirname(clip_pretrain_path), exist_ok=True)
        try:
            with open(clip_pretrain_path, 'w') as f:
                f.write("dummy clip model data for torch.jit.load")
        except Exception as e:
            print(f"Could not create dummy clip model file at {clip_pretrain_path}: {e}")
            # Decide if to proceed or exit if dummy file is essential and failed

    try:
        model = CRIS(cfg)
        model.eval() 
    except Exception as e:
        print(f"Error initializing CRIS model: {e}")
        print("Please ensure that the YAML config provides all necessary parameters for CRIS model (e.g., clip_pretrain, word_len, fpn_in, etc.)")
        raise # Re-raise after logging for clearer error
        return

    # --- Fetch and Process One Batch --- #
    print("\nFetching one batch...")
    try:
        img_batch, word_batch, mask_batch, interaction_point_batch = next(iter(item_dataloader))
        print(f"Image batch shape: {img_batch.shape}")
        print(f"Mask batch shape: {mask_batch.shape}")
        print(f"Interaction point batch shape: {interaction_point_batch.shape}")
        print(f"Word batch: {word_batch}")
        print(f"Interaction point batch: {interaction_point_batch}")
    except StopIteration:
        print("Dataloader is empty. Cannot fetch a batch.")
        return
    
    print(f"Image batch shape: {img_batch.shape}")
    print(f"Mask batch shape: {mask_batch.shape}")
    print(f"Interaction point batch shape: {interaction_point_batch.shape}")

    # Tokenize the word_batch using the utility from utils.dataset
    try:
        tokenized_words = tokenize(texts=word_batch, context_length=cfg.word_len, truncate=True)
    except Exception as e:
        print(f"Error tokenizing words: {e}")
        print("Ensure cfg.word_len is set in your config file.")
        raise

    print(f"Tokenized words shape: {tokenized_words.shape}")

    print("\nRunning batch through CRIS model...")
    with torch.no_grad(): 
        try:
            mask_pred, point_pred_logits, coords_hat = model(img_batch, tokenized_words, mask_batch, interaction_point_batch)
            print("\n--- Model Output ---")
            print(f"Predicted mask shape: {mask_pred.shape}")
            print(f"Predicted point logits shape: {point_pred_logits.shape}")
            print(f"Predicted coordinates (normalized) shape: {coords_hat.shape}")
            if coords_hat.numel() > 0:
                 print(f"Predicted coordinates (first item in batch): {coords_hat[0]}")
            else:
                print("Predicted coordinates tensor is empty.")

        except Exception as e:
            print(f"Error during model forward pass: {e}")
            raise

    print("\nTest script finished.")

if __name__ == "__main__":
    main()
