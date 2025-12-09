import argparse
import os
import pickle
import torch
import warnings
from tqdm import tqdm

from train_SF3D_better import SF3DTrainingModule
from datasets.scenefun3d import SF3DDataset, get_default_transforms
from utils.dataset import tokenize

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


def find_sample_by_image_path(dataset, image_path):
    """
    Finds a sample in the dataset by its image filename.
    """
    image_filename_to_find = os.path.basename(image_path)
    print(f"Searching for sample with image filename: {image_filename_to_find}...")
    print("This may take a while as it requires scanning dataset metadata.")

    for i in tqdm(range(len(dataset)), desc="Scanning dataset"):
        item_key_bytes = dataset.item_keys[i]
        with dataset.env.begin(write=False) as txn:
            item_data_bytes = txn.get(item_key_bytes)

        if item_data_bytes:
            item_data = pickle.loads(item_data_bytes)
            if item_data.get("rgb_image_path") == image_filename_to_find:
                print(f"\nFound sample at index {i}")
                return dataset[i], i

    return None, -1


def main(args):
    """
    Runs single-sample inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.ckpt_path}")
    model = SF3DTrainingModule.load_from_checkpoint(
        args.ckpt_path, map_location=device, strict=False
    )
    model.eval()
    model.to(device)

    print(f"Loading dataset from {args.data_dir}")
    input_size = (256, 256)
    rgb_transform, mask_transform, depth_transform = get_default_transforms(
        image_size=input_size
    )

    dataset = SF3DDataset(
        lmdb_data_root=args.data_dir,
        rgb_transform=rgb_transform,
        mask_transform=mask_transform,
        depth_transform=depth_transform,
        image_size_for_mask_reconstruction=input_size,
        load_camera_intrinsics=True,
    )

    sample, sample_index = find_sample_by_image_path(dataset, args.image_path)

    if sample is None:
        print(
            f"Error: Could not find a sample with image {os.path.basename(args.image_path)} in the dataset."
        )
        return

    (
        rgb_image_tensor,
        depth_image_tensor,
        original_description,
        gt_mask_tensor,
        _bbox_tensor,
        gt_point_norm,
        gt_motion,
        _motion_type_tensor,
        _image_size_tensor,
        rgb_image_filename,
        _motion_origin_3d_camera_coords,
        _camera_intrinsic_matrix,
    ) = sample

    full_image_path = args.image_path
    if not os.path.exists(full_image_path):
        print(f"Error: Image file not found at {full_image_path}")
        return
    print(f"Using image: {full_image_path}")

    img_input = rgb_image_tensor.unsqueeze(0).to(device)
    depth_input = depth_image_tensor.unsqueeze(0).to(device)

    text_description = (
        args.text_description if args.text_description else original_description
    )
    print(f"Using text description: '{text_description}'")

    tokenized_text = tokenize(
        [text_description], model.model_params.word_len, truncate=True
    ).to(device)

    with torch.no_grad():
        (
            mask_pred_logits,
            point_pred_logits,
            _coords_hat,
            motion_pred,
            motion_type_logits,
            _mu,
            _log_var,
        ) = model(img_input, depth_input, tokenized_text, None, None, None)

    point_pred_prob = torch.sigmoid(point_pred_logits[0])
    mask_pred_prob = torch.sigmoid(mask_pred_logits[0])
    motion_pred_sample = motion_pred[0]
    pred_motion_type = torch.argmax(motion_type_logits, dim=1)[0].item()

    print(f"Saving visualizations to {args.output_dir}")
    SF3DTrainingModule._save_sf3d_test_debug_visualizations(
        full_image_path=full_image_path,
        point_pred_prob_tensor=point_pred_prob.cpu(),
        mask_pred_prob_tensor=mask_pred_prob.cpu(),
        motion_pred=motion_pred_sample.cpu(),
        pred_motion_type=pred_motion_type,
        gt_point_norm=gt_point_norm.cpu(),
        gt_mask_tensor=gt_mask_tensor.cpu(),
        gt_motion=gt_motion.cpu(),
        description=text_description,
        output_dir=args.output_dir,
        sample_index=sample_index,
    )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run single-sample inference for SegAffordance model on SF3D dataset."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root of the processed SF3D LMDB dataset.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Full path to the image file to use for inference.",
    )
    parser.add_argument(
        "--text_description",
        type=str,
        default=None,
        help="Custom text description to use. If not provided, uses the one from the dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="inference_output",
        help="Directory to save visualization results.",
    )
    args = parser.parse_args()
    main(args)
