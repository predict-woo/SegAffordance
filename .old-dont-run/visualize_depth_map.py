import argparse
import os
import matplotlib.pyplot as plt
import torch
import warnings

from datasets.scenefun3d import SF3DDataset, get_default_transforms

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


def main(args):
    """
    Loads the first sample from the SF3D dataset and visualizes its depth map.
    """
    print(f"Loading dataset from {args.data_dir}")
    input_size = (256, 256)
    # We only need the dataset, so transforms can be minimal or default
    rgb_transform, mask_transform, depth_transform = get_default_transforms(
        image_size=input_size
    )

    try:
        dataset = SF3DDataset(
            lmdb_data_root=args.data_dir,
            rgb_transform=rgb_transform,
            mask_transform=mask_transform,
            depth_transform=depth_transform,
            image_size_for_mask_reconstruction=input_size,
            load_camera_intrinsics=False,
        )
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the --data_dir path is correct.")
        return

    if len(dataset) == 0:
        print("Error: The dataset is empty.")
        return

    print("Getting the first sample from the dataset...")
    sample = dataset[0]
    depth_image_tensor = sample[1]  # Depth tensor is the second element

    # Convert tensor to numpy array for plotting
    # The tensor is (C, H, W), so we squeeze to get (H, W)
    depth_map_numpy = depth_image_tensor.squeeze().numpy()

    print(f"Depth map shape: {depth_map_numpy.shape}")
    print(f"Depth map min value: {depth_map_numpy.min():.4f} meters")
    print(f"Depth map max value: {depth_map_numpy.max():.4f} meters")

    # Visualize the depth map
    plt.figure(figsize=(10, 8))
    im = plt.imshow(depth_map_numpy, cmap="viridis")
    plt.title("Depth Map Visualization (in Meters)")
    plt.xlabel("Width (pixels)")
    plt.ylabel("Height (pixels)")

    # Add a color bar to show the depth scale
    cbar = plt.colorbar(im)
    cbar.set_label("Depth (meters)")

    # Save the plot to a file
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, bbox_inches="tight")
    plt.close()  # Close the plot to free memory

    print(f"Depth map visualization saved to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a depth map from the SF3D dataset."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the root of the processed SF3D LMDB dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="depth_visualization.png",
        help="Path to save the output visualization PNG file.",
    )
    args = parser.parse_args()
    main(args)
