import json
import argparse
import os
import textwrap

import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import h5py


def draw_overlays(image, origin_xy, motion_dir_3d, description, is_valid, segmentation=None):
    """
    Draws overlays on the image for visualization.
    """
    origin_x, origin_y = int(origin_xy[0]), int(origin_xy[1])

    # Draw the mask
    if segmentation:
        # Assuming segmentation is a list of polygons
        for poly in segmentation:
            pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
            cv2.drawContours(image, [pts], -1, (0, 255, 0), 2)  # Green contour

    # Draw the motion origin point
    point_color = (0, 255, 0) if is_valid else (0, 0, 255)  # Green if valid, Red if not
    cv2.circle(image, (origin_x, origin_y), 8, point_color, -1)

    # Draw the motion direction arrow
    if motion_dir_3d:
        # Using x and y components for 2D visualization
        v_dir_2d = np.array([motion_dir_3d[0], motion_dir_3d[1]])
        norm = np.linalg.norm(v_dir_2d)
        if norm > 1e-6:
            v_dir_2d_norm = v_dir_2d / norm
            arrow_length = 50
            target_x = origin_x + int(v_dir_2d_norm[0] * arrow_length)
            target_y = origin_y + int(v_dir_2d_norm[1] * arrow_length)
            cv2.arrowedLine(image, (origin_x, origin_y), (target_x, target_y), (255, 0, 0), 2, tipLength=0.3)  # Blue arrow

    # Add the description text
    if description:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (0, 255, 255)  # Yellow
        line_type = 1
        for i, line in enumerate(textwrap.wrap(description, width=60)):
            cv2.putText(image, line, (10, 20 + i * 20), font, font_scale, font_color, line_type)

    return image


def project_point(
    point_3d: np.ndarray, intrinsic_matrix: np.ndarray
) -> tuple[float | None, float | None]:
    """
    Projects a 3D point in camera coordinates to 2D image coordinates.

    Args:
        point_3d: The 3D point (x, y, z).
        intrinsic_matrix: The 3x3 camera intrinsic matrix.

    Returns:
        A tuple (x, y) of the 2D projected point, or (None, None) if projection is not possible.
    """
    point_camera = np.array(point_3d)
    point_2d_homo = np.dot(intrinsic_matrix, point_camera[:3])

    # Avoid division by zero if the point is on the camera plane
    if point_2d_homo[2] == 0:
        return None, None

    x = point_2d_homo[0] / point_2d_homo[2]
    y = point_2d_homo[1] / point_2d_homo[2]
    return x, y


def main(args: argparse.Namespace):
    """
    Main function to filter annotations.
    """
    # Load the input JSON annotation file
    with open(args.input_json, "r") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]

    # Create a mapping from image ID to image information for quick access
    image_id_to_image = {img["id"]: img for img in images}

    filtered_annotations = []
    removed_count = 0

    h5_file = None
    images_dset = None
    filenames_map = None

    # Set up for visualization if a debug directory is provided
    if args.debug_viz_dir:
        os.makedirs(args.debug_viz_dir, exist_ok=True)
        dataset_key = os.path.splitext(os.path.basename(args.input_json))[0]
        split_name = dataset_key.split("_")[-1]

        if args.data_dir:
            data_path = args.data_dir
        else:
            # Infer data path from the input JSON path
            data_path = os.path.dirname(os.path.dirname(args.input_json))

        h5_path = os.path.join(data_path, f"{split_name}.h5")
        if not os.path.exists(h5_path):
            print(f"Error: H5 file not found at {h5_path}")
            print("Please provide the correct data directory using --data_dir")
            return

        h5_file = h5py.File(h5_path, "r")
        images_dset = h5_file[f"{split_name}_images"]
        assert isinstance(images_dset, h5py.Dataset)
        filenames_dset = h5_file[f"{split_name}_filenames"]
        assert isinstance(filenames_dset, h5py.Dataset)
        filenames_map = {
            name.decode("utf-8"): i for i, name in enumerate(list(filenames_dset))
        }

    # Iterate over annotations and filter them
    for i, anno in enumerate(tqdm(annotations, desc="Filtering annotations")):
        image_info = image_id_to_image[anno["image_id"]]
        height = image_info["height"]
        width = image_info["width"]

        # Handle different intrinsic matrix structures
        if args.is_multi:
            intrinsic_matrix = np.reshape(
                image_info["camera"]["intrinsic"], (3, 3), order="F"
            )
        else:
            intrinsic_matrix = np.reshape(
                image_info["camera"]["intrinsic"]["matrix"], (3, 3), order="F"
            )

        motion = anno["motion"]
        motion_origin_3d = motion.get("current_origin", motion.get("origin"))

        origin_x, origin_y = project_point(motion_origin_3d, intrinsic_matrix)

        if origin_x is None or origin_y is None:  # Projection failed
            removed_count += 1
            # NOTE: No visualization for projection failures for now.
            continue

        margin_x = 0.05 * width
        margin_y = 0.05 * height

        # Check if the point is within the valid area (5% margin from edges)
        is_valid = (margin_x <= origin_x < width - margin_x) and (
            margin_y <= origin_y < height - margin_y
        )

        if is_valid:
            filtered_annotations.append(anno)
        else:
            removed_count += 1

        if args.debug_viz_dir and filenames_map and images_dset:
            status = "unremoved" if is_valid else "removed"
            img_filename = os.path.basename(image_info["file_name"])
            if img_filename in filenames_map:
                img_index = filenames_map[img_filename]
                img_array = images_dset[img_index][:, :, :3]
                img_to_draw = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Get additional data for visualization
                motion_dir_3d = motion.get("direction")
                description = anno.get("description")
                segmentation = anno.get("segmentation")

                # Draw overlays
                img_to_draw = draw_overlays(
                    img_to_draw,
                    (origin_x, origin_y),
                    motion_dir_3d,
                    description,
                    is_valid,
                    segmentation,
                )

                # Save the debug image
                debug_image_path = os.path.join(
                    args.debug_viz_dir, f"{i}_{status}_{anno['id']}.jpg"
                )
                cv2.imwrite(debug_image_path, img_to_draw)

    if h5_file:
        h5_file.close()

    print(f"\nRemoved {removed_count} annotations.")
    print(f"Kept {len(filtered_annotations)} annotations.")

    # Rebuild the dataset, keeping only images that still have annotations
    kept_image_ids = {ann["image_id"] for ann in filtered_annotations}
    filtered_images = [img for img in images if img["id"] in kept_image_ids]

    # Create the new data structure, preserving other metadata
    new_data = {}
    for key, value in data.items():
        if key not in ["images", "annotations"]:
            new_data[key] = value
    new_data["images"] = filtered_images
    new_data["annotations"] = filtered_annotations

    # Save the filtered data to the output JSON file
    with open(args.output_json, "w") as f:
        json.dump(new_data, f, indent=4)

    print(f"Filtered JSON saved to {args.output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter annotations from OPD/OPD-Multi JSON files based on projected interaction points.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_json", type=str, help="Path to the input annotation JSON file."
    )
    parser.add_argument(
        "output_json", type=str, help="Path to save the filtered annotation JSON file."
    )
    parser.add_argument(
        "--is_multi",
        action="store_true",
        help="Flag for OPD-Multi dataset format to handle camera intrinsics correctly.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the root data directory containing H5 image files. If not provided, it is inferred from the input_json path.",
    )
    parser.add_argument(
        "--debug_viz_dir",
        type=str,
        default=None,
        help="Directory to save visualizations of removed annotations for debugging.",
    )

    args = parser.parse_args()
    main(args)
