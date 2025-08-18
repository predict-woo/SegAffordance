import os
import json
import numpy as np
import cv2  # For image manipulation (masks)
import tyro
from pathlib import Path
from typing import Annotated, List, Optional
from tqdm import tqdm
import lmdb
import pickle
from multiprocessing import Pool
from collections import defaultdict

from utils.data_parser import DataParser
from utils.fusion_util import PointCloudToImageMapper

# utils.homogenous is used by DataParser and for inverse transformation
import utils.homogenous as hm

# Default RGB asset to use for images and intrinsics
DEFAULT_RGB_ASSET = "hires_wide"
DEFAULT_INTRINSICS_ASSET = "hires_wide_intrinsics"
DEFAULT_DEPTH_ASSET = "hires_depth"  # For occlusion in projection

# It's good practice to define a version for your dataset format
LMDB_DATASET_VERSION = "1.0"

# Constants for debug visualization
ARROW_LENGTH_3D_TRANS = 0.1  # meters, for translational motion arrow
ROT_AXIS_VIS_LENGTH_3D = 0.05  # meters, for rotational axis visualization


def process_scene_video_by_frame(
    args: dict,
):
    """
    Processes a single scene video, designed to be called by a multiprocessing Pool.
    """
    # Unpack arguments
    (
        data_root_path,
        output_root_path,
        visit_id,
        video_id,
        rgb_asset,
        intrinsics_asset,
        depth_asset,
        skip_existing_frames,
        debug_visualizations,
        use_progress_bar,
        rgb_image_format,
        skip_items_without_motion_or_description,
        test_items,
        min_visibility_ratio,
        save_rgb_images,
    ) = (
        args["data_root_path"],
        args["output_root_path"],
        args["visit_id"],
        args["video_id"],
        args["rgb_asset"],
        args["intrinsics_asset"],
        args["depth_asset"],
        args["skip_existing_frames"],
        args["debug_visualizations"],
        args["use_progress_bar"],
        args["rgb_image_format"],
        args["skip_items_without_motion_or_description"],
        args.get("test_items"),
        args["min_visibility_ratio"],
        args["save_rgb_images"],
    )

    # This function will return a list of (key, value) pairs to be inserted into LMDB
    # by the main process, to avoid LMDB write contention.
    lmdb_records = []
    processed_item_count = 0
    skipped_item_count = 0

    print(f"Processing by Frame: Visit ID: {visit_id}, Video ID: {video_id}")
    data_parser = DataParser(str(data_root_path))

    # --- 1. Load scene-wide data (once per video) ---
    try:
        laser_scan_full = data_parser.get_laser_scan(visit_id)
        laser_scan_points_full = np.array(laser_scan_full.points)

        all_annotations = data_parser.get_annotations(
            visit_id, group_excluded_points=True
        )
        all_motions = data_parser.get_motions(visit_id)
        all_descriptions = data_parser.get_descriptions(visit_id)

        rgb_frame_paths_map = data_parser.get_rgb_frames(
            visit_id, video_id, data_asset_identifier=rgb_asset
        )
        depth_frame_paths_map = data_parser.get_depth_frames(
            visit_id, video_id, data_asset_identifier=depth_asset
        )
        intrinsics_paths_map = data_parser.get_camera_intrinsics(
            visit_id, video_id, data_asset_identifier=intrinsics_asset
        )
        poses_from_traj = data_parser.get_camera_trajectory(
            visit_id, video_id, pose_source="colmap"
        )

    except FileNotFoundError as e:
        print(
            f"  ERROR: Missing critical data for {visit_id}/{video_id}. Skipping. Details: {e}"
        )
        return None
    except Exception as e:
        print(
            f"  ERROR: Unexpected error loading scene-wide data for {visit_id}/{video_id}. Skipping. Details: {e}"
        )
        return None

    if not all(
        [
            all_annotations,
            rgb_frame_paths_map,
            poses_from_traj,
            intrinsics_paths_map,
            depth_frame_paths_map,
        ]
    ):
        print(
            f"  WARNING: Some critical data components (annotations, frames, poses, intrinsics, depth) are missing for {visit_id}/{video_id}. Processing will likely fail or be incomplete."
        )
        return None

    timestamps_to_process_map = None
    if test_items:
        timestamps_to_process_map = defaultdict(set)
        for timestamp, annot_id in test_items:
            timestamps_to_process_map[timestamp].add(annot_id)

    # --- 2. Iterate through each frame in the video sequence ---
    frame_timestamps_to_iterate = (
        list(timestamps_to_process_map.keys())
        if timestamps_to_process_map
        else [item[0] for item in rgb_frame_paths_map.items()]
    )
    total_frames = len(frame_timestamps_to_iterate)

    # Create iterator with or without progress bar
    if use_progress_bar:
        frame_iterator = tqdm(
            frame_timestamps_to_iterate, desc=f"  Frames in {video_id}"
        )
    else:
        frame_iterator = frame_timestamps_to_iterate
        print(f"  Processing {total_frames} frames in {video_id}...")

    frames_processed = 0
    frames_with_items = 0

    processed_item_count = 0
    skipped_item_count = 0

    for frame_idx, timestamp in enumerate(frame_iterator):
        rgb_frame_source_path_str = rgb_frame_paths_map.get(timestamp)
        if not rgb_frame_source_path_str:
            print(
                f"    WARNING: Timestamp {timestamp} from test file not found in RGB frames map for {video_id}. Skipping frame."
            )
            continue
        # Print progress every 10% when not using progress bar
        if (
            not use_progress_bar
            and frame_idx > 0
            and (frame_idx % max(1, total_frames // 10) == 0)
        ):
            print(
                f"    Progress: {frame_idx}/{total_frames} frames ({100 * frame_idx // total_frames}%)"
            )

        lmdb_key_prefix = f"{visit_id}/{video_id}/{timestamp}"
        # A basic check to see if we've already processed this frame for this video.
        # This isn't foolproof if processing was interrupted mid-frame.
        # A more robust check might involve querying for keys with this prefix.
        # However, for a simple skip, we assume if one key exists, all do.
        if skip_existing_frames:
            # We can't easily check for a directory, so we check for a sentinel item.
            # This is not perfect. A better approach is to not skip, or to build a list of already processed frames.
            # For this conversion, we will assume we are processing from scratch or that `skip_existing_frames` implies not re-processing anything.
            # The logic to check for existence is complex with LMDB without reading keys, so we might rely on user to not re-process.
            pass  # Skipping logic is complex with LMDB, for now we will re-process or rely on user to specify non-overlapping jobs.

        # --- Moved Up: Read and encode RGB image for the frame ---
        # We load the image once per frame, before checking items, so we can use it for failure case visualizations.
        try:
            rgb_image = cv2.imread(rgb_frame_source_path_str)
            if rgb_image is None:
                raise ValueError(
                    f"Image not found or could not be read at {rgb_frame_source_path_str}"
                )
        except Exception as e:
            print(
                f"    ERROR reading image for timestamp {timestamp}: {e}. Skipping frame."
            )
            continue

        # --- 2a. Load frame-specific data (pose, intrinsics, depth) ---
        try:
            camera_to_world_pose = data_parser.get_interpolated_pose(
                timestamp, poses_from_traj, time_distance_threshold=0.1
            )
            if camera_to_world_pose is None:
                camera_to_world_pose = data_parser.get_nearest_pose(
                    timestamp, poses_from_traj, time_distance_threshold=0.1
                )

            if camera_to_world_pose is None:
                print(
                    f"    WARNING: No pose found for frame {timestamp}. Skipping frame."
                )
                continue

            intrinsics_path = intrinsics_paths_map.get(timestamp)
            depth_path = depth_frame_paths_map.get(timestamp)

            if not intrinsics_path or not depth_path:
                print(
                    f"    WARNING: Missing intrinsics or depth for frame {timestamp}. Skipping frame."
                )
                continue

            w, h, fx, fy, cx, cy = data_parser.read_camera_intrinsics(intrinsics_path)
            K_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            current_depth_frame = data_parser.read_depth_frame(depth_path)
            world_to_camera_pose = hm.inverse(
                camera_to_world_pose
            )  # Using hm.inverse for clarity

        except Exception as e:
            print(
                f"    ERROR loading pose/intrinsics/depth for frame {timestamp}: {e}. Skipping frame."
            )
            continue

        point_to_image_mapper = PointCloudToImageMapper(
            image_dim=(int(w), int(h)), visibility_threshold=0.25, cut_bound=0
        )

        # --- 2b. Pre-check: Find if any items are visible in this frame ---
        has_visible_items = False
        visible_items_info = []  # Store info about visible items for later processing

        for ann_idx, annotation in enumerate(all_annotations):
            if annotation.get("label") == "exclude":
                continue

            annot_id = annotation.get("annot_id")

            # If we are in test mode, only process annot_ids specified for this timestamp
            if timestamps_to_process_map:
                if annot_id not in timestamps_to_process_map[timestamp]:
                    continue

            item_label = annotation.get("label", f"unknown_label_{ann_idx}")
            item_indices_in_scan = annotation.get("indices")

            if not annot_id or not item_indices_in_scan:
                continue

            item_points_3d_world = laser_scan_points_full[item_indices_in_scan]
            if item_points_3d_world.size == 0:
                continue

            # Project item points to current frame
            mapping_result = point_to_image_mapper.compute_mapping(
                camera_to_world=camera_to_world_pose,
                coords=item_points_3d_world,
                depth=current_depth_frame,
                intrinsic=K_matrix,
            )

            visible_mask_indices = mapping_result[:, 2] == 1
            visible_item_points_2d_yx = mapping_result[
                visible_mask_indices, :2
            ]  # These are (y,x) or (row,col)

            # Calculate the ratio of visible points to total points for the object
            total_item_points = len(item_points_3d_world)
            visible_item_points = len(visible_item_points_2d_yx)
            visibility_ratio = (
                visible_item_points / total_item_points if total_item_points > 0 else 0
            )

            if (
                visible_item_points >= 3 and visibility_ratio >= min_visibility_ratio
            ):  # Need at least 3 points and must meet visibility ratio
                has_visible_items = True
                visible_items_info.append(
                    {
                        "annotation": annotation,
                        "ann_idx": ann_idx,
                        "annot_id": annot_id,
                        "item_label": item_label,
                        "mapping_result": mapping_result,
                        "visible_item_points_2d_yx": visible_item_points_2d_yx,
                        "visible_mask_indices": visible_mask_indices,
                    }
                )
            elif debug_visualizations:
                failure_reason = ""
                if visible_item_points < 3:
                    failure_reason = f"NotEnoughPoints_{visible_item_points}"
                elif visibility_ratio < min_visibility_ratio:
                    failure_reason = f"LowVisibility_{visibility_ratio:.2f}"

                print(
                    f"      [DEBUG] Item {visit_id}/{video_id}/{timestamp}/{annot_id} skipped: {failure_reason}"
                )

                try:
                    debug_image = rgb_image.copy()
                    debug_dir = Path("debug")
                    debug_dir.mkdir(exist_ok=True)

                    # Draw the few visible points that were found
                    for y_vis, x_vis in visible_item_points_2d_yx:
                        cv2.circle(
                            debug_image, (int(x_vis), int(y_vis)), 3, (0, 255, 255), -1
                        )

                    # Add text for the failure
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_label = f"Label: {item_label}"
                    reason_text = f"Reason: {failure_reason}"
                    cv2.putText(
                        debug_image, text_label, (10, 20), font, 0.6, (255, 255, 255), 2
                    )
                    cv2.putText(
                        debug_image, reason_text, (10, 40), font, 0.6, (0, 0, 255), 2
                    )

                    debug_filename = f"{visit_id}_{video_id}_{timestamp}_{annot_id}_FAIL_{failure_reason}.png"
                    debug_image_path = debug_dir / debug_filename
                    cv2.imwrite(str(debug_image_path), debug_image)
                except Exception as e_debug:
                    print(
                        f"      ERROR generating failure debug image for item {annot_id}: {e_debug}"
                    )

        if not has_visible_items:
            frames_processed += 1
            continue

        # --- Save RGB image to file (only if there are visible items) ---
        image_filename = f"{visit_id}_{video_id}_{timestamp}.{rgb_image_format}"
        if save_rgb_images:
            image_save_path = output_root_path / "images" / image_filename
            try:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                # cv2.imwrite uses the file extension to determine format.
                # The encode_param is mainly for formats like JPEG.
                is_success = cv2.imwrite(str(image_save_path), rgb_image, encode_param)
                if not is_success:
                    raise ValueError(f"Failed to save image to {image_save_path}")
            except Exception as e:
                print(
                    f"    ERROR saving RGB image to {image_save_path} for timestamp {timestamp}: {e}. Skipping frame."
                )
                continue

        frames_processed += 1
        frames_with_items += 1

        # --- 2d. Process all visible items in this frame ---
        for item_info in visible_items_info:
            annotation = item_info["annotation"]
            ann_idx = item_info["ann_idx"]
            annot_id = item_info["annot_id"]
            item_label = item_info["item_label"]
            visible_item_points_2d_yx = item_info["visible_item_points_2d_yx"]

            # --- Get Mask Coordinates ---
            mask_image = np.zeros((int(h), int(w)), dtype=np.uint8)
            cv_points_xy = visible_item_points_2d_yx[:, [1, 0]].astype(
                np.int32
            )  # Convert (y,x) to (x,y) for OpenCV
            hull = cv2.convexHull(cv_points_xy)
            cv2.fillPoly(mask_image, [hull], (255,))
            mask_coordinates = np.argwhere(mask_image > 0).tolist()

            if not mask_coordinates:
                if debug_visualizations:
                    print(
                        f"      [DEBUG] Item {visit_id}/{video_id}/{timestamp}/{annot_id} skipped: Empty mask."
                    )
                skipped_item_count += 1
                continue

            # --- Get Label Info ---
            label_info_data = {
                "annot_id": annot_id,
                "label": item_label,
                "raw_annotation_data": annotation,
            }

            # --- Get Description ---
            item_descriptions_text = []
            if all_descriptions:  # Ensure all_descriptions was loaded
                for desc in all_descriptions:
                    if annot_id in desc.get("annot_id", []):
                        item_descriptions_text.append(desc.get("description", ""))
            description = "\n---\n".join(filter(None, item_descriptions_text))

            # --- Get Motion Info ---
            motion_info_data = {
                "annot_id": annot_id,
                "original_motion_data": None,
                "frame_specific_motion_data": None,
            }
            item_motion = None
            if all_motions:  # Ensure all_motions was loaded
                for m in all_motions:
                    if m.get("annot_id") == annot_id:
                        item_motion = m
                        break

            if item_motion:
                motion_origin_idx = item_motion.get("motion_origin_idx")
                motion_dir_world_list = item_motion.get(
                    "motion_dir"
                )  # This is already a list

                motion_info_data["original_motion_data"] = {
                    "motion_type": item_motion.get("motion_type"),
                    "motion_dir_world": motion_dir_world_list,
                    "motion_origin_3d_world": None,  # Will fill if valid idx
                    "motion_origin_idx_in_laserscan": motion_origin_idx,
                    "motion_viz_orient": item_motion.get("motion_viz_orient"),
                    "raw_motion_data": item_motion,
                }

                if (
                    motion_origin_idx is not None
                    and motion_dir_world_list is not None
                    and 0 <= motion_origin_idx < len(laser_scan_points_full)
                ):

                    motion_origin_3d_world_np = laser_scan_points_full[
                        motion_origin_idx
                    ]
                    motion_info_data["original_motion_data"][
                        "motion_origin_3d_world"
                    ] = motion_origin_3d_world_np.tolist()

                    # Transform to camera coordinates
                    origin_world_homo = np.append(motion_origin_3d_world_np, 1)
                    origin_cam_homo = world_to_camera_pose @ origin_world_homo

                    origin_3d_cam_coords_np = np.array([0.0, 0.0, 0.0])
                    if abs(origin_cam_homo[3]) > 1e-6:  # Avoid division by zero
                        origin_3d_cam_coords_np = (
                            origin_cam_homo[:3] / origin_cam_homo[3]
                        )

                    # Project origin to 2D image plane
                    origin_2d_image_coords_np = np.array([0.0, 0.0])
                    if (
                        abs(origin_3d_cam_coords_np[2]) > 1e-6
                    ):  # Zc must be non-zero (and positive for visibility)
                        origin_img_homo = K_matrix @ origin_3d_cam_coords_np
                        origin_2d_image_coords_np = (
                            origin_img_homo[:2] / origin_img_homo[2]
                        )

                    # Transform direction vector to camera coordinates
                    motion_dir_world_np = np.array(motion_dir_world_list)
                    motion_dir_3d_cam_coords_np = (
                        world_to_camera_pose[:3, :3] @ motion_dir_world_np
                    )

                    motion_info_data["frame_specific_motion_data"] = {
                        "motion_origin_2d_image_coords": origin_2d_image_coords_np.tolist(),
                        "motion_origin_3d_camera_coords": origin_3d_cam_coords_np.tolist(),
                        "motion_dir_3d_camera_coords": motion_dir_3d_cam_coords_np.tolist(),
                    }
                else:
                    motion_info_data["original_motion_data"][
                        "error"
                    ] = "Invalid motion_origin_idx or missing motion_dir."
            else:
                motion_info_data["message"] = "No motion data found for this annot_id."

            # --- Data Cleanup Filters ---
            x_min, x_max = w * 0.05, w * 0.95
            y_min, y_max = h * 0.05, h * 0.95

            # Filter 1: Check if motion origin is within the 5-95% bounding box
            motion_origin_in_bounds = True
            if motion_info_data.get("frame_specific_motion_data"):
                motion_origin_2d = motion_info_data["frame_specific_motion_data"].get(
                    "motion_origin_2d_image_coords"
                )
                if motion_origin_2d:
                    origin_x, origin_y = motion_origin_2d[0], motion_origin_2d[1]
                    if not (x_min <= origin_x <= x_max and y_min <= origin_y <= y_max):
                        motion_origin_in_bounds = False

            if not motion_origin_in_bounds:
                if debug_visualizations:
                    failure_reason = "MotionOriginOutOfBounds"
                    print(
                        f"      [DEBUG] Item {visit_id}/{video_id}/{timestamp}/{annot_id} skipped: {failure_reason}"
                    )
                    try:
                        debug_image = rgb_image.copy()
                        debug_dir = Path("debug")
                        debug_dir.mkdir(exist_ok=True)
                        # Draw Mask
                        mask_coords_np_fail = np.array(mask_coordinates)
                        cv_points_xy_fail = mask_coords_np_fail[:, [1, 0]].astype(
                            np.int32
                        )
                        hull_fail = cv2.convexHull(cv_points_xy_fail)
                        cv2.drawContours(debug_image, [hull_fail], -1, (0, 255, 0), 2)
                        # Draw Motion
                        if motion_info_data.get("frame_specific_motion_data"):
                            p_origin_2d = np.array(
                                motion_info_data["frame_specific_motion_data"][
                                    "motion_origin_2d_image_coords"
                                ]
                            )
                            cv2.circle(
                                debug_image,
                                (int(p_origin_2d[0]), int(p_origin_2d[1])),
                                8,
                                (0, 0, 255),
                                -1,
                            )  # Red circle for OOB origin
                        # Add text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(
                            debug_image,
                            f"FAIL: {failure_reason}",
                            (10, 20),
                            font,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        debug_filename = f"{visit_id}_{video_id}_{timestamp}_{annot_id}_FAIL_{failure_reason}.png"
                        debug_image_path = debug_dir / debug_filename
                        cv2.imwrite(str(debug_image_path), debug_image)
                    except Exception as e_debug:
                        print(
                            f"      ERROR generating failure debug image for item {annot_id}: {e_debug}"
                        )
                skipped_item_count += 1
                continue

            # Filter 2: Check if all mask coordinates are within the 5-95% bounding box
            mask_coords_np = np.array(mask_coordinates)
            x_coords, y_coords = mask_coords_np[:, 1], mask_coords_np[:, 0]
            if not (
                x_coords.min() >= x_min
                and x_coords.max() <= x_max
                and y_coords.min() >= y_min
                and y_coords.max() <= y_max
            ):
                if debug_visualizations:
                    failure_reason = "MaskOutOfBounds"
                    print(
                        f"      [DEBUG] Item {visit_id}/{video_id}/{timestamp}/{annot_id} skipped: {failure_reason}"
                    )
                    try:
                        debug_image = rgb_image.copy()
                        debug_dir = Path("debug")
                        debug_dir.mkdir(exist_ok=True)
                        # Draw Mask
                        cv_points_xy_fail = mask_coords_np[:, [1, 0]].astype(np.int32)
                        hull_fail = cv2.convexHull(cv_points_xy_fail)
                        cv2.drawContours(
                            debug_image, [hull_fail], -1, (0, 0, 255), 2
                        )  # Red contour for OOB mask
                        # Draw Bounding Box
                        cv2.rectangle(
                            debug_image,
                            (int(x_min), int(y_min)),
                            (int(x_max), int(y_max)),
                            (255, 255, 0),
                            1,
                        )
                        # Add text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(
                            debug_image,
                            f"FAIL: {failure_reason}",
                            (10, 20),
                            font,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        debug_filename = f"{visit_id}_{video_id}_{timestamp}_{annot_id}_FAIL_{failure_reason}.png"
                        debug_image_path = debug_dir / debug_filename
                        cv2.imwrite(str(debug_image_path), debug_image)
                    except Exception as e_debug:
                        print(
                            f"      ERROR generating failure debug image for item {annot_id}: {e_debug}"
                        )
                skipped_item_count += 1
                continue

            # Filter 3: Check for empty description
            if not description:
                if debug_visualizations:
                    failure_reason = "EmptyDescription"
                    print(
                        f"      [DEBUG] Item {visit_id}/{video_id}/{timestamp}/{annot_id} skipped: {failure_reason}"
                    )
                    try:
                        # Create a debug image for this failure case
                        debug_image = rgb_image.copy()
                        debug_dir = Path("debug")
                        debug_dir.mkdir(exist_ok=True)
                        # Draw Mask
                        mask_coords_np_fail = np.array(mask_coordinates)
                        cv_points_xy_fail = mask_coords_np_fail[:, [1, 0]].astype(
                            np.int32
                        )
                        hull_fail = cv2.convexHull(cv_points_xy_fail)
                        cv2.drawContours(debug_image, [hull_fail], -1, (0, 255, 0), 2)
                        # Add text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(
                            debug_image,
                            f"FAIL: {failure_reason}",
                            (10, 20),
                            font,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                        debug_filename = f"{visit_id}_{video_id}_{timestamp}_{annot_id}_FAIL_{failure_reason}.png"
                        debug_image_path = debug_dir / debug_filename
                        cv2.imwrite(str(debug_image_path), debug_image)
                    except Exception as e_debug:
                        print(
                            f"      ERROR generating failure debug image for item {annot_id}: {e_debug}"
                        )
                skipped_item_count += 1
                continue

            # --- Validation (from convert_to_lmdb.py) ---
            if skip_items_without_motion_or_description:
                if (
                    not motion_info_data.get("frame_specific_motion_data")
                    or motion_info_data["frame_specific_motion_data"].get(
                        "motion_origin_2d_image_coords"
                    )
                    is None
                    or motion_info_data["frame_specific_motion_data"].get(
                        "motion_dir_3d_camera_coords"
                    )
                    is None
                ):
                    if debug_visualizations:
                        print(
                            f"      [DEBUG] Item {visit_id}/{video_id}/{timestamp}/{annot_id} skipped: Incomplete motion data."
                        )
                    skipped_item_count += 1
                    continue

            # --- Prepare and save to LMDB ---
            lmdb_key_str = f"{visit_id}/{video_id}/{timestamp}/{annot_id}"
            lmdb_key_bytes = lmdb_key_str.encode("utf-8")

            # NOTE: We don't check for unique keys here because this worker doesn't have the shared set.
            # The main process will handle duplicates. It's assumed duplicates are rare.

            data_to_store = {
                "rgb_image_path": image_filename,
                "mask_coordinates_yx": mask_coordinates,  # List of [y,x]
                "description": description,
                "motion_info": motion_info_data,
                "label_info": label_info_data,
                "camera_intrinsics": K_matrix.tolist(),
                "camera_extrinsics_world_to_cam": world_to_camera_pose.tolist(),
                "camera_extrinsics_cam_to_world": camera_to_world_pose.tolist(),
                "image_dimensions_wh": (int(w), int(h)),
            }

            # Append the record to be written by the main process
            lmdb_records.append((lmdb_key_bytes, pickle.dumps(data_to_store)))
            processed_item_count += 1

            # --- Generate and Save Debug Visualization (if enabled) ---
            if (
                debug_visualizations
                and item_motion
                and motion_info_data.get("frame_specific_motion_data")
                and motion_info_data["frame_specific_motion_data"].get(
                    "motion_origin_2d_image_coords"
                )
                is not None
            ):
                try:
                    # Use the already loaded image for drawing. Make a copy to not alter the original.
                    debug_image = rgb_image.copy()

                    # Create a 'debug' directory in the current working directory
                    debug_dir = Path("debug")
                    debug_dir.mkdir(exist_ok=True)

                    p_origin_2d = np.array(
                        motion_info_data["frame_specific_motion_data"][
                            "motion_origin_2d_image_coords"
                        ]
                    )
                    p_origin_3d_cam = np.array(
                        motion_info_data["frame_specific_motion_data"][
                            "motion_origin_3d_camera_coords"
                        ]
                    )
                    v_dir_3d_cam = np.array(
                        motion_info_data["frame_specific_motion_data"][
                            "motion_dir_3d_camera_coords"
                        ]
                    )
                    motion_type = motion_info_data["original_motion_data"][
                        "motion_type"
                    ]

                    # Draw origin circle
                    cv2.circle(
                        debug_image,
                        (int(p_origin_2d[0]), int(p_origin_2d[1])),
                        5,
                        (255, 0, 0),
                        -1,
                    )  # Blue circle for origin

                    if motion_type == "trans":
                        p_target_3d_cam = (
                            p_origin_3d_cam + v_dir_3d_cam * ARROW_LENGTH_3D_TRANS
                        )
                        if abs(p_target_3d_cam[2]) > 1e-6:
                            target_img_homo = K_matrix @ p_target_3d_cam
                            p_target_2d = target_img_homo[:2] / target_img_homo[2]
                            cv2.arrowedLine(
                                debug_image,
                                (int(p_origin_2d[0]), int(p_origin_2d[1])),
                                (int(p_target_2d[0]), int(p_target_2d[1])),
                                (0, 255, 0),
                                2,
                                tipLength=0.3,
                            )  # Green arrow
                    elif motion_type == "rot":
                        cv2.circle(
                            debug_image,
                            (int(p_origin_2d[0]), int(p_origin_2d[1])),
                            10,
                            (0, 0, 255),
                            2,
                        )  # Red circle for rotation
                        p_target_axis_3d_cam = (
                            p_origin_3d_cam + v_dir_3d_cam * ROT_AXIS_VIS_LENGTH_3D
                        )
                        if abs(p_target_axis_3d_cam[2]) > 1e-6:
                            target_axis_img_homo = K_matrix @ p_target_axis_3d_cam
                            p_target_axis_2d = (
                                target_axis_img_homo[:2] / target_axis_img_homo[2]
                            )
                            cv2.line(
                                debug_image,
                                (int(p_origin_2d[0]), int(p_origin_2d[1])),
                                (int(p_target_axis_2d[0]), int(p_target_axis_2d[1])),
                                (0, 0, 255),
                                2,
                            )  # Red line for axis dir

                    # Add text
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    font_color = (255, 255, 0)  # Cyan
                    line_type = 1
                    text_label = label_info_data.get("label", "N/A")
                    short_desc = ""
                    if item_descriptions_text:
                        first_desc = item_descriptions_text[0]
                        short_desc = (
                            (first_desc[:50] + "...")
                            if len(first_desc) > 50
                            else first_desc
                        )

                    text_y_offset = 20
                    cv2.putText(
                        debug_image,
                        text_label,
                        (int(p_origin_2d[0]) + 10, int(p_origin_2d[1]) + text_y_offset),
                        font,
                        font_scale,
                        font_color,
                        line_type,
                    )
                    text_y_offset += 15
                    cv2.putText(
                        debug_image,
                        short_desc,
                        (int(p_origin_2d[0]) + 10, int(p_origin_2d[1]) + text_y_offset),
                        font,
                        font_scale,
                        font_color,
                        line_type,
                    )

                    # Create a descriptive filename and save it as PNG
                    debug_filename = f"{visit_id}_{video_id}_{timestamp}_{annot_id}.png"
                    debug_image_path = debug_dir / debug_filename
                    cv2.imwrite(str(debug_image_path), debug_image)
                except Exception as e_debug:
                    print(
                        f"      ERROR generating debug visualization for item {annot_id}, frame {timestamp}: {e_debug}"
                    )
    print(
        f"  Finished processing video {visit_id}/{video_id}: "
        f"{frames_with_items}/{total_frames} frames were saved after filtering."
    )
    # Return counts and records to be aggregated in main
    return processed_item_count, skipped_item_count, lmdb_records


def main(
    data_dir: Annotated[
        Path, tyro.conf.arg(help="Path to the root of the SceneFun3D dataset")
    ],
    output_dir: Annotated[
        Path,
        tyro.conf.arg(
            help="Path to save the new LMDB dataset and associated files (e.g., images, debug overlays)."
        ),
    ],
    visit_ids: Annotated[
        Optional[List[str]],
        tyro.conf.arg(
            help="List of visit_ids to process. If None, processes all found."
        ),
    ] = None,
    video_ids: Annotated[
        Optional[List[str]],
        tyro.conf.arg(
            help="List of video_ids to process. If multiple visit_ids, these videos are sought under each. If None, processes all found videos for specified visits."
        ),
    ] = None,
    csv_file: Annotated[
        Optional[Path],
        tyro.conf.arg(
            help="Path to a CSV file with visit_id,video_id pairs to process."
        ),
    ] = None,
    test_file: Annotated[
        Optional[Path],
        tyro.conf.arg(
            help="Path to a text file with specific items to process. Each line should be: visit_id,video_id,timestamp,annot_id. This overrides other scene selection methods."
        ),
    ] = None,
    rgb_asset_name: Annotated[
        str, tyro.conf.arg(help="RGB data asset type (e.g., hires_wide, lowres_wide)")
    ] = DEFAULT_RGB_ASSET,
    intrinsics_asset_name: Annotated[
        str, tyro.conf.arg(help="Intrinsics data asset type")
    ] = DEFAULT_INTRINSICS_ASSET,
    depth_asset_name: Annotated[
        str, tyro.conf.arg(help="Depth data asset type")
    ] = DEFAULT_DEPTH_ASSET,
    skip_existing_frames: Annotated[
        bool,
        tyro.conf.arg(
            help="If true, skips processing for a frame if its output directory and RGB image already exist. Note: with LMDB, this is less effective and may not prevent re-processing."
        ),
    ] = True,
    debug_visualizations: Annotated[
        bool,
        tyro.conf.arg(
            help="If true, generates and saves debug images with motion and text overlays in a 'debug_images' subfolder."
        ),
    ] = False,
    use_progress_bar: Annotated[
        bool,
        tyro.conf.arg(
            help="If true, uses tqdm progress bar. If false, uses simple logging-friendly progress messages."
        ),
    ] = True,
    map_size: Annotated[
        int, tyro.conf.arg(help="Maximum size of the LMDB database in bytes.")
    ] = 1024
    * 1024
    * 1024
    * 20,  # Default 50GB,
    rgb_image_format: Annotated[
        str, tyro.conf.arg(help="Format to save RGB images (e.g., jpg, png).")
    ] = "jpg",
    save_rgb_images: Annotated[
        bool,
        tyro.conf.arg(
            help="If true, saves RGB image files to disk. Set to false to skip this step if images are already generated."
        ),
    ] = True,
    skip_items_without_motion_or_description: Annotated[
        bool,
        tyro.conf.arg(help="Skip items if motion or description is missing/invalid."),
    ] = True,
    num_workers: Annotated[
        int,
        tyro.conf.arg(
            help="Number of parallel worker processes to use. If 0 or 1, runs sequentially in the main process."
        ),
    ] = 4,
    maxtasksperchild: Annotated[
        Optional[int],
        tyro.conf.arg(
            help="If specified, worker processes will be restarted after completing this many tasks. Useful for releasing memory."
        ),
    ] = 1,
    min_visibility_ratio: Annotated[
        float,
        tyro.conf.arg(
            help="Minimum ratio of an item's points that must be visible (not occluded) for it to be processed."
        ),
    ] = 0.1,
):
    """
    Processes the SceneFun3D dataset and saves it directly into an LMDB database.
    For each frame, it processes all visible items, storing their masks, motion info,
    labels, and descriptions in the LMDB. RGB images are stored separately.
    Optionally generates debug visualizations.
    """
    if not data_dir.is_dir():
        print(f"Error: Data directory {data_dir} not found.")
        return

    # Setup output directories and LMDB environment
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    lmdb_path = output_dir / "data.lmdb"

    print(f"Output LMDB will be at: {lmdb_path}")
    print(f"RGB images will be stored in: {images_dir}")

    env = lmdb.open(str(lmdb_path), map_size=map_size, writemap=True)

    scenes_to_process_map = {}
    test_items_map = defaultdict(set)

    if test_file:
        print(f"Processing specific items from test file: {test_file}")
        if not test_file.is_file():
            print(f"Error: Test file {test_file} not found.")
            env.close()
            return
        with open(test_file, "r") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) != 4:
                    print(
                        f"Warning: Skipping malformed line {line_num+1} in test file: {line}"
                    )
                    continue
                visit_id, video_id, timestamp, annot_id = parts
                key = (visit_id, video_id)
                scenes_to_process_map[key] = {
                    "visit_id": visit_id,
                    "video_id": video_id,
                }
                test_items_map[key].add((timestamp, annot_id))

    elif csv_file:
        if not csv_file.is_file():
            print(f"Error: CSV file {csv_file} not found.")
            env.close()
            return
        with open(csv_file, "r") as f:
            lines = f.readlines()
            for line_idx, line in enumerate(lines):
                if line_idx == 0 and (
                    "visit_id" in line.lower() and "video_id" in line.lower()
                ):
                    continue
                parts = line.strip().split(",")
                if len(parts) == 2:
                    scenes_to_process_map[(parts[0], parts[1])] = {
                        "visit_id": parts[0],
                        "video_id": parts[1],
                    }
                elif line.strip():
                    print(f"Warning: Could not parse line in CSV: {line.strip()}")
    elif visit_ids:
        for v_id_input in visit_ids:
            visit_path_check = data_dir / v_id_input
            if not visit_path_check.is_dir():
                print(
                    f"Warning: Specified visit_id {v_id_input} not found in {data_dir}. Skipping."
                )
                continue

            target_video_ids_for_visit = video_ids
            if not target_video_ids_for_visit:
                target_video_ids_for_visit = sorted(
                    [
                        p.name
                        for p in visit_path_check.iterdir()
                        if p.is_dir()
                        and (visit_path_check / p / rgb_asset_name).is_dir()
                    ]
                )

            for vid_id_input in target_video_ids_for_visit:
                video_path_check = visit_path_check / vid_id_input / rgb_asset_name
                if video_path_check.is_dir():
                    scenes_to_process_map[(v_id_input, vid_id_input)] = {
                        "visit_id": v_id_input,
                        "video_id": vid_id_input,
                    }
                else:
                    print(
                        f"Warning: Video {vid_id_input} (asset: {rgb_asset_name}) not found under visit {v_id_input}. Skipping."
                    )
    else:
        for visit_path in data_dir.iterdir():
            if visit_path.is_dir() and visit_path.name.isdigit():
                v_id = visit_path.name
                for video_path_in_visit in visit_path.iterdir():
                    rgb_asset_dir_check = video_path_in_visit / rgb_asset_name
                    if video_path_in_visit.is_dir() and rgb_asset_dir_check.is_dir():
                        vid_id = video_path_in_visit.name
                        scenes_to_process_map[(v_id, vid_id)] = {
                            "visit_id": v_id,
                            "video_id": vid_id,
                        }

    unique_scenes_to_process = list(scenes_to_process_map.values())

    if not unique_scenes_to_process:
        print("No scenes found or specified to process based on input criteria.")
        env.close()
        return

    print(
        f"Found {len(unique_scenes_to_process)} unique scene/video combinations to process."
    )

    total_processed_items = 0
    total_skipped_items = 0
    unique_keys = set()

    # Prepare arguments for multiprocessing
    processing_args = [
        {
            "data_root_path": data_dir,
            "output_root_path": output_dir,
            "visit_id": scene_info["visit_id"],
            "video_id": scene_info["video_id"],
            "test_items": test_items_map.get(
                (scene_info["visit_id"], scene_info["video_id"])
            ),
            "rgb_asset": rgb_asset_name,
            "intrinsics_asset": intrinsics_asset_name,
            "depth_asset": depth_asset_name,
            "skip_existing_frames": skip_existing_frames,
            "debug_visualizations": debug_visualizations,
            "use_progress_bar": False,  # Disable nested progress bars
            "rgb_image_format": rgb_image_format,
            "skip_items_without_motion_or_description": skip_items_without_motion_or_description,
            "min_visibility_ratio": min_visibility_ratio,
            "save_rgb_images": save_rgb_images,
        }
        for scene_info in unique_scenes_to_process
    ]

    # --- Write metadata to DB first in a separate transaction ---
    with env.begin(write=True) as txn:
        metadata = {
            "version": LMDB_DATASET_VERSION,
            "source_dataset_path": str(data_dir.resolve()),
            "rgb_image_format": rgb_image_format,
            "images_stored_in_lmdb": False,
        }
        txn.put(b"__metadata__", pickle.dumps(metadata))

    # --- Process data and write to LMDB ---
    if num_workers > 1:
        print(f"Starting parallel processing with {num_workers} workers...")
        with Pool(processes=num_workers, maxtasksperchild=maxtasksperchild) as pool:
            results_iterator = pool.imap_unordered(
                process_scene_video_by_frame, processing_args
            )
            pbar = tqdm(
                results_iterator,
                total=len(processing_args),
                desc="Overall Progress",
            )
            for result in pbar:
                if result:
                    processed_count, skipped_count, lmdb_records = result
                    total_processed_items += processed_count
                    total_skipped_items += skipped_count
                    # Write results for this video in a new transaction
                    with env.begin(write=True) as txn:
                        for key, value in lmdb_records:
                            if key not in unique_keys:
                                txn.put(key, value)
                                unique_keys.add(key)
                            else:
                                total_skipped_items += 1
    else:
        print("Starting sequential processing...")
        pbar = tqdm(processing_args, desc="Overall Progress")
        for args in pbar:
            result = process_scene_video_by_frame(args)
            if result:
                processed_count, skipped_count, lmdb_records = result
                total_processed_items += processed_count
                total_skipped_items += skipped_count
                # Write results for this video in a new transaction
                with env.begin(write=True) as txn:
                    for key, value in lmdb_records:
                        if key not in unique_keys:
                            txn.put(key, value)
                            unique_keys.add(key)
                        else:
                            total_skipped_items += 1

    env.close()
    print("\n----------------------------------------")
    print(f"Finished creating LMDB dataset at {lmdb_path}")
    print(f"  Total items processed and stored: {total_processed_items}")
    print(f"  Total items skipped: {total_skipped_items}")
    print(f"  Please verify the dataset and image paths.")
    print("----------------------------------------")


if __name__ == "__main__":
    tyro.cli(main)
