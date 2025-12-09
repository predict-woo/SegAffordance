import lmdb
import pickle
import numpy as np
import cv2
import tyro
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import base64
from typing import Annotated

# --- Constants for Visualization (from visualize_lmdb_item.py) ---
MOTION_ORIGIN_COLOR = (255, 0, 0)  # Blue in BGR
MOTION_TRANS_COLOR = (0, 255, 0)  # Green in BGR
MOTION_ROT_COLOR = (0, 0, 255)    # Red in BGR
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_THICKNESS = 2
ARROW_LENGTH_2D = 50  # pixels

app = Flask(__name__, static_folder="static")
CORS(app)

# Global variables to hold dataset info
env = None
image_dir = None
all_keys = []


def draw_visualization(image: np.ndarray, item_data: dict) -> np.ndarray:
    """
    Draws mask and motion information onto an image.
    """
    vis_image = image.copy()

    # Draw Mask by inverting colors
    mask_coords_yx = item_data.get("mask_coordinates_yx")
    if mask_coords_yx:
        mask = np.zeros(vis_image.shape[:2], dtype=np.uint8)
        points = np.array(mask_coords_yx)[:, [1, 0]]  # convert to (x,y) for OpenCV
        hull = cv2.convexHull(points)
        cv2.fillPoly(mask, [hull], (255,))
        vis_image[mask > 0] = 255 - vis_image[mask > 0]

    # Draw Motion
    motion_info = item_data.get("motion_info", {})
    frame_motion_data = motion_info.get("frame_specific_motion_data")
    if frame_motion_data:
        motion_origin_xy_list = frame_motion_data.get("motion_origin_2d_image_coords")
        if motion_origin_xy_list:
            p_origin = tuple(map(int, motion_origin_xy_list))
            cv2.circle(vis_image, p_origin, 10, MOTION_ORIGIN_COLOR, -1)

            motion_type = motion_info.get("original_motion_data", {}).get("motion_type")
            if motion_type == "trans":
                v_dir_3d_cam = np.array(frame_motion_data.get("motion_dir_3d_camera_coords", [0, 0, 0]))
                v_dir_2d = v_dir_3d_cam[:2]
                norm = np.linalg.norm(v_dir_2d)
                if norm > 1e-6:
                    v_dir_2d /= norm
                p_target = (
                    int(p_origin[0] + v_dir_2d[0] * ARROW_LENGTH_2D),
                    int(p_origin[1] + v_dir_2d[1] * ARROW_LENGTH_2D),
                )
                cv2.arrowedLine(vis_image, p_origin, p_target, MOTION_TRANS_COLOR, LINE_THICKNESS, tipLength=0.3)
            elif motion_type == "rot":
                v_axis_3d_cam = np.array(frame_motion_data.get("motion_axis_3d_camera_coords", [0, 0, 0]))
                v_axis_2d = v_axis_3d_cam[:2]
                norm = np.linalg.norm(v_axis_2d)
                if norm > 1e-6:
                    v_axis_2d /= norm
                p_target_rot = (
                    int(p_origin[0] + v_axis_2d[0] * ARROW_LENGTH_2D),
                    int(p_origin[1] + v_axis_2d[1] * ARROW_LENGTH_2D),
                )
                cv2.arrowedLine(vis_image, p_origin, p_target_rot, MOTION_ROT_COLOR, LINE_THICKNESS, tipLength=0.3)

    return vis_image


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/items")
def get_items():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)

    start_index = (page - 1) * per_page
    end_index = start_index + per_page
    
    if start_index >= len(all_keys):
        return jsonify([])

    keys_to_fetch = all_keys[start_index:end_index]
    items = []

    with env.begin(write=False) as txn:
        for key_bytes in keys_to_fetch:
            value_bytes = txn.get(key_bytes)
            if not value_bytes:
                continue
            
            item_data = pickle.loads(value_bytes)
            key_str = key_bytes.decode("utf-8")

            # The filename in the DB is flat: visit_id_video_id_timestamp.ext
            # The actual file structure is nested: visit_id/video_id/timestamp.ext
            image_filename_flat = item_data["rgb_image_path"]
            print("image_filename_flat", image_filename_flat)
            try:
                parts = image_filename_flat.rsplit('_', 2)
                visit_id = parts[0]
                video_id = parts[1]
                timestamp_filename = parts[2]
                # img_path = image_dir / visit_id / video_id / timestamp_filename
                img_path = image_dir / f"{visit_id}_{video_id}_{timestamp_filename}"
            except IndexError:
                # Fallback to the original flat structure if parsing fails
                img_path = image_dir / image_filename_flat

            if img_path.exists():
                img = cv2.imread(str(img_path))
                vis_img = draw_visualization(img, item_data)
                _, buffer = cv2.imencode(".jpg", vis_img)
                img_base64 = base64.b64encode(buffer).decode("utf-8")
            else:
                print(f"[DEBUG] Image not found. Checked path: {img_path.resolve()}")
                img_base64 = None

            items.append(
                {
                    "key": key_str,
                    "image": img_base64,
                    "description": item_data.get("description", "No description."),
                    "label": item_data.get("label_info", {}).get("label", "N/A"),
                }
            )

    return jsonify(items)


def main(
    images_dir_path: Annotated[
        Path, tyro.conf.arg(help="Path to the directory containing the RGB images.")
    ],
    lmdb_dir_path: Annotated[
        Path,
        tyro.conf.arg(help="Path to the directory containing the LMDB 'data.lmdb' file."),
    ],
    port: Annotated[int, tyro.conf.arg(help="Port to run the web server on.")] = 5000,
):
    """
    Launches a web-based LMDB dataset explorer.
    """
    global env, image_dir, all_keys

    image_dir = images_dir_path
    lmdb_path = lmdb_dir_path / "data.lmdb"

    if not image_dir.is_dir():
        print(f"Error: Image directory not found at {image_dir}")
        return
    if not lmdb_path.exists():
        print(f"Error: LMDB database not found at {lmdb_path}")
        return

    env = lmdb.open(
        str(lmdb_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=126, # A high number for web servers
    )

    with env.begin(write=False) as txn:
        print("Loading all keys from LMDB. This may take a moment...")
        all_keys = [key for key, _ in txn.cursor() if key != b"__metadata__"]
    
    print(f"Loaded {len(all_keys)} keys.")
    print(f"Starting web server at http://127.0.0.1:{port}")
    app.run(port=port, debug=True)


if __name__ == "__main__":
    tyro.cli(main)
