import json

import numpy as np

from tools.baselines_sf3d import common as C
from tools.baselines_sf3d.sf3d_to_opd import (
    CATS,
    OUT_H,
    OUT_W,
    build_image_entry,
    colmajor16,
    convert_record,
    element_diagonal,
    scale_intrinsics,
)


def _rec():
    return {
        "mask_coordinates_yx": [[y, x] for y in range(700, 760) for x in range(900, 1000)],
        "label_info": {"label": "hook_pull"},
        "motion_info": {
            "original_motion_data": {"motion_type": "rot"},
            "frame_specific_motion_data": {
                "motion_dir_3d_camera_coords": [0.0, 2.0, 0.0],
                "motion_origin_3d_camera_coords": [0.1, 0.2, 3.0],
            },
        },
        "camera_intrinsics": [[1600.0, 0, 960.0], [0, 1600.0, 720.0], [0, 0, 1]],
        "camera_extrinsics_cam_to_world": np.eye(4).tolist(),
        "image_dimensions_wh": (1920, 1440),
    }


def test_convert_record_geometry_and_mask():
    ann = convert_record(_rec(), "s/v/1.0/a", ann_id=7, image_id=3, categories={"hook_pull": 2})
    assert ann["category_id"] == 2 and ann["image_id"] == 3 and ann["id"] == 7
    assert ann["motion"]["type"] == "rotation"
    assert np.allclose(ann["motion"]["axis"], [0.0, -1.0, 0.0])  # unit + y/z flipped
    assert np.allclose(ann["motion"]["origin"], [0.1, -0.2, -3.0])
    assert ann["segmentation"]["size"] == [192, 256]
    x, y, w, h = ann["bbox"]
    assert 115 <= x <= 122 and 90 <= y <= 95 and 10 <= w <= 15 and 6 <= h <= 10
    assert ann["area"] == ann["motion"]["pixel_num"] > 0
    assert ann["motion"]["partId"] == 7 and ann["motion"]["part_label"] == "hook_pull"
    assert ann["motion"]["rangeMax"] > 1.5 and ann["height"] == OUT_H and ann["width"] == OUT_W
    # the RLE decodes to the same mask our fast pipeline would produce at 192x256
    m = C.nearest_resize(C.mask_from_coords(_rec()["mask_coordinates_yx"]), (OUT_H, OUT_W)).astype(bool)
    assert (C.rle_decode(ann["segmentation"]) == m).all()
    json.dumps(ann)  # must be serialisable as-is


def test_convert_record_translation():
    rec = _rec()
    rec["motion_info"]["original_motion_data"]["motion_type"] = "trans"
    ann = convert_record(rec, "k", ann_id=1, image_id=1, categories={"hook_pull": 1})
    assert ann["motion"]["type"] == "translation" and np.isclose(ann["motion"]["rangeMax"], 0.7)


def test_scale_intrinsics_and_colmajor():
    K = scale_intrinsics(np.array([[1600.0, 0, 960.0], [0, 1600.0, 720.0], [0, 0, 1]]), (1920, 1440), (256, 192))
    assert np.allclose(K[0, 0], 1600 * 256 / 1920) and np.allclose(K[1, 2], 720 * 192 / 1440)
    assert np.allclose(K[0, 2], 960 * 256 / 1920) and np.allclose(K[2, 2], 1.0)
    M = np.arange(16.0).reshape(4, 4)
    flat = colmajor16(M)
    assert len(flat) == 16 and flat[1] == 4.0 and np.allclose(np.array(flat).reshape(4, 4).T, M)


def test_element_diagonal_from_depth():
    rec = _rec()
    depth = np.zeros((1440, 1920), np.uint16)
    depth[700:760, 900:1000] = 2000  # 2 m plane
    d, mn, mx = element_diagonal(rec, depth)
    # 100 px wide at f=1600, z=2 -> ~0.124 m; 60 px tall -> ~0.074 m; dz = 0
    assert np.isclose(d, np.hypot(99 * 2 / 1600, 59 * 2 / 1600), atol=1e-3)
    assert len(mn) == 3 and len(mx) == 3 and mx[2] == mn[2] == 2.0
    # no valid depth -> clamped fallback, still a 3-tuple
    d0, mn0, mx0 = element_diagonal(rec, np.zeros((1440, 1920), np.uint16))
    assert d0 == 0.05 and mn0 == [0.0, 0.0, 0.0] and mx0 == [0.0, 0.0, 0.0]
    # tiny element -> clamped to >= 0.05
    rec["mask_coordinates_yx"] = [[700, 900], [700, 901], [701, 900], [701, 901]]
    assert element_diagonal(rec, depth)[0] == 0.05


def test_build_image_entry_camera_layout():
    rec = _rec()
    c2w = np.eye(4)
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    rec["camera_extrinsics_cam_to_world"] = c2w.tolist()
    img = build_image_entry(rec, image_id=5, visit="420673", frame_idx=12)
    assert img["id"] == 5 and img["file_name"] == "420673-000012.png"
    assert img["depth_file_name"] == "420673-000012_d.png"
    assert img["height"] == OUT_H and img["width"] == OUT_W
    intr = img["camera"]["intrinsic"]
    assert len(intr) == 9 and np.isclose(intr[0], 1600 * 256 / 1920) and np.isclose(intr[6], 960 * 256 / 1920)
    ext = img["camera"]["extrinsic"]
    assert len(ext) == 16 and ext[12:15] == [1.0, 2.0, 3.0]  # translation lives in the last column (col-major)
    E = np.array(ext).reshape(4, 4).T
    assert np.allclose(E, c2w @ C.F_YZ)


def test_categories_are_sorted_sf3d_labels():
    assert CATS == sorted(CATS) and len(CATS) == 8


def _portrait_rec():
    rec = _rec()
    rec["image_dimensions_wh"] = (1440, 1920)
    rec["camera_intrinsics"] = [[1500.0, 0, 700.0], [0, 1600.0, 900.0], [0, 0, 1]]
    rec["mask_coordinates_yx"] = [[y, x] for y in range(1600, 1640) for x in range(100, 200)]  # y beyond 1440
    rec["motion_info"]["frame_specific_motion_data"]["motion_dir_3d_camera_coords"] = [0.0, 1.0, 0.0]
    rec["motion_info"]["frame_specific_motion_data"]["motion_origin_3d_camera_coords"] = [0.3, -0.2, 2.0]
    c2w = np.eye(4)
    c2w[:3, :3] = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    c2w[:3, 3] = [1.0, 2.0, 3.0]
    rec["camera_extrinsics_cam_to_world"] = c2w.tolist()
    rec["camera_extrinsics_world_to_cam"] = np.linalg.inv(c2w).tolist()
    return rec


def test_roll_record_is_an_exact_camera_roll():
    import cv2

    from tools.baselines_sf3d.sf3d_to_opd import R_ROLL, needs_roll, roll_record

    rec = _portrait_rec()
    assert needs_roll(rec) and not needs_roll(_rec())
    r = roll_record(rec)
    assert tuple(r["image_dimensions_wh"]) == (1920, 1440) and r["rolled"] is True
    fs = r["motion_info"]["frame_specific_motion_data"]
    assert np.allclose(fs["motion_dir_3d_camera_coords"], [-1.0, 0.0, 0.0])  # before the OPD flip
    assert np.allclose(fs["motion_origin_3d_camera_coords"], [0.2, 0.3, 2.0])
    K_old, K_new = np.array(rec["camera_intrinsics"]), np.array(r["camera_intrinsics"])
    assert K_new[0, 0] == 1600 and K_new[1, 1] == 1500 and K_new[0, 2] == 1919 - 900 and K_new[1, 2] == 700
    # mask: rolled coordinates == cv2.ROTATE_90_CLOCKWISE of the native mask
    m_old = C.mask_from_coords(rec["mask_coordinates_yx"], 1920, 1440)
    m_new = C.mask_from_coords(r["mask_coordinates_yx"], 1440, 1920)
    assert (m_new == cv2.rotate(m_old, cv2.ROTATE_90_CLOCKWISE)).all()
    assert (np.rot90(m_new, 1) == m_old).all()  # what opd_preds_to_jsonl undoes
    # projection: old pixel (u, v) -> new pixel (h - 1 - v, u) for every camera point
    p_old = np.array([0.3, -0.2, 2.0])
    uv = (K_old @ p_old) / p_old[2]
    uv2 = (K_new @ (R_ROLL @ p_old)) / p_old[2]
    assert np.allclose(uv2[:2], [1919 - uv[1], uv[0]])
    # pose: the same world point through either camera
    c2w_old, c2w_new = np.array(rec["camera_extrinsics_cam_to_world"]), np.array(r["camera_extrinsics_cam_to_world"])
    assert np.allclose(c2w_old @ np.append(p_old, 1), c2w_new @ np.append(R_ROLL @ p_old, 1))
    w2c_new = np.array(r["camera_extrinsics_world_to_cam"])
    assert np.allclose(w2c_new @ c2w_new, np.eye(4))
    # convert_record on the rolled record: axis after the OPD flip is still (-1, 0, 0), mask lands in the right place
    ann = convert_record(r, "k", ann_id=1, image_id=1, categories={"hook_pull": 2})
    assert np.allclose(ann["motion"]["axis"], [-1.0, 0.0, 0.0])
    assert np.allclose(ann["motion"]["origin"], [0.2, -0.3, -2.0])
    x, y, w, h = ann["bbox"]
    # native rows 1600..1639 -> new cols 280..319 -> /7.5 -> ~37..42; native cols 100..199 -> new rows -> ~13..26
    assert 36 <= x <= 39 and 12 <= y <= 14 and 4 <= w <= 7 and 12 <= h <= 15
    # element diagonal is roll-invariant
    depth = np.full((1920, 1440), 2000, np.uint16)
    d_old = element_diagonal(rec, depth)[0]
    d_new = element_diagonal(r, cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE))[0]
    assert np.isclose(d_old, d_new) and d_old > 0.05
