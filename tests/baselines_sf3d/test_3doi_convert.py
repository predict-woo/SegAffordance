import numpy as np
from pycocotools import mask as mask_util

from tools.baselines_sf3d import common as C
from tools.baselines_sf3d.sf3d_to_3doi import (
    OUT_H,
    OUT_W,
    axis_line_2d,
    bbox_norm,
    clip_line_to_rect,
    element_point,
    hfov_rads,
    img_name,
    instance_from_record,
    mask_polygon,
    resize_mask,
)

K_OUT = [[1600.0 * OUT_W / 1920, 0, 960.0 * OUT_W / 1920], [0, 1600.0 * OUT_H / 1440, 720.0 * OUT_H / 1440], [0, 0, 1]]


def test_names_parse_like_their_loader():
    n = img_name("420673", 17)
    s = n.split("_")
    assert s[0] == "taskonomy" and s[1] == "420673" and int(s[3]) == 17 and int(s[5]) == 0


def test_hfov_matches_focal():
    fov = hfov_rads(K_OUT)
    assert abs(OUT_W / 2 / np.tan(fov / 2) - K_OUT[0][0]) < 1e-6  # test.py:463 recovers fx from the fov


def _blob(y0=300, y1=360, x0=600, x1=700):
    coords = [[y, x] for y in range(y0, y1) for x in range(x0, x1)]
    return resize_mask(C.mask_from_coords(coords, 1440, 1920))


def test_polygon_rasterises_to_the_mask():
    m = _blob()
    poly = mask_polygon(m)
    assert len(poly) >= 4 and all(0 <= x <= 1 and 0 <= y <= 1 for x, y in poly)
    flat = np.array(poly)
    flat[:, 0] *= OUT_W
    flat[:, 1] *= OUT_H
    rle = mask_util.merge(mask_util.frPyObjects([flat.astype(int).reshape(-1).tolist()], OUT_H, OUT_W))
    r = mask_util.decode(rle).astype(bool)
    inter, union = (r & m.astype(bool)).sum(), (r | m.astype(bool)).sum()
    assert inter / union > 0.7  # dilation by 3 px pads a 53x32 px blob
    assert mask_polygon(np.zeros((OUT_H, OUT_W), np.uint8)) == []


def test_element_point_and_bbox():
    m = _blob()
    x, y = element_point(m)
    assert m[int(y * OUT_H), int(x * OUT_W)] == 1
    x1, y1, x2, y2 = bbox_norm(m)
    assert 0 < x1 < x < x2 <= 1 and 0 < y1 < y < y2 <= 1
    # a ring: the centroid is outside the mask, the point snaps into it
    ring = np.zeros((OUT_H, OUT_W), np.uint8)
    ring[300:400, 500:600] = 1
    ring[320:380, 520:580] = 0
    x, y = element_point(ring)
    assert ring[int(round(y * OUT_H)), int(round(x * OUT_W))] == 1


def test_clip_line_to_rect():
    seg = clip_line_to_rect([0.5, 0.5], [1.0, 0.0])
    assert np.allclose(seg, [0.01, 0.5, 0.99, 0.5])
    seg = clip_line_to_rect([0.5, 0.5], [0.0, 1.0])
    assert np.allclose(seg, [0.5, 0.01, 0.5, 0.99])
    seg = clip_line_to_rect([0.0, 0.0], [1.0, 1.0])
    assert np.allclose(seg, [0.01, 0.01, 0.99, 0.99]) and seg[0] > 0
    assert clip_line_to_rect([2.0, 0.5], [0.0, 1.0]) is None


def test_axis_line_2d_vertical_axis():
    seg = axis_line_2d([0.0, 0.0, 2.0], [0.0, 1.0, 0.0], K_OUT)  # vertical axis through the principal point
    assert seg[0] > 0 and abs(seg[0] - 0.5) < 1e-6 and abs(seg[2] - 0.5) < 1e-6 and seg[1] < seg[3]
    # axis along the viewing ray through the principal point projects to a point -> invalid
    assert axis_line_2d([0.0, 0.0, 2.0], [0.0, 0.0, 1.0], K_OUT) == [-1.0, -1.0, -1.0, -1.0]
    # a slanted axis: both endpoints on the line through its projection
    seg = axis_line_2d([0.3, -0.2, 2.5], [0.6, 0.8, 0.0], K_OUT)
    K = np.array(K_OUT)
    o = np.array([0.3, -0.2, 2.5])
    u0 = np.array([K[0, 0] * o[0] / o[2] + K[0, 2], K[1, 1] * o[1] / o[2] + K[1, 2]]) / [OUT_W, OUT_H]
    a, b = np.array(seg[:2]), np.array(seg[2:])
    d = (b - a) / np.linalg.norm(b - a)
    off = (u0 - a) - np.dot(u0 - a, d) * d
    assert np.linalg.norm(off) < 1e-6 and seg[0] > 0


def test_instance_from_record():
    rec = {
        "mask_coordinates_yx": [[y, x] for y in range(300, 360) for x in range(600, 700)],
        "motion_info": {
            "original_motion_data": {"motion_type": "rot"},
            "frame_specific_motion_data": {
                "motion_dir_3d_camera_coords": [0.0, 1.0, 0.0],
                "motion_origin_3d_camera_coords": [0.1, 0.0, 2.0],
            },
        },
        "image_dimensions_wh": (1920, 1440),
    }
    inst = instance_from_record(rec, K_OUT)
    assert inst["kinematic"] == "rotation" and inst["movable"] == "one_hand" and inst["rigid"] == "yes"
    assert inst["pull_or_push"] == "n/a" and inst["affordance"] == inst["keypoint"]
    assert inst["axis"][0] > 0 and len(inst["mask"]) >= 4 and len(inst["bbox"]) == 4
    rec["motion_info"]["original_motion_data"]["motion_type"] = "trans"
    assert instance_from_record(rec, K_OUT)["kinematic"] == "translation"
    rec["mask_coordinates_yx"] = []
    assert instance_from_record(rec, K_OUT) is None
