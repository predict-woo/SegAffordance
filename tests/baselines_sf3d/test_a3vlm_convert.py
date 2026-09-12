import numpy as np

from tools.baselines_sf3d import common as C
from tools.baselines_sf3d.sf3d_to_a3vlm import (
    AXIS_RE,
    BOX_RE,
    box_points,
    det_answer,
    element_geometry,
    fmt_axis,
    fmt_box,
    pad_params,
    parse_axis,
    parse_box,
    project_uvd,
    unproject_uvd,
)

K = np.array([[1600.0, 0, 960.0], [0, 1600.0, 720.0], [0, 0, 1]])


def test_pad_params_matches_pad_to_square():
    assert pad_params(1920, 1440) == (0, 240, 1920)  # landscape: paste at (0, (w-h)//2)
    assert pad_params(1440, 1920) == (240, 0, 1920)  # portrait: paste at ((h-w)//2, 0)


def test_project_unproject_round_trip():
    pad = pad_params(1920, 1440)
    pts = np.array([[0.1, -0.2, 2.0], [-0.5, 0.3, 3.5], [0.0, 0.0, 1.0]])
    uvd = project_uvd(pts, K, pad, 1.0, 4.0)
    assert uvd.shape == (3, 3) and (uvd >= 0).all() and (uvd <= 1).all()
    # principal point of the native frame lands at the padded-square centre
    assert np.allclose(uvd[2], [0.5, 0.5, 0.0])
    back = unproject_uvd(uvd, K, pad, 1.0, 4.0)
    assert np.allclose(back, pts, atol=1e-9)


def test_box_points_order_is_theirs():
    p = box_points([0, 0, 0], [2, 4, 6])
    assert np.allclose(p[0], [-1, -2, -3]) and np.allclose(p[1], [1, -2, -3]) and np.allclose(p[2], [-1, 2, -3])
    assert np.allclose(p[3], [-1, -2, 3]) and np.allclose(p[4], [1, 2, 3]) and np.allclose(p[5], [-1, 2, 3])
    assert np.allclose(p[6], [1, -2, 3]) and np.allclose(p[7], [1, 2, -3])


def test_format_parse_round_trip_and_regexes():
    uvd = np.round(np.random.RandomState(0).rand(8, 3), 2)
    s = fmt_box(uvd)
    assert BOX_RE.fullmatch(s) and np.allclose(parse_box("blah " + s + "."), uvd)
    a = np.round(np.random.RandomState(1).rand(2, 3), 2)
    t = fmt_axis("revolute", a)
    assert t.startswith("<axis>revolute</axis>[") and AXIS_RE.fullmatch(t)
    jt, pts = parse_axis(t)
    assert jt == "revolute" and np.allclose(pts, a)
    assert parse_axis("<axis>revolute</axis>[0.1,0.2]") is None and parse_box("[[1,2]]") is None


def _rec(mtype="rot"):
    coords = [[y, x] for y in range(700, 760) for x in range(900, 1000)]
    return {
        "mask_coordinates_yx": coords,
        "label_info": {"label": "hook_pull"},
        "description": "open the drawer",
        "motion_info": {
            "original_motion_data": {"motion_type": mtype},
            "frame_specific_motion_data": {
                "motion_dir_3d_camera_coords": [0.0, 1.0, 0.0],
                "motion_origin_3d_camera_coords": [0.1, 0.0, 2.0],  # the foot point takes the centroid's y
            },
        },
        "camera_intrinsics": K.tolist(),
        "image_dimensions_wh": (1920, 1440),
    }


def test_element_geometry_on_a_flat_patch():
    depth = np.zeros((1440, 1920))
    depth[700:760, 900:1000] = 2.0  # a fronto-parallel patch at z = 2 m
    g = element_geometry(_rec(), depth)
    assert g["joint_type"] == "revolute" and g["n_valid"] == 6000
    # centre/extent from the back-projected pixels
    cx = ((900 + 999) / 2 - 960) * 2.0 / 1600
    cy = ((700 + 759) / 2 - 720) * 2.0 / 1600
    assert abs(g["center"][0] - cx) < 0.01 and abs(g["center"][1] - cy) < 0.01 and abs(g["center"][2] - 2.0) < 1e-6
    assert abs(g["extent"][0] - 99 * 2 / 1600 * 0.96) < 0.02 and g["extent"][2] == 0.02
    # axis endpoints: on the GT line (x = 0.1, z = 2), symmetric about the centroid's foot, direction = +y
    p0, p1 = g["axis_pts"]
    assert np.allclose([p0[0], p0[2]], [0.1, 2.0]) and np.allclose([p1[0], p1[2]], [0.1, 2.0])
    assert p1[1] > p0[1] and abs((p0[1] + p1[1]) / 2 - cy) < 0.01 and abs(p1[1] - p0[1] - 0.6) < 1e-9  # longest in-frame segment
    # a segment that would leave the padded square shrinks to the longest one that stays inside
    r2 = _rec()
    r2["motion_info"]["frame_specific_motion_data"]["motion_dir_3d_camera_coords"] = [1.0, 0.0, 0.0]
    r2["mask_coordinates_yx"] = [[y, x] for y in range(700, 760) for x in range(1850, 1920)]
    depth2 = np.zeros((1440, 1920))
    depth2[700:760, 1850:1920] = 2.0  # patch at the right edge: +-0.3 m in x leaves the frame
    g2 = element_geometry(r2, depth2)
    assert g2["axis_len"] < 0.6 and g2["axis_len"] in (0.5, 0.4, 0.3, 0.2, 0.1)
    q0, q1 = g2["axis_pts"]
    assert abs(q1[0] - q0[0] - g2["axis_len"]) < 1e-9 and q0[1] == q1[1] and q0[2] == q1[2] == 2.0
    # prismatic: anchored at the centroid instead of the axis foot
    gt = element_geometry(_rec("trans"), depth)
    assert gt["joint_type"] == "prismatic" and abs(gt["axis_pts"].mean(0)[0] - cx) < 0.01


def test_element_geometry_needs_depth():
    assert element_geometry(_rec(), np.zeros((1440, 1920))) is None


def test_gt_round_trip_through_serialisation():
    """What the model must reproduce: axis string -> our JSONL geometry within 2-decimal quantisation."""
    depth = np.zeros((1440, 1920))
    depth[700:760, 900:1000] = 2.0
    depth[0, 0] = 1.0
    depth[1, 1] = 4.0  # d_min = 1, d_max = 4
    g = element_geometry(_rec(), depth)
    pad = pad_params(1920, 1440)
    s = fmt_axis(g["joint_type"], project_uvd(g["axis_pts"], K, pad, 1.0, 4.0))
    jt, uvd = parse_axis(s)
    pts = unproject_uvd(uvd, K, pad, 1.0, 4.0)
    axis = pts[1] - pts[0]
    axis /= np.linalg.norm(axis)
    ang = np.degrees(np.arccos(abs(np.dot(axis, [0, 1, 0]))))
    assert ang < 3.0  # 0.01 quantisation of u,v,d over a 60 cm segment at 2 m


def test_det_answer_format():
    s = det_answer([("hook pull", "[[0.1,0.2,0.3]]"), ("tip push", "[[0.4,0.5,0.6]]")])
    assert s == "There are two manipulable object parts with their 3d bounding boxes: <box>hook pull</box>[[0.1,0.2,0.3]],<box>tip push</box>[[0.4,0.5,0.6]]."
    assert det_answer([("a", "[[0]]")]).startswith("There is one manipulable object part with its 3d bounding box: ")
    assert det_answer([("a", "[[0]]")] * 12).count("<box>") == 10


def test_nearest_resize_unaffected():
    m = C.mask_from_coords([[0, 0]], 4, 4)
    assert m.sum() == 1
