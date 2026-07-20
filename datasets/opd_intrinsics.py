"""
Camera-intrinsics handling shared by the OPD dataset loaders and
annotation-filtering tools.

The three OPD datasets store camera intrinsics differently:
  - OPDReal:  camera.intrinsic = {"matrix": [9 floats, column-major]}
  - OPDMulti: camera.intrinsic = [9 floats, column-major]
  - OPDSynth: camera.intrinsic = {"fov": degrees, "aspect": w/h} — no matrix.

For OPDSynth the upstream OPD repo (motlib/utils/motion_visualizer.py,
getFocalLength/camera_to_image) projects with a fixed vertical FOV and an
OpenGL-style camera that looks down -z with y up:

    f  = height / (2 * tan(FOV/2))
    x  =   px * fx / (-pz) + cx
    y  = -(py * fy / (-pz)) + cy

That projection is exactly reproduced by the single matrix

    M = [[fx, 0, -cx], [0, -fy, -cy], [0, 0, -1]]

applied as homo = M @ p; x = homo[0]/homo[2]; y = homo[1]/homo[2] — the same
code path used for the real/multi matrices, so callers never need to branch.
"""

import math

import numpy as np


def synth_intrinsic_matrix(fov_deg: float, width: int, height: int) -> np.ndarray:
    """Equivalent 3x3 projection matrix for OPDSynth's {fov, aspect} cameras."""
    fov = math.radians(fov_deg)
    # Upstream convention (getFocalLength): fx from the vertical FOV and
    # height, fy scaled by the aspect ratio. For square images fx == fy.
    fx = height / (2 * math.tan(fov / 2))
    fy = fx / height * width
    cx = width / 2.0
    cy = height / 2.0
    return np.array(
        [
            [fx, 0.0, -cx],
            [0.0, -fy, -cy],
            [0.0, 0.0, -1.0],
        ]
    )


def intrinsic_matrix_from_camera(image_dict: dict, is_multi: bool = False) -> np.ndarray:
    """Return a 3x3 projection matrix for any OPD image dict."""
    intrinsic = image_dict["camera"]["intrinsic"]
    if is_multi or isinstance(intrinsic, list):
        return np.reshape(intrinsic, (3, 3), order="F")
    if "matrix" in intrinsic:
        return np.reshape(intrinsic["matrix"], (3, 3), order="F")
    if "fov" in intrinsic:
        return synth_intrinsic_matrix(
            intrinsic["fov"], image_dict["width"], image_dict["height"]
        )
    raise KeyError(f"Unrecognized camera intrinsic format: {intrinsic!r}")
