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
import shutil
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

# --- Mask rasterisation ---------------------------------------------------
# The annotated element is a set of laser-scan points; the mask is their
# footprint in the image. The original writer took the CONVEX HULL of the
# projected points, which is decided entirely by the extremal points: it
# over-covered concave hardware (D-handles, taps, lever latches), collapsed to
# a sliver whenever the surviving points were near-collinear (thin bars), and
# swung ~4x in area between frames of the same element. Measured on the
# rebuilt LMDB: median mask 704 px, min 18 px, and 258 -> 993 px across frames
# of one cabinet handle.
#
# Instead, splat every point at the radius its own scan patch subtends at its
# own depth, then close the gaps between neighbouring splats. The radius comes
# from the known scan resolution rather than a tuned parameter.
LASER_SCAN_SPACING_M = 0.005  # the downloaded asset is laser_scan_5mm
SPLAT_RADIUS_SCALE = 0.75  # patch half-width -> splat radius
SPLAT_MIN_RADIUS_PX = 1
SPLAT_MAX_RADIUS_PX = 12
# Longest MST edge that may be drawn as a bridge between splats. Beyond this,
# two visible fragments are assumed to be genuinely separated (an occluder
# between them) and are left as separate components rather than welded.
MST_MAX_BRIDGE_PX = 60.0
MASK_METHODS = ("splat", "hull")


# --- Sensor-depth (ARKit) occlusion gate ---------------------------------
# hires_depth is NOT a measurement: SceneFun3D renders it from the laser scan
# into the hires camera. So the visibility test in the visibility pass compares
# projected laser points against a render of that same laser data, and any
# surface MISSING from the scan is invisible to it by construction. Verified
# 2026-07-27 on visit 464982: a dark glossy tiled wall is absent from the scan,
# so hires_depth looks straight through it into the next room and a sink
# cabinet's pinch-pull scored "100% visible, gap 0.000 m" in frames whose photo
# shows only tile and a toilet.
#
# lowres_depth is the actual ARKit LiDAR capture and physically cannot see
# through a wall, so it is the only independent evidence in the dataset. Points
# whose laser depth sits well BEHIND the sensor surface are genuinely occluded.
SENSOR_DEPTH_ABS_TOL_M = 0.10   # allow for sensor noise / registration slop
SENSOR_DEPTH_REL_TOL = 0.05     # plus 5% of range (vs the 25% used laser-side)
SENSOR_TIME_TOL_S = 0.10        # hires <-> lowres frame pairing window
SENSOR_MIN_CONFIDENCE = 1       # ARKit confidence: 0 low, 1 medium, 2 high
SENSOR_MAX_OCCLUDED_FRAC = 0.5  # reject the item above this


class LowresSource:
    """Per-video access to the ARKit lowres assets, from .zip or a directory.

    tools/sf3d_fetch_lowres_zips.py downloads these assets as archives WITHOUT
    extracting them: unpacking ~3-6k tiny PNGs per video onto the MooseFS
    volume runs at ~0.8 MB/s and does not parallelise, while fetching the
    archives is network-bound at ~168 MB/s. Reading members straight out of
    the zip keeps that win and leaves ~609 files on the volume instead of
    millions.

    Falls back to an extracted directory when no archive is present, so a
    partially-extracted tree from an earlier run still works.
    """

    def __init__(self, data_root, visit_id, video_id):
        import zipfile

        base = Path(data_root) / str(visit_id) / str(video_id)
        self._zips = {}
        self.depth = self._index(base, "lowres_depth", ".png", zipfile)
        self.intrinsics = self._index(base, "lowres_wide_intrinsics", ".pincam", zipfile)
        self.timestamps = np.array(
            sorted(float(t) for t in self.depth), dtype=np.float64
        )

    def _index(self, base, asset, suffix, zipfile_mod):
        """timestamp -> ('zip', member) | ('file', path). Timestamps are the
        trailing '<video_id>_<timestamp>.<ext>' component, as DataParser keys."""
        out = {}
        archive = base / f"{asset}.zip"
        if archive.is_file():
            zf = zipfile_mod.ZipFile(archive)
            self._zips[asset] = zf
            for member in zf.namelist():
                if member.endswith(suffix):
                    stem = member.rsplit("/", 1)[-1][: -len(suffix)]
                    out[stem.rsplit("_", 1)[-1]] = ("zip", asset, member)
            return out
        directory = base / asset
        if directory.is_dir():
            for path in directory.iterdir():
                if path.suffix == suffix:
                    out[path.stem.rsplit("_", 1)[-1]] = ("file", asset, str(path))
        return out

    def _read_bytes(self, entry):
        kind, asset, ref = entry
        if kind == "zip":
            return self._zips[asset].read(ref)
        with open(ref, "rb") as fh:
            return fh.read()

    def read_depth(self, timestamp_key):
        """Sensor depth in METRES (the PNGs are uint16 millimetres)."""
        entry = self.depth.get(timestamp_key)
        if entry is None:
            return None
        buf = np.frombuffer(self._read_bytes(entry), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return img.astype(np.float32) / 1000.0

    def read_intrinsics(self, timestamp_key):
        """3x3 K for the lowres camera. .pincam is one line: w h fx fy cx cy."""
        entry = self.intrinsics.get(timestamp_key)
        if entry is None:
            return None
        parts = self._read_bytes(entry).decode().split()
        if len(parts) < 6:
            return None
        _, _, fx, fy, cx, cy = (float(p) for p in parts[:6])
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

    def nearest_timestamp(self, t, tolerance):
        if self.timestamps.size == 0:
            return None
        nearest = self.timestamps[int(np.argmin(np.abs(self.timestamps - t)))]
        return nearest if abs(nearest - t) <= tolerance else None

    def close(self):
        for zf in self._zips.values():
            try:
                zf.close()
            except Exception:
                pass


def sensor_occlusion_stats(
    points_laser,
    laser_to_arkit,
    arkit_cam_to_world,
    intrinsics_lowres,
    depth_lowres,
    confidence_lowres=None,
):
    """Compare laser points against the ARKit sensor depth for one frame.

    Args:
        points_laser: (N,3) element points in laser-scan coordinates.
        laser_to_arkit: 4x4 from DataParser.get_transform (laser -> ARKit).
        arkit_cam_to_world: 4x4 ARKit camera-to-world pose (ARKit frame).
        intrinsics_lowres: 3x3 K of the lowres camera.
        depth_lowres: (h,w) sensor depth in metres, 0 where invalid.
        confidence_lowres: (h,w) ARKit confidence, optional.

    Returns:
        dict with counts, or None if nothing could be evaluated.
    """
    pts = np.asarray(points_laser, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0 or depth_lowres is None:
        return None

    homo = np.hstack([pts, np.ones((len(pts), 1))])
    pts_arkit = (np.asarray(laser_to_arkit, dtype=np.float64) @ homo.T).T
    world_to_cam = hm.inverse(np.asarray(arkit_cam_to_world, dtype=np.float64))
    pts_cam = (world_to_cam @ pts_arkit.T).T
    w4 = pts_cam[:, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        pts_cam = np.where(np.abs(w4) > 1e-9, pts_cam[:, :3] / w4, pts_cam[:, :3])

    z = pts_cam[:, 2]
    in_front = z > 1e-6
    if not in_front.any():
        return None

    uv = (np.asarray(intrinsics_lowres, dtype=np.float64) @ pts_cam[in_front].T).T
    uv = uv[:, :2] / uv[:, 2:3]
    zf = z[in_front]

    h, w = depth_lowres.shape[:2]
    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not inside.any():
        return None
    u, v, zf = u[inside], v[inside], zf[inside]

    sensor = depth_lowres[v, u]
    valid = sensor > 1e-6
    if confidence_lowres is not None and confidence_lowres.shape[:2] == depth_lowres.shape[:2]:
        valid &= confidence_lowres[v, u] >= SENSOR_MIN_CONFIDENCE
    if not valid.any():
        return None

    sensor, zf = sensor[valid], zf[valid]
    tol = SENSOR_DEPTH_ABS_TOL_M + SENSOR_DEPTH_REL_TOL * sensor
    occluded = sensor < zf - tol       # a real surface in front of the point
    agreeing = np.abs(sensor - zf) <= tol
    n = int(valid.sum())
    return {
        "sensor_points_evaluated": n,
        "sensor_occluded_frac": float(occluded.sum()) / n,
        "sensor_agreeing_frac": float(agreeing.sum()) / n,
        "sensor_median_gap_m": float(np.median(sensor - zf)),
    }


def annotation_bounding_spheres(all_annotations, laser_scan_points):
    """World-space (centre, radius) per annotation, computed once per video.

    Feeds the per-frame frustum rejection below. Entries are None for
    annotations that can never produce a record (excluded, or no indices).
    """
    spheres = []
    for annotation in all_annotations:
        indices = annotation.get("indices")
        if annotation.get("label") == "exclude" or not indices:
            spheres.append(None)
            continue
        pts = laser_scan_points[indices]
        if pts.size == 0:
            spheres.append(None)
            continue
        centre = pts.mean(axis=0)
        radius = float(np.linalg.norm(pts - centre, axis=1).max())
        spheres.append((centre, radius))
    return spheres


def _frame_may_contain_items(spheres, world_to_camera_pose, K, width, height, margin_px=4.0):
    """Can any annotated element possibly project into this frame?

    Roughly 40% of hires frames contain no annotated element at all (345k
    frames across the 609 videos versus 206k that produced records), yet the
    original ordering read every frame's multi-MB depth map from the network
    volume before finding that out. This test uses only the pose and
    intrinsics, both already in hand.

    Deliberately conservative -- it asks whether the annotation's bounding
    SPHERE overlaps the image, and PointCloudToImageMapper's visible set is a
    strict subset of that (it additionally requires the depth check). So a
    frame this rejects could not have produced a record. Points behind the
    camera, and cameras inside the sphere, both fall through to "keep".
    """
    R = world_to_camera_pose[:3, :3]
    t = world_to_camera_pose[:3, 3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    for sphere in spheres:
        if sphere is None:
            continue
        centre, radius = sphere
        pc = R @ centre + t
        z = pc[2]
        if z + radius <= 1e-6:  # entirely behind the camera
            continue
        # Use the sphere's NEAREST depth so the projected disc is the largest
        # it could be; when the camera is inside the sphere this explodes and
        # the frame is kept, which is the safe direction.
        z_near = max(z - radius, 1e-3)
        z_safe = max(z, 1e-6)
        u = fx * pc[0] / z_safe + cx
        v = fy * pc[1] / z_safe + cy
        ru = abs(fx) * radius / z_near + margin_px
        rv = abs(fy) * radius / z_near + margin_px
        if (u + ru >= 0) and (u - ru < width) and (v + rv >= 0) and (v - rv < height):
            return True
    return False


def project_trajectory_to_2d(trajectory_cam, K, width, height):
    """Project a camera-frame trajectory into the image.

    The 2D track is just the 3D trajectory seen by this frame's camera, with
    the SAME point ordering, so a reader that subsamples both (the SF3D reader
    takes 20 uniformly) keeps 2D point i corresponding to 3D point i.

    Points behind the camera cannot be projected at all; points in front may
    still land outside the frame (a 90 deg arc on a large door regularly
    leaves it). Both cases are reported through the validity mask rather than
    silently clamped, because clamping would invent a plausible-looking pixel
    that the 2D head would then be trained to reproduce.

    Returns:
        (coords, valid) where coords is a length-N list of [u, v] in PIXELS of
        the full-resolution frame (0,0 for unprojectable points) and valid is a
        length-N list of bools meaning "in front of the camera AND inside the
        image".
    """
    pts = np.asarray(trajectory_cam, dtype=np.float64).reshape(-1, 3)
    coords = np.zeros((len(pts), 2), dtype=np.float64)
    if len(pts) == 0:
        return coords.tolist(), []

    z = pts[:, 2]
    in_front = z > 1e-6
    if in_front.any():
        homo = (np.asarray(K, dtype=np.float64) @ pts[in_front].T).T
        coords[in_front] = homo[:, :2] / homo[:, 2:3]

    valid = (
        in_front
        & (coords[:, 0] >= 0)
        & (coords[:, 0] < float(width))
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < float(height))
    )
    return coords.tolist(), valid.tolist()


def _world_to_camera(points_world, world_to_camera_pose):
    """(N,3) world points -> (N,3) camera-frame points."""
    pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    cam = (world_to_camera_pose @ homo.T).T
    return cam[:, :3] / cam[:, 3:4]


def _bridge_along_mst(mask_image, pts_xy, radii):
    """Join the splats into one region along the points' minimum spanning tree.

    Blanket morphological closing cannot do this job: sizing the kernel from
    the splat radius, or from the MEDIAN nearest-neighbour distance, both leave
    disconnected chains of blobs on thin hardware (a tightly clustered point
    set with a few far outliers has a small median NN but large real gaps).
    Measured on 98 samples: 3/98 and 9/98 masks fragmented respectively.

    The MST is by construction the shortest set of edges that connects every
    point, so drawing each edge at the local splat thickness guarantees a
    single component while adding minimal area and leaving genuine concavities
    (e.g. the hole of a D-handle) intact. Edges longer than
    MST_MAX_BRIDGE_PX are left undrawn so that two genuinely separate visible
    fragments of an occluded element are not welded across the occluder.
    """
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    if n < 2:
        return
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial import cKDTree
        from scipy.spatial.distance import pdist, squareform

        if n <= 1500:
            # Dense MST. minimum_spanning_tree treats its input as undirected
            # but does NOT symmetrise it, so a one-directional kNN graph
            # silently loses most edges (this left a 46-point drawer pull in 3
            # pieces). A full symmetric distance matrix has no such trap and is
            # cheap at these sizes.
            mst = minimum_spanning_tree(squareform(pdist(pts))).tocoo()
        else:
            k = min(n, 6)
            d, idx = cKDTree(pts).query(pts, k=k)
            rows = np.repeat(np.arange(n), k - 1)
            cols = idx[:, 1:].ravel()
            vals = np.maximum(d[:, 1:].ravel(), 1e-6)
            graph = csr_matrix((vals, (rows, cols)), shape=(n, n))
            graph = graph.maximum(graph.T)  # make it undirected
            mst = minimum_spanning_tree(graph).tocoo()
        edges = list(zip(mst.row, mst.col, mst.data))
    except Exception:
        # Fallback: chain points in nearest-neighbour order along x, which
        # still connects a thin bar (the case that matters most).
        order = np.argsort(pts[:, 0])
        edges = (
            (order[i], order[i + 1], float(np.linalg.norm(pts[order[i]] - pts[order[i + 1]])))
            for i in range(n - 1)
        )

    for i, j, weight in edges:
        if weight > MST_MAX_BRIDGE_PX:
            continue
        thickness = max(1, int(round((radii[i] + radii[j]) * 0.5)))
        cv2.line(
            mask_image,
            (int(pts[i][0]), int(pts[i][1])),
            (int(pts[j][0]), int(pts[j][1])),
            (255,),
            thickness,
        )


def rasterize_mask(points_2d_yx, points_3d_cam, fx, h, w, method="splat"):
    """Rasterise the visible laser points of one element into a binary mask.

    Args:
        points_2d_yx: (N,2) projected (row, col) of the visible points.
        points_3d_cam: (N,3) the same points in camera coords (for depth), or
            an empty array to fall back to a constant splat radius.
        fx: focal length in pixels, used to convert the metric scan spacing
            into a pixel radius.
        method: "splat" (default) or "hull" (the original behaviour, kept so
            the two can be compared on identical inputs).
    """
    mask_image = np.zeros((int(h), int(w)), dtype=np.uint8)
    pts_xy = np.asarray(points_2d_yx)[:, [1, 0]].astype(np.int32)  # (y,x)->(x,y)
    if len(pts_xy) == 0:
        return mask_image

    if method == "hull":
        cv2.fillPoly(mask_image, [cv2.convexHull(pts_xy)], (255,))
        return mask_image

    pts_cam = np.asarray(points_3d_cam, dtype=np.float64).reshape(-1, 3)
    if len(pts_cam) == len(pts_xy) and len(pts_cam):
        z = np.maximum(pts_cam[:, 2], 1e-3)
    else:
        # No depth available: one radius for all, from the median scene depth
        # a hires frame typically sees. Only hit on malformed input.
        z = np.full(len(pts_xy), 2.0)

    radii = np.clip(
        LASER_SCAN_SPACING_M * float(fx) / z * SPLAT_RADIUS_SCALE,
        SPLAT_MIN_RADIUS_PX,
        SPLAT_MAX_RADIUS_PX,
    )
    for (px, py), rad in zip(pts_xy, radii):
        cv2.circle(mask_image, (int(px), int(py)), int(round(rad)), (255,), -1)

    # Join the splats into a single region along the points' MST, then a small
    # radius-sized close to smooth the seams. See _bridge_along_mst for why a
    # plain morphological close is not sufficient here.
    _bridge_along_mst(mask_image, pts_xy, radii)
    close_k = int(np.clip(round(float(np.median(radii))), 3, 15)) | 1
    mask_image = cv2.morphologyEx(
        mask_image,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)),
    )
    return mask_image

# --- Constants for GT trajectory generation (trajectory_3d_camera_coords) ---
TRAJECTORY_NUM_POINTS = 100  # reader subsamples 20 uniformly (datasets/scenefun3d.py)
TRAJECTORY_TRANS_LENGTH_M = ARROW_LENGTH_3D_TRANS  # 0.1 m straight travel for "trans"
TRAJECTORY_ROT_ARC_RAD = np.pi / 2.0  # 90 deg sweep for "rot"
TRAJECTORY_MIN_ROT_RADIUS_M = 0.01  # below this the arc is considered degenerate
TRAJECTORY_DEGENERATE_SEGMENT_M = 0.01  # tiny straight fallback so traj is never empty


def compute_trajectory_3d_camera_coords(
    motion_type,
    motion_origin_3d_cam,
    motion_dir_3d_cam,
    visible_item_points_3d_cam,
    num_points=TRAJECTORY_NUM_POINTS,
):
    """
    Reconstructs the "trajectory_3d_camera_coords" field of the original
    (lost) Euler LMDB writer: an ordered list of `num_points` 3D points, in
    the SAME camera frame as motion_origin_3d_camera_coords /
    motion_dir_3d_camera_coords, describing the path swept by the functional
    element when the motion is executed.

    Evidence this construction is based on (ground-truth contract):

    * Reader (datasets/scenefun3d.py:244-257): tensorizes the raw list and
      uniformly subsamples 20 points -> any N >= 20 ordered points work;
      docstring at :274 says "3D trajectory points in camera coordinates".
    * Old visualizer (.old-dont-run/visualize_trajectory_motion_mlp.py:86-126)
      projects the ABSOLUTE trajectory points with the frame's intrinsics
      (u = fx*x/z + cx) onto the image -> the stored trajectory is absolute
      camera-frame geometry located at the object (positive z), not a
      normalized/relative curve.
    * Trajectory loss (train_OPDReal_better.py:196-205, inherited by
      SF3DTrainingModule): MSE(pred, gt - gt[0]) -> the GT is re-centered on
      its first point before use, so the absolute offset cancels but the
      METRIC SCALE and point ordering/spacing of the curve are learned.
    * Geometric consistency loss "pred vector <-> gt traj"
      (train_OPDReal_better.py:224-230 calling _geometric_consistency_loss,
      :758-827) is evaluated ON THE GT trajectory (relative), with
      C/P_0 = motion_origin_3d - traj[0] and n/v = the motion direction.
      For this loss to be ~0 for a correct motion vector (well-posedness of
      the GT contract) the GT must satisfy exactly:
        - trans (:782-794, line loss ||(Q_i - P_0) x v||^2): every absolute
          trajectory point lies on the 3D line through the motion origin
          along motion_dir  -> straight segment starting at the motion
          origin, direction = motion_dir. Length has no loss constraint; we
          use ARROW_LENGTH_3D_TRANS = 0.1 m, the translation-arrow length
          this script has always used for its debug visualization (:29),
          i.e. the only scale convention in the codebase.
        - rot: CHANGED 2026-07-28. The arc is now centred on the axis at the
          ELEMENT'S OWN HEIGHT (the centroid projected onto the axis), so
          traj[0] == the element centroid. Previously it was centred on the
          motion origin, which slid the start point along the axis to the
          hinge's height -- physically wrong (a door handle sweeps at its own
          height, not the hinge's) and confusing to look at. Radius, plane
          normal, start phase and arc extent are unchanged, so the RELATIVE
          trajectory the loss consumes (gt - gt[0]) is bit-identical; only the
          absolute placement moves.
          IMPORTANT: this breaks the circle loss's plane term
          ((Q_i - C) . n)^2, which the old placement zeroed by construction.
          With the corrected arc that term equals the squared along-axis
          offset (measured 0.02 m and 0.21 m on two annotations) and is
          minimised by tilting the predicted axis AWAY from truth. The plane
          term must be dropped from model/losses/geometric.py (the radius
          term already encodes the correct "equidistant from the axis"
          constraint) before training on this GT.
        - rot, historical: (a) plane term ((Q_i - C) . n)^2 = 0
          -> every absolute point lay in the plane through the motion
          origin perpendicular to the axis; (b) radius term uses
          r = || perp-component of (traj[0] - origin) ||, and requires all
          points at that same axis distance -> a circular arc centered at
          the motion origin, in that plane, radius = axis-distance of the
          arc's first point (traj[0] itself must be in the plane, since the
          plane term also applies to i=0).
    * SceneFun3D motion schema: for "rot" the motion origin lies ON the
      rotation axis (hinge/knob center), so the origin itself cannot be the
      rotating point (radius 0). The rotating point is taken from the
      annotated functional element's own geometry: the centroid of its
      visible laser-scan points (camera frame), projected into the rotation
      plane. If the centroid sits on the axis (round knobs), the mean axis
      distance of all visible points is used as radius instead (physical
      knob radius), with the arc phased at the farthest point. Arc extent
      (90 deg) and sweep sign are NOT constrained by any loss (the circle
      loss is invariant to both); 90 deg is chosen as the typical
      door/lid/valve opening travel and, at knob/handle radii of a few cm,
      gives a curve whose chord length is commensurate with the 0.1 m
      translation segments.
    * Degenerate inputs (zero direction, no usable radius) fall back to a
      tiny 0.01 m straight segment from the origin so the reader's
      non-empty requirement (datasets/scenefun3d.py:257 raises ValueError)
      always holds.

    Args:
        motion_type: "trans" or "rot" (SceneFun3D motions.json motion_type).
        motion_origin_3d_cam: (3,) motion origin in camera coordinates.
        motion_dir_3d_cam: (3,) motion direction / rotation axis in camera
            coordinates (same vector stored as motion_dir_3d_camera_coords).
        visible_item_points_3d_cam: (M, 3) camera-frame 3D points of the
            annotated element that are visible in this frame (may be empty).
        num_points: number of trajectory samples to generate.

    Returns:
        List of [x, y, z] lists, length num_points, ordered from motion
        start to motion end. Never empty.
    """
    origin = np.asarray(motion_origin_3d_cam, dtype=np.float64)
    direction = np.asarray(motion_dir_3d_cam, dtype=np.float64)
    dir_norm = np.linalg.norm(direction)

    def _straight_segment(unit_dir, length):
        ts = np.linspace(0.0, length, num_points)
        return (origin[None, :] + ts[:, None] * unit_dir[None, :]).tolist()

    if dir_norm < 1e-8:
        # Degenerate direction: tiny straight segment along camera +z.
        return _straight_segment(np.array([0.0, 0.0, 1.0]), TRAJECTORY_DEGENERATE_SEGMENT_M)

    dir_unit = direction / dir_norm

    if motion_type != "rot":
        # "trans" (and any unknown type, mirroring the reader's default of
        # treating unknown motion types as translation).
        return _straight_segment(dir_unit, TRAJECTORY_TRANS_LENGTH_M)

    # --- "rot": circular arc about the axis (origin, dir_unit) ---
    radius = 0.0
    e1 = None  # in-plane unit vector from the arc centre towards the arc start
    arc_center = origin  # point on the axis that the arc circles around
    pts = np.asarray(visible_item_points_3d_cam, dtype=np.float64)
    if pts.size > 0:
        pts = pts.reshape(-1, 3)
        # Candidate 1: centroid of the visible element points.
        centroid_rel = pts.mean(axis=0) - origin
        centroid_perp = centroid_rel - np.dot(centroid_rel, dir_unit) * dir_unit
        centroid_radius = np.linalg.norm(centroid_perp)
        if centroid_radius >= TRAJECTORY_MIN_ROT_RADIUS_M:
            radius = centroid_radius
            e1 = centroid_perp / centroid_radius
            # Circle about the axis AT THE ELEMENT'S OWN HEIGHT: project the
            # centroid onto the axis instead of using the motion origin. The
            # arc then starts exactly at the centroid rather than at the
            # centroid slid along the axis to the hinge's height.
            arc_center = origin + np.dot(centroid_rel, dir_unit) * dir_unit
        else:
            # Candidate 2 (centroid on axis, e.g. round knobs): mean axis
            # distance of the points = physical radius of the element.
            rel = pts - origin[None, :]
            perp = rel - np.outer(rel @ dir_unit, dir_unit)
            dists = np.linalg.norm(perp, axis=1)
            mean_radius = float(dists.mean())
            max_idx = int(np.argmax(dists))
            if mean_radius >= TRAJECTORY_MIN_ROT_RADIUS_M and dists[max_idx] > 1e-8:
                radius = mean_radius
                e1 = perp[max_idx] / dists[max_idx]
                # Same correction for the round-knob case: circle about the
                # axis at the height of the point the arc is phased on.
                arc_center = origin + np.dot(rel[max_idx], dir_unit) * dir_unit

    if radius < TRAJECTORY_MIN_ROT_RADIUS_M or e1 is None:
        # Point on axis / no visible geometry: fall back to a tiny straight
        # segment so the trajectory is never empty.
        return _straight_segment(dir_unit, TRAJECTORY_DEGENERATE_SEGMENT_M)

    e2 = np.cross(dir_unit, e1)  # completes the in-plane orthonormal basis
    angles = np.linspace(0.0, TRAJECTORY_ROT_ARC_RAD, num_points)
    arc = (
        arc_center[None, :]
        + radius * np.cos(angles)[:, None] * e1[None, :]
        + radius * np.sin(angles)[:, None] * e2[None, :]
    )
    return arc.tolist()


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
        save_depth_images,
        mask_method,
        sensor_depth_check,
        sensor_max_occluded_frac,
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
        args["save_depth_images"],
        args.get("mask_method", "splat"),
        args.get("sensor_depth_check", False),
        args.get("sensor_max_occluded_frac", SENSOR_MAX_OCCLUDED_FRAC),
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

    # --- ARKit sensor-depth assets for the occlusion gate (optional) ---
    sensor = None
    if sensor_depth_check:
        try:
            lowres = LowresSource(data_root_path, visit_id, video_id)
            sensor_poses = data_parser.get_camera_trajectory(
                visit_id, video_id, pose_source="arkit"
            )
            laser_to_arkit = data_parser.get_transform(visit_id, video_id)
            if lowres.timestamps.size and sensor_poses and lowres.intrinsics:
                sensor = {
                    "lowres": lowres,
                    "poses": sensor_poses,
                    "laser_to_arkit": laser_to_arkit,
                }
            else:
                print(
                    f"  WARNING: lowres assets incomplete for {visit_id}/{video_id}; "
                    f"sensor-depth gate disabled for this video."
                )
        except Exception as e:
            print(
                f"  WARNING: sensor-depth gate unavailable for {visit_id}/{video_id} "
                f"({type(e).__name__}: {e}); continuing without it."
            )

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

    # Bounding sphere per annotation, in world coords, for the per-frame
    # frustum rejection. Computed once here rather than per frame.
    annotation_spheres = annotation_bounding_spheres(
        all_annotations, laser_scan_points_full
    )

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

        # The RGB frame is NOT read here. It is only ever needed for writing
        # images out or for debug overlays -- w/h come from the intrinsics
        # file, not the pixels -- so decoding it up front cost one multi-MB
        # JPEG read per frame (345k of them) that the default reprocessing run
        # throws away. It is now loaded lazily, below, only if actually used.
        rgb_image = None

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
            world_to_camera_pose = hm.inverse(
                camera_to_world_pose
            )  # Using hm.inverse for clarity

            # Reject frames that cannot contain any annotated element BEFORE
            # paying for the depth map. Conservative by construction, so this
            # cannot change which records are produced -- only how many
            # multi-MB reads the network volume serves.
            if not _frame_may_contain_items(
                annotation_spheres, world_to_camera_pose, K_matrix, w, h
            ):
                frames_processed += 1
                continue

            current_depth_frame = data_parser.read_depth_frame(depth_path)
            if debug_visualizations:
                rgb_image = cv2.imread(rgb_frame_source_path_str)

        except Exception as e:
            print(
                f"    ERROR loading pose/intrinsics/depth for frame {timestamp}: {e}. Skipping frame."
            )
            continue

        point_to_image_mapper = PointCloudToImageMapper(
            image_dim=(int(w), int(h)), visibility_threshold=0.25, cut_bound=0
        )

        # --- Pair this hires frame with the nearest ARKit sensor frame ---
        frame_sensor = None
        if sensor is not None:
            try:
                lowres = sensor["lowres"]
                t_hires = float(timestamp)
                nearest = lowres.nearest_timestamp(t_hires, SENSOR_TIME_TOL_S)
                if nearest is not None:
                    skey = f"{nearest:.3f}"
                    s_pose = data_parser.get_nearest_pose(
                        skey, sensor["poses"], time_distance_threshold=SENSOR_TIME_TOL_S
                    )
                    s_K = lowres.read_intrinsics(skey)
                    s_depth = lowres.read_depth(skey)
                    if s_pose is not None and s_K is not None and s_depth is not None:
                        frame_sensor = {
                            "K": s_K,
                            "depth": s_depth,
                            # `confidence` is deliberately not downloaded: it
                            # doubles the archive count for a marginal gain,
                            # and the depth agreement is already ~5 mm median.
                            "confidence": None,
                            "cam_to_world": s_pose,
                            "dt": float(abs(nearest - t_hires)),
                        }
            except Exception as e:
                if debug_visualizations:
                    print(
                        f"    [DEBUG] sensor frame pairing failed at {timestamp}: "
                        f"{type(e).__name__}: {e}"
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
                        # Camera-frame coords of the same visible points.
                        # Used twice below: per-point splat radius for the
                        # mask, and the rotating point for the GT trajectory.
                        "visible_item_points_3d_cam": _world_to_camera(
                            item_points_3d_world[visible_mask_indices],
                            world_to_camera_pose,
                        ),
                        "visibility_ratio": float(visibility_ratio),
                        "total_item_points": int(total_item_points),
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
            if rgb_image is None:
                rgb_image = cv2.imread(rgb_frame_source_path_str)
                if rgb_image is None:
                    print(
                        f"    ERROR reading image for timestamp {timestamp} at "
                        f"{rgb_frame_source_path_str}. Skipping frame."
                    )
                    continue
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

        # --- Save depth image to file (one per frame, shared by all annotations
        # of the frame, mirroring how RGB images are stored). The raw
        # hires_depth frames are already 16-bit PNGs in millimeters, matching
        # what the reader expects (datasets/scenefun3d.py loads
        # <lmdb_data_root>/depth/<filename> as uint16 mm and divides by 1000),
        # so a plain file copy suffices for .png sources.
        depth_image_filename = f"{visit_id}_{video_id}_{timestamp}.png"
        if save_depth_images:
            depth_save_path = output_root_path / "depth" / depth_image_filename
            try:
                if not depth_save_path.exists():
                    if str(depth_path).lower().endswith(".png"):
                        shutil.copyfile(str(depth_path), str(depth_save_path))
                    else:
                        # Unexpected source format: re-encode as 16-bit mm PNG.
                        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                        if depth_raw is None:
                            raise ValueError(
                                f"Depth image not found or unreadable at {depth_path}"
                            )
                        if depth_raw.dtype != np.uint16:
                            depth_raw = depth_raw.astype(np.uint16)
                        is_success = cv2.imwrite(str(depth_save_path), depth_raw)
                        if not is_success:
                            raise ValueError(
                                f"Failed to save depth image to {depth_save_path}"
                            )
            except Exception as e:
                print(
                    f"    ERROR saving depth image to {depth_save_path} for timestamp {timestamp}: {e}. Skipping frame."
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

            # --- ARKit sensor-depth occlusion gate ---
            # The laser-side visibility test cannot see a surface that is
            # missing from the scan (hires_depth is a render of that scan), so
            # cross-check against the real LiDAR capture. Stats are stored on
            # every record whether or not they cause a rejection, so the
            # threshold can be retuned later without the raw data.
            sensor_stats = None
            if frame_sensor is not None:
                sensor_stats = sensor_occlusion_stats(
                    points_laser=laser_scan_points_full[annotation.get("indices")][
                        item_info["visible_mask_indices"]
                    ],
                    laser_to_arkit=sensor["laser_to_arkit"],
                    arkit_cam_to_world=frame_sensor["cam_to_world"],
                    intrinsics_lowres=frame_sensor["K"],
                    depth_lowres=frame_sensor["depth"],
                    confidence_lowres=frame_sensor["confidence"],
                )
            if (
                sensor_stats is not None
                and sensor_stats["sensor_occluded_frac"] > sensor_max_occluded_frac
            ):
                if debug_visualizations:
                    print(
                        f"      [DEBUG] Item {visit_id}/{video_id}/{timestamp}/{annot_id} "
                        f"skipped: SensorOccluded "
                        f"{sensor_stats['sensor_occluded_frac']:.2f} "
                        f"(median gap {sensor_stats['sensor_median_gap_m']:+.2f} m)"
                    )
                skipped_item_count += 1
                continue

            # --- Get Mask Coordinates ---
            mask_image = rasterize_mask(
                visible_item_points_2d_yx,
                item_info["visible_item_points_3d_cam"],
                fx,
                h,
                w,
                method=mask_method,
            )
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
            trajectory_3d_camera_coords = None
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

                    # --- Compute GT trajectory in the same camera frame ---
                    # Visible element points (camera coords) provide the
                    # rotating point / radius for "rot" motions; see
                    # compute_trajectory_3d_camera_coords for the full
                    # evidence-based contract.
                    visible_pts_cam = item_info["visible_item_points_3d_cam"]

                    trajectory_3d_camera_coords = compute_trajectory_3d_camera_coords(
                        motion_type=item_motion.get("motion_type"),
                        motion_origin_3d_cam=origin_3d_cam_coords_np,
                        motion_dir_3d_cam=motion_dir_3d_cam_coords_np,
                        visible_item_points_3d_cam=visible_pts_cam,
                    )
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

            if trajectory_3d_camera_coords is None:
                # Records without valid motion data (only stored when
                # skip_items_without_motion_or_description=False) still get a
                # degenerate tiny segment: the reader
                # (datasets/scenefun3d.py:257) raises on a missing/empty
                # trajectory.
                trajectory_3d_camera_coords = compute_trajectory_3d_camera_coords(
                    motion_type="trans",
                    motion_origin_3d_cam=np.zeros(3),
                    motion_dir_3d_cam=np.zeros(3),
                    visible_item_points_3d_cam=np.zeros((0, 3)),
                )

            # 2D track for the trajectory_2d head: the same trajectory as seen
            # by this frame's camera. Stored in full-resolution pixels (the
            # frame's own size is in image_dimensions_wh), so a reader can
            # normalise or rescale to whatever input size it uses.
            trajectory_2d_image_coords, trajectory_2d_valid = project_trajectory_to_2d(
                trajectory_3d_camera_coords, K_matrix, w, h
            )

            data_to_store = {
                "rgb_image_path": image_filename,
                "depth_image_path": depth_image_filename,
                "mask_coordinates_yx": mask_coordinates,  # List of [y,x]
                # Recorded so the visibility filter can be tightened later
                # WITHOUT the raw SceneFun3D data (which lives only on the
                # temporary scratch volume). These genuinely vary per record;
                # run-wide settings (mask_method, thresholds) live in the
                # __metadata__ entry instead of being repeated 461k times.
                "visibility_ratio": item_info.get("visibility_ratio"),
                "visible_point_count": int(len(visible_item_points_2d_yx)),
                "total_point_count": item_info.get("total_item_points"),
                # ARKit sensor-depth cross-check (None if lowres assets were
                # unavailable or no sensor frame paired within the time window).
                "sensor_check": sensor_stats,
                "description": description,
                "motion_info": motion_info_data,
                "label_info": label_info_data,
                "camera_intrinsics": K_matrix.tolist(),
                "camera_extrinsics_world_to_cam": world_to_camera_pose.tolist(),
                "camera_extrinsics_cam_to_world": camera_to_world_pose.tolist(),
                "image_dimensions_wh": (int(w), int(h)),
                "trajectory_3d_camera_coords": trajectory_3d_camera_coords,
                # Same curve, same ordering, projected into this frame.
                # trajectory_2d_valid marks points that are in front of the
                # camera AND inside the image; invalid ones carry [0, 0].
                "trajectory_2d_image_coords": trajectory_2d_image_coords,
                "trajectory_2d_valid": trajectory_2d_valid,
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
    save_depth_images: Annotated[
        bool,
        tyro.conf.arg(
            help="If true, saves per-frame depth PNGs (16-bit, millimeters) to <output_dir>/depth/. Set to false to skip this step if depth images are already generated."
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
    sensor_depth_check: Annotated[
        bool,
        tyro.conf.arg(
            help="Cross-check each item against the ARKit lowres_depth sensor frame "
            "and reject items whose laser points sit behind a real measured surface. "
            "hires_depth is rendered FROM the laser scan, so the default visibility "
            "test cannot detect surfaces missing from the scan (verified: a glossy "
            "tiled wall absent from visit 464982's scan let a cabinet handle in "
            "another room score 100%% visible). Requires the lowres_depth, "
            "lowres_wide_intrinsics, lowres_poses and transform assets; degrades "
            "gracefully per-video if they are absent."
        ),
    ] = False,
    sensor_max_occluded_frac: Annotated[
        float,
        tyro.conf.arg(
            help="Reject an item when more than this fraction of its sensor-evaluated "
            "points lie behind the measured surface."
        ),
    ] = SENSOR_MAX_OCCLUDED_FRAC,
    mask_method: Annotated[
        str,
        tyro.conf.arg(
            help="How to rasterise the mask from the visible laser points: "
            "'splat' (default; per-point disc sized by the 5mm scan spacing at "
            "that point's depth, then morphologically closed) or 'hull' (the "
            "original convex hull, kept for A/B comparison)."
        ),
    ] = "splat",
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

    if mask_method not in MASK_METHODS:
        print(f"Error: --mask-method must be one of {MASK_METHODS}, got {mask_method!r}.")
        return
    print(f"Mask rasterisation method: {mask_method}")

    # Setup output directories and LMDB environment
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(exist_ok=True)
    lmdb_path = output_dir / "data.lmdb"

    print(f"Output LMDB will be at: {lmdb_path}")
    print(f"RGB images will be stored in: {images_dir}")
    print(f"Depth images will be stored in: {depth_dir}")

    env = lmdb.open(str(lmdb_path), map_size=map_size, writemap=False)

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
            "save_depth_images": save_depth_images,
            "mask_method": mask_method,
            "sensor_depth_check": sensor_depth_check,
            "sensor_max_occluded_frac": sensor_max_occluded_frac,
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
            "mask_method": mask_method,
            "min_visibility_ratio": min_visibility_ratio,
            "sensor_depth_check": sensor_depth_check,
            "sensor_max_occluded_frac": sensor_max_occluded_frac,
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
