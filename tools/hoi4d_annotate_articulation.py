"""Web-based articulation annotator for the HOI4D processed-2D dataset.

Spec: docs/superpowers/specs/2026-09-01-hoi4d-articulation-annotator-design.md
Serve (dev pod):  python3 tools/hoi4d_annotate_articulation.py serve \
    --data /workspace/hoi4d_processed_2d --hands /workspace/datasets/hands354 \
    --out /workspace/hoi4d_processed_2d/annotations [--port 8080]
Export:           python3 tools/hoi4d_annotate_articulation.py export \
    --data ... --hands ... --out ...   (writes annotations/export_sf3d.pkl)

viser is imported only inside serve(); the geometry/IO core is
importable and unit-tested without it.
"""
import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np

OPEN_EVENTS = {"open", "pull", "pullout"}
_KEY_RE = re.compile(r"^(.+)/(.+)_w(\d+)_f(\d+)$")


def parse_key(key: str):
    """LMDB key -> (sequence id, window index, 0-based frame)."""
    m = _KEY_RE.match(key)
    pre, mid, w, f = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    return f"{pre}_{mid}", w, f


def wxyz_to_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, float) / np.linalg.norm(wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_wxyz(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-8:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                         (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    i = int(np.argmax(np.diag(R)))          # w ~ 0: pick dominant axis
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4.0
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q / np.linalg.norm(q)


def backproject(depth_u16, rgb_bgr, K, stride: int = 4):
    """Depth (uint16 mm) + BGR image + K -> (N,3) camera-frame metres, (N,3) RGB [0,1]."""
    d = depth_u16[::stride, ::stride].astype(np.float64) / 1000.0
    ys, xs = np.nonzero(d > 0)
    z = d[ys, xs]
    xs_full, ys_full = xs * stride, ys * stride
    x = (xs_full - K[0, 2]) / K[0, 0] * z
    y = (ys_full - K[1, 2]) / K[1, 1] * z
    pts = np.stack([x, y, z], axis=1)
    cols = rgb_bgr[ys_full, xs_full, ::-1].astype(np.float64) / 255.0
    return pts, cols


def to_world(points_cam, pose_c2w):
    return points_cam @ pose_c2w[:3, :3].T + pose_c2w[:3, 3]


def world_axis_to_camera(axis_w, origin_w, pose_c2w, K):
    """World axis/origin -> SF3D-convention camera-frame dict (2D in pixels at K's res)."""
    Rcw = pose_c2w[:3, :3].T
    dir_c = Rcw @ (np.asarray(axis_w, float) / np.linalg.norm(axis_w))
    org_c = Rcw @ (np.asarray(origin_w, float) - pose_c2w[:3, 3])
    uvw = K @ org_c
    return {
        "motion_dir_3d_camera_coords": dir_c,
        "motion_origin_3d_camera_coords": org_c,
        "motion_origin_2d_image_coords": np.array([uvw[0] / uvw[2], uvw[1] / uvw[2]]),
    }


def sweep_points(points, mtype, axis, origin, t):
    """Move points along the screw: trans = t metres along axis; rot = t rad about (axis, origin)."""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    if mtype == "trans":
        return points + t * axis
    c, s = np.cos(t), np.sin(t)
    Kx = np.array([[0, -axis[2], axis[1]],
                   [axis[2], 0, -axis[0]],
                   [-axis[1], axis[0], 0]])
    R = np.eye(3) * c + s * Kx + (1 - c) * np.outer(axis, axis)
    return (points - origin) @ R.T + origin


def principal_direction(points):
    """Unit first principal component, oriented from first toward last point."""
    p = np.asarray(points, float)
    centered = p - p.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    d = vt[0] / np.linalg.norm(vt[0])
    if d @ (p[-1] - p[0]) < 0:
        d = -d
    return d


class AnnotationStore:
    """Per-sequence articulation annotations (atomic JSON files)."""

    def __init__(self, out_dir):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, seq):
        return self.dir / f"{seq}.json"

    def load(self, seq):
        p = self._path(seq)
        return json.load(open(p)) if p.exists() else None

    def save(self, seq, category, parts):
        rec = {"seq": seq, "category": category,
               "annotator": os.environ.get("USER", "unknown"),
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "parts": parts}
        p = self._path(seq)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(rec, f, indent=1)
        os.replace(tmp, p)
        return p

    def status(self):
        out = {}
        for p in sorted(self.dir.glob("*.json")):
            rec = json.load(open(p))
            flagged = rec["parts"] and all(pt.get("flag") for pt in rec["parts"])
            out[rec["seq"]] = "flagged" if flagged else "annotated"
        return out


class Dataset:
    """Read-only view over the processed-2D LMDBs + hands package."""

    def __init__(self, data_dir, hands_dir):
        import lmdb
        self.data_dir, self.hands_dir = Path(data_dir), Path(hands_dir)
        self._env_d = lmdb.open(str(self.data_dir / "data.lmdb"),
                                readonly=True, lock=False)
        self._env_f = lmdb.open(str(self.data_dir / "frames.lmdb"),
                                readonly=True, lock=False)
        self.sequences = {}
        with self._env_d.begin() as txn:
            for key, _ in txn.cursor():
                k = key.decode()
                if k.startswith("__"):
                    continue
                seq, _, _ = parse_key(k)
                self.sequences.setdefault(seq, []).append(k)
        for v in self.sequences.values():
            v.sort()
        self._poses = {}

    def record(self, key):
        with self._env_d.begin() as txn:
            return pickle.loads(txn.get(key.encode()))

    def frame(self, key):
        """-> (rgb_bgr, depth_u16, K scaled to the stored resolution, orig_size)."""
        import cv2
        with self._env_f.begin() as txn:
            fr = pickle.loads(txn.get(key.encode()))
        rgb = cv2.imdecode(np.frombuffer(fr["jpeg"], np.uint8), cv2.IMREAD_COLOR)
        depth = cv2.imdecode(np.frombuffer(fr["depth_png"], np.uint8),
                             cv2.IMREAD_UNCHANGED)
        K = np.array(self.record(key)["camera_intrinsics"], float)
        ow, oh = fr["orig_size"]
        K = K.copy()
        K[0] *= rgb.shape[1] / ow
        K[1] *= rgb.shape[0] / oh
        return rgb, depth, K, (ow, oh)

    def pose(self, seq, frame):
        if seq not in self._poses:
            self._poses[seq] = np.load(
                self.hands_dir / seq / "camera" / "official_poses.npy")
        return self._poses[seq][frame]


def build_scene(ds, seq, cache_dir, max_frames=15, target_points=300_000):
    """Fused world-frame cloud + per-window overlays for one sequence (npz-cached)."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_dir / f"{seq}.npz"
    if cpath.exists():
        z = np.load(cpath, allow_pickle=True)
        return {"points": z["points"], "colors": z["colors"],
                "windows": pickle.loads(z["meta"].tobytes())["windows"],
                "frusta": [p for p in z["frusta"]]}

    keys = ds.sequences[seq]
    step = max(1, len(keys) // max_frames)
    chosen = keys[::step][:max_frames]
    pts_all, col_all, frusta = [], [], []
    for key in chosen:
        rgb, depth, K, _ = ds.frame(key)
        _, _, f = parse_key(key)
        pose = ds.pose(seq, f)
        pts, cols = backproject(depth, rgb, K, stride=4)
        pts_all.append(to_world(pts, pose)); col_all.append(cols)
        frusta.append(pose)
    points = np.concatenate(pts_all); colors = np.concatenate(col_all)
    if len(points) > target_points:
        idx = np.random.default_rng(0).choice(len(points), target_points, replace=False)
        points, colors = points[idx], colors[idx]

    windows = {}
    for key in keys:            # one representative sample per window
        rec = ds.record(key)
        w = rec["hoi4d"]["window"]
        if w in windows:
            continue
        rgb, depth, K, (ow, oh) = ds.frame(key)
        _, _, f = parse_key(key)
        pose = ds.pose(seq, f)
        sy, sx = depth.shape[0] / oh, depth.shape[1] / ow
        mask_pts = []
        for y, x in rec["mask_coordinates_yx"]:
            yy, xx = int(y * sy), int(x * sx)
            z = depth[yy, xx] / 1000.0
            if z > 0:
                mask_pts.append([(xx - K[0, 2]) / K[0, 0] * z,
                                 (yy - K[1, 2]) / K[1, 1] * z, z])
        mask_w = (to_world(np.array(mask_pts), pose)
                  if mask_pts else np.zeros((0, 3)))
        traj_w = to_world(np.array(rec["trajectory_3d_camera_coords"], float), pose)
        windows[w] = {"event": rec["hoi4d"]["event"], "mask_points": mask_w,
                      "traj": traj_w, "sample_key": key}

    meta = np.frombuffer(pickle.dumps({"windows": windows}), np.uint8)
    np.savez_compressed(cpath, points=points, colors=colors,
                        frusta=np.array(frusta), meta=meta)
    return {"points": points, "colors": colors, "windows": windows,
            "frusta": frusta}


def export_sf3d(ds, store, out_path):
    """Project world-frame annotations into every sample's camera frame (SF3D fields)."""
    out = {}
    status = store.status()
    for seq, keys in ds.sequences.items():
        if status.get(seq) != "annotated":
            continue
        rec0 = store.load(seq)
        for key in keys:
            r = ds.record(key)
            event = r["hoi4d"]["event"]
            part = next((p for p in rec0["parts"]
                         if event in p["window_events"]), rec0["parts"][0])
            _, _, f = parse_key(key)
            pose = ds.pose(seq, f)
            K = np.array(r["camera_intrinsics"], float)   # orig-res K
            cam = world_axis_to_camera(np.array(part["axis_world"], float),
                                       np.array(part["origin_world"], float),
                                       pose, K)
            out[key] = {"motion_type": part["type"],
                        "motion_dir_3d_camera_coords": cam["motion_dir_3d_camera_coords"].tolist(),
                        "motion_origin_3d_camera_coords": cam["motion_origin_3d_camera_coords"].tolist(),
                        "motion_origin_2d_image_coords": cam["motion_origin_2d_image_coords"].tolist()}
    out_path = Path(out_path)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    os.replace(tmp, out_path)
    return len(out)


def serve(ds, store, cache_dir, port):
    import viser
    server = viser.ViserServer(port=port)
    state = {"seq": None, "scene": None, "window": None}

    status = store.status()
    seq_names = sorted(ds.sequences)

    def label(s):
        mark = {"annotated": "[done] ", "flagged": "[flag] "}.get(status.get(s), "")
        return mark + s

    seq_dd = server.gui.add_dropdown("Sequence", [label(s) for s in seq_names],
                                     initial_value=label(seq_names[0]))
    win_dd = server.gui.add_dropdown("Window", ["0"], initial_value="0")
    type_dd = server.gui.add_dropdown("Type", ["trans", "rot"], initial_value="trans")
    align_btn = server.gui.add_button("Align axis to trajectory")
    preview = server.gui.add_slider("Motion preview", min=-1.0, max=1.0,
                                    step=0.02, initial_value=0.0)
    save_btn = server.gui.add_button("Save")
    flag_dd = server.gui.add_dropdown("Flag", ["none", "bad-poses", "ambiguous"],
                                      initial_value="none")
    info = server.gui.add_markdown("")

    tc = server.scene.add_transform_controls("/gizmo", scale=0.25)

    def axis_vec():
        return wxyz_to_matrix(np.array(tc.wxyz)) @ np.array([0.0, 0.0, 1.0])

    def draw_axis():
        a, o = axis_vec(), np.array(tc.position)
        seg = np.stack([o - 2.0 * a, o + 2.0 * a])[None]
        server.scene.add_line_segments("/axis_line", seg,
                                       colors=np.array([[[255, 60, 60]] * 2]))

    def mask_cloud(points):
        server.scene.add_point_cloud("/mask", points,
                                     colors=np.tile([[1.0, 0.15, 0.15]],
                                                    (len(points), 1)),
                                     point_size=0.007)

    def load_seq(_=None):
        seq = seq_names[[label(s) for s in seq_names].index(seq_dd.value)]
        state["seq"], state["window"] = seq, None
        state["scene"] = build_scene(ds, seq, cache_dir)
        sc = state["scene"]
        server.scene.add_point_cloud("/cloud", sc["points"],
                                     colors=sc["colors"], point_size=0.004)
        for i, pose in enumerate(sc["frusta"]):   # small orientation frusta
            server.scene.add_camera_frustum(
                f"/frusta/{i}", fov=1.0, aspect=1.0, scale=0.06,
                wxyz=matrix_to_wxyz(pose[:3, :3]), position=tuple(pose[:3, 3]))
        wins = sorted(sc["windows"])
        win_dd.options = [str(w) for w in wins]
        win_dd.value = str(wins[0])
        cat = seq.split("_")[2]
        type_dd.value = "trans" if cat == "C4" else "rot"
        existing = store.load(seq)
        if existing and existing["parts"]:
            p = existing["parts"][0]
            tc.position = tuple(p["origin_world"])
            type_dd.value = p["type"]
        else:  # start the gizmo at the first window's mask centroid
            m = sc["windows"][wins[0]]["mask_points"]
            if len(m):
                tc.position = tuple(m.mean(axis=0))
        load_window()

    def load_window(_=None):
        sc, w = state["scene"], int(win_dd.value)
        state["window"] = w
        wd = sc["windows"][w]
        mask_cloud(wd["mask_points"])
        if len(wd["traj"]) >= 2:
            segs = np.stack([wd["traj"][:-1], wd["traj"][1:]], axis=1)
            server.scene.add_line_segments(
                "/traj", segs,
                colors=np.tile([[40, 120, 255]], (len(segs), 2, 1)))
        info.content = f"**{state['seq']}** — window {w}: *{wd['event']}*"
        draw_axis()

    def do_align(_):
        wd = state["scene"]["windows"][state["window"]]
        if len(wd["traj"]) >= 2:
            d = principal_direction(wd["traj"])
            z = np.array([0.0, 0.0, 1.0]); v = np.cross(z, d)
            w = 1.0 + float(z @ d)
            if np.linalg.norm([w, *v]) < 1e-8:      # antiparallel: flip about x
                q = np.array([0.0, 1.0, 0.0, 0.0])
            else:
                q = np.array([w, *v]); q /= np.linalg.norm(q)
            tc.wxyz = tuple(q)
            draw_axis()

    def do_preview(_):
        wd = state["scene"]["windows"][state["window"]]
        t = preview.value * (0.4 if type_dd.value == "trans" else np.pi / 2)
        moved = sweep_points(wd["mask_points"], type_dd.value,
                             axis_vec(), np.array(tc.position), t)
        mask_cloud(moved)

    def do_save(_):
        wd_events = sorted({w["event"] for w in state["scene"]["windows"].values()})
        part = {"window_events": wd_events, "type": type_dd.value,
                "axis_world": axis_vec().tolist(),
                "origin_world": list(map(float, tc.position)),
                "flag": None if flag_dd.value == "none" else flag_dd.value}
        store.save(state["seq"], state["seq"].split("_")[2], [part])
        status.update(store.status())
        info.content = f"saved {state['seq']}"

    seq_dd.on_update(load_seq)
    win_dd.on_update(load_window)
    align_btn.on_click(do_align)
    preview.on_update(do_preview)
    save_btn.on_click(do_save)
    tc.on_update(lambda _: draw_axis())

    load_seq()
    print(f"annotator up: forward the port and open http://localhost:{port}")
    while True:
        time.sleep(3600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["serve", "export"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--hands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    ds = Dataset(args.data, args.hands)
    store = AnnotationStore(args.out)
    if args.mode == "export":
        n = export_sf3d(ds, store, Path(args.out) / "export_sf3d.pkl")
        print(f"exported {n} keys")
    else:
        serve(ds, store, Path(args.out) / "cache", args.port)


if __name__ == "__main__":
    main()
