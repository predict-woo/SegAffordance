"""SF3D validation panels in the sharp style: [GT | model A | model B ...] at
2x resolution with thin overlays, PNG.

GT panel: GT mask (green), GT 2D track (cyan), GT interaction point (white
ring), GT axis (green line: hinge line through the GT origin for rot, a
direction ray from the track's first point for trans). Prediction panels:
tools/predict_image.py's draw_prediction (mask red, point ring, trajectory
light green, axis red, orbit yellow) + the axis error vs GT in the text.

  python tools/sf3d_vis_val.py --model joint4_dct <cfg> <ckpt> [--model ...] \
      --out viz/<batch> --num 16 [--seed 7] [--scale 2] [--dump viz/<batch>/preds.jsonl]
Sampling: stratified rot/trans over the g19 val split (same filters and
seed as the SF3D configs); --idx picks explicit val indices instead.
--dump writes one JSONL record per model x frame (tools/sf3d_preds_io.py) so
tools/sf3d_render_preds.py can redraw these (or other) frames with no GPU.
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from datasets.scenefun3d import SF3DDataset, get_default_transforms, split_dataset_by_scene  # noqa: E402
from model.losses.geometric import apply_trajectory_scale, normalized_intrinsics, trajectory_scale_factor  # noqa: E402
from predict_image import draw_prediction  # noqa: E402
from sf3d_preds_io import out_to_record, write_record  # noqa: E402
from sf3d_vis_predictions import draw_axis_3d, draw_points_norm, draw_polyline_norm, load_model, overlay_mask, put_lines  # noqa: E402
from viz_manifest import write_manifest  # noqa: E402

DATA_ROOT = "/workspace/datasets/sf3d_processed_v3"
FRAME_CACHE = "/workspace/datasets/sf3d_frames_512.lmdb"
KEY_CACHE = "/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl"


def add_data_args(ap):
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--frame-cache-path", default=FRAME_CACHE)
    ap.add_argument("--key-cache", default=KEY_CACHE)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--ray-len", type=float, default=0.5)


def load_val(a):
    """The g19 val split (scene split, seed 42) with the SF3D config filters. Returns (ds, val_subset)."""
    r, m, d = get_default_transforms(image_size=(512, 512))
    ds = SF3DDataset(
        lmdb_data_root=a.data_root, lmdb_path=f"{a.data_root}/data.lmdb", rgb_transform=r, mask_transform=m,
        depth_transform=d, image_size_for_mask_reconstruction=(512, 512), point_source="element",
        key_cache_path=a.key_cache, return_trajectory_2d=True, frame_cache_path=a.frame_cache_path,
        fast_pipeline=True, load_depth=False, min_revolute_radius=0.10, min_mask_area_frac=0.001, edge_margin_frac=0.05,
    )
    _, va = split_dataset_by_scene(ds, 0.1, 42)
    return ds, va


def stratified_picks(ds, va, num, seed):
    """num val indices, half prismatic half revolute, from a seeded RNG (reads only the LMDB records)."""
    import pickle
    rng = np.random.default_rng(seed)
    types = []
    with ds.env.begin() as t:
        for i in va.indices:
            rec = pickle.loads(t.get(ds.item_keys[i]))
            types.append(rec["motion_info"]["original_motion_data"].get("motion_type", "trans"))
    rot = [j for j, ty in enumerate(types) if ty in ("rot", "rotation")]
    trans = [j for j, ty in enumerate(types) if ty not in ("rot", "rotation")]
    return [int(x) for x in rng.choice(trans, num // 2, replace=False)] + [int(x) for x in rng.choice(rot, num - num // 2, replace=False)]


def load_frame(va, j, scale):
    """Unpack val item j into the pieces the panels need."""
    (img_t, depth_t, desc, mask_t, _bbox, point_gt, motion_gt, type_gt, img_size, fname,
     origin_3d, K, traj3d, traj2d_px, valid2d) = va[j]
    W, H = float(img_size[0]), float(img_size[1])
    frame = img_t.permute(1, 2, 0).numpy()[:, :, ::-1].copy()
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
    gt_dir = motion_gt.float().numpy(); gt_dir = gt_dir / max(float(np.linalg.norm(gt_dir)), 1e-8)
    return dict(img_t=img_t, depth_t=depth_t, desc=desc, mask_t=mask_t, point_gt=point_gt, gt_dir=gt_dir,
                gt_type=int(type_gt), W=W, H=H, frame=frame, K_norm=K_norm, origin_3d=origin_3d,
                traj3d=traj3d, traj2d_px=traj2d_px, valid2d=valid2d, j=j)


def gt_panel(fr, ray_len):
    frame, S = fr["frame"], fr["frame"].shape[0]
    gt = overlay_mask(frame, cv2.resize(fr["mask_t"][0].numpy(), (S, S), interpolation=cv2.INTER_NEAREST), (0, 200, 0))
    tuv = (fr["traj2d_px"] / torch.tensor([fr["W"], fr["H"]])).numpy()
    gt = draw_polyline_norm(gt, tuv, fr["valid2d"].numpy(), (255, 255, 0), thickness=2)
    gt = draw_points_norm(gt, tuv, fr["valid2d"].numpy(), (255, 255, 0), radius=3)
    p0 = fr["traj3d"][0].float().numpy()
    if fr["gt_type"] == 1:
        gt = draw_axis_3d(gt, fr["K_norm"], fr["origin_3d"].float().numpy(), fr["gt_dir"], p0, (0, 200, 0), t0=-ray_len, t1=ray_len, thickness=3)
    else:
        gt = draw_axis_3d(gt, fr["K_norm"], p0, fr["gt_dir"], p0, (0, 200, 0), t0=0.0, t1=ray_len, thickness=3)
    cv2.circle(gt, (int(fr["point_gt"][0] * S), int(fr["point_gt"][1] * S)), 10, (255, 255, 255), 2, cv2.LINE_AA)
    return put_lines(gt, [f"GT [{'rot' if fr['gt_type'] == 1 else 'trans'}]  val {fr['j']}", fr["desc"][:52]])


def pred_panel(fr, out, name, ray_len):
    """draw_prediction + the axis-error / type line; works on a live output or a record_to_out() namespace."""
    panel, cls_type, dpred = draw_prediction(fr["frame"], out, fr["K_norm"], name, ray_len)
    if dpred is not None:
        ang = math.degrees(math.acos(float(np.clip(np.dot(dpred, fr["gt_dir"]), -1.0, 1.0))))
        ok = "type OK" if (cls_type == ("rot" if fr["gt_type"] == 1 else "trans")) else "type WRONG"
        for col, th in (((0, 0, 0), 4), ((255, 255, 255), 2)):
            cv2.putText(panel, f"axis err {ang:.0f} deg  {ok}", (8, 26 * 4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, th, cv2.LINE_AA)
    return panel


def panel_filename(n_i, fr):
    return f"{n_i:03d}_{'rot' if fr['gt_type'] == 1 else 'trans'}_val{fr['j']}.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", required=True, metavar=("NAME", "CONFIG", "CKPT"))
    add_data_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--idx", type=int, nargs="*", default=None, help="explicit val indices instead of the stratified draw")
    ap.add_argument("--field-names", default="", help="comma-separated model NAMEs that are FieldModel checkpoints (model/field_model.py)")
    ap.add_argument("--dump", default=None, help="JSONL path: write every prediction (tools/sf3d_preds_io.py records)")
    ap.add_argument("--no-panels", action="store_true", help="only dump, draw nothing")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, va = load_val(a)
    picks = list(a.idx) if a.idx else stratified_picks(ds, va, a.num, a.seed)
    from arctic_axis_probe import load_field_model
    field_names = {x for x in a.field_names.split(",") if x}
    models = [(n, *(load_field_model if n in field_names else load_model)(c, k, device)) for n, c, k in a.model]
    os.makedirs(a.out, exist_ok=True)
    fdump = open(a.dump, "w") if a.dump else None
    for n_i, j in enumerate(picks):
        fr = load_frame(va, j, a.scale)
        key = ds.item_keys[va.indices[j]].decode()
        panels = [gt_panel(fr, a.ray_len)] if not a.no_panels else []
        for name, model, mp in models:
            with torch.no_grad():
                word = model.tokenize([fr["desc"]], 77).to(device)
                out = model(fr["img_t"][None].to(device), fr["depth_t"][None].to(device), word, None, None, None, None, fr["K_norm"].to(device).float())
                if getattr(mp, "trajectory_scale_free", False):
                    out = apply_trajectory_scale(out, trajectory_scale_factor("pred_z_p", out, None))
            if fdump:
                write_record(fdump, out_to_record(out, name, j, key, fr["desc"], fr["gt_type"]))
            if not a.no_panels:
                panels.append(pred_panel(fr, out, name, a.ray_len))
        if panels:
            cv2.imwrite(f"{a.out}/{panel_filename(n_i, fr)}", np.hstack(panels))
        print("wrote", n_i, j, "|", fr["desc"][:60])
    if fdump:
        fdump.close()
    write_manifest(a.out, models=[{"name": n, "config": c, "ckpt": k} for n, c, k in a.model], picks=[int(x) for x in picks],
                   dump=a.dump)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
