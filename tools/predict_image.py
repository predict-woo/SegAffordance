"""In-the-wild inference: run a checkpoint on arbitrary photos with text
prompts and draw the predictions on the full-resolution image.

Intrinsics come from the 35 mm-equivalent focal length (EXIF; iPhone main
camera = 26 mm): f_px = f35 / 36 * long_side, principal point at the centre.
The model input is the photo stretched to 512x512 (exactly how the training
LMDBs' 4:3 / 16:9 frames were resized), and every 2D prediction is mapped
back through the same stretch. RGB-only models get a zero depth map.

Overlays: predicted mask (red), point_uv (white ring), projected trajectory
(light green; z_p-scaled for scale-free heads), predicted axis (red: hinge
line through origin_pred for rot, direction ray from the lifted point for
trans; --ray-len metres), 90-deg orbit (yellow) for rot, origin-heatmap uv
(small red circle); text = model, type, p_rev, axis direction, z_p.

  python tools/predict_image.py \
      --model joint4_dct config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml <ckpt> \
      --case viz/<batch>/inputs/IMG_0874.jpg "close the laptop" \
      --case viz/<batch>/inputs/IMG_0876.jpg "open the door" \
      --out viz/<batch> [--f35 26] [--ray-len 0.5]
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
from model.losses.geometric import (  # noqa: E402
    apply_trajectory_scale, normalized_intrinsics, project_points, trajectory_scale_factor,
)
from sf3d_vis_predictions import (  # noqa: E402
    draw_axis_3d, draw_points_norm, draw_polyline_norm, load_model, overlay_mask, put_lines,
)
from viz_manifest import write_manifest  # noqa: E402

GREEN = (120, 255, 120)


def draw_prediction(bgr, out, K_norm, name, ray_len=0.5, extra_lines=()):
    """Draw one model's prediction on a copy of `bgr` (H, W, 3). Returns the panel."""
    H, W = bgr.shape[:2]
    p = bgr.copy()
    pm = torch.sigmoid(out.mask_logits)[0, 0].cpu()
    pm = torch.nn.functional.interpolate(pm[None, None], size=(H, W), mode="bilinear")[0, 0].numpy()
    p = overlay_mask(p, (pm > 0.5).astype(np.float32), (0, 0, 230))
    lines = [name]
    anchor = out.point_3d_pred[0:1].cpu().float() if out.point_3d_pred is not None else None
    if out.point_uv is not None:
        pu = out.point_uv[0].cpu().float()
        cv2.circle(p, (int(pu[0] * W), int(pu[1] * H)), 10, (255, 255, 255), 2, cv2.LINE_AA)
    cls_type, p_rev = "n/a", float("nan")
    if out.motion_type_logits is not None:
        p_rev = float(torch.softmax(out.motion_type_logits[0].float(), -1)[1])
        cls_type = "rot" if p_rev > 0.5 else "trans"
    d = None
    if anchor is not None and out.motion_pred is not None:
        d = out.motion_pred[0].cpu().float().numpy()
        d = d / max(float(np.linalg.norm(d)), 1e-8)
        a3 = anchor[0].numpy()
        if cls_type == "rot" and out.origin_pred is not None:
            q3 = out.origin_pred[0].cpu().float().numpy()
            c3 = q3 + np.dot(a3 - q3, d) * d
            r_vec = a3 - c3
            if float(np.linalg.norm(r_vec)) > 1e-4:
                th = np.linspace(0.0, math.pi / 2.0, 48)
                arc = c3[None] + np.cos(th)[:, None] * r_vec[None] + np.sin(th)[:, None] * np.cross(d, r_vec)[None]
                p = draw_polyline_norm(p, project_points(K_norm, torch.from_numpy(arc).float()[None])[0].clamp(-2, 3).numpy(),
                                       arc[:, 2] > 0.05, (0, 230, 230), thickness=2)
            p = draw_axis_3d(p, K_norm, q3, d, a3, (0, 0, 255), t0=-ray_len, t1=ray_len, thickness=3)
            lines.append(f"rot  p_rev={p_rev:.2f}  r={float(np.linalg.norm(r_vec)):.2f}m")
        else:
            p = draw_axis_3d(p, K_norm, a3, d, a3, (0, 0, 255), t0=0.0, t1=ray_len, thickness=3)
            lines.append(f"trans  p_rev={p_rev:.2f}")
        lines.append(f"axis=({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f})  z_p={float(anchor[0, 2]):.2f}m")
    if out.origin_uv is not None:
        ou = out.origin_uv[0].cpu().float()
        cv2.circle(p, (int(ou[0] * W), int(ou[1] * H)), 7, (0, 0, 255), 2, cv2.LINE_AA)
    # trajectory LAST so it stays visible over the orbit/axis (the analytic decoder's
    # curve lies exactly on the orbit)
    if anchor is not None and out.trajectory_pred is not None:
        traj_abs = anchor.unsqueeze(1) + out.trajectory_pred[0:1].cpu().float()
        tv = (traj_abs[0, :, 2] > 0.05).numpy()
        tuv = project_points(K_norm, traj_abs)[0].clamp(-2, 3).numpy()
        p = draw_polyline_norm(p, tuv, tv, GREEN, thickness=2)
        p = draw_points_norm(p, tuv, tv, GREEN, radius=3)
    lines.extend(extra_lines)
    return put_lines(p, lines), cls_type, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", required=True, metavar=("NAME", "CONFIG", "CKPT"))
    ap.add_argument("--case", nargs=2, action="append", required=True, metavar=("IMAGE", "PROMPT"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--f35", type=float, default=26.0, help="35 mm-equivalent focal length of the photos")
    ap.add_argument("--ray-len", type=float, default=0.5)
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--pad-to-landscape", action="store_true",
                    help="letterbox a portrait photo onto a 4:3 landscape canvas (grey sides) before the 512 stretch — the "
                         "training frames are all landscape; intrinsics shift accordingly (cx += pad)")
    ap.add_argument("--K", nargs=4, type=float, default=None, metavar=("fx", "fy", "cx", "cy"),
                    help="explicit intrinsics in pixels of the input image (overrides --f35); e.g. an SF3D frame's K")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = [(n, *load_model(c, k, device)) for n, c, k in a.model]
    os.makedirs(a.out, exist_ok=True)
    S = a.input_size
    for i, (path, prompt) in enumerate(a.case):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # honours EXIF orientation
        if bgr is None:
            raise SystemExit(f"cannot read {path}")
        H, W = bgr.shape[:2]
        if a.K is not None:
            fx, fy, cx, cy = a.K
        else:
            fx = fy = a.f35 / 36.0 * max(W, H); cx, cy = W / 2.0, H / 2.0
        if a.pad_to_landscape and H > W:
            W2 = int(round(H * 4 / 3)); pad = (W2 - W) // 2
            canvas = np.full((H, W2, 3), 114, np.uint8); canvas[:, pad:pad + W] = bgr
            bgr = canvas; cx += pad; W = W2
        f_px = fx
        K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        img_size = torch.tensor([float(W), float(H)])
        K_norm = normalized_intrinsics(K[None], img_size[None])
        small = cv2.resize(bgr, (S, S), interpolation=cv2.INTER_AREA)
        img_t = torch.from_numpy(np.ascontiguousarray(small[:, :, ::-1].transpose(2, 0, 1)))  # uint8 CHW RGB
        depth_t = torch.zeros(1, S, S)
        panels = [put_lines(bgr.copy(), [os.path.basename(path), f'"{prompt}"', f"{W}x{H}, f={f_px:.0f}px"])]
        for name, model, mp in models:
            with torch.no_grad():
                word = model.tokenize([prompt], 77).to(device)
                out = model(img_t[None].to(device), depth_t[None].to(device), word,
                            None, None, None, None, K_norm.to(device).float())
                if getattr(mp, "trajectory_scale_free", False):
                    out = apply_trajectory_scale(out, trajectory_scale_factor("pred_z_p", out, None))
            panel, _, _ = draw_prediction(bgr, out, K_norm, name, a.ray_len)
            panels.append(panel)
        stem = os.path.splitext(os.path.basename(path))[0]
        cv2.imwrite(f"{a.out}/{i:02d}_{stem}.png", np.hstack(panels))
        print("wrote", i, stem, "|", prompt, "|", W, "x", H)
    write_manifest(a.out, models=[{"name": n, "config": c, "ckpt": k} for n, c, k in a.model],
                   cases=[{"image": im, "prompt": pr} for im, pr in a.case], f35=a.f35)
    print("done ->", a.out)


if __name__ == "__main__":
    main()
