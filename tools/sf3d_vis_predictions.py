"""Side-by-side prediction panels for SF3D twist-arm checkpoints.

For N deterministic validation samples (stratified rot/trans), renders one
composite JPEG per sample:

    [ GT | model A pred | model B pred ... ]

GT panel: mask (green), element point, projected GT trajectory (cyan; a
note replaces it when no GT point is valid in this frame).
Pred panels: predicted mask (red), predicted point (point_uv), and three
curves anchored exactly the way the losses anchor them (point_uv lifted
with the INPUT depth — no GT):
  * magenta — the 3D trajectory head's swept path, projected into the frame
  * orange  — the 2D track head's path (2d_twist arm only)
  * yellow  — the decoded twist's forward SWEEP from the predicted point:
    90 deg for revolute / 0.1 m for prismatic (the GT conventions), in the
    committed direction — the predicted counterpart of the cyan GT track
plus text: classifier type, type-from-|omega|, |omega|.

Split-arm (gen-6, point_prediction_3d) checkpoints render the same panels
with the predicted 3D point as marker/anchor (no point_uv, no depth
lifting), the axis-head prediction as the red axis (origin_pred +
motion_pred, typed by the classifier), and no twist orbit / |w| readout.

gen-7 (use_origin_heatmap/predict_point_depth) checkpoints take the same
split-arm rendering path via their lifted point_3d_pred/origin_pred (the
tool passes the sample's normalized intrinsics into the forward — a no-op
for older arms); their ABSOLUTE trajectory is projected directly (no anchor
translation) and the origin heatmap's uv gets a small red circle marker.

Inference is deployment-condition: CVAE prior sampling (motion_gt=None) and
NO type hint (motion_type_input=None -> NULL token).

Run from the repo root on a pod. --out must be a dated batch directory
under viz/ (see CLAUDE.md, "Visualization organization"); a manifest.yaml
with the exact command/commit/checkpoints is written next to the images:
    python tools/sf3d_vis_predictions.py \
        --model twist experiments/20260728_sf3d_twist/config.yaml \
                experiments/20260728_sf3d_twist/checkpoints/best-epoch04-valloss0.9891.ckpt \
        --model 2d_twist experiments/20260728_sf3d_2d_twist/config.yaml \
                experiments/20260728_sf3d_2d_twist/checkpoints/best-epoch15-valloss1.0906.ckpt \
        --out viz/20260803_sf3d_twist_vs_2d_twist_panels --num 16
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.opd_train import ModelParams
from viz_manifest import write_manifest
from datasets.scenefun3d import SF3DDataset, get_default_transforms, split_dataset_by_scene
from model.losses.geometric import backproject_points, normalized_intrinsics, project_points
from model.losses.twist import decode_twist, screw_orbit
from model.segmenter import CRIS

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
PANEL_W = 640  # each panel is the frame resized to this width


def load_model(config_path, ckpt_path, device):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    mp = ModelParams(**cfg["model"]["model_params"])
    model = CRIS(mp)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    return model.to(device).eval(), mp


def overlay_mask(img, mask01, color, alpha=0.45):
    layer = np.zeros_like(img)
    layer[mask01 > 0.5] = color
    return cv2.addWeighted(img, 1.0, layer, alpha, 0)


def put_lines(img, lines, color=(255, 255, 255)):
    y = 26
    for line in lines:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        y += 26
    return img


def draw_points_norm(img, uv_norm, valid, color, radius=4):
    """uv_norm: (M, 2) in [0,1]-ish; one dot per valid point, start point
    ringed white so sweep direction is readable."""
    h, w = img.shape[:2]
    first = True
    for i in range(len(uv_norm)):
        if not valid[i]:
            continue
        p = (int(round(uv_norm[i, 0] * w)), int(round(uv_norm[i, 1] * h)))
        if -w < p[0] < 2 * w and -h < p[1] < 2 * h:
            cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)
            if first:
                cv2.circle(img, p, radius + 3, (255, 255, 255), 2, cv2.LINE_AA)
        first = False
    return img


def draw_polyline_norm(img, uv_norm, valid, color, thickness=3):
    """uv_norm: (M, 2) in [0,1]-ish; draws only consecutive valid pairs."""
    h, w = img.shape[:2]
    pts = np.stack([uv_norm[:, 0] * w, uv_norm[:, 1] * h], axis=1)
    for i in range(len(pts) - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        p0 = tuple(np.round(pts[i]).astype(int))
        p1 = tuple(np.round(pts[i + 1]).astype(int))
        if all(-w < c < 2 * w for c in (p0[0], p1[0])) and all(-h < c < 2 * h for c in (p0[1], p1[1])):
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
    return img


def draw_axis_3d(img, K_norm, pt, d, center_at, color, t0=-0.35, t1=0.35,
                 thickness=3):
    """Project the 3D axis line ``pt + t*d`` as the segment t in [t0, t1]
    (metres) centred at the axis point closest to ``center_at``, with a
    filled dot at the +direction end (the canonical sign). For prismatic
    "axes" pass t0=0 to draw a direction ray from the element instead."""
    d = np.asarray(d, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-8:
        return img
    d = d / n
    pt = np.asarray(pt, dtype=np.float64)
    rel = np.asarray(center_at, dtype=np.float64) - pt
    qc = pt + float(np.dot(rel, d)) * d
    ts = np.linspace(t0, t1, 9)
    pts3 = qc[None] + ts[:, None] * d[None]
    valid = pts3[:, 2] > 0.05
    uv = project_points(
        K_norm, torch.from_numpy(pts3[None]).float()
    )[0].clamp(-2, 3).numpy()
    img = draw_polyline_norm(img, uv, valid, color, thickness)
    if valid[-1]:
        h, w = img.shape[:2]
        e = (int(round(uv[-1, 0] * w)), int(round(uv[-1, 1] * h)))
        if -w < e[0] < 2 * w and -h < e[1] < 2 * h:
            cv2.circle(img, e, 6, color, -1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs=3, action="append", required=True,
                    metavar=("NAME", "CONFIG", "CKPT"))
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v2")
    ap.add_argument("--key-cache", default="/workspace/cache/sf3d_v2_keys_cutoff05.pkl")
    ap.add_argument("--out", required=True,
                    help="dated batch dir under viz/, e.g. viz/YYYYMMDD_<subject>_panels")
    ap.add_argument("--min-revolute-radius", type=float, default=0.0,
                    help="drop knob-class rotations (must match the key cache)")
    ap.add_argument("--min-mask-area-frac", type=float, default=0.0,
                    help="must match the key cache / training config")
    ap.add_argument("--edge-margin-frac", type=float, default=0.0,
                    help="must match the key cache / training config")
    ap.add_argument("--num", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--traj-only", action="store_true",
                    help="trajectory-focused panels: GT track and predicted 3D "
                         "trajectory drawn as POINTS (start ringed white); no "
                         "mask overlay, no twist orbit, no 2D track")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = []
    for name, cfg, ckpt in args.model:
        print(f"loading {name}: {ckpt}")
        model, mp = load_model(cfg, ckpt, device)
        models.append((name, model, mp))

    r, m, d = get_default_transforms((256, 256))
    ds = SF3DDataset(
        lmdb_data_root=args.data_root,
        key_cache_path=args.key_cache,
        frame_cache_path=os.path.join(args.data_root, "frames.lmdb"),
        rgb_transform=r, mask_transform=m, depth_transform=d,
        image_size_for_mask_reconstruction=(256, 256),
        point_source="element", return_trajectory_2d=True,
        min_revolute_radius=args.min_revolute_radius,
        min_mask_area_frac=args.min_mask_area_frac,
        edge_margin_frac=args.edge_margin_frac,
    )
    # The SAME val split as training (seed 42, ratio 0.1, split by scene).
    _, val = split_dataset_by_scene(ds, val_split_ratio=0.1, manual_seed=42)

    # Stratify: half rot, half trans, deterministic.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(val))
    picks, want = [], {0: args.num // 2, 1: args.num - args.num // 2}
    for i in order:
        s = val[int(i)]
        t = int(s[7])
        if want[t] > 0:
            picks.append((int(i), s))
            want[t] -= 1
        if not want[0] and not want[1]:
            break

    os.makedirs(args.out, exist_ok=True)
    for rank, (vi, s) in enumerate(picks):
        (img_t, depth_t, desc, mask_t, _bbox, point_gt, motion_gt, type_gt,
         img_size, rgb_name, origin_3d, K, traj3d, traj2d_px, traj2d_valid) = s
        w0, h0 = int(img_size[0].item()), int(img_size[1].item())
        frame = cv2.imread(os.path.join(args.data_root, "images", rgb_name))
        if frame is None:
            continue
        scale = PANEL_W / frame.shape[1]
        frame = cv2.resize(frame, (PANEL_W, int(frame.shape[0] * scale)))
        H, W = frame.shape[:2]

        # ---- GT panel
        gt = frame.copy()
        if not args.traj_only:
            gt_mask = F.interpolate(mask_t[None].float(), size=(H, W), mode="nearest")[0, 0].numpy()
            gt = overlay_mask(gt, gt_mask, (0, 200, 0))
        traj_uv = (traj2d_px / img_size[None, :]).numpy()
        gt_valid = traj2d_valid.numpy()
        gt_lines = [f"GT [{ 'rot' if int(type_gt) else 'trans' }]", desc[:52]]
        if args.traj_only:
            gt = draw_points_norm(gt, traj_uv, gt_valid, (255, 220, 0))
            if not gt_valid.any():
                gt_lines.append("(GT track: no point visible in frame)")
        elif gt_valid.any():
            # DOTS only (matches the predicted-trajectory rendering).
            gt = draw_points_norm(gt, traj_uv, gt_valid, (255, 220, 0), radius=3)
            # end-of-sweep marker so the motion direction is readable
            end = np.flatnonzero(gt_valid)[-1]
            ep = (int(traj_uv[end, 0] * W), int(traj_uv[end, 1] * H))
            cv2.circle(gt, ep, 5, (255, 220, 0), -1)
        else:
            gt = draw_polyline_norm(gt, traj_uv, gt_valid, (255, 220, 0))
            gt_lines.append("(GT track: no point visible in frame)")
        K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
        gt_dir = motion_gt.float().numpy()
        if not args.traj_only:
            # GT articulation axis (green): revolute = the hinge LINE through
            # the annotated origin; prismatic = a direction ray from the
            # element (no axis position exists). Dot marks the + sign end.
            if int(type_gt) == 1:
                gt = draw_axis_3d(gt, K_norm, origin_3d.float().numpy(), gt_dir,
                                  traj3d[0].float().numpy(), (0, 255, 0))
            else:
                gt = draw_axis_3d(gt, K_norm, traj3d[0].float().numpy(), gt_dir,
                                  traj3d[0].float().numpy(), (0, 255, 0),
                                  t0=0.0, t1=0.25)
        gp = (int(point_gt[0] * W), int(point_gt[1] * H))
        cv2.circle(gt, gp, 6, (255, 255, 255), -1)
        cv2.circle(gt, gp, 6, (0, 0, 0), 2)
        gt = put_lines(gt, gt_lines)

        panels = [gt]
        for name, model, mp in models:
            with torch.no_grad():
                word = model.tokenize([desc], 77).to(device)
                # Intrinsics into the forward: gen-7 arms lift their 2D heads
                # (point_uv/origin_uv + depth) into point_3d_pred/origin_pred
                # with K_norm. A no-op for twist and gen-6 checkpoints (the
                # segmenter only lifts when its z_p/z_q depth heads exist).
                out = model(img_t[None].to(device), depth_t[None].to(device),
                            word, None, None, None, None,
                            K_norm.to(device).float())
            p = frame.copy()
            if not args.traj_only:
                pm = torch.sigmoid(out.mask_logits)[0, 0].cpu()
                pm = F.interpolate(pm[None, None], size=(H, W), mode="bilinear")[0, 0].numpy()
                p = overlay_mask(p, (pm > 0.5).astype(np.float32), (0, 0, 230))
            # Prediction-side point + 3D anchor. Twist-era arms: point_uv
            # is the marker, lifted with the INPUT depth into the anchor
            # (exactly as the screw self/track terms anchor it). Split arm
            # (gen-6, point_prediction_3d): the predicted 3D point IS the
            # anchor and its projection is the marker — no depth lifting.
            K_norm = normalized_intrinsics(K[None].float(), img_size[None].float())
            split_arm = out.point_3d_pred is not None
            if split_arm:
                anchor = out.point_3d_pred[0:1].cpu().float()  # (1, 3)
                anchor_ok = anchor[0, 2].item() > 0.05
                coords = project_points(K_norm, anchor[:, None])[0, 0].clamp(-2, 3)
            else:
                coords = out.point_uv[0].cpu()
                grid = (coords[None] * 2.0 - 1.0).view(1, 1, 1, 2)
                z = F.grid_sample(depth_t[None].float(), grid, align_corners=False).view(1)
                anchor = backproject_points(K_norm, coords[None].float(), z)
                anchor_ok = z.item() > 1e-3
            pp = (int(coords[0] * W), int(coords[1] * H))
            cv2.circle(p, pp, 6, (255, 255, 255), -1)
            cv2.circle(p, pp, 6, (0, 0, 230), 2)

            lines = [name]
            if out.motion_type_logits is not None:
                cls_type = "rot" if int(out.motion_type_logits[0].argmax()) == 1 else "trans"
            else:
                cls_type = "n/a"

            # 3D trajectory head: swept path, projected. Relative heads
            # (twist / gen-6) are anchored at the predicted point exactly the
            # way the losses anchor them; gen-7 ABSOLUTE heads predict camera-
            # frame points directly — project them with no anchor translation.
            traj_is_absolute = bool(getattr(mp, "trajectory_absolute", False))
            # GT track on the MODEL panel too (thin, under the prediction),
            # so pred-vs-GT trajectory comparison happens within one panel.
            if not args.traj_only and gt_valid.any():
                p = draw_points_norm(p, traj_uv, gt_valid, (255, 220, 0),
                                     radius=2)
            if out.trajectory_pred is not None and (anchor_ok or traj_is_absolute):
                if traj_is_absolute:
                    traj_abs = out.trajectory_pred[0:1].cpu().float()
                else:
                    traj_abs = anchor.unsqueeze(1) + out.trajectory_pred[0:1].cpu().float()
                tv = (traj_abs[0, :, 2] > 0.05).numpy()
                tuv = project_points(K_norm, traj_abs)[0].clamp(-2, 3).numpy()
                # DOTS only, start ringed white (user decision 2026-08-15):
                # discrete points keep sequence spacing and ordering honest —
                # line rendering visually smooths disordered sequences (the
                # 2026-08-03 zigzag lesson).
                p = draw_points_norm(p, tuv, tv, (255, 0, 255),
                                     radius=4 if args.traj_only else 3)

            # 2D track head (relative to its own first point, anchored at
            # point_uv — the same element point the track starts from).
            if out.trajectory_2d_pred is not None and not args.traj_only:
                track = (coords[None] + out.trajectory_2d_pred[0].cpu().float()).numpy()
                p = draw_polyline_norm(p, track, np.ones(len(track), bool), (0, 140, 255))

            if out.twist_pred is not None:
                tw = out.twist_pred[0].cpu().float()
                is_rev, direction, axis_point = decode_twist(tw[None])
                om = tw[:3].norm().item()
                if not args.traj_only:
                    # Forward-only sweep of the GT extent (90 deg revolute /
                    # 0.1 m prismatic — the preprocessor's conventions), so
                    # the orbit IS the predicted counterpart of the cyan GT
                    # track: same start (the predicted point), same length,
                    # direction as committed by the sign-sensitive twist.
                    if bool(is_rev[0]):
                        t_max = (math.pi / 2.0) / max(om, 1e-3)
                    else:
                        t_max = 0.1 / max(tw[3:].norm().item(), 1e-3)
                    ts = torch.linspace(0.0, t_max, 48)[None]
                    orbit = screw_orbit(tw[None], anchor, ts)[0]
                    ov = (orbit[:, 2] > 0.05).numpy() if anchor_ok else np.zeros(48, bool)
                    ouv = project_points(K_norm, orbit[None])[0].clamp(-2, 3).numpy()
                    p = draw_polyline_norm(p, ouv, ov, (0, 230, 230), thickness=4)
                    # Decoded axis (red) vs the GT axis (thin green) on the
                    # same panel: hinge placement + sign at a glance.
                    anchor3d = anchor[0].cpu().numpy() if anchor_ok else traj3d[0].float().numpy()
                    if bool(is_rev[0]):
                        p = draw_axis_3d(p, K_norm, axis_point[0].numpy(),
                                         direction[0].numpy(), anchor3d, (0, 0, 255))
                    else:
                        p = draw_axis_3d(p, K_norm, anchor3d, direction[0].numpy(),
                                         anchor3d, (0, 0, 255), t0=0.0, t1=0.25)
                    if int(type_gt) == 1:
                        p = draw_axis_3d(p, K_norm, origin_3d.float().numpy(), gt_dir,
                                         traj3d[0].float().numpy(), (0, 255, 0),
                                         thickness=2)
                    else:
                        p = draw_axis_3d(p, K_norm, traj3d[0].float().numpy(), gt_dir,
                                         traj3d[0].float().numpy(), (0, 255, 0),
                                         t0=0.0, t1=0.25, thickness=2)
                gt_dir_n = gt_dir / max(float(np.linalg.norm(gt_dir)), 1e-8)
                ang = math.degrees(math.acos(
                    float(np.clip(np.dot(direction[0].numpy(), gt_dir_n), -1.0, 1.0))))
                lines.append(f"cls={cls_type}  |w|={om:.2f} -> {'rot' if bool(is_rev[0]) else 'trans'}  ax={ang:.0f}deg")
            elif split_arm and out.motion_pred is not None:
                # Split arm: no twist to decode. Predicted axis (red, same
                # convention as the twist-decoded axis) is origin_pred +
                # normalized motion_pred; type comes from the classifier.
                d = out.motion_pred[0].cpu().float().numpy()
                d = d / max(float(np.linalg.norm(d)), 1e-8)
                if not args.traj_only:
                    anchor3d = anchor[0].numpy() if anchor_ok else traj3d[0].float().numpy()
                    if cls_type == "rot" and out.origin_pred is not None:
                        # 90-deg predicted orbit (yellow), the split-arm
                        # counterpart of the twist orbit: sweep the lifted
                        # point right-handed about the predicted axis —
                        # same extent and sign convention as the GT arcs
                        # (the preprocessor sweeps 90 deg right-handed).
                        q3 = out.origin_pred[0].cpu().float().numpy()
                        c3 = q3 + np.dot(anchor3d - q3, d) * d
                        r_vec = anchor3d - c3
                        if float(np.linalg.norm(r_vec)) > 1e-4:
                            th = np.linspace(0.0, math.pi / 2.0, 48)
                            arc = (c3[None]
                                   + np.cos(th)[:, None] * r_vec[None]
                                   + np.sin(th)[:, None] * np.cross(d, r_vec)[None])
                            av = arc[:, 2] > 0.05
                            auv = project_points(
                                K_norm, torch.from_numpy(arc).float()[None]
                            )[0].clamp(-2, 3).numpy()
                            p = draw_polyline_norm(p, auv, av, (0, 230, 230),
                                                   thickness=4)
                        p = draw_axis_3d(p, K_norm,
                                         out.origin_pred[0].cpu().float().numpy(),
                                         d, anchor3d, (0, 0, 255))
                    else:
                        p = draw_axis_3d(p, K_norm, anchor3d, d, anchor3d,
                                         (0, 0, 255), t0=0.0, t1=0.25)
                    # GT axis (thin green) on the same panel, as on twist arms.
                    if int(type_gt) == 1:
                        p = draw_axis_3d(p, K_norm, origin_3d.float().numpy(), gt_dir,
                                         traj3d[0].float().numpy(), (0, 255, 0),
                                         thickness=2)
                    else:
                        p = draw_axis_3d(p, K_norm, traj3d[0].float().numpy(), gt_dir,
                                         traj3d[0].float().numpy(), (0, 255, 0),
                                         t0=0.0, t1=0.25, thickness=2)
                    # gen-7: the origin heatmap's soft-argmax uv (small red
                    # circle) alongside the lifted red axis — shows where the
                    # 2D origin channel fired before the depth lift.
                    if out.origin_uv is not None:
                        ou = out.origin_uv[0].cpu().float()
                        op = (int(round(float(ou[0]) * W)),
                              int(round(float(ou[1]) * H)))
                        if -W < op[0] < 2 * W and -H < op[1] < 2 * H:
                            cv2.circle(p, op, 5, (0, 0, 255), 2, cv2.LINE_AA)
                gt_dir_n = gt_dir / max(float(np.linalg.norm(gt_dir)), 1e-8)
                ang = math.degrees(math.acos(
                    float(np.clip(np.dot(d, gt_dir_n), -1.0, 1.0))))
                lines.append(f"cls={cls_type}  ax={ang:.0f}deg")
            else:
                lines.append(f"cls={cls_type}")
            legend = "mag=traj3d cyn=GTtraj orn=trk2d yel=orbit grn=GTaxis red=predaxis"
            if getattr(out, "origin_uv", None) is not None:
                # gen-7 arm: the origin heatmap's 2D readout, drawn as a red
                # ring (meaningful on revolute rows only — unsupervised on
                # prismatic by design).
                legend += " redring=origin_uv"
            lines.append("mag=traj3d pts" if args.traj_only else legend)
            p = put_lines(p, lines)
            panels.append(p)

        combo = np.concatenate(panels, axis=1)
        t_lbl = "rot" if int(type_gt) else "trans"
        cv2.imwrite(os.path.join(args.out, f"{rank:02d}_{t_lbl}_val{vi}.jpg"),
                    combo, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"wrote {rank:02d}_{t_lbl}_val{vi}.jpg  ({desc[:48]})")

    write_manifest(
        args.out,
        models=[{"name": n, "config": c, "ckpt": k} for n, c, k in args.model],
        num=args.num, seed=args.seed,
        data_root=args.data_root, key_cache=args.key_cache,
    )
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
