"""Empirical collapse-reversal test for the gen-16 normalized trajectory loss.

Loads a checkpoint whose revolute trajectory predictions have collapsed
(gen-13: ~4-8 cm net sweep vs ~0.7 m GT arcs), fine-tunes it for a few
hundred steps, and measures the predicted rot net-sweep before/after on
real val samples. Run twice:

  --mode normalized   the gen-16 loss (trajectory_loss_normalized, w=0.5)
                      -> rot sweeps must GROW substantially
  --mode old015       the collapsed recipe unchanged (control)
                      -> sweeps must stay collapsed

Together the two arms show the fix (and only the fix) reverses the
collapse. Dev-pod sized: bs 8, backbone frozen, no compile (~6 min/arm).

  /opt/venv/bin/python -u tools/diag_traj_collapse_test.py \
    --config config/sf3d_train_runpod_g13_res512.yaml \
    --ckpt experiments/20260817_sf3d_g13_res512/checkpoints/best-epoch27-valloss0.8288.ckpt \
    --mode normalized
"""
import argparse
import dataclasses
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, "/workspace/SegAffordance")

from config.opd_train import Config, LossParams, ModelParams, OptimizerParams  # noqa: E402
from datasets.scenefun3d import (  # noqa: E402
    SF3DDataset,
    get_default_transforms,
    split_dataset_by_scene,
)
from train_SF3D_better import SF3DTrainingModule  # noqa: E402


def _dc(cls, d, **overrides):
    names = {f.name for f in dataclasses.fields(cls)}
    kw = {k: v for k, v in d.items() if k in names}
    kw.update(overrides)
    return cls(**kw)


def rot_extents(module, ds, indices, device):
    module.eval()
    out = {}
    with torch.no_grad():
        for vi in indices:
            s = ds[int(vi)]
            img, depth, desc, type_gt = s[0], s[1], s[2], int(s[7])
            if type_gt != 1:
                continue
            word = module.model.tokenize([desc], 77).to(device)
            o = module.model(img[None].to(device), depth[None].to(device),
                             word, None, None, None, None, None)
            t = o.trajectory_pred[0].float().cpu()
            out[int(vi)] = float(t[-1].norm())
    module.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=["normalized", "old015"], required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--data-root", default="/workspace/datasets/sf3d_processed_v3")
    ap.add_argument("--frame-cache-path",
                    default="/workspace/datasets/sf3d_frames_512.lmdb")
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--key-cache",
                    default="/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl")
    ap.add_argument("--probe-rot", type=int, default=24,
                    help="rot val samples for the extent measurement")
    args = ap.parse_args()

    device = torch.device("cuda")
    raw = yaml.safe_load(open(args.config))["model"]
    mp = _dc(ModelParams, raw["model_params"], compile_model=False)
    if args.mode == "normalized":
        lp = _dc(LossParams, raw["loss_params"],
                 trajectory_weight=0.5, trajectory_loss_normalized=True)
    else:
        lp = _dc(LossParams, raw["loss_params"])  # w=0.15, flag off — control
    op = _dc(OptimizerParams, raw["optimizer_params"], lr=args.lr)
    cfg = _dc(Config, raw["config"],
              log_image_interval_steps=0, enable_wandb=False, val_vis_samples=0)

    module = SF3DTrainingModule(mp, lp, op, cfg)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = module.load_state_dict(state, strict=False)
    print(f"ckpt loaded: {len(missing)} missing, {len(unexpected)} unexpected")
    module.log = lambda *a, **k: None            # unattached module
    for p in module.model.backbone.parameters():
        p.requires_grad_(False)                  # frozen, as in training
    module.to(device).train()

    sz = (args.input_size, args.input_size)
    r, m, d = get_default_transforms(sz)
    ds = SF3DDataset(
        lmdb_data_root=args.data_root, key_cache_path=args.key_cache,
        frame_cache_path=args.frame_cache_path,
        rgb_transform=r, mask_transform=m, depth_transform=d,
        image_size_for_mask_reconstruction=sz,
        point_source="element", return_trajectory_2d=True,
        min_revolute_radius=0.10, min_mask_area_frac=0.001,
        edge_margin_frac=0.05,
    )
    train_ds, val_ds = split_dataset_by_scene(ds, val_split_ratio=0.1, manual_seed=42)

    # Probe set: the two reference rot samples + the first N rot val rows.
    probe = [93, 1684]
    for i in range(len(val_ds)):
        if len(probe) >= 2 + args.probe_rot:
            break
        if int(val_ds[i][7]) == 1 and i not in probe:
            probe.append(i)

    before = rot_extents(module, val_ds, probe, device)
    print(f"BEFORE  mean rot extent {np.mean(list(before.values())):.4f} m  "
          f"val93={before.get(93, float('nan')):.4f}  "
          f"val1684={before.get(1684, float('nan')):.4f}", flush=True)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=6,
        drop_last=True, generator=torch.Generator().manual_seed(0),
    )
    trainable = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)

    step = 0
    while step < args.steps:
        for batch in loader:
            batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            loss = module._common_step(batch, step, "train")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{args.steps}  loss {float(loss):.4f}", flush=True)
            if step >= args.steps:
                break

    after = rot_extents(module, val_ds, probe, device)
    print(f"AFTER   mean rot extent {np.mean(list(after.values())):.4f} m  "
          f"val93={after.get(93, float('nan')):.4f}  "
          f"val1684={after.get(1684, float('nan')):.4f}")
    grew = [k for k in before if after[k] > 1.5 * before[k]]
    print(f"mode={args.mode}: {len(grew)}/{len(before)} probe samples grew >1.5x")


if __name__ == "__main__":
    main()
