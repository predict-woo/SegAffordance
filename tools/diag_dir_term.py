"""Per-row diagnosis of WHERE the midpoint screw-direction term hurts.

Runs the standard 5,088-sample test split through a checkpoint and dumps one
JSON record per row: GT/predicted type, P(revolute), signed axis error and
flip flag (per the harness's _axis_error_deg), the net-sweep traj_dir cosine
(the harness's test/traj_dir_cos definition: last-first, pred vs GT), a
per-segment direction score (mean cosine over the 19 GT segments > 1 mm),
and sweep magnitudes. Comparing the dumps of two checkpoints that differ
only in pred_pred_art_dir_weight (g19_dct vs g21_dct_dir) localizes the
term's damage: rot vs trans rows, direction vs magnitude, axis vs curve.

Usage (dev pod, volume paths — no shm staging; eval-scale FUSE mmap is fine):
  /opt/venv/bin/python tools/diag_dir_term.py \
      --config config/sf3d_train_runpod_g21_dct_dir.yaml \
      --ckpt experiments/20260823_sf3d_g21_dct_dir/checkpoints/<best>.ckpt \
      --out /root/diag_dir_A.jsonl [--limit N]

The config supplies the model architecture and the data filters; the data
paths are taken from the config as-is (volume), never /dev/shm.
"""
import argparse
import dataclasses
import json

import torch
import torch.nn.functional as F
import yaml

from config.opd_train import ModelParams
from datasets.scenefun3d_datamodule import SF3DDataModule
from model.losses.geometric import normalized_intrinsics
from model.segmenter import CRIS


def load_model(cfg_path, ckpt_path, device):
    cfg = yaml.safe_load(open(cfg_path))
    raw = dict(cfg["model"]["model_params"])
    raw["compile_model"] = False  # eval-only; compile pays off over epochs
    known = {f.name for f in dataclasses.fields(ModelParams)}
    mp = ModelParams(**{k: v for k, v in raw.items() if k in known})
    model = CRIS(mp).to(device).eval()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    msd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(msd, strict=False)
    print(f"loaded {ckpt_path}: {len(missing)} missing / {len(unexpected)} unexpected keys")
    assert not unexpected, unexpected[:5]
    return model, cfg


def build_loader(cfg, batch_size, workers):
    d = cfg["data"]
    dm = SF3DDataModule(
        train_data_dir=d["train_data_dir"],
        val_split_ratio=d["val_split_ratio"],
        input_size=tuple(d["input_size"]),
        batch_size_train=batch_size, batch_size_val=batch_size,
        num_workers_train=workers, num_workers_val=workers,
        manual_seed=d["manual_seed"],
        point_source=d.get("point_source", "motion_origin"),
        key_cache_path=d.get("key_cache_path"),
        frame_cache_path=d.get("frame_cache_path"),
        fast_pipeline=d.get("fast_pipeline", False),
        min_revolute_radius=d.get("min_revolute_radius", 0.0),
        min_mask_area_frac=d.get("min_mask_area_frac", 0.0),
        edge_margin_frac=d.get("edge_margin_frac", 0.0),
    )
    dm.setup("test")
    return dm.test_dataloader()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="stop after N rows (0 = all)")
    args = ap.parse_args()

    device = "cuda"
    model, cfg = load_model(args.config, args.ckpt, device)
    loader = build_loader(cfg, args.batch_size, args.workers)
    word_len = cfg["model"]["model_params"]["word_len"]

    n = 0
    out = open(args.out, "w")
    for bi, batch in enumerate(loader):
        (img, depth, desc, mask, bbox, point_norm, motion_gt, motion_type,
         img_size, fname, origin_3d, K, traj_gt, *extra) = batch
        img = img.to(device, non_blocking=True)
        depth = depth.to(device, non_blocking=True)
        words = model.tokenize(list(desc), word_len).to(device)
        K_norm = normalized_intrinsics(K.to(device).float(), img_size.to(device).float())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(img, depth, words, None, None, None, None, K_norm)

        tp = o.trajectory_pred.float().cpu()          # (B, 20, 3) relative
        tg = traj_gt.float()                          # (B, 20, 3) absolute
        tg_rel = tg - tg[:, :1]
        mp = o.motion_pred.float().cpu()              # row-selected axis
        # fp32 logits for the gate
        logits = o.motion_type_logits.float().cpu()
        p_rev = logits.softmax(-1)[:, 1]
        pred_type = logits.argmax(-1)

        gt_ax = F.normalize(motion_gt.float(), dim=-1)
        pr_ax = F.normalize(mp, dim=-1)
        cos_ax = (gt_ax * pr_ax).sum(-1).clamp(-1, 1)
        err_signed = torch.rad2deg(torch.acos(cos_ax))

        net_p = tp[:, -1] - tp[:, 0]
        net_g = tg_rel[:, -1] - tg_rel[:, 0]
        denom = net_p.norm(dim=-1) * net_g.norm(dim=-1)
        net_cos = torch.where(
            denom > 1e-8,
            (net_p * net_g).sum(-1) / denom.clamp(min=1e-12),
            torch.zeros_like(denom),
        )

        # per-segment direction score: mean cos over GT segments > 1 mm
        sp = torch.diff(tp, dim=1)                    # (B, 19, 3)
        sg = torch.diff(tg_rel, dim=1)
        ok = sg.norm(dim=-1) > 1e-3
        seg_cos = F.cosine_similarity(sp, sg, dim=-1, eps=1e-8)
        seg_score = torch.where(
            ok.any(-1),
            (seg_cos * ok).sum(-1) / ok.sum(-1).clamp(min=1),
            torch.zeros(len(tp)),
        )

        for i in range(len(tp)):
            if denom[i] <= 1e-8:
                continue                              # harness skips these rows
            out.write(json.dumps({
                "gt_type": int(motion_type[i]),        # 1 = rot
                "pred_type": int(pred_type[i]),
                "p_rev": round(float(p_rev[i]), 4),
                "err_signed_deg": round(float(err_signed[i]), 2),
                "flip": bool(err_signed[i] > 90),
                "net_cos": round(float(net_cos[i]), 4),
                "seg_cos": round(float(seg_score[i]), 4),
                "net_pred_m": round(float(net_p[i].norm()), 4),
                "net_gt_m": round(float(net_g[i].norm()), 4),
            }) + "\n")
            n += 1
        if bi % 20 == 0:
            print(f"batch {bi}/{len(loader)}  rows {n}", flush=True)
        if args.limit and n >= args.limit:
            break
    out.close()
    print(f"wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
