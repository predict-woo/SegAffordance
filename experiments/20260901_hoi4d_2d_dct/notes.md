# 20260901_hoi4d_2d_dct — the 2D recipe on real HOI4D hands: perception excels, geometry doesn't emerge

**Question:** does real hand-interaction video (WiLoR wrist tracks +
moving-part masks, zero articulation GT) teach the 2D pipeline what
SF3D's derived 2D tracks taught it? From-scratch (user-approved init).

**Data:** 354 furniture seqs → 15,612 samples (C4 7,389 / C6 8,223,
all 4 cameras; tools/hoi4d_process_2d.py, spec 2026-08-31). Formats
survived: shift_mask dirs (cams 2–4), markResult action JSONs (8),
10s-clock action times (13, found by sibling session), FFV1 16-bit
depth, gather-grid mask coords. Scene split by physical object, 15%.

**Recipe:** identical to 20260822_sf3d_g17_2d_dct. 30 epochs (~1h,
210 steps/ep), best = epoch 28 (val 0.3584).

## Val (15% held-out objects), best-epoch28

| metric | HOI4D (this run) | SF3D 2D arm (g17_2d_dct) |
|---|---|---|
| mIoU / PDet | **0.477 / 53.4** | 0.2655 / ~20 |
| point error (norm) | **0.0084** | ~0.10 |
| traj shape / err / anchor | **0.027** / 0.028 / 0.012 | 0.0947 / — / — |
| traj_dir acc / cos | **49.0 / 0.03 (CHANCE)** | 81–84 / — |
| p_rev C4-vs-C6 AUC | **0.463 (CHANCE)** | (self-organized partially) |
| GT track roughness | 0.72 (jittery wrist) | ~0 (derived curves) |
| zero-shot on SF3D val | mIoU 0.003, all ~chance | — |

(3D/axis test rows are meaningless by construction — placeholder GT.)

## Reading

1. **Perception learns beautifully from real video.** Masks at 0.477
   mIoU / 53% PDet — nearly double the SF3D arms (bigger parts, salient
   hands, 354 objects); the wrist point is nailed at <1% of the image;
   projected-trajectory SHAPE beats the SF3D arm 3.5×.
2. **Motion geometry does NOT emerge: direction and type both at
   chance.** On SF3D-2D, direction emerged (81–84%) from long clean
   derived tracks; here the supervision is short, jittery real wrist
   tracks (GT roughness 0.72 vs ~0) whose NET direction is noisy at
   window scale — and arcs vs lines are indistinguishable in that
   regime, so L_pp has no geometric signal to funnel into the type gate
   either. The 2D pipeline's geometry-emergence story is a property of
   CLEAN track supervision, not of tracks per se.
3. **Zero-shot transfer to SF3D is null** (mIoU 0.003): from scratch on
   354 objects + 6 prompt templates = domain-locked features. The
   fine-tune direction (SF3D-pretrained + HOI4D) is the natural next
   arm and was deliberately deferred at commissioning.
4. **Obvious remedies for a v2** (recorded, not commissioned): temporal
   smoothing of wrist tracks before supervision (kill the 0.72
   roughness); longer windows (Reachout+manipulation) for net-direction
   SNR; palm centroid instead of wrist; mixed SF3D+HOI4D training;
   fine-tune from g17_2d_dct.

**Ops footnotes:** train_pod.sh gained the *hoi4d* config family
(7c23c3d). Panels rendered to viz/20260901_hoi4d_2d_dct_val_panels/ on
the VOLUME; images not yet on the Mac (pod deleted first — fetch on
next dev-pod start). Vis tool used its own 0.1 split (150/17 scenes) vs
training's 0.15 — panel "val" may overlap train; treat as qualitative.

test passes: logs/test.log (HOI4D val), logs/test_sf3d_zeroshot.log.
ckpt best-epoch28-valloss0.3584. Pod deleted; LMDBs at
/workspace/datasets/hoi4d_processed_2d (main volume) + backup copy on
the hoi4d volume.
