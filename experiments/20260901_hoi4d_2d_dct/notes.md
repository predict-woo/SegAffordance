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
| p_rev C4-vs-C6 AUC | **0.657 (weak but real)** | (self-organized partially) |
| GT track roughness | 0.72 (jittery wrist) | ~0 (derived curves) |
| zero-shot on SF3D val | mIoU 0.003, all ~chance | — |

(3D/axis test rows are meaningless by construction — placeholder GT.)

## Reading

1. **Perception learns beautifully from real video.** Masks at 0.477
   mIoU / 53% PDet — nearly double the SF3D arms (bigger parts, salient
   hands, 354 objects); the wrist point is nailed at <1% of the image;
   projected-trajectory SHAPE beats the SF3D arm 3.5×.
2. **Direction does NOT emerge (49%, chance); type WEAKLY does
   (p_rev AUC 0.657, uncalibrated — both category means ~0.22).** On SF3D-2D, direction emerged (81–84%) from long clean
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

## CORRECTIONS (2026-09-02, after panel review)

1. **Data bug found via panels: the moving-part selection often picked
   the HAND.** HOI4D 2Dseg includes the hand as a class, and during
   open/close windows the hand's mask frequently maximizes the motion-
   energy criterion. So a substantial share of training masks are
   hand/arm, not the furniture part — the 0.477 mIoU partly measures
   hand segmentation. Fix for v2: exclude any color containing/near the
   WiLoR wrist in most window frames, then apply motion energy.
   (Interestingly the corrected panels show the model sometimes
   predicting the DRAWER against a hand GT label — 00_trans panel.)
2. **My first panels/probe fed the model float 0-255 images** —
   CRIS.forward normalizes ONLY uint8 (segmenter.py:393). The garbage
   panels and the original "p_rev AUC 0.46 = chance" claim were this
   bug; the harness metrics were never affected. Corrected probe:
   **AUC 0.657** (weak real type signal, uncalibrated). traj_dir 49%
   stands (harness-measured).
3. Panel batch (viz/20260901_hoi4d_2d_dct_val_panels, 12 imgs, now on
   the Mac too) uses the training split (0.15/seed 42) via
   tools/hoi4d_vis_2d_panels.py. The magenta trajectory overlay now
   reuses the trainer's EXACT projection (normalized K + depth-lifted
   anchor, train_SF3D_better.py:625) — the first render's hand-rolled
   pixel-K/z_p-anchor projection drew off-scale garbage, a vivid demo
   of the 2D arm's projection gauge freedom (the curve is only defined
   under the training-time projection).
