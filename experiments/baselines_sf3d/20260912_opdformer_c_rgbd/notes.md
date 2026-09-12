# 20260912_opdformer_c_rgbd — OPDFormer-C (BMCC), RGB-D, retrained on SF3D

**Goal.** External baseline for the paper table: OPDMulti's OPDFormer-C retrained faithfully on the
SF3D train split and scored with our test protocol. Plan: `docs/superpowers/plans/2026-09-12-sf3d-external-baselines.md`.

**Upstream.** github.com/3dlg-hcvc/OPDMulti (opdformer), config `configs/opd_c_real.yaml` (R50
Mask2Former, 100 queries, batch 16, AdamW 1e-4, 60k iters, steps 36k/48k, COCO Mask2Former init).
Only fit-to-data overrides: `MODEL.SEM_SEG_HEAD.NUM_CLASSES 8` (SF3D affordance labels),
`MODEL.PIXEL_MEAN/STD` from our train split (RGB + depth mm), `INPUT.MASK_FORMAT bitmask`
(sparse splat masks as RLE; mapper patched, `runpod/baselines/opd/patch_mapper.py`),
`DATALOADER.NUM_WORKERS 8`. Numpy-alias patch in their evaluator (`patch_numpy_aliases.py`).
Data: `tools/baselines_sf3d/sf3d_to_opd.py` (256x192, portrait frames rolled 90 deg = exact camera
roll, OPD y/z-flipped camera frame); train 182 scenes / valid 20 (their eval cadence only) / test =
our 22-scene 5,088-sample split (`experiments/baselines_sf3d/splits.json`).
Pod bl-opd-c (RTX PRO 6000 Blackwell, torch 2.8.0+cu128 rebuilt, 0.29 s/iter), 00:29-06:10 UTC
2026-09-12 (~5.7 h incl. valid evals, test, export; ~$12). Final model = `model_final.pth` (60k).

**Their evaluator (validation split, segm, 10k..60k):** AP50 4.80 / 4.55 / 4.45 / 4.78 / 4.80 / 4.90 —
plateau from 10k; +axis (motion_axis50) 0.38 -> 0.98-1.1. Test split (segm): AP 0.49, AP50 2.58,
AP75 0.03, all_motion50 0.58, type50 6.07, origin50 0.84, axis50 2.87; bbox AP50 5.77.
Detection at 256x192 is the binding constraint: SF3D elements are ~12 px at that size (their own
parts are doors/drawers).

**Our protocol (`tools/baselines_sf3d/score_predictions.py`, 5,088 test samples):** per GT element
the predicted instance with the highest mask IoU (native res) among the detector's outputs is taken
(ORACLE matching, no text grounding; 3,379/5,088 elements have any overlapping instance, only 229 of
those with confidence > 0.5), then scored exactly like our INDEX rows.

| PDet | mIoU | type % | MA | MA signed | axis all / matched | flips all / rot | origin_err_m | origin_line_err_m |
|---|---|---|---|---|---|---|---|---|
| 27.6 | 0.280 | 62.2 | 22.1 | 20.1 | 47.2 / 21.9 deg | 9.7 / 24.0 | 0.381 | 0.347 |

Reference (our best, 20260912_joint4_decoder_cfframe_seed7): PDet 23.9, mIoU 0.264, type 93.3,
MA 36.0, all-axis 22.5, origin 0.305.

**Reading.** With oracle instance selection OPDFormer-C's masks are competitive (PDet 27.6 vs our
23.9) but its articulation is far weaker (MA 22.1 vs 36.0, type 62 vs 93, all-axis 47 vs 22.5 deg,
origin +8 cm). Unmatched elements (33%) count as failures on every column. Confidence-thresholded /
top-1 protocols are a follow-up (their `instances_predictions.pth` is kept under
`/workspace/datasets/baselines/runs/opd_c_rgbd/test/inference/`).
