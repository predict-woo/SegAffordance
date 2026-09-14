# 20260914_opdformer_c_rgbd_512 — OPDFormer-C (RGB-D) at 512x384, retrained on SF3D

**Goal.** Resolution-matched rerun of `20260912_opdformer_c_rgbd`: SF3D functional elements are ~12 px
at the recipe's 256x192, which the Sep-12 results suggested was the binding constraint (their own
test AP50 4.8). Requested by the paper session, approved by the user 2026-09-14 00:1x UTC.

**Setup.** Identical to `20260912_opdformer_c_rgbd` (OPDMulti `configs/opd_c_real.yaml`, R50
Mask2Former, batch 16, AdamW 1e-4, 60k iters, steps 36k/48k, COCO init, 8 SF3D classes, per-dataset
pixel stats) except the data: `data/opd_sf3d_512` from `sf3d_to_opd.py --size 512 384` (recentred
object poses, same splits: 32,171 train frames / 48,560 annotations). Their `MotionDatasetMapper`
applies no resize (flip / brightness / contrast only), so frames are consumed at the stored h5
size; `INPUT.*SIZE*` (256 in `opd_base.yaml`) are lifted to 512 for anything that reads them
(`IMG_SIZE=512` in `runpod/baselines/opd/chain.sh`). Pixel stats recomputed on the 512 h5.

**Run.** Pod `bl-opd512-c` (6w98k0sr3v4cma, A100-SXM4-80GB, $1.59/h, main volume EU-RO-1), chain
00:26-17:21 UTC 2026-09-14 (~0.95 s/iter incl. 6 validation passes, peak 11.4 GB; ~$27). Run dir
`runs/opd512_c_rgbd`, results `results/opd512_c_rgbd/` (preds.jsonl, train/test logs, config). Pod
deleted by the watcher at 17:22.

**Their evaluator (validation segm AP50 at 10k..60k):** 7.6, 7.9, 9.7, 8.3, 7.2, 7.3 (256 run: 4.8,
4.5, 4.5, 4.8, 4.8, 4.9). **Test:** segm AP 3.20 / AP50 10.29 / AP75 1.11, all_motion50 2.59, type50
12.09, origin50 0.81, axis50 4.54; bbox AP50 15.06 (256: segm AP50 2.58, bbox AP50 5.77, axis50 2.87).

**Our protocol (oracle best-IoU instance per GT element, 5,088 elements; signed MA first):**

| signed MA | MA (unsigned) | type % | axis all / matched | origin_err_m | origin_line_err_m | PDet | mIoU | flips all / rot |
|---|---|---|---|---|---|---|---|---|
| **26.0** | 27.2 | 65.6 | 41.3 / 19.1 deg | 0.370 | 0.326 | **37.7** | **0.340** | 7.4 / 22.6 |
| 256 run: 20.1 | 22.1 | 62.2 | 47.2 / 21.9 | 0.381 | 0.347 | 27.6 | 0.280 | 9.7 / 24.0 |

1,917 of 5,088 elements matched at IoU >= 0.5 (256: 1,406).

**Confidence-thresholded (non-oracle) PDet / mIoU** (`thresholded.json`; an element counts only if its
best-IoU instance has detector confidence above the threshold, otherwise IoU 0):

| threshold | elements kept | PDet | mIoU |
|---|---|---|---|
| oracle (any conf) | 5,088 | 37.7 | 0.340 |
| conf > 0.3 | 433 | 4.9 | 0.043 |
| conf > 0.5 | 398 | 4.6 | 0.040 |
| conf > 0.7 | 369 | 4.4 | 0.038 |
| 256 run, conf > 0.5 | 229 | 1.7 | 0.020 |

**Reading.** At 512x384 OPDFormer-C improves on every column: masks (PDet 27.6 -> 37.7, mIoU 0.280 ->
0.340; their segm AP50 x4 on test), type (62.2 -> 65.6) and — unlike the P variant — articulation
too (signed MA 20.1 -> 26.0, matched axis 21.9 -> 19.1 deg, origin 0.381 -> 0.370 m). The C head
predicts the axis directly in the camera frame, so sharper features help it; P's world-frame axis
via the predicted object pose does not benefit. It is now the strongest of the detector baselines on
articulation (signed MA 26.0 vs 14.4 for P-RGB 512), still well below A3VLM (43.8) and ours (42.7).
Confidence-thresholded numbers remain tiny (conf > 0.5 on a matching instance for 7.8 % of the
elements), as for every OPD variant: report the oracle columns with the thresholded ones alongside.
Per-sample CSV: `per_sample_metrics.csv`.
