# 20260914_opdformer_p_rgb_512 — OPDFormer-P (RGB) at 512x384, retrained on SF3D

**Goal.** Resolution-matched rerun of `20260912_opdformer_p_rgb` (see
`20260914_opdformer_c_rgbd_512/notes.md` for the motivation and the shared data/config facts).

**Setup.** OPDMulti `configs/opd_p_real.yaml`, `--input-format RGB`, recipe unchanged; data
`data/opd_sf3d_512`; `IMG_SIZE=512` (`runpod/baselines/opd/chain.sh p_rgb`).

**Run.** Pod `bl-opd512-prgb` (9r6p29cxu5gwtt, A100 80GB PCIe, $1.59/h), 00:36-13:39 UTC 2026-09-14
(~0.75 s/iter, peak 11.3 GB; ~$21). Run dir `runs/opd512_p_rgb`, results `results/opd512_p_rgb/`
(preds.jsonl, train/test logs, config). Pod deleted by the watcher at 13:40.

**Their evaluator (validation segm AP50 at 10k..60k):** 12.9, 15.6, 15.0, 14.9, 14.0, 13.6 (256 run:
7.7, 10.3, 6.4, 5.9, 5.5, 5.4) — same shape (peak at 20k, slow decline to the final model, which the
recipe uses), roughly 2.3x higher. **Test:** segm AP 3.09 / AP50 11.62 / AP75 0.39, all_motion50 1.54,
type50 11.27, origin50 1.05, axis50 2.59; bbox AP50 15.57 (256: segm AP50 5.50, bbox AP50 8.87).

**Our protocol (oracle best-IoU instance per GT element, 5,088 elements; signed MA first):**

| signed MA | MA (unsigned) | type % | axis all / matched | origin_err_m | origin_line_err_m | PDet | mIoU | flips all / rot |
|---|---|---|---|---|---|---|---|---|
| **14.4** | 15.4 | 74.7 | 45.4 / 30.4 deg | 0.725 | 0.655 | **40.2** | **0.372** | 11.8 / 20.0 |
| 256 run: 14.4 | 15.6 | 67.9 | 47.6 / 31.7 | 0.743 | 0.685 | 30.9 | 0.320 | 13.7 / 21.8 |

2,047 of 5,088 elements matched at IoU >= 0.5 (256: 1,573).

**Confidence-thresholded (non-oracle) PDet / mIoU**, from `per_sample_metrics.csv` (an element counts
only if its best-IoU instance has detector confidence above the threshold; otherwise IoU 0):

| threshold | elements kept | PDet | mIoU |
|---|---|---|---|
| oracle (any conf) | 5,088 | 40.2 | 0.372 |
| conf > 0.3 | 362 | 4.1 | 0.036 |
| conf > 0.5 | 337 | 4.0 | 0.033 |
| conf > 0.7 | 303 | 3.7 | 0.030 |
| 256 run, conf > 0.5 | 316 | 2.5 | 0.027 |

**Reading.** Resolution does exactly what the Sep-12 analysis predicted and nothing more: the masks
and the type head improve (PDet +9.3, mIoU +0.05, type +6.8; their own segm AP50 x2.1), while the
articulation numbers are unchanged within noise (signed MA 14.4 -> 14.4, matched axis 31.7 -> 30.4
deg, origin 0.74 -> 0.73 m). OPDFormer-P's world-frame axis + predicted object pose remains the
bottleneck, not the pixels. The confidence-thresholded columns stay tiny at both resolutions: the
detector puts confidence > 0.5 on a matching instance for only ~6.5 % of the elements, so the oracle
protocol (best-IoU instance regardless of score) remains the only way to read its masks — the paper
should report both. Per-sample CSV: `per_sample_metrics.csv`; thresholds: `thresholded.json`.
