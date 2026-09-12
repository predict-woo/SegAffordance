# 20260912_opdformer_p_rgb — OPDFormer-P (BMOC_V1), RGB only, retrained on SF3D

**Goal / setup.** Same as `20260912_opdformer_p_rgbd/notes.md` but `--input-format RGB` (their
RGB variant; also the required init for MOPD, whose model is RGB-only). Per-scene recentred
scene pose as object pose. Pod bl-opd-prgb (A100 80GB PCIe, torch 2.1.1+cu121, 0.53 s/iter),
01:03-11:20 UTC 2026-09-12 (~10.3 h incl. evals/test/export, ~$16). Final model = `model_final.pth`.

**Their evaluator (validation, segm, 10k..60k):** AP50 7.68 / 10.31 / 6.40 / 5.89 / 5.49 / 5.42
(peak at 20k, then the LR steps do not recover it); axis50 0.99 -> 0.90. Test split (segm): AP 1.22,
AP50 5.50, AP75 0.07, all_motion50 1.19, type50 6.45, origin50 0.01, axis50 0.97; bbox AP50 8.87.
RGB-only detects better than RGB-D at 256x192 for both C and P (the rendered-depth channel at
this resolution seems to hurt their R50 stem more than it helps).

**Our protocol (oracle best-IoU instance; 3,859/5,088 with any overlap, 316 with conf > 0.5):**

| PDet | mIoU | type % | MA | MA signed | axis all / matched | flips all / rot | origin_err_m | origin_line_err_m |
|---|---|---|---|---|---|---|---|---|
| 30.9 | 0.320 | 67.9 | 15.6 | 14.4 | 47.6 / 31.7 deg | 13.7 / 21.8 | 0.743 | 0.685 |

vs C RGB-D 27.6 / 0.280 / 62.2 / 22.1 / 21.9 deg / 0.381; vs P RGB-D 24.6 / 0.276 / 66.3 / 12.9 /
34.3 deg / 0.771; ours (cfframe seed 7) 23.9 / 0.264 / 93.3 / 36.0 / 17.6 deg / 0.305.

**Reading.** Best masks of the OPD family under oracle matching (PDet 30.9 > ours 23.9, mIoU 0.320
> 0.264) but the P-head articulation stays poor (MA 15.6, origin 0.74 m): the world-frame + predicted
pose design is the limiting factor, not the input. Init of `20260912_mopd_rgb`.
