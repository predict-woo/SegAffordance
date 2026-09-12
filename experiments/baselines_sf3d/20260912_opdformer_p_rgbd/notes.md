# 20260912_opdformer_p_rgbd — OPDFormer-P (BMOC_V1), RGB-D, retrained on SF3D

**Goal / setup.** Same as `20260912_opdformer_c_rgbd/notes.md` (upstream OPDMulti code, 60k-iter
recipe, 8 SF3D classes, per-dataset pixel stats, bitmask masks, 256x192, portrait roll) with
`configs/opd_p_real.yaml` (BMOC_V1: axis/origin regressed in an object frame + a per-part
object-pose head, EXTRINSIC_WEIGHT 30). Our object pose for every part = the frame's
cam-to-world (scene pose), recentred PER SCENE so translations are metre-scale
(`tools/baselines_sf3d/opd_recenter_extrinsics.py`; the raw laser-scan frame put cameras 35-240 m
from the origin and made loss_extrinsic ~1500). A first launch on the un-recentred data was killed
at ~3k iters and restarted from scratch at 01:00 UTC.
Pod bl-opd-p (RTX PRO 6000 Blackwell, torch 2.8.0+cu128), 01:00-08:05 UTC 2026-09-12 (~7 h, ~$15).
Final model = `model_final.pth` (60k). Export used the parallel `opd_preds_to_jsonl.py`.

**Their evaluator (validation, segm, 10k..60k):** AP50 1.96 / 2.73 / 4.01 / 3.42 / 3.25 / 3.12;
axis50 0.07 -> 0.21. Test split (segm): AP 0.26, AP50 1.49, AP75 0.003, all_motion50 0.34, type50
2.89, origin50 0.03, axis50 0.60; bbox AP50 2.92.

**Our protocol (oracle best-IoU instance per GT element; 3,682/5,088 with any overlap, 279 of them
with confidence > 0.5):**

| PDet | mIoU | type % | MA | MA signed | axis all / matched | flips all / rot | origin_err_m | origin_line_err_m |
|---|---|---|---|---|---|---|---|---|
| 24.6 | 0.276 | 66.3 | 12.9 | 11.7 | 51.7 / 34.3 deg | 14.4 / 19.7 | 0.771 | 0.736 |

vs OPDFormer-C RGB-D: PDet 27.6 / mIoU 0.280 / type 62.2 / MA 22.1 / matched 21.9 deg / origin 0.381.

**Reading.** Masks on par with C; articulation much worse: predicting axis/origin in a scene
frame and mapping back through a predicted 6-DoF pose (their design for object-centric OPD data)
does not transfer to room-scale SF3D frames — the origin error doubles (0.77 m) and MA halves.
Faithful to their recipe; the object-pose definition (scene pose) is the one fit-to-data choice.
