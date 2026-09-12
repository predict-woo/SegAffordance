# 20260912_usdnet — USDNet (Articulate3D, ICCV'25) retrained on SF3D scenes

**Goal.** Scene-level 3D baseline: USDNet's movable-part + articulation model (Mask3D backbone,
MinkowskiEngine) trained on our SF3D train scenes and scored per frame with our protocol.
Plan: `docs/superpowers/plans/2026-09-12-sf3d-external-baselines.md`.

**Upstream.** github.com/insait-institute/USDNet @ 0ba303d, recipe = `scripts/train_mov.sh`
(articulate3d_challenge_mov config, batch 1, voxel 2 cm, crop 5.5 m, c2f 0.1/0.4/100, 100 queries,
predict_articulation_mode 2, losses labels+masks+articulations, lr 1e-4 OneCycle, init = Mask3D
scannet200_benchmark.ckpt). Deviation: `trainer.max_epochs=200` (upstream yaml: 10,000; Mask3D
ScanNet: 601) = 11 h on an A100. Env: `runpod/baselines/usdnet/setup_env.sh` (torch 2.1.1+cu121,
MinkowskiEngine 02fc608 with thrust patches, volumentations from git).
Data: `tools/baselines_sf3d/sf3d_to_usdnet.py` — SceneFun3D 5 mm laser scans (re-downloaded,
`runpod/baselines/usdnet/download_scans.sh`) voxel-downsampled to 2 cm in a z-up ARKit frame,
one instance per annotated element with a motion (sem 1 rot / 2 trans, inter = inst), axis/origin
from motions.json; train 182 / validation 20 / test 22 scenes (our split). Median element = 11
points at 2 cm (p90 28; 97% under 50) — an order of magnitude below Articulate3D's doors/drawers.
Pod bl-usdnet (A100 80GB PCIe), train 00:50-11:45 UTC 2026-09-12 (~1.05 it/s, 182 it/epoch), export
+ per-frame projection 12:00-12:12 (~$19 incl. downloads/conversions). Chain:
`runpod/baselines/usdnet/chain.sh`; export from the best-val checkpoint (epoch 199, val AP50 0.035)
through a sibling data dir whose "validation" set is our test split (their eval path ignores the
mode override).

**Their evaluator (validation scenes, 20 of the train scenes held out):** mean AP50 0.000 at
epochs 19-119, 0.030 at 139, 0.035 at 199; rotation AP/AP50/AP25 0.016/0.051/0.199 at 199. The test
run's AP table is not in its log (no numbers to report at scene level for the test split).

**Our protocol (`tools/baselines_sf3d/usdnet_preds_to_jsonl.py` + scorer):** per test frame each
predicted instance (score >= 0.05, >= 20 points) is projected into the frame (splat radius 4 px)
and the one with the highest IoU vs the GT element mask is taken (oracle matching, no text);
2,465/5,088 elements have any overlapping instance (2,306 of them with confidence > 0.5).

| PDet | mIoU | type % | MA | MA signed | axis all / matched | flips all / rot | origin_err_m | origin_line_err_m |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0.071 | 47.2 | 23.1 | 21.8 | 55.7 / n.a. | 5.6 / 9.4 | 0.988 | 0.977 |

vs OPDFormer-C RGB-D 27.6 / 0.280 / 62.2 / 22.1 / 21.9 deg / 0.381; ours (cfframe seed 7)
23.9 / 0.264 / 93.3 / 36.0 / 17.6 deg / 0.305.

**Reading.** Faithful 2 cm USDNet cannot segment SF3D functional elements (no projected mask
reaches IoU 0.5; scene-level AP50 ~0.03-0.05), yet the axis/type of the overlapping instance is
right 23% of the time (MA 23.1, on par with OPDFormer-C) with the lowest flip rate of all
baselines — its per-point axis voting transfers, its masks do not. Origins are ~1 m off (scene-level
votes on 11-point instances). A 1 cm variant dataset was started (`data/usdnet_sf3d_v1cm`, partial)
for a resolution-matched follow-up; not trained.
