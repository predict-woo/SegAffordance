# 20260914_usdnet_v1cm — USDNet on SF3D scenes at 1 cm voxels

**Goal.** `20260912_usdnet` (2 cm voxels, as in their Articulate3D setup) gave SF3D functional
elements a median of 11 points (97 % under 50) and AP50 0.000; PDet 0.0 on our protocol. This run
halves the voxel size so elements have ~8x the points. Requested by the paper session, approved
2026-09-14.

**Setup.** Same as `20260912_usdnet` (USDNet @ 0ba303d, `scripts/train_mov.sh` recipe from the
Mask3D scannet200 backbone, 200 epochs, val every 20, best `val_mean_ap_50` checkpoint, crop 5.5 m /
min 75k points, c2f radius 0.1) except `VOXEL=0.01`: converter `sf3d_to_usdnet.py --voxel 0.01` →
`data/usdnet_sf3d_v1cm`, `data.voxel_size=0.01` (`runpod/baselines/usdnet/chain.sh`). The conversion
(CPU) was 38 % done from Sep 12 and was finished on the dev pod (16 workers) while the training pod
came up; the chain waits for `.done_convert` (`CONVERT_HERE=0`).

**Conversion (1 cm).** Started on the dev pod (16 workers) at 00:2x UTC, moved to the training pod at
00:55 (the dev pod has a 31 GB cgroup cap and the workers OOM-killed the paper session's probes), done
05:48 UTC with 32 workers: 182 / 20 / 22 scenes, 0 failures; e.g. scene 421015 has 2.48 M points at
1 cm; colour mean/std recomputed (0.529/0.503/0.490, 0.248/0.249/0.256).

**Run.** Pod `bl-usdnet1cm` (d2z37fxh1uwutg, A100 80GB PCIe, $1.59/h). First launch (05:50 UTC) OOMed at
epoch 0, iteration 11: 78.9 GB in use of which only 61.3 GB allocated — the caching allocator fragments on
MinkowskiEngine's variable-size sparse tensors. Relaunched 05:53 with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (allocator setting only, no numerical effect; now the
chain's default): passed that point at ~71 GB, 2.28 s/it, then OOMed again at iteration 93 — this time
`cudaErrorMemoryAllocation` inside MinkowskiEngine's own allocator, i.e. the largest 5.5 m crops at
1 cm genuinely exceed 80 GB (millions of voxels; Mask3D/USDNet was built for ScanNet-scale counts).
**Fit-to-memory deviation:** third launch 06:08 UTC with `data.crop_length=4.0` still OOMed (iteration
41, 77 GB allocated by PyTorch): the 1 cm scenes hold a median 1.24 M / max 4.5 M points (2 cm: 0.36 M
/ 1.27 M) and Mask3D's per-decoder-layer mask logits and matching costs scale with points x queries.
Fourth launch 06:14 UTC with `data.crop_length=3.0` OOMed at iteration 1 (78 GB). The traceback points
at USDNet's own articulation head (`models/mask3d.py:575`, `predict_articulation_mode 2`), which
materialises (points x queries x 3) tensors per decoder layer; measured crops on the densest 1 cm scene
(4.5 M points, 30 x 10 m) still hold 0.12-0.71 M points at 3.0 m (1.4-2.2 M at the recipe's 5.5 m), so
no single GPU takes the recipe's crops at 1 cm — a bound on training points is inherent to running this
architecture at 1 cm. **Fifth launch 06:20 UTC: `data.crop_length=3.0` + a training-only cap of 450 k
points per sample** (uniform random subsample above the cap, all per-point arrays kept aligned;
`runpod/baselines/usdnet/patch_max_points.py`, env `SF3D_MAX_POINTS`; the 2 cm run's largest crops were
~0.6 M points). The cap binds only on the densest crops, so the 1 cm density is kept almost everywhere.
`crop_min_size` 75k, c2f and everything else unchanged; the 2 cm run keeps 5.5 m / no cap. Test-time
inference is on whole scenes (`data.cropping=false`), unchanged. Run dir `runs/usdnet_v1cm`, results
`results/usdnet_v1cm/`; the OOM logs are kept as `logs/train_usdnet_v1cm.oom{1,2,3,4}.log`.

**Result (2026-09-15 07:03 UTC CHAIN_DONE; pod deleted, ~24.7 h, ~$39).** Trained 200 epochs; the
exported checkpoint is the best-validation one, `epoch=139-val_mean_ap_50=0.019.ckpt` (the 2 cm run's
best was 0.035 at its own epoch). Our scorer on the 5,088 test elements (`metrics.json`,
`per_sample_metrics.csv`, `thresholded.json`):

| | 2 cm (20260912_usdnet) | 1 cm, crop 3.0 m + 450 k cap |
|---|---|---|
| PDet (mask IoU >= 0.5) | 0.0 | 1.1 (58 elements) |
| mean mask IoU | 0.071 | 0.135 |
| type accuracy | 47.2 | 52.3 |
| MA (unsigned / signed) | 23.1 / 21.8 | 11.2 / 7.8 |
| axis err all / matched (deg) | 55.7 / n.a. | 55.7 / 47.7 |
| axis flips all / rot | 5.6 / 9.4 | 17.8 / 32.0 |
| origin err (m) | 0.988 | 1.551 |

Confidence-thresholded (conf > 0.5): 2,247 elements keep a prediction, PDet 1.1, mean IoU 0.101.
Reading: the finer voxels let the network produce a few correct element masks (the 2 cm run matched
none) and double the mean IoU, but the articulation columns get WORSE -- the axis votes that carried
the 2 cm run's MA (per-frame projection of whole-scene predictions) now flip three times as often
and the origins drift further. The two runs are not a clean ablation (the 1 cm run needed the crop /
point-cap deviations above, and USDNet has no seed control), so the table reports the 1 cm run as the
resolution-matched row and keeps the 2 cm row for reference; neither is competitive on SF3D
elements. Scored with `runpod/baselines/score_run.sh usdnet_v1cm 20260914_usdnet_v1cm usdnet_v1cm`.
