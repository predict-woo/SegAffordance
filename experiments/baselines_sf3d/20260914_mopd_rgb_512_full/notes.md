# 20260914_mopd_rgb_512_full — MOPD (RGB) TRAINED on SF3D at 512x384 with OPDFormer's schedule

**Goal.** `20260912_mopd_rgb` ran MOPD's released recipe — a 1,000-iteration lr 5e-6 fine-tune —
from a composed init whose MOPD-specific layers were random, and landed within noise of its
OPDFormer-P init. This run trains the MOPD architecture for the full detector schedule so the added
EfficientSAM / EfficientNet-B5 branches actually learn. Requested by the paper session (their
Baidu-checkpoint route was dropped by the user), approved 2026-09-14.

**Setup.** MOPD repo (`lisiqi-zju/MOPD`, `configs/opd_p_real.yaml` = OPDMulti's byte for byte) with
`SCHEDULE=opdformer` in `runpod/baselines/mopd/chain.sh`: MOPD's `opd_base.yaml` differs from
OPDMulti's ONLY in `BASE_LR 5e-6 / MAX_ITER 1000 / CHECKPOINT_PERIOD 200 / EVAL_PERIOD 50` (diffed
2026-09-14), so the override restores exactly OPDFormer's values: lr 1e-4, 60k iters, steps
36k/48k, batch 16, checkpoint + eval every 10k. Data `data/opd_sf3d_512`, `IMG_SIZE=512`, RGB.
**Init:** composed by `tools/baselines_sf3d/mopd_compose_ckpt.py` from the EXISTING 256x192
`20260912_opdformer_p_rgb` `model_final.pth` (not the 512 P-RGB run, which was launched at the same
time — waiting for it would have cost a day; conv weights are resolution-agnostic and 60k iterations
at 512 have ample time to adapt; confirmed with the paper session 2026-09-14 00:3x UTC) + public
EfficientSAM ViT-S + geffnet `tf_efficientnet_b5_ap` ImageNet weights; MOPD's 120 own parameters
random (non-strict load). EfficientSAM pads inputs to its 1024 canvas, so the 512 frames need no
extra handling. Fit-to-env patches as in `20260912_mopd_rgb` (matcher Sinkhorn skip, bitmask, numpy).

**Run.** Pod `bl-mopd512` (az2ad8iqoruhd5, A100 80GB PCIe, $1.59/h), chain started ~00:33 UTC
2026-09-14. Run dir `runs/mopd512_rgb`, results `results/mopd512_rgb/`.

**Speed-up attempt (2026-09-14 21:10 - 22:20 UTC, abandoned).** The run is CPU-bound: one Python
thread at ~107 %, GPU 18 % / 90 W, `data_time` 0.2 s of a 2.6 s step (their per-image / per-query
loops launch small kernels and synchronise). To halve the wall clock with identical maths, a second
pod `bl-mopd512x2` (2 x A100 PCIe) was brought up with `runpod/baselines/mopd/patch_ddp.py` (`--resume`,
SyncBatchNorm for the B5 encoder's 116 BN layers so batch statistics stay over 16 images, the
EfficientSAM `nn.DataParallel` wrapper replaced by an equivalent pass-through, DDP with
`find_unused_parameters`). The resume path is correct — 40 iterations from `model_0019999.pth` on 2
GPUs gave validation segm AP50 9.7 vs 10.1 for the 1-GPU run at the same point — but not faster:
3.1 s/iter on 2 GPUs. Measured on that host: single process 8 images 2.5 s, 16 images 4.5 s (a
virtualised "EPYC-Genoa" host with a CPU quota, 1.7x slower per thread than the original pod's
bare-metal EPYC 7543), so the per-step cost is ~0.25 s/image + ~0.5 s fixed, and DDP adds ~0.6 s of
synchronisation (SyncBN + gradient all-reduce). Even on a fast host the projected gain was ~1.3x;
not worth a second lottery. Pod deleted 22:20 UTC (~$3); the original single-GPU run was never
interrupted. The DDP tooling is kept (it works) for future multi-GPU MOPD runs.

**Moved to an H200 (2026-09-15 02:02 UTC).** User asked for a single-H200 test. Pod `bl-mopd-h200`
(t0ht94598bv263, AP-JP-1 on the `bl-apjp` volume, $4.59/h; Xeon Platinum 8460Y+ host, 3.7 GHz boost):
`runpod/baselines/mopd/time_smoke.sh` resumed `model_0029999.pth` for 100 iterations on a 320-image
subset -> 1.90 s/iter (2.73 s/iter on the A100 at the same point; the CPU-bound step benefits from the
faster host CPU as much as from the GPU). The dataset needed for RGB training (`train/valid/test.h5`
+ annotations, 23 GB; the RGB mapper never opens `depth.h5`) was copied to the JP volume with 10
parallel ssh streams (~80 MB/s aggregate) and md5-verified against the main volume. The chain was
then resumed on the H200 with `RESUME=1 NPROC=1` (same config, same batch 16, `--resume` from
`last_checkpoint`; detectron2 does not restore the sampler state, so the shuffle order after 30k
differs from what the A100 would have drawn -- the same situation as any resume; losses at
30019-30199 in the same band as the A100 log). With the user's OK the A100 run was killed at iter
~32.5k (2,500 iterations redone, ~1.3 h) once the H200 had passed iteration 30,180 with 2.0 s/iter,
no errors and identical memory. Training end ~18:00 UTC 2026-09-15 (was ~23:00 UTC); ~$44 more.
Results (`model_final.pth`, `test/`, `preds.jsonl`, logs) are copied back to the main volume's
`runs/mopd512_rgb` / `results/mopd512_rgb` when the chain finishes.

**Result (2026-09-15; training ended 18:26 UTC on the H200, test 18:35, export + scoring on the dev pod
19:17 UTC; H200 pod deleted 18:45 UTC, ~17.5 h x $4.59 = ~$80 + ~$41 of A100 time = ~$121 total).**
The chain's own export step died on the AP-JP-1 volume (`opd_preds_to_jsonl.py` needs the SF3D GT
cache under `/workspace/cache`, main volume only), so `model_final.pth`, `test/inference/*` and the logs
were rsynced back to the main volume and `runpod/baselines/mopd/finish_mopd512.sh` did the export +
scoring there. Their evaluator on MotionNet_test: bbox AP50 12.5, segm AP50 8.9 (OPDFormer-P RGB 512:
segm AP50 11.6). Ours (`metrics.json`, `per_sample_metrics.csv`, `thresholded.json`):

| | OPDFormer-P RGB 512 (init's recipe) | MOPD 512, 60k OPDFormer schedule | MOPD 256, their 1k fine-tune |
|---|---|---|---|
| PDet / mean IoU | 40.2 / 0.372 | 37.6 / 0.357 | 30.9 / 0.321 |
| type accuracy | 74.7 | 71.6 | 67.6 |
| MA unsigned / signed | 15.4 / 14.4 | 19.0 / 17.2 | 13.3 / 12.2 |
| axis err all / matched (deg) | 45.4 / 30.4 | 45.1 / 30.1 | 47.8 / 32.2 |
| flips all / rot | 11.8 / 20.0 | 13.0 / 24.5 | 13.6 / 22.6 |
| origin err (m) | 0.725 | 0.666 | 0.715 |
| conf > 0.5: kept / PDet | -- / 4.0 | 339 / 3.6 | -- |

Reading: trained for the full schedule at 512x384, MOPD's extra branches (EfficientSAM prompt encoder
+ B5 "normal" encoder) buy +2.8 signed MA and a 6 cm better origin over the OPDFormer-P RGB 512 run it
was initialised from, at the cost of slightly weaker masks / type (its 256 fine-tune had been within
noise of its init). Validation segm AP50 along the run: 10.1 (20k) -> 11.7 (40k). Note MOPD trains
from the 256 P-RGB init (see Setup); the 512 P-RGB run is the resolution-matched comparison.
Hand-video (Fig. 5): `results/mopd512_rgb/handvideo_preds.jsonl` (oracle, 87/150 frames with a
matching instance) and `handvideo_preds_nearest.jsonl` (5 figure samples, 3 nearest-centroid
fallbacks), produced on the H200 pod by `runpod/baselines/handvideo_opd.sh mopd512_rgb mopd_rgb`.
