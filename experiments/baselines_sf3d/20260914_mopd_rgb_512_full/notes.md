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

**Result.** pending.
