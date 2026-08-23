# 20260823_sf3d_s10_3d_dir — label-efficiency v2 arm A': scratch on 10%, g21 recipe

**Recipe:** the g21 recipe (g17 split-axis + DCT head + dir term 0.1)
FROM SCRATCH on the same ~10% scene partition as v1 (19/202 scenes,
5,502 samples, seed 4242). Full 30 epochs (early stopping never fired),
best = epoch 24 (val 1.7506 — below v1 scratch-10's 1.8192 despite an
extra loss term: the DCT head helps in the small-data regime too).

## Test (5,088) vs v1 scratch-10 (g17 recipe) and C' (100%, g21)

| metric | s10 v1 (g17) | **A' (g21)** | C' (100%) |
|---|---|---|---|
| mIoU / PDet | 0.021 / 0.37 | 0.029 / 0.98 | 0.271 / 23.2 |
| axis matched / all | 25.0° / 44.4° | 38.4° / 44.3° | 20.8° / 27.0° |
| flips all / rot | 21.1 / 35.1 | **19.4 / 25.6** | 11.6 / 15.4 |
| MA / MA_signed | 4.9 / 4.4 | 4.9 / 4.3 | 26.6 / 26.4 |
| origin / radius (m) | 0.555 / 0.253 | 0.516 / 0.221 | 0.278 / 0.140 |
| traj_dir acc / cos | 80.3 / 0.477 | 80.2 / 0.477 | 88.8 / 0.724 |
| roughness (m) | 0.263 | **0.018** | 0.0085 |

## Reading

Control story unchanged from v1: **10% alone cannot buy a trunk** (mIoU
0.029, PDet 1.0 — MA pinned at 4.9 by absent detection) while
articulation-adjacent quantities degrade far more gracefully. Recipe
effects at 10% data: the DCT head gives its usual massive smoothness win
(0.263 → 0.018) and a slightly less-dead trunk; flips are lower than
v1's scratch (25.6 vs 35.1 rot) but matched-axis error is worse (38.4°
vs 25.0°) — at this data scale those articulation columns are noisy,
don't over-read. Serves as the v2 lower bound for B'2.

test pass: logs/test.log (ckpt best-epoch24-valloss1.7506); CSV logger
version_0 (clean single launch). Pod deleted after the pass.
