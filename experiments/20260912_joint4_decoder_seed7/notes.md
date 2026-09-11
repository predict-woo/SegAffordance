# 20260912_joint4_decoder_seed7 — seed replicate of the joint decoder base run

**Goal.** Same recipe as `20260911_joint4_decoder_rgb_scalefree`, seed_everything 7 (split seed unchanged). Calibrates the single-seed noise behind the decoder comparisons (l2anchor: MA -0.24, rot flips +5; variants pending).

**Comparison row.** base seed 42: MA 31.29 / 30.86, matched 18.4, rot flips 13.9, origin 0.273, mIoU 0.266 / PDet 22.5, HOI4D 0.615.

**Result (2026-09-11 14:05 local, pod jdec-seed7, 4 h 20 min + tests, pod deleted — verified).** Best
`val/sf3d/loss_total` 1.0964 at epoch 19 (seed 42: 1.0089 at 16).

| SF3D metric | seed 42 | **seed 7** | Δ |
|---|---|---|---|
| MA / signed | 31.29 / 30.86 | 33.26 / 32.21 | +2.0 / +1.4 |
| type pass | 91.8 | 91.9 | |
| matched / all / signed-all | 18.4 / 25.3 / 32.1 | 18.8 / 25.3 / 33.3 | |
| flips all / rot | 10.7 / 13.9 | 10.5 / 15.2 | rot +1.2 |
| origin / line | 0.273 / 0.247 | 0.311 / 0.271 | +3.8 cm |
| radius / point 3D | 0.124 / 0.289 | 0.164 / 0.325 | +4 cm |
| mIoU / PDet | 0.266 / 22.5 | 0.247 / 20.0 | −0.019 / −2.5 |
| traj_dir | 91.1 | 91.6 | |
| HOI4D / EPIC / ARCTIC mIoU | 0.615 / 0.267 / 0.607 | 0.573 / 0.244 / 0.573 | −0.04 / −0.02 / −0.03 |

**Reading — the calibration the night needed.** The same recipe at a second seed moves MA by 2 points,
origin and radius by 4 cm, masks by 0.02 and hinge flips by 1.2. Consequences for the other single-seed
comparisons: (1) the l2anchor / h1anchor "MA −0.2" and "origin +2 cm" readings are inside seed noise;
their +5 rot-flip points are above the ±1.2 seen here but not far above — the sign story stays a
hypothesis. (2) The base recipe's origin/radius/mask "big win" (0.273 / 0.124 / 0.266) was partly seed
luck: seed 7 lands at 0.311 / 0.164 / 0.247, i.e. where the cfframe run is. (3) The cfframe record
(36.36) is +4.1 over the two-seed base mean (32.3), still well outside this noise; its own replicate
(`cfframe_seed7`) is the direct test. (4) HOI4D masks 0.57–0.62 for every decoder arm: the hand-source
mask regression vs the head-based joint4 (0.676) is robust to seed.

**Verdict.** Report decoder-arm differences below ~2 MA / 4 cm / 0.02 mIoU as inconclusive at one seed.

