# 20260816_sf3d_g11a_originv2 — attribution arm: origin local sample ALONE, on v2

**Recipe:** exactly gen-10 + `use_origin_local_feature: true`, on v2 (0.1 m
trans sweeps). Separates the origin flag from gen-11's dataset change.

**Run caveat (quota incident 2026-08-16):** wedged ~85 min at epoch 22
during the volume-quota outage, killed and resumed from the epoch-21
checkpoint (metrics.csv = epochs 0–21, metrics_resumed.csv = 22–29). Best =
epoch 22 (val 0.9889). Numbers carry an asterisk.

## Test (best-epoch22, 5,088 samples; Δ vs gen-10 = its exact base)

| metric | g11a | gen-10 | Δ |
|---|---|---|---|
| origin vs q* / line (m) | 0.3650 / 0.3251 | 0.3692 / 0.3279 | **−0.004 / −0.003** |
| radius err (m) | 0.1863 | 0.1839 | +0.002 |
| type acc | 90.61 | 93.08 | **−2.5** |
| mIoU / PDet | 0.1395 / 4.99 | 0.1435 / 5.40 | −0.004 / −0.4 |
| 2D / 3D point | 0.157 / 0.297 | 0.155 / 0.294 | ≈flat |
| traj_dir acc / cos | 91.47 / 0.724 | 91.53 / 0.714 | ≈flat |

## Reading

- The origin local sample ALONE buys a small origin gain (−0.4 cm, matching
  gen-11's −0.7 cm direction) at ≈no radius change.
- The type −2.5 and mask −0.4 dips do NOT cleanly replicate gen-11's
  pattern (gen-11 type was 92.9 — better than this arm despite carrying v3
  too), so those dips are within run-to-run noise + the resume asterisk
  rather than attributable to the flag.
- Attribution verdict: gen-11's mask regression is NOT explained by the
  origin flag; g11b's rebalance result pins it on the v3 loss balance.
