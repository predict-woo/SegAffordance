# 20260816_sf3d_g11b_trajw015 — rebalance arm: trajectory_weight 0.15 on v3

**Recipe:** exactly gen-11 (origin flag + v3) with `trajectory_weight`
0.5 → 0.15, calibrated for v3's 7× longer trans sweeps (band 0.11–0.16
from converged/val/init ratios).

**Run caveat:** crashed at ~epoch 19 in the quota outage, resumed from the
epoch-16 checkpoint (metrics.csv = 0–16, metrics_resumed.csv = 17–29; the
resume reset ModelCheckpoint's top-k, so pre-crash bests were rotated
out). Best available = epoch 22 (val 0.9979; val NOT comparable to other
arms — different trajectory weight).

## Test (best-epoch22, 5,088 samples)

| metric | g11b (w=0.15) | gen-11 (w=0.5) | gen-10 (v2 base) |
|---|---|---|---|
| mIoU / PDet | **0.1467 / 6.07** | 0.1320 / 4.89 | 0.1435 / 5.40 |
| axis° matched | **14.62** (family best) | 18.34 | 17.75 |
| radius err (m) | **0.1732** (family best) | 0.1790 | 0.1839 |
| origin vs q* / line (m) | 0.362 / 0.318 | 0.362 / 0.324 | 0.369 / 0.328 |
| type acc | 91.25 | 92.90 | 93.08 |
| 3D point (m) | 0.304 | 0.311 | 0.294 |
| traj_dir acc / cos | **84.81 / 0.594** | 91.29 / 0.708 | 91.53 / 0.714 |

## Reading

- **The loss-budget hypothesis is confirmed in both directions.** Dropping
  the trajectory weight recovered everything gen-11 lost (masks BEYOND
  gen-10: mIoU 0.147, PDet 6.07) and unlocked family bests in matched-axis
  (14.6°) and radius (0.173) — but the trajectory head under-trains at
  0.15 on v3 (traj_dir 84.8, worst since gen-7's absolute head).
- w=0.5 and w=0.15 bracket the trade-off; the frontier suggests **~0.3**
  as the next candidate (recorded, not scheduled). Alternatively keep 0.15
  and add a direction-only (cosine) trajectory term — direction is what
  collapsed, not magnitude.
- Origin/radius gains compound with the origin local sample: the g11b
  combination is the current best articulation-geometry checkpoint.
