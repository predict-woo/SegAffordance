# 20260815_sf3d_g9_closeup010 — gen-9: close-up at 0.1% mask cutoff + 5% edge margin

**Recipe:** identical to gen-8 (`20260814_sf3d_g8_closeup`) except the data
split and schedule: `min_mask_area_frac` 0.0025 → **0.001**, NEW
`edge_margin_frac: 0.05` (GT interaction point and projected q* ≥5% of W/H
from every border), 30 epochs (g8: 60 — g8's curves had flattened by ~30 on
3× less data), milestones [24, 28]. 59,174 records (sensor −40,499, radius
−61,018, mask<0.1% −297,132, edge −441); val/test split 5,088. Cache
`sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl`.

**FRESH baseline:** 3.1× gen-8's data but a strictly harder split (masks
down to 0.1% vs 0.25%) — comparable to NEITHER gen-8 nor full-split arms.
Also serves as **arm A (joint)** of the supervision ablation
(`docs/superpowers/specs/2026-08-15-supervision-ablation-design.md`).

**Run:** RTX PRO 4500 ($0.72/hr), ~1h train (422 batches/ep ≈ 1:57),
best = epoch 23 (val 0.9191; top-3: ep23/24/26). Training exited cleanly
(no teardown hang this time). Cost ≈ $1.1 + eval.

## Test (best-epoch23, 5,088 samples)

| metric | gen-9 | gen-8 (0.25% split) |
|---|---|---|
| mIoU | 0.1463 | 0.178 |
| PDet (IoU>0.5) | 5.39% | 9.4% |
| type acc | 94.97% | 92.0% |
| axis° all / matched | 27.56 / 15.81 | 32.8 / — |
| 2D point err | 0.1525 | — |
| 3D point err (m) | 0.2920 | 0.30 |
| origin vs q* (m) | 0.3614 | 0.35 |
| origin→GT-axis line (m) | 0.3220 | — |
| radius err (m) | 0.1914 | 0.16 |
| traj_dir acc / cos | 93.10% / 0.736 | 94.8% / 0.783 |
| legacy pseudo-origin (m) | 1.0880 | — |

## Reading

- **The 0.1% split behaves like a mix of g8's close-ups and the old hard
  tail.** mIoU/PDet sit between g8 (0.178/9.4) and full-split (~0.10/~3):
  masks in the 0.1–0.25% band are still hard at 256². Not a regression —
  a different, harder denominator with 3.1× the data.
- **Semantics improved on 3× data despite the harder split:** type 95.0
  (g8 92.0), axis-all 27.6° (g8 32.8°) — data volume helps the
  classifier/axis heads more than the mask head.
- **3D localization ≈ flat** (0.29 m point, 0.36 m origin) — grounding
  tail still dominates these means.
- traj_dir 93.1% confirms the relative direct readout at scale.

## Next

- Arms B (art-only) and C (traj-only) of the supervision ablation train on
  THIS split — gen-9 is their joint-supervision reference.
- Comparison caveats (from the spec's implementation notes): absent CSV
  columns = no data; `test/mean_origin_error_m` is the legacy pseudo-origin
  (computed identically in all arms, NOT an origin-head metric);
  val/loss_total is not comparable across arms.
