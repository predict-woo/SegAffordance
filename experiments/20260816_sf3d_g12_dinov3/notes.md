# 20260816_sf3d_g12_dinov3 — dinov3 backbone baseline in the modern recipe

**Recipe:** g11b (v3 data, trajectory_weight 0.15, origin local sample,
normalized L_pp) with the DINOv3 ViT-L/16 + dino.txt backbone, frozen,
exactly as implemented for gen-3 (SimpleFeaturePyramid from the final
layer, aligned tokens at /32). Baseline for the standard-stack stages
(gens 13–15). RTX PRO 6000 WK; restarted from scratch after the quota
outage killed the first attempt at epoch 2. Best = epoch 20 (val 0.9624).

## Test (best-epoch20, 5,088 samples; vs g11b = same recipe on CLIP)

| metric | g12 (dinov3) | g11b (CLIP) | Δ |
|---|---|---|---|
| mIoU | **0.1599** | 0.1467 | **+1.3 pts (family best)** |
| 2D point | **0.1138** | 0.1591 | **−28% (family best)** |
| 3D point (m) | **0.2529** | 0.3045 | **−17% (family best)** |
| origin vs q* / line (m) | **0.3259 / 0.2964** | 0.3617 / 0.3183 | **−3.6 / −2.2 cm (family bests)** |
| radius err (m) | 0.1734 | 0.1732 | tied family best |
| type acc | 91.45 | 91.25 | flat |
| axis° all / matched | **37.13 / 29.08** | 28.08 / 14.62 | **+9.0 / +14.5 — badly degraded** |
| MA pass | 12.11 | 22.56 | **halved** |
| PDet | 5.09 | 6.07 | −1.0 (better mean IoU, fewer >0.5 hits) |
| traj_dir acc / cos | 84.65 / 0.601 | 84.81 / 0.594 | flat (the w=0.15 signature) |

Consistency probe: normalized L_pp mean **0.199** (CLIP family: ~0.054) —
the axis/origin/trajectory triad is far less mutually consistent, dominated
by the axis-direction error.

## Reading

- **dino.txt's grounding/localization advantage is real and large**: masks,
  2D point, 3D point, origin, radius — every "where is it" metric is a
  family best, echoing and amplifying gen-3's "dinov3 wins geometry".
- **The axis-direction head is the casualty** (37° vs 28°): it regresses a
  3D direction from pooled condition features, and the dinotxt feature
  space appears worse for that than CLIP's — or the single-final-layer
  pyramid starves it. This is exactly what the standard-stack stages will
  test: taps (gen-14) change the feature diet; if axis recovers there, the
  pyramid was the problem.
- traj_dir carries g11b's w=0.15 under-training signature unchanged —
  backbone-independent, consistent with the rebalance analysis.
- PDet vs mIoU divergence suggests dinov3 masks are better on average but
  flatter (fewer confident >0.5 hits) — worth a viz look during gen-13.

vis: (pending — viz batch deferred to the gen-13 comparison)
