# 20260814_sf3d_g7 — heatmap + depth lifts, absolute trajectory, frozen CLIP

**Goal:** fix gen-6's localization half (spec 2026-08-14): both 3D
points become heatmap + scalar-depth predictions lifted with intrinsics
(3-channel projector: mask + point + origin heatmaps; z_p with a local
feature sample at point_uv, z_q condition-only; composed 3D losses;
Gaussian BCE on both channels, origin at projected q*); classical 2D
point machinery + GT-mask pooling teacher forcing restored; trajectory
switched to a direct ABSOLUTE 20-point readout (user decision, zigzag
+ absolute-regression risks pre-registered); CLIP FROZEN.

**Setup:** config.yaml (= config/sf3d_train_runpod_g7.yaml @ launch).
RTX PRO 4500 ($0.72/hr, PRO 6000 out of stock x3 SKUs), NVMe LMDBs,
24 workers, batch 128, 16 epochs ~4.5h (~$3.5 + eval). torch.compile
handled the new lift ops (grid_sample, linalg.inv) cleanly. Teardown
hung after final val AND after eval (killed both; known pattern).

**Best: epoch 14, val 1.3171** — kept improving through the LR drop
(no gen-6-style late overfit). last.ckpt md5-deduped to ep14; single
evaluated checkpoint, 36k filtered test samples.

**Result vs gen-6 (same filtered eval):**

| metric | gen-7 | gen-6 | read |
|---|---|---|---|
| point_err_3d_m | **0.454** | 0.582 | the lift win (-22%) |
| mean_point_error (2D) | **0.138** | n/a | restored to gen-4 par (0.135) |
| radius_err_m | **0.349** | 0.368 | slightly better |
| type pass_rate_m | **98.3** | 97.5 | best ever |
| axis dir all/matched | 23.6/16.9 | 22.1/15.1 | held (~flat) |
| origin_err_m / line | 0.681/0.625 | 0.688/0.630 | FLAT — see below |
| mIoU | 0.083 | 0.098 | WORSE — frozen-CLIP watch item confirmed |
| p_det | 3.0 | 3.5 | mask-driven drop |
| pass_rate_ma | 28.3 | 42.2 | dragged by matching (masks), not axis |
| traj_dir_acc / cos | 84.0/0.533 | 92.0/0.742 | REGRESSED — absolute head |
| point_traj0_gap_m | 0.293 | n/a | traj[0] vs lifted p_hat disagree 29 cm |

- **The lift works where it was aimed:** 3D interaction point -22%,
  2D localization back to the best classical level, type at 98.3%.
- **Origin: mean metrics flat despite the heatmap.** The panels show
  the fix works when grounding works (viz highlight: axis 6 deg,
  origin ON the hinge, 90-deg orbit tracing the GT arc) — the flat
  MEAN is consistent with a heavy tail of relational-grounding misses
  ("second drawer next to the washing machine" grounds the oven).
  Grounding, not geometry, is now the origin bottleneck.
- **Absolute trajectory: both pre-registered risks materialized and
  are now QUANTIFIED.** Point-rendered scribbles (zigzag — the
  e6d8cef disease at ~20x amplitude: each point carries ~0.4 m
  absolute uncertainty over ~2 cm steps), traj_dir 92->84, and the
  29 cm p_hat-vs-traj[0] gap. The geometric decode (yellow 90-deg
  orbit from point+axis) visibly beats the trajectory head's own
  output on the same panels.
- **Masks: worst yet (0.083)** under frozen CLIP even with the
  restored heatmap co-training — the gen-5 frozen-mask correlation
  now has a second data point.

**Decision:** keep the heatmap+depth lift stack (points, origin
channel, composed losses) — it is the new localization baseline. For
gen-8, the evidence points at: (a) trajectory back to relative +
delta-cumsum anchored at the lifted p_hat (or derived geometrically
from point+axis+type — the orbit already outperforms the head), (b)
unfreeze CLIP or otherwise attack masks (two frozen runs, two mask
regressions), (c) grounding is the dominant error source for
everything origin/point-related — a language/grounding-focused
iteration may beat any further geometry work.

vis: viz/20260814_sf3d_g7_e14_panels (random draw seed 30662; 90-deg
orbit overlay added same day)

Eval log: logs/test_best.log (best only; last.ckpt = same bytes, see
ckpt_md5.txt). mean_origin_error_m (1.80) is the stale legacy metric
(unprojects the ELEMENT-point pixel against the GT origin — measures
~the radius by construction with point_source=element); ignore it.
Spec: docs/superpowers/specs/2026-08-14-heatmap-depth-lift-gen7-design.md.
