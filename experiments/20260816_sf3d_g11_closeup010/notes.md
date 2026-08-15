# 20260816_sf3d_g11_closeup010 — gen-11: origin local sample, on v3

**Recipe:** gen-10 (`20260815_sf3d_g10_closeup010`) with two bundled deltas
(user decision — attribution between them is not separable from this run):

1. `use_origin_local_feature: true` — ẑ_q consumes `grid_sample(fq,
   origin_uv)` (the hinge-seam pixel), mirroring ẑ_p; reverses gen-7's
   condition-only choice.
2. Dataset `sf3d_processed_v3` — prismatic sweeps 0.7 m (the revolute
   median) instead of 0.1 m. Same records/filters/caches as v2.

Same split (59,174), 30 epochs, milestones [24, 28], seed 42, frozen CLIP.
Spec: `docs/superpowers/specs/2026-08-16-origin-local-sample-gen11-design.md`.

**Run:** RTX PRO 6000 Server ($2.09/hr — 4500s were skipped by stock this
time; ~1:05/epoch, ~35 min train, cost ≈ $1.6). Best = epoch 16 (val
0.9978 — NOT comparable to earlier arms: v3's 7× longer trans sweeps raise
the trajectory term). Val rose after ep16 (1.008 @ 22, 1.020 @ 28) — the
early best is new; earlier arms peaked at ep 21–26.

## Test (best-epoch16, same 5,088 samples; Δ vs gen-10)

Trans-row trajectory MAGNITUDE metrics are a fresh baseline (0.7 m GT);
direction metrics comparable.

| metric | gen-11 | gen-10 | Δ |
|---|---|---|---|
| origin vs q* (m) | 0.3620 | 0.3692 | **−0.007** |
| origin line (m) | 0.3240 | 0.3279 | −0.004 |
| radius err (m) | 0.1790 | 0.1839 | **−0.005** (family best) |
| traj_dir acc / cos | 91.29 / 0.708 | 91.53 / 0.714 | ≈flat |
| type acc | 92.90 | 93.08 | ≈flat |
| axis° all / matched | 28.22 / 18.34 | 27.62 / 17.75 | +0.6 / +0.6 |
| MA pass | 22.56 | 23.35 | −0.8 |
| 2D point | 0.1610 | 0.1545 | +0.007 |
| 3D point (m) | 0.3113 | 0.2939 | +0.017 |
| mIoU / PDet | 0.1320 / 4.89 | 0.1435 / 5.40 | **−0.011 / −0.5** |

## Consistency probe (ref-512, v3 GT)

Normalized L_pp mean 0.0592 (gen-10: 0.0539 — within band, success
criterion 2 met); p50 0.0163. Tail slightly heavier (p99 0.63 vs 0.46).
val93: radial RMS 0.038 (g10: 0.050), L_pp_norm 0.049 — best yet.
val1684: radial 0.065, axial 0.125, L_pp_norm 0.088 — still tail-ish.

## Reading

- **The origin heads got their win, but a small one:** origin −0.7 cm,
  line −0.4 cm, radius −0.5 cm (best in family). The hinge-pixel sample
  helps; it does not move the grounding-dominated 0.36 m mean much.
- **v3's 0.7 m trans sweeps did no harm to direction** (traj_dir flat) —
  the head absorbed the 7× magnitude change.
- **Cost shows up in masks/points:** mIoU −0.011, PDet −0.5, 3D point
  +1.7 cm, axis +0.6°. With TWO variables bundled, the likely driver is
  the loss-balance shift (the trajectory term is ~7× larger on 78% of
  rows, taxing the shared decoder), not the origin sample — but this run
  cannot separate them.
- Early best epoch (16) + rising val after: the enlarged trajectory term
  changes val-loss composition; worth revisiting trajectory_weight or the
  schedule if v3 stays.

## Candidates recorded (user's call, not scheduled)

- Rebalance `trajectory_weight` for v3's scale (e.g. 0.5 → 0.1) and/or
  extend the schedule; the ep-16 peak suggests the current recipe
  under-trains the other heads on v3.
- Origin-sample-only arm on v2 if attribution matters.

vis: viz/20260816_sf3d_g11_vs_g10_panels
