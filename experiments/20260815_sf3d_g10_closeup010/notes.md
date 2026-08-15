# 20260815_sf3d_g10_closeup010 — gen-10: normalized, full-strength L_pp

**Recipe:** gen-9 (`20260815_sf3d_g9_closeup010`) with ONE change: the
pred-pred consistency loss runs its normalized (dimensionless) branches at
weight 0.1 — line = off-axis energy fraction (≤1), circle = orbit residual
/ max(r̂, 0.10 m)². Calibrated so the weighted contribution sits at the top
of the trajectory/axis/origin band (was ~0.2% of the total in gen-9). Same
split (59,174), 30 epochs, milestones [24,28], seed 42, frozen CLIP. Spec:
`docs/superpowers/specs/2026-08-15-lpp-normalized-gen10-design.md`.

**Run:** RTX PRO 4500, ~1h05, best = epoch 24 (val 0.9471 — NOT comparable
to gen-9's total: different L_pp scale). Stable from epoch 0 (the O(1)-at-
init watch item never bit). One aborted launch (wrong default trainer —
fixed in train_pod.sh). Cost ≈ $1.3.

## Consistency (the target metric) — SUCCESS

`tools/diag_lpp_samples.py`, 512-sample val reference, both checkpoints:

| normalized L_pp | gen-9 | gen-10 |
|---|---|---|
| mean | 0.1635 | **0.0539** (−67%) |
| p50 | 0.0369 | 0.0161 |
| p90 | 0.3232 | 0.1572 |
| p99 | 1.4793 | 0.4649 |
| max | 8.4791 | 0.7094 |

The two motivating samples: val93 radial RMS 0.124→0.050 m (trajectory now
hugs its own orbit; 3D plot in `viz/20260816_sf3d_g10_lpp_val93_3d`),
val1684 radial 0.057→0.025 m. val1684's axial drift worsened (0.111→0.171 m
— the sweep got longer and still climbs the axis); it remains a tail sample
(~p90) rather than p97.

## Test (best-epoch24, same 5,088 samples; Δ vs gen-9)

| metric | gen-10 | gen-9 | Δ |
|---|---|---|---|
| mIoU / PDet | 0.1435 / 5.40 | 0.1463 / 5.39 | ≈flat |
| type acc | 93.08% | 94.97% | −1.9 |
| axis° all / matched | 27.62 / 17.75 | 27.56 / 15.81 | flat / +1.9 |
| MA pass | 23.35 | 21.91 | **+1.4** |
| 2D pt / 3D pt (m) | 0.1545 / 0.2939 | 0.1525 / 0.2920 | ≈flat |
| origin vs q* / line (m) | 0.3692 / 0.3279 | 0.3614 / 0.3220 | +0.008 |
| radius err (m) | 0.1839 | 0.1914 | **−0.008** |
| traj_dir acc / cos | 91.53 / 0.714 | 93.10 / 0.736 | −1.6 |

## Reading

- **Consistency: 3× better on the mean, 12× on the max** — the loss now
  does its job; predicted trajectories and articulations describe the same
  motion (visible in `viz/20260816_sf3d_g10_vs_g9_panels`: val1684's
  trajectory follows its orbit + the GT arc instead of curling away).
- **No denominator gaming:** the final review's predicted fingerprint (a
  positive radius-error bias from inflating r̂) did NOT appear — radius err
  *improved* (−0.008 m), MA pass rate improved (+1.4).
- **Cost of the coupling:** small dips in type (−1.9), matched-axis
  (+1.9°), traj_dir (−1.6). The val93 3D plot shows one mechanism:
  consistency partly achieved by *shortening* the predicted sweep (net
  extent 0.41→0.19 m there) — an under-sweep pressure worth watching if
  L_pp is ever raised further.
- Net: accepted trade at weight 0.1 — consistency was the stated goal and
  the direct-metric cost is ≤2 points everywhere, with MA pass (the
  combined type+axis metric) actually up.

vis: viz/20260816_sf3d_g10_vs_g9_panels, viz/20260816_sf3d_g10_lpp_val93_3d
