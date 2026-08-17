# 20260817_sf3d_g16_trajnorm — GT-energy-normalized trajectory loss

**Recipe:** gen-13 (dinov3 @512, v3, no taps/cost-map) with the trajectory
loss normalized per-row by GT sweep energy and the weight restored to 0.5
(`trajectory_loss_normalized`, spec 2026-08-17). Purpose: fix the
rot-sweep collapse of every trajectory_weight=0.15 arm.

**Pre-launch validation:** a 300-step fine-tune probe on g13's collapsed
checkpoint grew rot sweeps 0.083→0.192 m (22/26 samples >1.5×) while the
old-recipe control shrank them further 0.083→0.022 m (0/26) —
`tools/diag_traj_collapse_test.py`.

**Run:** best = epoch 21 (val 1.0123 — NOT comparable across arms: new
loss composition). Clean exit.

## Success gates (spec) — ALL PASS

1. Rot sweep restored: val93 **0.436 m** (g13: 0.067), val1684 **0.791 m**
   (g13: 0.041) — both ≫ the 0.2 m gate, at gen-11's healthy scale.
2. traj_dir: **94.46% / cos 0.798** (gate ≥90; g13: 86.2/0.633) —
   best trajectory direction in the project (cos best-ever; acc within
   0.3 of gen-8's 94.8 but with full-scale sweeps and the 512 stack).
3. g13 records hold: mIoU 0.2636 (0.2655), PDet 21.44 (21.38), 2D point
   0.0988 (best-ever, was 0.0998), axis 28.3/17.8° (27.9/16.7 — +1.1
   matched), origin 0.3033 (0.2939, +0.9 cm), radius 0.139 (0.133),
   type 92.3 (93.0), MA 23.1 (26.2 — the one notable dip, −3.1 via the
   slightly softer matched axis).

## Test (best-epoch21, 5,088 samples)

mIoU 0.2636 · PDet 21.44 · type 92.32 · axis 28.25/17.83° · MA 23.05 ·
2D pt 0.0988 · 3D pt 0.2488 · origin 0.3033/0.2743 · radius 0.1391 ·
traj_dir 94.46/0.798 · L_pp_norm probe mean 0.1554 (g13: 0.104 — higher
because full-scale sweeps EXPOSE axis/origin error the collapsed sweeps
hid; the old number was flattered by the collapse).

## Sign-aware axis re-test (2026-08-18, same ckpt/samples)

The legacy axis metric takes |cos| and cannot see flipped opening
directions; re-ran the test pass after adding signed columns (commit
c53974a). Legacy numbers reproduced exactly. New: **signed axis error
33.91°** (vs 28.25 unsigned) · **flip rate 10.18% all / 12.73%
rotational** (signed > 90° — predicted axis in the wrong hemisphere, i.e.
door swings the wrong way) · MA_signed 22.92 (vs 23.09 — flips almost
never co-occur with a <10° unsigned match, so MA barely moves; the flips
live in the already-unmatched tail). Baseline for the gen-17 split-head
arm.

## Reading

The normalization did exactly what it was built for: full-scale revolute
trajectories AND project-best direction, at ~zero cost to the g13 mask/
point records (2D point even improved). The −3 MA / +1° matched-axis dip
is the real price — plausibly the trajectory gradients competing again,
but at O(1) scale rather than gen-11's runaway. GEN-16 IS THE NEW OVERALL
BEST CHECKPOINT (best-epoch21): g13-class masks + real trajectories.

Candidates recorded: cost-map-without-taps on the g16 base (geometry
gains from g15 may stack); TALENT-style contrastive for grounding.

vis: viz/20260817_sf3d_g16_vs_g13_panels
