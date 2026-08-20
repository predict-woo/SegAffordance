# Gen-19: Smooth Trajectory Heads (DCT basis vs first-difference losses)

**Date:** 2026-08-21
**Status:** APPROVED (user) — implement + run both arms overnight
**Motivation:** predicted trajectories are noisy/non-smooth; survey
(knowledge/trajectory-parameterization-survey.md) shows the field's two
low-risk answers. Both arms build on the 3D gen-17 base (the 2D line is
wrapped up for now). Naming: gen-18 is skipped — old commit messages use
"g18" for what was renamed gen-17-2D.

## Arm 1 — g19_dct: truncated-DCT trajectory head

`ModelParams.trajectory_dct_coeffs: int = 0` (0 = off, legacy direct
readout byte-identical).

- `TrajectoryMLP`: when K>0, the final linear emits `num_hypotheses*K*3`;
  decode `traj = idct[:, :K] @ coeffs` with a fixed orthonormal DCT-II
  matrix buffer (Mao et al. convention, N=20). Output shape unchanged
  (B, K_hyp, 20, 3) — every consumer (losses, L_pp, viz, metrics) is
  untouched. Jitter above frequency K is unrepresentable by construction.
- Guards: ValueError with `trajectory_delta_cumsum` (two competing
  smoothing parameterizations).
- K = **6** (survey: 5–8 for 20 points; HumanMAC/siMLPe compression
  ratios). Loss config identical to g17 (normalized trajectory loss on
  DECODED points — the field's "loss on points, never coefficients" rule
  comes for free since the head returns points).
- Config `sf3d_train_runpod_g19_dct.yaml` = g17 + the K knob + paths
  `experiments/20260821_sf3d_g19_dct`.

## Arm 2 — g19_fdiff: first-difference losses on the unchanged head

Three LossParams knobs (all default 0.0 = off), computed on consecutive
segment vectors Δp/Δg (`diff` along time; translation-invariant, so the
relative-vs-absolute frame question is moot):

- `trajectory_velocity_weight: 1.0` — siMLPe convention:
  `mean(‖Δpred − Δgt‖₂)` (norm, not squared, per their code).
- `trajectory_angle_weight: 0.5` — MADiff: `mean(1 − cos(Δpred, Δgt))`
  over segments with ‖Δgt‖ > 1 mm (degenerate segments skipped; if none
  survive, graph-connected zero).
- `trajectory_length_weight: 0.5` — MADiff: `mean(|‖Δpred‖ − ‖Δgt‖|)`.

Logged as `L_traj_velocity` / `L_traj_angle` / `L_traj_length`. Applied in
the trajectory-loss block (skipped when the WTA twist loss handled the
trajectory). Config `sf3d_train_runpod_g19_fdiff.yaml` = g17 + the three
weights + paths `experiments/20260821_sf3d_g19_fdiff`.

## New smoothness metric (both arms + baseline)

Test pass gains `test/traj_rough_pred` and `test/traj_rough_gt`: mean
second-difference magnitude `mean‖p[i+1] − 2p[i] + p[i−1]‖` over the 18
interior points — the quantitative version of "the trajectories look
noisy". The g17 baseline test is RERUN to get its numbers for comparison.

## Success criteria

1. `traj_rough_pred` drops toward `traj_rough_gt` (GT arcs/rays are
   near-perfectly smooth; the gap is the noise we're removing).
2. Trajectory quality holds or improves: traj_dir ≥ 94.9/cos ≥ 0.811,
   normalized trajectory loss not worse.
3. Nothing else regresses: mIoU ~0.265, PDet ~20.6, type ~95.3, matched
   axis ~16.9°, origin ~0.257 (the heads are untouched; only the
   trajectory pathway changes).
4. Standard wrap: test pass each + g17 rerun, viz panels (the visual
   smoothness check), notes/INDEX.

## Tests

DCT: decode matrix orthonormality (idct@dct=I at K=20); flag-off
byte-identity; K=6 output is in the span (re-encoding decoded output with
6 coeffs reproduces it); smoothness by construction (second-diff of
decoded random coeffs ≪ direct random readout); ValueError with
delta_cumsum; shape with num_hypotheses>1. Fdiff: velocity loss zero iff
equal diffs; angle loss invariant to per-segment speed scaling; length
loss invariant to direction; degenerate-segment skip; weights-off
byte-identity of total loss. Config chains ×2 (= g17 + knobs + paths).
Training-step integration per arm.
