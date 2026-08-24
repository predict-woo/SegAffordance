# Why does trajectory supervision help articulation? — mechanism study

**Date:** 2026-08-25. **Commissioned:** user ("figure out exactly why…
design and launch new experiments as you learn new things without asking…
when you fully understand, create a toy example").

## The puzzle

SF3D trajectories are DERIVED from articulation ∧ element geometry by the
dataset writer — zero new label bits given the other supervision. Yet
supabl2 arm D vs arm B (identical, ± trajectory head/loss, no consistency
terms anywhere) measures trajectory supervision at **MA +7.8, matched
axis −6.2°, rot flips −7.0**.

## Hypotheses

- **H1 — loss geometry / reparameterization.** The same axis enters the
  trajectory MSE sign-linearly and without saddles (a flipped axis makes
  an oppositely-swept arc: large, well-conditioned point-space error),
  while 1−cos sees sign through a vanishing-gradient antipode. Dense (60
  numbers vs a few scalars) and conjunction-coupled (curve placement
  requires axis ∧ origin ∧ element point jointly).
- **H2 — auxiliary feature shaping.** The trajectory head's dense
  regression enriches the shared trunk; articulation heads then read
  better features. Evidence: arm C (traj-only) has the best masks; 2D
  arms improved trunk but articulation stayed dead without its own anchor.

Not mutually exclusive; the question is the split.

## Experiment 1 (the discriminator): analytic screw decode

`20260825_sf3d_analytic_decode` — arm B's exact config (NO trajectory
head) + `analytic_trajectory_weight: 0.5`: decode the trajectory
differentiably from predicted (type-routed axis, origin, 3D point) with
an exact mirror of the GT writer (well-posedness locked against the
AST-extracted writer in tests/test_analytic_decode.py), supervise with
the gen-16 normalized trajectory loss. Zero new parameters vs arm B.

Readout on the fixed anchors, arm B (MA 20.4 / matched 23.3° / flips
21.8) and arm D (MA 28.2 / 17.1° / ~14.8):
- lands ≈ D → **H1**: the gain is the loss's geometry, and the decode is
  a strictly cleaner way to get it (gradients hit the articulation heads
  DIRECTLY, not via shared features).
- lands ≈ B → **H2**: the head's feature shaping is the carrier.
- in between → both; the position gives the split.

Secondary readouts: flip rate (H1's sign-linearity predicts flips drop
hardest); radius/origin (conjunction sub-hypothesis); masks/PDet (H2
predicts the decode does NOT reproduce arm C/D's mask gains, since no
dense signal reaches the trunk through a 9-DoF bottleneck).

## Follow-ups (branch on E1; launched autonomously)

- If H1: sweep nothing; instead test H1's sign claim in isolation — the
  toy example (below) + per-type metric decomposition of E1 suffice. The
  practical product: analytic decode as a permanent loss (maybe combined
  with a real head: decode teaches heads, head teaches features).
- If H2: variant with a trajectory head whose INPUT features are
  detached (head can learn, trunk gets no trajectory gradient) — if that
  also fails to help articulation, H2 is confirmed as trunk-mediated.
- Either way, if E1 ≈ D: candidate gen-22 = analytic decode + fdiff-style
  losses on the decode + no L_pp (folds the ablation's findings in).

## Toy example (after the mechanism is pinned)

Minimal synthetic: an MLP predicts an angle/axis from noisy features;
loss A = 1−cos to GT; loss B = point-MSE on an arc decoded from the
prediction. Same information, same labels. Show: (a) flipped inits —
loss A stalls at the antipodal saddle, loss B escapes; (b) convergence
speed and final sign-error histograms. Pure CPU, `tools/toy_*` + a
slide in docs/slides/.

## Also running (independent, user-commissioned earlier)

fdiff family grid: g19_fdiff (exists) / fdiff_nolpp / fdiff_dir — L_pp
and dir isolation off the DCT head. Wrapped as they land.

## Cost

E1 ≈ $8 (one 30-epoch run + test). Follow-ups ≤ 2 more runs. Pods:
reuse freed fdiff pods or poll (PRO 6000 only).
