# Articulation head: annealed winner-takes-all hypotheses (mode-commitment fix)

> 2026-08-11 revision: hypotheses extended from twists to JOINT
> (twist, trajectory) bundles after measuring that the trajectory head
> converged exactly to the zero-motion baseline (val L_trajectory 0.0182
> vs 0.0181 for predicting "nothing moves") — the same sign-averaging
> hedge, which no weight amplification can fix (the MSE minimizer is
> weight-invariant). Each hypothesis is one coherent story about how the
> object moves; bundling keeps twist and trajectory structurally
> consistent instead of leaving screw_self to fight an inconsistent
> selection.

**Status: IMPLEMENTED 2026-08-11 (commit 7c6b677; smoke-validated on the dev pod).** Companion to
`2026-08-11-twist-body-metric-design.md`; the two are one change set and
should be implemented together (the body metric is the distortion measure
this loss quantizes under).

## Problem

The remaining half of the large-radius diagnosis: a single twist head
trained with any fixed quadratic metric predicts the posterior-*mean* twist
under ambiguity (the minimizer of a quadratic is the affine mean regardless
of the metric). Measured on clip_g3: revolute `‖ω‖` median 0.295, radius
inflated 10×, direction accuracy 71%, emergent type 60% — all consequences
of averaging over discrete ambiguities (sweep sign, motion type, hinge
side) instead of committing to one.

Modern-practice research (2026-08-11 session, four literature sweeps):
motion forecasting converged on multi-hypothesis winner-takes-all heads;
the state of the art trains them with an **annealed softmin** (aMCL,
NeurIPS 2024, arXiv:2407.15580; deployed for forecasting as aWTA, ICRA
2025, arXiv:2409.11172), which fixes classic WTA hypothesis collapse and
dead heads. Theory: WTA hypotheses converge to a centroidal Voronoi
quantization of the conditional distribution — they sit ON modes, never
between them (Rupprecht arXiv:1612.00197; Voronoi-WTA, ICML 2024,
arXiv:2406.04706).

## Design

**K = 4 joint articulation hypotheses + K logits.** Hypothesis k is a
bundle `(xi_k, tau_k)`: a twist AND its matching trajectory. `TwistMLP`'s
final linear widens to `K·6 + K` (pitch-free projection per hypothesis,
unchanged); `TrajectoryMLP`'s final linear widens to `K·(N−1)·3`
(delta-cumsum per hypothesis, unchanged). The K logits live on the twist
head and select the whole bundle.

Training loss (SF3D, sign-sensitive path), with `L_body` the body-metric
twist loss from the companion spec and `L_traj` the per-point trajectory
MSE (same m² units, same sum-over-xyz norm convention):

```
d_k    = w_tw * L_body(xi_k, xi_gt) + w_traj * L_traj(tau_k, tau_gt)
k*     = argmin_k d_k                              (winner)
q_T(k) = softmax_k(-d_k / T)                       (annealed weights)

L_wta    = sum_k stopgrad(q_T(k)) * d_k            (regression term)
L_hyp_ce = cross_entropy(logits, k*)               (selection term)
```

with `w_tw = twist_weight = 0.5` and `w_traj = trajectory_weight = 4.0`
(raised from 0.5: at 0.5 the trajectory term carried ~1% of the total
loss its entire life — init scale 0.018 m² vs 0.6 for the twist — and
was doubly capped by the hedge; amplification is only meaningful now
that WTA makes the term reducible. 4.0 gives it an ~8x larger gradient
scale — roughly 4% of the early-training total, vs ~1% before).

- **Temperature schedule**: exponential decay per epoch, `T(e) = T0 * rho^e`
  with `T0 = 10`, `rho` chosen so `T ≈ 0.01` at ~80% of `max_epochs`
  (aWTA's recipe); hard WTA (pure winner) for the remainder. High T =
  every hypothesis trains (no dead heads); low T = full specialization.
- **Inference/eval**: `(xi*, tau*) = bundles[argmax logits]` —
  deterministic, one twist + one trajectory that tell the same story;
  decode, viz, and all metrics unchanged downstream.
- **Consistency losses** — the gating rule is: GT-anchored terms follow
  the annealed winner weights; GT-free terms apply to every hypothesis.
  - `screw_gt` (twist field vs GT trajectory, GT-anchored): per-bundle
    residuals weighted by the SAME `stopgrad(q_T(k))` as the regression
    distortion — all bundles early, winner-only as T→0. A hard
    winner-only gate would be inconsistent with the soft regression
    weighting; an ungated version would drag every expert toward every
    sample's GT mode and erode specialization.
  - `screw_self` (twist vs its OWN bundled trajectory, GT-free): mean
    over ALL K bundles, every sample, unweighted. It never pulls toward
    GT, so it cannot fight specialization — and every bundle must be
    internally coherent because argmax-logit can select any of them at
    eval. (Winner-only would leave losing bundles free to drift into
    twist/trajectory disagreement off their home modes.) All bundles
    share the single predicted-point anchor. Compute is negligible
    (analytic residual, ×K).
  - At val/test, metrics read the argmax-logit bundle as elsewhere.
- **OPD / sign-agnostic path**: distortion becomes
  `min(L_body(xi_k, xi_gt), L_body(xi_k, -xi_gt))` inside `d_k`; the rest
  is identical. (OPD batches lack a 3D origin so the loss is a no-op
  there today; the formulation stays well-defined.)
- **K = 4 rationale**: per-sample posteriors typically carry ONE live
  binary ambiguity (2 modes), but which one varies (type, sweep sign,
  hinge side), and worst-case samples carry two at once. K is a coverage
  budget, not an ontology: surplus hypotheses converge to low-logit
  near-duplicates that argmax ignores, and cost K−1 rows of one linear
  layer. Log per-hypothesis win/argmax rates; persistent dead heads are
  evidence K can shrink.

What this deliberately does NOT touch:

- No type head, no branches: each hypothesis is a full unified twist;
  type stays emergent from `‖ω‖` of the selected hypothesis.
- The 2D trajectory head and the 2D-only arm: untouched. Whatever the
  joint-WTA run teaches transfers there later.
- The delta-cumsum parameterization, anchoring, and all non-articulation
  heads (mask, point, coords): untouched.

## Config

- `ModelParams.twist_num_hypotheses: int = 1` (1 = legacy single heads;
  SF3D twist arms set 4; gates BOTH the twist and trajectory widening).
- `LossParams.trajectory_weight: 4.0` in the SF3D twist configs (raised
  from 0.5, see above; also the `w_traj` inside the WTA distortion).
- `LossParams.twist_wta_T0: float = 10.0`,
  `LossParams.twist_wta_anneal_frac: float = 0.8` (fraction of max_epochs
  to reach T ≈ 0.01).
- `LossParams.twist_hyp_ce_weight: float = 0.1` (weight of `L_hyp_ce`
  inside the twist term).
- Smoke-run acceptance check: log per-term gradient norms into the shared
  features (`‖∇_feat(w·L_term)‖`, one batch every few hundred steps) to
  verify the geometric terms are not drowned by the high-floor mask/
  point_map gradients late in training; rebalance only on that evidence.

## Outputs / wiring sketch

- `ModelOutputs`: `twist_pred (B, 6)` and `trajectory_pred (B, N, 3)`
  remain the argmax-logit selection (keeps every consumer working); add
  `twist_hyps (B, K, 6)`, `trajectory_hyps (B, K, N, 3)` and
  `twist_logits (B, K)` (None when K = 1).
- The WTA loss module computes `d_k` (twist + trajectory parts), the
  winner index, and both terms; exposes the winner index so the training
  step hands the winning bundle to the consistency losses. The legacy
  standalone `L_trajectory` term is subsumed by the WTA distortion (do
  not double-count it).
- Metrics: log `train/twist_T`, per-hypothesis win rates, selected `‖ω‖`
  mean by GT type, and selected-trajectory MSE vs the zero-motion
  baseline 0.018 (the direct readout of whether commitment works).

## Tests

- K = 1 reduces exactly to the current losses (regression term equals
  `w_tw·L_body + w_traj·L_traj`, CE term zero, outputs shaped as today).
- Bundle integrity: the winner index selects the SAME k for twist and
  trajectory; a sample whose GT trajectory matches bundle A but twist
  matches bundle B still trains one bundle (the joint-distortion winner).
- Consistency gating: `screw_gt` per-bundle residuals carry the same
  `q_T` weights as the regression term (verify at high and low T);
  `screw_self` averages over all K bundles regardless of winner, and
  each bundle's residual pairs twist k with trajectory k (never a
  cross-bundle pair).
- T → large: weights ≈ uniform (mean-regression start); T → 0: weight
  1 on the winner only.
- Stop-grad: no gradient through the weights themselves.
- Synthetic commitment test (port of the session's E3 study): tiny MLP on
  the rot/pris ambiguous mixture — with K = 4 the selected hypothesis
  recovers `‖ω‖ ≈ 1` and radius ≈ GT on revolute inputs where the K = 1
  head collapses to the inflated mean; CE-selected hypothesis matches the
  majority mode.
- Winner-index plumbing: consistency losses receive the winner twist.

## Expected outcome on SF3D (acceptance signals)

Selected-hypothesis `‖ω‖` on GT-revolute val samples ≈ 1 (was 0.295);
radius ratio median ≈ 1 (was 10×); emergent type and direction accuracy
up. Rendered twist sweeps curve at part scale instead of the flat
large-radius arcs. Selected-trajectory MSE clearly below the 0.018
zero-motion baseline the gen-3 head converged to — the trajectory head
finally predicting motion instead of the hedge.
