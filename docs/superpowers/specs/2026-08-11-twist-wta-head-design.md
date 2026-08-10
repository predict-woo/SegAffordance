# Twist head: annealed winner-takes-all hypotheses (mode-commitment fix)

**Status: approved design — NOT yet implemented.** Companion to
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

`TwistMLP` outputs **K = 4 hypotheses + K logits** (widen the final linear
to `K·6 + K`; the pitch-free projection applies per hypothesis, unchanged).

Training loss (SF3D, sign-sensitive path), with `L_body` the body-metric
twist loss from the companion spec:

```
d_k    = L_body(xi_k, xi_gt)                       k = 1..K
k*     = argmin_k d_k                              (winner)
q_T(k) = softmax_k(-d_k / T)                       (annealed weights)

L_twist_wta = sum_k stopgrad(q_T(k)) * d_k         (regression term)
L_hyp_ce    = cross_entropy(logits, k*)            (selection term)
```

- **Temperature schedule**: exponential decay per epoch, `T(e) = T0 * rho^e`
  with `T0 = 10`, `rho` chosen so `T ≈ 0.01` at ~80% of `max_epochs`
  (aWTA's recipe); hard WTA (pure winner) for the remainder. High T =
  every hypothesis trains (no dead heads); low T = full specialization.
- **Inference/eval**: `xi* = hypotheses[argmax logits]` — deterministic,
  single twist; decode, viz, and all metrics unchanged downstream.
- **Consistency losses** (`screw_gt`, `screw_self`): applied to the
  SELECTED twist — winner `k*` during training (GT-informed, consistent
  with the twist supervision), argmax-logit at val/test.
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
- The trajectory head keeps its single-mode delta-cumsum MSE. It hedges
  too (straightened arcs), but it is anchored and dense-supervised, and
  the screw-self term ties it to the committed twist. If needed later:
  shared mode queries producing (twist, trajectory) pairs, winner by
  joint loss — the forecasting pattern. Out of scope now.

## Config

- `ModelParams.twist_num_hypotheses: int = 1` (1 = legacy single head;
  SF3D twist arms set 4).
- `LossParams.twist_wta_T0: float = 10.0`,
  `LossParams.twist_wta_anneal_frac: float = 0.8` (fraction of max_epochs
  to reach T ≈ 0.01).
- `LossParams.twist_hyp_ce_weight: float = 0.1` (weight of `L_hyp_ce`
  inside the twist term).

## Outputs / wiring sketch

- `ModelOutputs`: `twist_pred` remains `(B, 6)` = argmax-logit selection
  (keeps every consumer working); add `twist_hyps (B, K, 6)` and
  `twist_logits (B, K)` (None when K = 1).
- `TwistLoss` computes `d_k`, the winner index, and both terms; exposes
  the winner index so the training step can hand the winning twist to the
  consistency losses.
- Metrics: log `train/twist_T`, per-hypothesis win rates, and selected
  `‖ω‖` mean by GT type (the direct readout of whether commitment works).

## Tests

- K = 1 reduces exactly to the current loss (regression term equals
  `L_body`, CE term zero).
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
large-radius arcs.
