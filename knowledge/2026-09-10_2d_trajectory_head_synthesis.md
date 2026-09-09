# Smooth 2D-supervised trajectory heads without DCT or first-difference losses — synthesis

2026-09-10. Three parallel literature sweeps (Fable agents) plus the in-house
record. Question from the user: with the plain 20-point head back (the DCT
basis is rejected, the uv-space first-difference losses were a wash), the
predicted trajectories are jittery again — what does the field do?

Reports: `2026-09-10_survey_egocentric_hand_trajectory_forecasting.md`
(OCT, VRB, USST, Diff-IP2D, MADiff, MMTwin, Uni-Hand, EgoH4, EgoMAN, ...),
`2026-09-10_survey_point_track_prediction_manipulation.md` (ATM, Track2Act,
Im2Flow2Act, General Flow, FLIP, GeoPredict, CoTracker3/TAPIR, ...),
`2026-09-10_survey_trajectory_parameterisations_smoothness.md` (Bezier /
B-spline heads, temporal decoders, physical decoders, Laplacian priors,
PCA bases, action chunking). Earlier sweep: `trajectory-parameterization-survey.md`
(2026-08-21; its options 1-2 = fdiff and DCT are the rejected ones).

## What the three sweeps agree on

1. **Nobody emits 20 points from one flat MLP and hopes** — but the fix
   differs by field. Egocentric hand forecasting uses per-step sequence
   decoders or diffusion/flow denoisers and SMOOTHS ITS LABELS (Hermite
   spline in OCT, Savitzky-Golay in VRB). Manipulation point-track work
   uses per-point whole-horizon MLP heads more often than not (ATM, General
   Flow, FLIP, GeoPredict), never measures jitter, and gets coherence from
   delta parameterisations and robust masked losses. Motion forecasting
   uses bases (DCT, Bezier, PCA), temporal mixers (siMLPe, SmoothNet) or
   physical decoders.
2. **Two mechanisms are ranked top-2 by all three reports:** (a) a small
   TEMPORAL decoder — 20 learned time queries with a sinusoidal time
   embedding, 1-2 attention layers (self-attention across the 20 queries,
   cross-attention to the spatial feature map), a shared linear head; or
   the cheaper siMLPe/SmoothNet temporal mixer (a 20x20 FC or depthwise
   1D conv after the MLP) — smoothness LEARNED through shared weights,
   no basis, no expressivity limit, no loss change; and (b) a PHYSICAL
   decoder — type / axis / pivot / sweep decoded to the arc or line by
   Rodrigues (FlowBot++, DKM, ScrewMimic) with a zero-initialised gated
   residual — smooth by construction and exact for our GT family.
3. **The rejected losses are what the egocentric line uses** (MADiff and
   Uni-Hand: angle + length on consecutive segments; the DCT line: velocity
   L2), and no egocentric paper shows them reducing roughness on a
   2D-supervised path. Our 2026-08-22 result matches the literature.
4. **The target side matters on 2D data:** index-matched L2 to a jittery
   hand-keypoint track (WiLoR on HOI4D/EPIC; ARCTIC is clean mocap) rewards
   chasing per-index noise. Every hand-forecasting paper low-passes the
   track first; trackers use Huber with confidence masking; USST's
   heteroscedastic NLL lets the head down-weight bad points.
5. **A GT-free second-difference (Laplacian) penalty in 3D** is different
   in kind from the first-difference losses we tried: those matched
   velocities to the noisy track (copying its jitter), this compares
   nothing to GT, is exactly zero for uniform straight motion and
   O((theta/19)^2 r) for an arc — it penalises only what the physics
   forbids, and never sees the projection. Ubiquitous in optimisation,
   rarely ablated in learned heads.

## In-house facts that bound the choice

- The physical decoder already exists: `analytic_screw_trajectory` (2026-08-25
  mechanism study) decodes 20 points from predicted type/axis/origin/point
  with zero parameters; on the 3D line it recovered 75-89% of the
  trajectory-supervision transfer and its continuous limit (`cf_h1only`)
  holds the articulation records — but it was never run on the 2D path,
  where its gradient routing (projection loss -> axis/origin/magnitude) is
  the standing candidate for the articulation deadlock (STATE "Open threads").
- With the analytic decoder the trajectory stops being independent evidence
  (L_pp becomes vacuous, axis errors become trajectory errors verbatim);
  the 3D study showed a MA cost of ~1-3 vs a joint head, with better
  origin/flips.
- Delta + cumsum was tried in gen-8 and is equivalent to the direct readout
  ("cumsum of a linear map is a linear map"); the new ingredient the
  sweeps add is the unit-direction x scalar-scale split (FLIP / General
  Flow), where magnitude jitter collapses into one scalar.
- The plain post-trained model (`20260909_sf3d_plain_rgb_scalefree_ft_multi3`)
  is 0.1 MA from the record with roughness 0.068 — the jitter is a visual /
  downstream problem, not a headline-metric problem; MA never reads the
  trajectory.

## Ranked recommendation for the 2D-first line (no DCT, no fdiff)

| rank | change | what it is | cost | risk |
|---|---|---|---|---|
| 1 | **temporal decoder head** | 20 learned time queries + sinusoidal t; 1-2 self-attention layers; cross-attention to fq (spatial), pooled condition added; shared Linear -> 3 (relative, first point pinned to 0). Replaces TrajectoryMLP's readout; all losses unchanged | ~0.5 M params, one class + config flag, ~1 day | learned, not guaranteed smooth; over-smooths sharp stops if too small |
| 2 | **GT-free Laplacian penalty** | `L_lap = mean_k ||q_{k-1} - 2 q_k + q_{k+1}||^2` on the 3D (scale-free) curve, small weight, sweep {0.01, 0.1, 1} | 5 lines + a test | over-weighting flattens arcs; watch the 3D-set trajectory val loss |
| 3 | **target-side fixes** | Savitzky-Golay (window 7-9, order 2) on the hand tracks BEFORE the 20-point resample (builder or reader flag); Huber (~6 px) with the valid mask in the projection loss | builder/reader flag + loss option, hours | changes the 2D targets for every arm (rebuild or a reader flag with a version note) |
| 4 | **physical decoder + gated residual** | reuse `analytic_screw_trajectory` with a new sweep-magnitude head; `q = decode(type, axis, origin, theta) + alpha * MLP(c)`, alpha init 0, ||r|| penalised; type-gated mix as in the axis heads | the decoder exists; a magnitude head + wiring + tests, ~1 day | 2D-only axis-depth ambiguity; trajectory no longer independent evidence (3D MA -1..3 in the mechanism study); this is a different MODEL, not a smoother |
| 5 | direction x scale split | unit directions per step + one scalar length (3D batches supervise the scalar), cumulated-position loss | small | error accumulation; scale unobservable on 2D-only |
| 6 | diffusion / flow-matching head | the egocentric field's consensus | large | multi-step inference; projection loss through x0 |
| 0 | **diagnostic first** | least-squares fit of the 20 predicted points to the best arc-or-line at inference; report roughness and 2D shape before/after | an evening | none — tells how much error is jitter vs bias |

**Proposed next experiment (A/B on the multi3 recipe, ~10 epochs at x8 each, ~1 h per arm):**
arm T = temporal decoder (1) + Laplacian (2) + smoothed/Huber targets (3);
arm P = physical decoder with residual (4) + smoothed/Huber targets (3);
control = the current plain head + (3) only. Judge by roughness, 2D shape,
HOI4D/EPIC/ARCTIC held-out mIoU/PDet, then post-train the winner on SF3D
with the plain-compatible loader (the temporal decoder changes the head's
parameter names — the SF3D config must use the same head, as learned on
2026-09-09). Run the diagnostic (0) on the existing plain checkpoint first:
if the arc/line fit removes most of the 2D shape error, the jitter is
cosmetic and (2)+(3) may be enough; if not, the head is the problem and (1)
or (4) is the fix.
