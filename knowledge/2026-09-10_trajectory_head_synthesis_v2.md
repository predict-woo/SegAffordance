# Trajectory-head structure alternatives — synthesis v2 (2026-09-10)

Four clean-slate literature sweeps (Fable 5.1 subagents, web-only, no repo
knowledge), one per field, each with its own comparison table and ranked list:

| sweep | file | papers |
|---|---|---|
| egocentric hand / HOI forecasting | `2026-09-10_survey_v2_egocentric_hand_forecasting.md` | 25 (FHOI, OCT, USST, Diff-IP2D, MADiff, MMTwin, HandsOnVLM, EgoMAN, SFHand, ForeHand4D, LatentAct, VidBot, ...) |
| point-track / flow affordances for manipulation | `2026-09-10_survey_v2_point_track_affordance.md` | 35 (ATM, Track2Act, Im2Flow2Act, General Flow, VidBot, FlowBotHD, What-Happens-Next, TraceGen, BEAST, FAST, siMLPe, TAPIR, CoTracker3, Track-On, ...) |
| trajectory parameterisations, motion-forecasting heads | `2026-09-10_survey_v2_trajectory_parameterisations.md` | ~40 (LTD, siMLPe, HumanMAC, V2-Net, MotionDiffuser, SIMPL, MultiPath++, MTR, QCNet, TNT, MotionLM, DKM, BezierLaneNet, aWTA, FlowBot++, Yao'24 OOD study, ...) |
| articulated-object / affordance / VLM traces | `2026-09-10_survey_v2_articulation_affordance_trajectories.md` | 45 (Where2Act, VAT-Mart, FlowBot3D/++/HD, GAMMA, RPMArt, OPD/OPDMulti, 3DOI, SceneFun3D, USDNet, AFUN, A0, HAMSTER/3D-HAMSTER, MolmoAct, Magma, Embodied-R1, ...) |

Our head under review: pooled 512-d fused vector -> 2-layer MLP -> either
20x3 direct ("plain", roughness ~0.07) or 6 DCT-II coefficients/axis through a
fixed IDCT (roughness ~0.008). Losses: SF3D normalised 3D trajectory loss +
axis consistency; on HOI4D/EPIC/ARCTIC a unit-anchor 2D projection loss.
Failure modes: plain jitter; out of domain (iPhone photos, HOI4D laptops) the
sweep collapses to a squiggle while mask / type / axis / z_p stay plausible.

## 1. What the four sweeps agree on

1. **Nobody reads a whole trajectory out of one pooled vector with a bare
   MLP.** Every field's regressors give each timestep or point its own token
   attending to *spatial* features (ATM, 3PoinTr, SFHand, ManiTrend), or
   decode from a low-dimensional structured parameterisation. The pooled
   MLP is the implicit baseline everyone beats; two same-backbone ablations
   put numbers on it: MADiff's 0-block denoiser is "substantially worse"
   than 4-6 sequence blocks; OCT MLP 0.21 -> CVAE 0.12 ADE (min-of-20).
2. **The basis is worth little on its own; the readout convention and the
   loss carry the gains.** siMLPe: removing DCT costs 0.3-1 mm, removing the
   residual-to-anchor readout costs 3-4 mm, the velocity loss helps long
   horizons. BEAST: pinned first control point + N=10 B-spline for 20 steps;
   General Flow: shape/scale decomposition (+0.2-0.3 cm). Our DCT-6 is
   mainstream (LTD, siMLPe, HumanMAC, V2-Net, PatchTraj, FAST) but is
   missing the pinned start, an explicit scale scalar, and a velocity term.
3. **The articulated-object field does not regress free-form trajectories
   at all.** FlowBot++, GAMMA, RPMArt, UniAff/A3VLM and the real-world door
   system all predict (type, axis, origin) and *render* the arc or line
   analytically. The single cleanest OOD number in all four sweeps is
   FlowBot++ on unseen categories, same backbone: step-wise flow 0.73,
   direct screw-parameter regression 0.26, analytic rollout 0.18
   normalised distance. GAMMA beats VAT-Mart's cVAE trajectories 2-5x on
   unseen categories. DKM / MultiPath++ "Control": a physical integrator
   costs nothing in ADE and removes every infeasible output.
4. **Low-DoF constrained outputs degrade least out of distribution.**
   Yao et al. 2024 (train Argoverse 2, test Waymo): a 345k-parameter
   degree-5 Bernstein-polynomial model has the smallest ID->OOD increase of
   every model tested (+24.7 % vs +61.5 % minADE6 for Forecast-MAE) and beats
   QCNet OOD while losing in-domain. Monomial coefficients are uniformly the
   worst parameterisation (BezierLaneNet F1 1.49 vs 68.9; SIMPL 1.738 vs
   1.457); Bezier control points match raw points in-domain.
5. **Velocity / consistency losses are the cheapest OOD lever.** USST:
   velocity head + cumsum consistency does almost nothing in-domain
   (0.189 -> 0.183) but cuts unseen-scene 3D ADE 0.168 -> 0.120 (-29 %).
   MADiff / MMTwin / Uni-Hand all add angle (cosine between successive
   displacements) and length losses on top of displacement L2.
6. **Generative heads win where the data are multimodal, and the gain is
   largest OOD.** ForeHand4D (single image, same transformer): regressor
   15.3/30.0/29.2 vs diffusion 14.8/27.9/24.0 in-domain / held-out /
   zero-shot. EgoMAN: 0.273 -> 0.162 (flow matching) -> 0.151 (+ explicit
   start/contact/end waypoints), -27 % on OOD HOT3D. FlowBotHD doors
   23 -> 77 %. What-Happens-Next 23.07 -> 16.70 mean (10.99 best-of-8) vs
   the ATM regressor. Affordance-FM: flow matching 1.01 cm vs transformer
   BC 4.91 cm, and 1-step FM is 1.03 cm, so the compute objection is weak.
   Caveat: several of these are min-of-K numbers; TNT shows that *given the
   target* completion is unimodal and a 2-layer MLP suffices.
7. **Iterative delta refinement beats one-shot regression in tracking**
   (TAPIR AJ 41.6 / 55.0 / 61.3 for 0 / 1 / 4 iterations; MTR +1.7 mAP from
   refinement; Track-On's classify-then-offset delta-1px 45.5 vs 27.6).
   Autoregressive decoders are not recommended anywhere (Martinez'17,
   NAT, DKM UM-LSTM, USST's own limitation list).
8. **Two warnings for our current recipe.** (a) ForeHand4D found auxiliary
   2D heads neutral-to-harmful (26.8/25.9, 31.9/24.8 vs 27.9/24.0) while
   lifting 2D labels to pseudo-3D helped a lot (20.3/18.8): our 2D
   projection loss leaves depth unconstrained (3D HAMSTER: 2D traces
   "inherit whatever depth lies beneath them"). (b) AFUN (2026, evaluated
   on SceneFun3D, n=721) found an ungated OPD-style axis/type
   parameterisation *loses* to an anchored Bezier curve (0.282 vs 0.254
   ADE) because a single parameterisation across motion types is
   ambiguous: an analytic decoder needs a soft type gate and a residual.

## 2. Why our OOD squiggle happens, per the literature

Two mechanisms, both fixable without touching the backbone:

- **The head has no access to the geometry the other heads got right.** The
  trajectory MLP re-derives the motion from a pooled feature that no longer
  carries it out of domain; the axis/origin heads see the same feature but
  their outputs are low-dimensional and constrained. FlowBot3D-vs-FlowBot++
  (0.73 vs 0.18) is exactly this gap closed by decoding the trajectory from
  the axis. Our consistency loss only ties them *in training*.
- **Mean collapse under L2 on ambiguous hand data.** Push/pull, left/right
  hinge, partial vs full opening: a unimodal regressor averages the modes
  (FlowBotHD, What-Happens-Next, MotionForesight's 0.72 magnitude ratio).
  Smoothing (DCT) hides the jitter but not the averaged shape.

## 3. Ranked shortlist (merged across the four sweeps)

All four sweeps independently ranked the same design first.

### #1 Analytic screw decoder as the OUTPUT head (predicted extent + monotone profile + small gated residual)

Head predicts: extent (theta or l, softplus), a monotone timing profile
s_k = cumsum(softmax(19 logits)) (the one place a cumsum is safe: bounded,
monotone), optionally 3 DCT residual coefficients per axis under an L2/L1
penalty and a late-start schedule. Decoder: revolute
x_k = R(s_k theta; n)(p0 - o) + o - p0, prismatic x_k = s_k l d; type mixed
softly by p_rev during training, argmax at test; scale-free output as now.
Loss: existing SF3D trajectory loss and existing 2D projection loss on the
decoded 20x3, plus a direct loss on extent derived from the GT curve
(least-squares theta(t) given the GT axis).

Why: the trajectory *inherits* the axis head's generalisation (the failure
"axis right, sweep wrong" becomes unrepresentable); roughness -> 0; the
axis-consistency loss becomes an identity; the 2D projection loss on hand
videos now back-propagates into axis / origin / extent, i.e. a new axis
supervision path from 2D video (FlowBot++ trains both from one L2); depth
along the arc is dictated by the geometry, which removes the 2D-loss blind
spot flagged by ForeHand4D / 3D HAMSTER.
Evidence: FlowBot++ 0.73 -> 0.18; GAMMA vs VAT-Mart; DKM 0 % infeasible at
equal ADE; "Opening Articulated Objects in the Real World" modular > e2e;
AFUN's warning -> keep the soft gate + residual.
Cost: ~50-100 lines, two scalar heads, no new data. Risk: origin error
propagates into the radius (mitigate with USDNet's origin-to-axis distance
loss); the 2D sources have no axis GT, so on them the residual must carry
the hand-slip deviations — keep it bounded.

**Repo note:** `analytic_screw_trajectory()` in `model/losses/geometric.py`
(2026-08-25 mechanism study) already implements this decoder differentiably
with FIXED extent (pi/2, 0.7 m), linear profile, GT-type routing, used only
as a loss on the predicted parameters with the head off
(`analytic_trajectory_weight`; MA 26.45 vs head 28.2, and the closed-form
continuous version reached MA 30.64 with no head at all). What is new here
is (a) predicted extent + profile, (b) the soft type gate, (c) the residual,
(d) using the decode as the *inference output* and under the 2D projection
loss. The plumbing (routing, tests against the GT writer) exists.

### #2 DCT done properly (pinned start, scale scalar, velocity + angle/length losses) — free, orthogonal, do alongside #1

Force x_0 = 0 (BEAST's c0 pin), predict log-scale + unit shape (General
Flow, +0.2-0.3 cm and the mechanism that lets one head cover 5 cm and 70 cm
motions), add an L2 velocity term (siMLPe 111.3 -> 109.4 mm) and MADiff's
cosine-angle + length terms on successive displacements, optionally USST's
cumsum-consistency. Zero inference cost. Evidence: siMLPe ablation table,
BEAST N=5/10/20 = 3.88/4.43/2.71 (6 coefficients with a pinned start is in
the right regime), USST unseen 0.168 -> 0.120. This is also the residual
basis for #1.

### #3 Anchored Bezier control-point head as the type-agnostic fallback / ablation partner

5-7 control points, P_0 pinned at the interaction point, trajectory = fixed
Bernstein basis matrix times control points (same cost as IDCT), L1 on
16-20 sampled points plus a tangent-cosine loss (SIMPL heading error
0.134 -> 0.055) tied to the predicted axis (perpendicular for revolute,
parallel for prismatic). Evidence: AFUN on SceneFun3D (Bezier 0.254 vs
OPD-param 0.282), Yao'24 OOD study, SIMPL (Bezier 1.457 = raw 1.452 >>
monomial 1.738), BezierLaneNet. Use for the non-articulated elements
(buttons, plugs) where an axis is meaningless, or as #1's residual.

### #4 Query-token decoder over the feature map with delta refinement

20 timestep tokens (Fourier features of t) + scale token, 2-3 layers d=256,
cross-attention to the *unpooled* fused map around the contact point,
self-attention along time, M=3 delta updates with 0.8^(M-m)-weighted L1
(CoTracker3 / TAPIR / PIPs recipe); tokens can emit DCT or Bezier
coefficients instead of positions. Evidence: ATM / Tra-MoE / 3PoinTr / SFHand
(4x USST), MADiff 0-block ablation, TAPIR 41.6 -> 61.3, MTR refinement
+1.7 mAP, Gen2Act motion-type generalisation 5 -> 30 % from track
supervision. Cost: 2-5 M params, <1 ms. Addresses the "pooled vector" half
of the diagnosis; #1 addresses the "no geometry" half — they compose.

### #5 Small flow-matching head with best-of-K + axis-consistency selection — only if the 2D sources are demonstrably multimodal

4-layer DiT or 1-D U-Net over the 20x3 (or the 18 DCT / low-dim #1
parameters), rectified-flow objective, K=8 samples x 1-4 Euler steps,
selected by the differentiable axis-consistency + smoothness cost (VidBot
85.6 vs 74.5 % without cost-based selection; Affordance-FM 1-step 1.03 cm;
ForeHand4D zero-shot 29.2 -> 24.0; EgoMAN; FlowBotHD). Diagnostic first:
measure the best-of-K oracle error of the current head vs its mean error on
HOI4D/EPIC/ARCTIC val — a large gap is the signal that a generative head
will pay; a small gap says the ambiguity is only in the extent, and a K=3
annealed-WTA / Laplace head over the extent inside #1 is enough (aWTA:
MR 0.30 -> 0.19 with 6 queries).

Not recommended: autoregressive GRU/transformer decoders; cVAE readouts
(OCT's gain is min-of-20 and the cVAE collapsed when USST moved it to 3D);
monomial polynomials; tokenised VLM traces (gains come from 3-13 B backbones
and 10^5-10^6 traces, and 2D traces distort depth); dense per-point heads.

## 4. Proposed experiment order

1. **#2 on the current DCT head** (pinned start, scale scalar, velocity +
   angle/length losses) — a one-day sanity run on the joint4 recipe;
   re-measure roughness, SF3D MA, and the iPhone / laptop probes.
2. **#1 as the output head**, extent + profile + DCT-3 residual, soft type
   gate; SF3D post-training from the multi3 DCT arm AND the joint recipe.
   New metrics: residual norm relative to the analytic part ("how much the
   model trusts the physics"), ID-vs-OOD delta per Yao'24, extent error.
3. **#3 Bezier** as the like-for-like ablation partner of #2 (same DoF).
4. **The best-of-K oracle diagnostic** on the 2D val splits to decide
   between #5 and a K=3 extent head.
5. **#4 query decoder** on top of the winner if the pooled feature is still
   the bottleneck (check: does the trajectory improve when the head sees
   the unpooled map?).
6. In parallel, re-examine the 2D projection loss per ForeHand4D: lifting
   HOI4D/EPIC/ARCTIC tracks to pseudo-3D with a monocular depth model
   before they reach the head is the tested alternative to a 2D auxiliary.

## 5. Relation to the earlier (2026-09-10 morning) sweep

The first sweep (`2026-09-10_2d_trajectory_head_synthesis.md`) was run to
decide plain vs DCT and concluded DCT is the human-motion standard. This
second, broader sweep (145 papers vs ~60, four fields incl. articulated
objects and driving) keeps that conclusion but moves the emphasis: the basis
is the smallest lever; the decisive designs are the analytic decoder from our
own axis heads, the readout conventions (pinned start, scale, velocity
losses), and spatial query decoding. The out-of-domain squiggle now has a
literature-backed diagnosis and a near-free fix.
