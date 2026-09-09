# Survey: trajectory output parameterisations and smoothness mechanisms (2019-2026)

Context: our head is a plain MLP, cond (~800-d) -> 20x3 relative points, trained by 3D L2 (3D sets) and
index-matched 2D projection L2 against a resampled hand track (2D sets). Output is zigzag (roughness ~7x a
smooth basis). Already tried and rejected: truncated DCT readout; first-difference (velocity/angle/length)
losses. Physics: revolute -> circular arc, prismatic -> line, i.e. the GT is a 4-7 parameter curve.

## 1. Driving / pedestrian forecasting

- **Bezier control points.** Hug et al., Bezier Curve GPs (arXiv:2205.01754): `X(t) = sum_l b_{l,L}(t) P_l`,
  `b_{l,L}(t) = C(L,l)(1-t)^{L-l} t^l`, Gaussian control points give `mu(t) = sum_l b_l(t) mu_l`,
  `Sigma(t) = sum_l b_l(t)^2 Sigma_l`; MDN NLL over all timesteps; L+1 points for any horizon, smooth by
  construction. DeepRacing (arXiv:2005.05178): Bezier head, loss = 1.0 L2 on curve sampled at GT times +
  0.1 derivative-vs-velocity + 0.05 L2 to least-squares control points; vs waypoint regression boundary
  failures 5.6 -> 1.8, "smoother velocity curve". BezierLaneNet (CVPR 2022, arXiv:2203.02431): cubic (4
  control points), sampling loss `L = 1/n sum_t ||B(t) - B_gt(t)||_1` (n=100), never on control points;
  raw polynomial coefficients were near-unlearnable (F1 1.49) while control points, which live in output
  space, trained fine.
- **Polynomial / kinematic-control bases.** MultiPath++ (arXiv:2111.14973, Table 4): raw coords minADE 0.978 /
  integrated controls 0.987 / polynomial 1.041 - polynomial "hurt performance, counter to PLOP". Controls:
  `v(t)=v0+int a`, `theta(t)=theta0+int theta_dot`, midpoint integration, curvature clipping; buys feasibility,
  not accuracy. Deep Kinematic Models (arXiv:1908.00219): predict (a, steer) per step, integrate a bicycle
  model; position error unchanged (4.21 vs 4.25 m @6 s) but heading error 7.69 -> 4.92 deg, infeasible 26% -> 0%,
  turning-rate Wasserstein distance halved ("smooth" vs "very noisy"). Trajectron++ (arXiv:2001.03093) does the
  same with a unicycle. Integration smooths only because controls are low-dimensional/clipped and coupled
  through heading; cumsum of free per-step deltas (what we tried) is a bijection and smooths nothing.
- **Anchors / heatmaps / Laplace.** MTR (NeurIPS 2022), MultiPath++, mmTransformer: K intention queries,
  per-timestep GMM NLL; HOME/GOHOME (arXiv:2109.01827): endpoint heatmap then a regressed trajectory;
  HiVT/QCNet: winner-take-all Laplace NLL per timestep. All per-point outputs; none address jitter.
- **PCA basis.** MotionDiffuser (CVPR 2023, arXiv:2306.03083): `s_hat = (s - s_bar) W^T`, 160-d (80x2) -> 10
  components (3 explain 99.7% variance, recon 0.06 m); minSADE 0.88 with PCA vs 1.03 without.

## 2. Human motion prediction

- siMLPe (WACV 2023, arXiv:2207.01567): DCT in / IDCT out, temporal FC blocks `W_i in R^{TxT}` (frame-mixing
  linear layers), residual w.r.t. last pose (9.6 vs 12.4 mm), auxiliary `L_v = ||v'_{t+1:t+N} - v_{t+1:t+N}||_2`,
  `v_t = x_{t+1} - x_t` (helps long horizon only).
- SmoothNet (ECCV 2022, arXiv:2112.13715): 0.03M-param temporal-only FC refiner, loss `L_pose + L_acc`;
  acceleration error 19.17 -> 1.03 mm/frame^2 on H3.6M: a temporal-mixing layer over per-frame outputs removes
  jitter almost entirely. "NeuralSpline"/"MoSpline" do not exist; keyframe in-betweening (CGF 2024,
  arXiv:2503.13859) is a spline head in disguise.

## 3. Robotics action chunking

- ACT (arXiv:2304.13705): chunk k=100, temporal ensembling `w_i = exp(-m i)` over overlapping chunks (+3-4%
  success, "smooth motion") - inference-time averaging.
- BEAST (arXiv:2506.06072): clamped cubic B-spline, N=10 control points per 20-step chunk, fitted by ridge
  regression `c = (Phi^T Phi + lambda I)^{-1} Phi^T a`, first point clamped to the last action; recon MSE 0.0004
  vs 0.0009 binning; 4-8x fewer tokens than FAST (DCT). B-spline Policy (arXiv:2607.09648): cubic,
  `a(u) = sum_i N_{i,3}(u) c_i`, 16 knots + control points predicted, "substantially smoother executions" at 4x
  speed. ABPolicy (arXiv:2602.23901): 8 cubic control points per 40 steps, flow matching on control points;
  velocity zero-crossings -29%, p95 acceleration -57%. Spline Policy (arXiv:2606.07386): piecewise quadratic
  Bernstein `f_i(tau) = (1-tau)^2 w1 + 2(1-tau)tau w2 + tau^2 w3`, 8 params vs 16-step chunk, MSE at sampled
  times, C1 by construction. LiPo (arXiv:2506.05165): post-hoc `int ||d^3/dt^3 (q_ref + eps)||^2` with bounded
  eps, ball-toss 75 -> 90%. SmoothVLA (arXiv:2603.13925): RL reward `1[success](1 - 0.2 mean|jerk|)`, jerk -13.8%.

## 4. Articulation parameters -> trajectory

- ScrewNet (arXiv:2008.10518): screw (l, m, theta, d) regressed directly. ANCSH (CVPR 2020): per-point NPCS +
  voted axis/pivot. Ditto (arXiv:2202.08227): per-point revolute (u, projection dir d, distance h, angle c) /
  prismatic (u, c), losses `arccos(u . u_hat)`, `|c - c_hat|`, plus a displacement loss through the rotation
  matrix. OPD (arXiv:2203.16421): RGB -> type, axis, origin (smooth-L1, weights [1,8,8]), no range. None
  decode a time series.
- FlowBot++ (arXiv:2306.12893) does: per-point flow f_p and *articulation projection* r_p (vector to the axis),
  axis/origin recovered by cross products, rollout `p' = R(phi)(p - v_hat) + v_hat` for phi in [0, phi_g] via
  Rodrigues; plain L2 on (f, r); "lower angular acceleration variation", no back-and-forth jitter, 17.1 -> 1.2 s
  per object vs per-step FlowBot3D. General Flow (arXiv:2401.11439: 3 steps, length-normalised displacement +
  scale, CVAE) and ATM (arXiv:2401.00025: 16 absolute 2D steps per point token, MSE) are per-step with no
  smoothness term.

| Method | Params vs points | Loss | Smooth by | Effect |
|---|---|---|---|---|
| Bezier GP / DeepRacing / BezierLaneNet | L+1 ctrl pts (4-6) for any T | sampled-point L1/L2 (+MDN) | construction | smoother velocity, fewer failures; poly coeffs unlearnable |
| MultiPath++ polynomial | deg+1 per axis | GMM NLL | construction | minADE 6% worse |
| DKM / MultiPath++ control | 2 clipped controls/step | position NLL | integration + clipping | heading err -36%, jitter halved, ADE same |
| MotionDiffuser PCA | 10 of 160 | L2 (diffusion) | construction (data basis) | minSADE 1.03 -> 0.88 |
| siMLPe / SmoothNet | TxT temporal FC | L2 + velocity / accel | learned mixing | accel error -80..95% |
| ACT temporal ensembling | none | - | inference averaging | +3-4% success |
| BEAST / BSP / ABPolicy / Spline Policy | 8-10 ctrl pts per 20-32 steps | L2 on fitted ctrl pts or sampled curve | construction | p95 accel -57%, zero-cross -29% |
| LiPo / SmoothVLA | - | jerk penalty (post-hoc / RL) | regularisation | jerk -14%, ball toss 75->90% |
| FlowBot++ | axis+origin (6) -> K-step arc | L2 on flow+projection | analytic Rodrigues | consistent multi-step motion |
| Laplace-NLL / MTR / HOME | per point | NLL / heatmap | none | n/a for jitter |

## What applies to our head (ranked)

1. **Physical decoder + gated residual (FlowBot++ / DKM pattern).** Head outputs type logits, unit axis u,
   pivot offset c, sweep theta (or distance d); decode `q_k = R(k theta/19; u)(p0 - c) + c - p0` (revolute) or
   `q_k = (k d/19) u` (prismatic), mix by type probability, add `r_k = alpha * MLP` with alpha init 0 and an L2
   penalty on r. Jitter vanishes because 20 points share 4-7 parameters and the arc/line is exact for the true
   motion; Rodrigues is differentiable and FlowBot++/DKM train through such decoders with position-only
   supervision, so the 2D projection loss stays unchanged. Risks: axis/plane depth ambiguity under 2D-only
   supervision (keep 3D sets in every batch, add Ditto's `arccos` axis loss where GT axes exist); the residual
   reintroducing jitter if alpha grows (penalise ||r|| or ||Delta^2 r||).
2. **Small temporal decoder instead of the per-point MLP.** 20 learned time queries (+ sinusoidal t), 1-2
   self-attention layers with the pooled condition added or cross-attended, linear -> 3; or cheaper, keep the MLP
   and append a siMLPe-style temporal FC `W in R^{20x20}` or a depthwise 1D conv (kernel 5). Removes jitter
   because every output depends on all others (SmoothNet: -95% accel error with 0.03M params); not by
   construction, so it needs data, but it restricts expressivity not at all. 1k-50k params.
3. **Bezier control points with sampling loss** (BezierLaneNet, DeepRacing). Cubic or quartic, P0 = 0 fixed,
   so 3-4 free points x 3 = 9-12 outputs, evaluated at t_k = k/19 and fed to the existing 20-point losses.
   Same dimensionality class as DCT-6 but different geometry: control points are in output units
   (initialisable, clippable, geometrically regularisable - BezierLaneNet found coefficient bases unlearnable),
   the last control point *is* the endpoint (direct FDE gradient), the start tangent is 3(P1 - P0), and the
   convex hull bounds the curve. A cubic matches a circular arc to <0.03% radius up to 90 deg; use quartic or two
   quadratic B-spline segments (Spline Policy) for larger sweeps. Risk: still a global basis - if the objection
   to DCT was "smoothness by fiat", this shares it.
4. **Second-difference (Laplacian) penalty, GT-free, in 3D:** `L_lap = mean_k ||q_{k-1} - 2 q_k + q_{k+1}||^2`.
   Differs from the first-difference losses tried in two ways: (a) those *matched* velocity/angle/length to the
   resampled hand track, so target jitter was copied into the prediction; this term compares nothing to GT, is
   exactly zero for uniform straight motion and O((theta/19)^2 r) for an arc, i.e. it only penalises what the
   physics forbids; (b) it acts in 3D camera space, not through the projection, so 2D track noise never enters
   it. It is the discrete thin-plate prior, and SmoothNet's "Accel" jitter metric is exactly its magnitude.
   Risk: over-weighting flattens arcs; sweep a small weight and watch 3D ADE on the 3D sets.
5. **Low-rank basis learned from GT** (MotionDiffuser PCA): fit PCA on the 60-d GT curves from the 3D sets
   (expect 4-6 components >99% variance), predict coefficients, decode linearly. Data-adaptive (contains the arc
   shapes) unlike DCT, but the same fixed-global-basis idea, and a 3D-set basis may not cover 2D-set motions.
6. **Inference-only projection** (LiPo/ACT flavour): least-squares fit the 20 predicted points to the best
   arc-or-line (our decoder inverted) and output the fit. Zero training cost; removes jitter for evaluation and
   tells us how much error is jitter vs bias before investing in 1-3, but does not fix the training signal.

Honest gaps: no paper compares Bezier vs DCT heads under a 2D-projection loss; the evidence for (1) is
FlowBot++/DKM (consistency and feasibility, accuracy roughly neutral), for (2) SmoothNet/siMLPe (pose, not
forecasting), for (3) 2025-26 robotics chunking papers on single-embodiment demos. GT-free second-difference
penalties are ubiquitous in optimisation but rarely ablated in learned forecasting heads.
