# Survey v2: trajectory parameterisations and regression heads (human-motion + vehicle/pedestrian forecasting)

Date: 2026-09-10. Clean-slate literature survey (web/arXiv only; no repo knowledge used).
Target: what to replace/augment our `pooled 512-d -> MLP -> (20x3 | 6 DCT coeffs/axis)` trajectory readout with.

Our setting for reference: one RGB image + text -> mask, interaction point, joint type, axis direction, hinge origin, and a 20-point 3D trajectory of the interaction point (camera frame, relative to the first point, scale-free). Failure modes: the plain 20x3 head is jittery (roughness ~0.07), the 6-coefficient DCT head is smooth (~0.008) but is only a basis truncation, and out-of-domain the trajectory collapses to squiggles while mask/type/axis stay plausible.

---

## 0. Executive summary

1. **Low-dimensional smooth bases are the norm, not the exception.** Every strong human-motion predictor since 2019 (LTD, HisRep, siMLPe, HumanMAC, PGBIG, FreqMRN) predicts DCT coefficients of the *padded* full sequence and reads out with a fixed IDCT; pedestrian forecasting has the same story with DFT (V2-Net) and DCT (PatchTraj); MotionDiffuser diffuses in a 10-component PCA space where 3 components already explain 99.7 % of Waymo trajectory variance. Our DCT head is therefore well aligned with the field. What the literature adds is (i) *residual-to-anchor* readouts, (ii) *velocity/derivative auxiliary losses*, (iii) *iterative refinement*, and (iv) *physically-analytic decoders*.
2. **Direct per-point regression is rarely the best but is not catastrophically worse when data are plentiful and in-distribution.** siMLPe's own ablation: removing DCT costs only ~0.3-1 mm MPJPE; removing the residual-to-last-frame readout costs ~3-4 mm. MultiPath++: raw coordinates beat polynomial outputs on Waymo (minFDE 2.305 vs 2.537) but have a "non-trivial rate of kinematic infeasibility".
3. **The clearest OOD evidence favours constrained, low-DoF parametric outputs.** Yao et al. 2024 (arXiv 2407.13431) train on Argoverse 2 and test on Waymo: a 345 k-parameter model with degree-5 Bernstein-polynomial outputs has the smallest ID->OOD degradation of all models tested and beats QCNet/Forecast-MAE in OOD minADE/minFDE despite losing to them in-distribution.
4. **Monomial polynomial coefficients are hard to regress; Bezier control points are not.** BezierLaneNet: F1 on CULane 68.9 (cubic Bezier) vs 1.49 (cubic polynomial coefficients, same network). SIMPL: minFDE6 1.457 (Bezier) vs 1.738 (monomial) vs 1.452 (raw points).
5. **For our physically-constrained motion (revolute or prismatic) the strongest analogue is FlowBot++ (CoRL 2023):** predict the axis (direction + origin) and *analytically* roll out the arc/line with Rodrigues; it beats per-step flow prediction on unseen categories (normalised distance 0.18 vs 0.73) and eliminates the "back-and-forth" jitter of step-wise prediction. This is the single most relevant design for our OOD-collapse problem.
6. **Mean collapse is a real, documented failure of unimodal L2 heads** (Martinez 2017: zero-velocity beats all RNNs; aWTA 2024: with soft assignment "hypotheses converge toward a conditional mean"). For our task the multimodality is small (given the instruction, the joint type and the axis, the motion is nearly deterministic up to opening extent), so a K-hypothesis/WTA head is second-order; a *magnitude* (opening angle / distance) uncertainty is the relevant residual ambiguity.

---

## 1. (a) Basis / frequency parameterisations

### DCT in human-motion prediction
- **LTD — Mao, Liu, Salzmann, Li, "Learning Trajectory Dependencies for Human Motion Prediction", ICCV 2019, arXiv:1908.05436.** Replicates the last observed pose T times, takes the DCT of the whole (N+T)-frame sequence, and a GCN predicts the *residual* DCT coefficients; IDCT gives the future poses. Loss: L1 on angles or MPJPE on 3D joints, summed over observed+future frames. Number of coefficients is truncated and ablated in the supplementary: 15 (3D, short-term) / 30 (3D, long-term) and 20/35 for angles out of 35-60 frames; truncating high frequencies is explicitly motivated by avoiding "jittery motion".
- **HisRep — Mao, Liu, Salzmann, "History Repeats Itself", ECCV 2020, arXiv:2007.11755.** Same DCT+residual recipe, with motion attention over history sub-sequences.
- **siMLPe — Guo et al., "Back to MLP: A Simple Baseline for Human Motion Prediction", WACV 2023, arXiv:2207.01567.** A 0.14 M-parameter all-linear MLP (FC + LayerNorm + transpose) operating on DCT coefficients of the 50-frame padded sequence, predicting residual displacement w.r.t. the last observed frame *after* IDCT, plus a velocity auxiliary loss. Human3.6M ablations (MPJPE mm at 80/…/1000 ms):

  | variant | 80 | 400 | 1000 |
  |---|---|---|---|
  | full siMLPe | 9.6 | 57.3 | 109.4 |
  | w/o DCT | 9.9 | 58.4 | 110.5 |
  | w/o residual (absolute poses) | 12.4 | 61.6 | 113.0 |
  | residual before IDCT (in DCT space, as LTD) | 10.4 | 59.1 | 110.5 |
  | consecutive-frame velocity residual | 9.7 | 57.8 | 110.1 |
  | w/o velocity loss Lv | 9.6 | 57.5 | 111.3 |

  Take-aways: the *residual-to-anchor* readout matters far more than the basis; DCT helps "slightly"; velocity loss helps long horizons only.
- **HumanMAC — Chen et al., ICCV 2023, arXiv:2302.03665.** Diffusion in DCT space; keeps L = 20 (H3.6M) / 10 (HumanEva) coefficients for 125-frame sequences; observed frames are injected at inference via "DCT-completion".
- **FreqMRN — "Towards Accurate Human Motion Prediction via Iterative Refinement", arXiv:2305.04443.** 3 refinement stages that convert back and forth between pose space and DCT space; 3 stages x 2 blocks beats 1 stage x 6 blocks at equal compute (avg MPJPE 62.1 vs 62.9).
- **Time-continuous (NeRF-style) encodings — NeRMo, Wei et al., ECCV 2024.** Motion as an implicit function of (Fourier-encoded) continuous time and a joint code. Ablation on CMU-MoCap: no Fourier features 14.8/58.6/109.8 mm (80/400/1000 ms) -> with Fourier features 8.1/38.4/83.8 -> plus codebook 8.05/37.2/80.5. The authors note that a large number of Fourier bands "may lead to degraded smoothness … similar to the case that all DCT coefficients are used".

### Fourier / DCT in pedestrian forecasting
- **V2-Net — Wong et al., "View Vertically", ECCV 2022, arXiv:2110.07288.** Trajectories are represented by DFT spectra; a coarse stage predicts N_key = 3 "keypoint" spectra, a fine stage interpolates the full spectrum; IDFT readout. Spectrum-based prediction beats a time-domain variant by ~10 % ADE/FDE (ETH-UCY/SDD), more with many samples. Follow-up (arXiv:2304.05106) generalises to arbitrary transforms (Haar wavelets etc.).
- **PatchTraj, arXiv:2507.19119.** Dual time/DCT branches, keeping l = 8-10 DCT coefficients; frequency branch alone gives ~6 % ADE gain on ETH-UCY (0.32 -> 0.30), most of the gain comes from the fusion.
- **PCA basis — MotionDiffuser, Jiang et al., CVPR 2023, arXiv:2306.03083.** Trajectories (80x2) are projected on a *learned* PCA basis: 3 components = 99.7 % variance, 10 components used (0.06 m reconstruction error, far below prediction error). Stated benefits: faster inference, better constrained sampling, "better accuracy". Their argument that high-order components are "perception noise" is the same argument as DCT truncation, but with a data-adapted basis.

### Practical numbers on K
- Human motion: 10-20 of 50-125 frames' worth of coefficients (siMLPe keeps all 50 but the LN-MLP effectively low-passes); LTD 15-35 of 35-60.
- Pedestrians: 3 keypoint spectra (V2-Net), 8-10 DCT (PatchTraj).
- Vehicles: 10 PCA components for 80 points (MotionDiffuser); polynomial degree 3-6 by AIC (below).
- Our 6 DCT coefficients for 20 points is at the low end but consistent; NeRMo and LTD both caution that *too many* coefficients reintroduce jitter.

## 2. (b) Polynomial / spline / keyframe parameterisations

- **How many DoF does a real trajectory need? Reichardt, "An Empirical Bayes Analysis of Object Trajectory Representation Models", arXiv:2211.01696.** Fits polynomial bases of degree 0-7 to Argoverse 1/2 and Waymo trajectories, selects degree by AIC/BIC: degree 3-6 for 5 s horizons, 6-7 for 8 s; a degree-6 fit has 3.7 cm longitudinal / 1.6 cm lateral error on 8 s Waymo vehicles, "negligible" relative to prediction error. Corollary: a degree-5 polynomial is fully determined by position/velocity/acceleration at start and end, i.e. prediction reduces to forecasting endpoint kinematics.
- **Polynomial outputs for OOD robustness — Yao et al., "Improving Out-of-Distribution Generalization of Trajectory Prediction … via Polynomial Representations", arXiv:2407.13431 (2024; follow-up dissertation arXiv:2608.03330).** Model "EP": Bernstein (Bezier) polynomials for agent history, map and *output* (degree 5 for agents, 3 for lanes); ~345 k parameters (4.5 % of QCNet). ID on Argoverse 2 it is slightly worse than QCNet/Forecast-MAE (minFDE1 3.4 % higher than QCNet). OOD (train A2, test homogenised Waymo): EP has the smallest absolute and relative error increase in every setting and *beats* both baselines in OOD minADE/minFDE; e.g. EP-F's OOD minADE6 increase is +0.228 m (24.7 %) vs Forecast-MAE +0.279 m (61.5 %); QCNet's minFDE1 increase is +1.07-1.80 m (46-72 %). Their explicit thesis: "limit the expressiveness of our model by constraining its input and output representation". This is the best available evidence that low-DoF parametric heads reduce OOD collapse.
- **Bezier vs monomial vs points — SIMPL, Zhang et al., RA-L 2024, arXiv:2402.02519.** MLP head -> (n+1) control points of a septic (degree-7) Bezier curve per mode; trajectory = fixed basis matrix B (Tx(n+1)) times control points, so it is a one-shot linear readout exactly like an IDCT. Argoverse 2 val ablation: raw coords minADE6/minFDE6 0.780/1.452; monomial polynomial 0.861/1.738; Bezier 0.780/1.457; Bezier + yaw loss 0.783/1.452 with yaw error 0.055 vs 0.134. Figure 8 shows monomial 1st-order coefficients have a far wider dynamic range than Bezier control points, explaining the optimisation gap. Note that a Bezier readout gives the velocity profile analytically (hodograph), useful for derivative losses.
- **MultiPath++ (Varadarajan et al., ICRA 2022, arXiv:2111.14973), sec. 3.6 / 5.6.2.** Compared raw per-step states, polynomial-in-time coefficients, and integrated kinematic controls (acceleration + heading rate). Waymo minFDE: raw 2.305, controls 2.319, polynomial 2.537; they conclude polynomials "hurt performance" at a 10 s horizon on a large dataset (while PLOP, arXiv:2003.08744, found gains at 4 s on small data), and that raw outputs have "a non-trivial rate of kinematic infeasibility" (TRI metrics) invisible to displacement metrics. Interpretation: parametric heads pay a small ID accuracy tax that shrinks with horizon and data scarcity — our regime (20 points, small datasets) is the favourable one.
- **Deep Kinematic Models — Cui et al., ICRA 2020, arXiv:1908.00219.** Also ablated monomial polynomials of degree 1-3 as outputs: poly-1 (constant velocity) 5.79 m @6 s, poly-2 4.39 m but 20.8 % kinematically infeasible, poly-3 4.66 m / 15.5 % infeasible, vs unconstrained 4.25 m / 26 % and the kinematic decoder 4.21 m / 0 %.
- **Lane-detection analogue.** LSTR (Liu et al., WACV 2021, arXiv:2011.04233) regresses cubic-polynomial-on-the-road parameters composed with camera geometry (an "analytic decoder" of 4 shape + 2 camera parameters) with Hungarian fitting loss; PolyLaneNet regresses coefficients with an FC layer. BezierLaneNet (Feng et al., CVPR 2022, arXiv:2203.02431) replaces this with 4 cubic-Bezier control points and a *sampling* L1 loss on curve points (not on parameters): TuSimple LPD fit error cubic Bezier 0.471 vs cubic polynomial 0.558; CULane F1 68.9 vs 1.49 for the same net with polynomial coefficients; converges 4-5x faster than LSTR. Lesson: supervise in *point space through the fixed decoder*, and parameterise with quantities that have a spatial meaning and comparable scale.
- **Keyframe / target-then-complete.** TNT (Zhao et al., CoRL 2020, arXiv:2008.08294): classify a discretised endpoint + regress an offset, then a 2-layer MLP regresses the full trajectory conditioned on the target with Huber loss; "given the target the distribution is unimodal" (cVAE for the completion stage gives no gain: minADE 0.73 vs 0.73). Endpoint offset regression matters (minFDE 0.53 vs 0.69 without). DenseTNT (arXiv:2108.09640), HOME/GOHOME (Gilles et al., arXiv:2105.10968 / 2109.01827) predict an endpoint *heatmap*, sample endpoints, then regress the path. V2-Net's 3 spectral keypoints and MotionLM's tokens are keyframe designs in other spaces.
- **Discrete tokens — MotionLM, Seff et al., ICCV 2023, arXiv:2309.16534.** Uniformly quantised (dx, dy) deltas at 2 Hz, "Verlet-wrapped" (a zero token repeats the previous delta), 13x13 = 169 tokens; autoregressive transformer with teacher forcing and cross-entropy. Authors suspect discretisation "hides some precision from the model, possibly mitigating compounding error"; no anchors or latent variables needed for multimodality.

## 3. (c) Sequential vs one-shot decoders, readout choices, refinement

- **Autoregressive RNNs and the mean-collapse / discontinuity problem — Martinez, Black, Romero, CVPR 2017, arXiv:1705.02445.** A zero-velocity baseline (repeat last frame) beats ERD/LSTM-3LR/SRNN at every horizon (avg angle error 0.42/0.74/1.12/1.20 vs e.g. SRNN 0.81/0.94/1.16/1.30 on Walking). Fixes: sampling-based loss (decoder fed its own outputs — the standard cure for exposure bias) and a *residual (velocity) readout*: "motion continuity … is easier to express in terms of velocities than in poses"; Residual sup. (MA) 0.36/0.67/1.02/1.15 average.
- **Non-autoregressive decoders.** NAT (Li et al., arXiv:2007.06426) argues error accumulation and exposure bias are intrinsic to AR decoding and predicts every future frame independently from a context encoder + positional encoding; matches/exceeds AR baselines on H3.6M/CMU. Every SOTA method since LTD is one-shot (the whole DCT vector at once). In driving, MultiPath++/Wayformer/MTR/SIMPL are all one-shot per mode; QCNet (Zhou et al., CVPR 2023) is the notable hybrid: an anchor-free *recurrent* proposal decoder (chunks of waypoints) followed by an anchor-based one-shot refinement that predicts offsets to the proposal.
- **Delta / cumulative-sum readouts.** Cui et al. "UM-velo" (predict per-step velocities, cumsum) is not better than positions (4.28 vs 4.25 m @6 s, 27 % infeasible); siMLPe "consecutive" residual is slightly worse than residual-to-last-frame; MotionLM tokens are deltas but wrapped so that repeated tokens encode *constant velocity*, which is the key to making cumsum readouts drift-free. Practical rule from these: predict displacement *relative to a fixed anchor* (last observed / first point), not chained deltas, unless there is a physical integrator behind it.
- **Anchor + residual, one-shot.** MultiPath (Chai et al., CoRL 2019, arXiv:1910.05449) regresses offsets to k-means trajectory anchors; MultiPath++ replaces static anchors with learned latent anchors (minFDE 2.305 vs 2.99 static). MTR (Shi et al., NeurIPS 2022, arXiv:2209.13508): 64 static *intention points* (k-means endpoints) as queries, 6 stacked decoder layers each with a GMM head; dynamic searching query re-centred on the previous layer's predicted endpoint (deformable-DETR-style). Ablation (Waymo val): latent learnable queries mAP 0.263 / MR 0.213; intention queries 0.306 / 0.185; + iterative refinement 0.317 / 0.178; + local movement refinement 0.323 / 0.176. Refinement-only modules (SmartRefine arXiv:2403.11492, R-Pred arXiv:2211.08609) add 1-2 % on top of any backbone by re-encoding features *along the predicted trajectory* and predicting offsets.
- **Iterative refinement in human motion.** FreqMRN above (62.1 vs 62.9 mm at equal compute). PGBIG (Ma et al., CVPR 2022) does progressive coarse-to-fine in DCT space.

## 4. (d) Multimodal / uncertainty heads

- **Mixture heads with hard assignment.** MultiPath++, Wayformer (Nayakanti et al., ICRA 2023, arXiv:2207.05844) and MTR all output per-timestep Gaussians (mean, log-sigma, rho) per mode plus a mode logit; loss = classification of the closest mode + NLL of GT under that mode only (winner-takes-all). HiVT/QCNet use Laplace mixtures (L1-flavoured NLL, "more robust to outliers"); the consensus in driving is that Laplace beats Gaussian for displacement metrics.
- **WTA pathologies and remedies.** Rupprecht et al. (ICCV 2017, arXiv:1612.00197) introduce MHP/WTA and its relaxed version. Makansi et al. (CVPR 2019, arXiv:1906.03631) document MDN mode collapse and NaNs, plain-WTA "dead" hypotheses stuck at equilibria, RWTA spurious modes; EWTA (top-k winners annealed from M to 1) then a fitting net: SDD NLL 9.33 vs MDN 9.71. Annealed WTA for motion forecasting (Xu et al., arXiv:2409.11172): softmin assignment with temperature schedule — at high temperature "the hypotheses converge toward a conditional mean, and the effective number of hypotheses is equal to 1" — lets MTR train with 6 instead of 64 queries: Argoverse 2 minADE 0.85 -> 0.77, MR 0.30 -> 0.19; Waymo minFDE 1.74 -> 1.34.
- **cVAE.** Trajectron++ (arXiv:2001.03093), VAT-Mart (Wu et al., ICLR 2022, arXiv:2106.14440: cVAE over <=5 residual 6-DoF waypoints, L1 position + 6D-rotation + KL) for articulated-object manipulation. DLow (Yuan & Kitani, ECCV 2020, arXiv:2003.08386) shows independent cVAE samples "may only produce samples that correspond to the major modes" and adds a learned diversity mapping; BeLFusion (Barquero et al., ICCV 2023, arXiv:2211.14304) shows VAE/GAN diversity is often "unrealistic and incoherent with past motion".
- **Diffusion / flow matching.** HumanMAC (DCT-space diffusion), TransFusion (arXiv:2307.16106), MotionDiffuser (PCA-space diffusion with a single L2 denoising loss, no anchors, minSADE 0.86 vs Wayformer 0.99 on Waymo interactive), MoFlow (Fu et al., CVPR 2025, arXiv:2503.09950: K-hypothesis flow matching with a WTA-style diverse loss, distilled to one step). Cost: many function evaluations unless distilled; benefit: no mode collapse by construction, controllable sampling via differentiable costs (MotionDiffuser's attractor/repeller).
- **When does unimodal regression collapse to the mean, and does it matter for us?** It matters when the conditional distribution is genuinely multimodal at the level of the *shape* (left vs right turn, walk vs stop). Given an instruction, a predicted joint type and an axis, our motion shape is essentially unimodal; the residual ambiguity is *how far* (opening extent) and the camera-frame sign. TNT's finding that target-conditioned completion is unimodal is the closest evidence: condition on the physical parameters and one-shot regression is enough. A 2-3 hypothesis head over the *extent* (or a Laplace NLL on it) is the cheap insurance.

## 5. (e) Smoothness and structure priors

- **Derivative losses.** siMLPe velocity loss (long-horizon gain 111.3 -> 109.4 mm). "Learning Velocity and Acceleration: Self-Supervised Motion Consistency for Pedestrian Trajectory Prediction" (arXiv:2503.24272): Huber losses on velocity and acceleration plus consistency terms; removing them degrades ETH-UCY ADE/FDE 0.16/0.28 -> 0.28/0.47 and SDD 7.08/10.47 -> 12.01/17.61. SIMPL's yaw (tangent) cosine loss cuts heading error 0.134 -> 0.055 with no displacement cost. Jerk penalties are standard in trajectory-adaptation/robot policy work (e.g. TrueAdapt arXiv:2006.00375) but rarely ablated in forecasting benchmarks.
- **Kinematic (physical) decoders.** Cui et al. DKM: the network predicts per-step longitudinal acceleration and steering; a differentiable bicycle-model layer integrates them. Heading error @6 s 4.92 deg vs 7.69 (unconstrained), position 4.21 vs 4.25 m, 0 % vs 26 % infeasible. MultiPath++ "Control" variant: same idea, minFDE 2.319 vs 2.305 raw, 0 % infeasible. Message: an integrator with the right physics costs ~nothing in accuracy and removes an entire class of failures; a *wrong or too-weak* prior (poly-2/3, constant velocity) does hurt.
- **Analytic decoders for articulated parts — FlowBot++, Zhang, Eisner, Held, CoRL 2023, arXiv:2306.12893.** Predicts per-point Articulation Flow f_p (instantaneous motion direction) and Articulation Projection r_p (vector from the point to the axis); axis direction = normalised f_p x r_p, origin = p + r_p, both averaged over the segmented part; the H-step trajectory of the contact point is then generated *analytically* with Rodrigues' rotation about (omega, v) for revolute joints or p + (i/K) l_g omega for prismatic joints. Loss: plain MSE on the dense f and r fields. Results (normalised distance to goal, lower is better): unseen test categories avg FlowBot3D (step-wise flow) 0.73, "Screw Parameters" direct regression 0.26, AP-only 0.27, FlowBot++ 0.18; train-category novel instances 0.29 -> 0.18 -> 0.11 -> 0.07. Qualitatively, step-wise prediction "begins to make the contact point go back and forth" under occlusion, while the analytic rollout is "consistent and smooth". Failure mode: if both axis estimates are wrong nothing can correct the rollout. Related: ScrewMimic (Bareddy et al., RSS 2024, arXiv:2405.03666) uses screw axes as the *action space* for bimanual manipulation; RPM-Net / Artic-O / ScrewSplat parameterise part motion as a screw (axis, origin, pitch, range).
- **Hybrid parameters + residual.** MultiPath++'s polynomial-with-regularised-constant-term and SmartRefine's offset-to-anchor are the generic pattern: a low-DoF structured prediction plus a small residual field, with the residual regularised (or delayed in training) so the structured part carries the load.

---

## 6. Comparison table

| Design | Parameterises | DoF (for T points, D dims) | Head | Loss | Evidence vs per-point regression | Known failure modes |
|---|---|---|---|---|---|---|
| Per-point one-shot (baseline) | positions/deltas at each step | T x D | MLP / query decoder | L1/L2/Huber, NLL | siMLPe w/o DCT +0.3-1 mm; MultiPath++ raw best ID minFDE | jitter, kinematic infeasibility (20-27 %), OOD squiggles |
| DCT / DFT truncation (LTD, siMLPe, HumanMAC, V2-Net, PatchTraj) | K low-freq coefficients of padded sequence | K x D (K=6-35) | MLP/GCN -> fixed IDCT | point-space loss after IDCT, + velocity loss | LTD supp.: truncation removes jitter; V2-Net ~10 % ADE/FDE; PatchTraj 6 % | not shape-aware; K too large -> jitter returns; still collapses OOD if the coefficients do |
| PCA basis (MotionDiffuser) | 10 data-driven components | 10 x 1 (joint xy) | diffusion denoiser | L2 on denoised latent | 0.06 m reconstruction; "better accuracy" | needs a trajectory dataset for the basis; basis is dataset-specific |
| Monomial polynomial coefficients (PolyLaneNet, LSTR, DKM poly-n, MultiPath++) | c_0..c_n per axis | (n+1) x D | FC / transformer | point-sampling loss | worse everywhere: BezierLaneNet F1 1.49 vs 68.9; SIMPL 1.738 vs 1.452 | ill-conditioned coefficient scales; poly-2/3 15-21 % infeasible |
| Bezier / Bernstein control points (BezierLaneNet, SIMPL, DeepRacing, EP) | n+1 control points | (n+1) x D (n=3-7) | MLP -> fixed basis matrix | sampling L1/smooth-L1 on curve points (+ tangent loss) | matches raw points ID; EP: best OOD robustness of all tested; DeepRacing: fewer boundary failures, smoother speed than waypoints | slight ID tax at long horizons/large data (MultiPath++) |
| Target-then-complete (TNT, DenseTNT, HOME) | endpoint (class+offset) + completion | 1 + T x D | 2-stage | CE + Huber | minFDE 0.53 vs 0.69 without offset; unimodal given target | needs candidate endpoints; scene-specific |
| Discrete tokens, AR (MotionLM) | quantised Verlet deltas | T tokens (169-way) | causal transformer | cross-entropy, teacher forcing | SOTA joint prediction; "hides precision", mitigates compounding | exposure bias, sequential latency, resolution floor |
| Anchor + one-shot residual (MultiPath++, MTR, QCNet) | offsets to K anchors | K x T x D | queries -> MLP/GMM | WTA-NLL + mode CE | MTR intention anchors +4.3 mAP; refinement +1.7 mAP | needs anchor set; mode collapse if K small (aWTA fixes) |
| Iterative refinement (MTR layers, FreqMRN, SmartRefine) | successive offsets | as above x L | stacked decoders | loss at every layer | FreqMRN 62.1 vs 62.9 mm at equal compute; SmartRefine +1-2 % | cost x L; can overfit to proposal errors |
| Kinematic integrator (DKM, MultiPath++ Control) | per-step controls (a, steer) | 2 x T | MLP + fixed ODE step | point loss through integrator | 0 % infeasible, heading err 4.9 vs 7.7 deg, same ADE | wrong model hurts; drift if controls noisy |
| Analytic screw/axis decoder (FlowBot++, ScrewMimic) | axis dir + origin + extent | 5-7 total | dense field or global regression -> Rodrigues rollout | MSE on fields (or on rolled-out points) | unseen-category normalised distance 0.18 vs 0.73 step-wise | uncorrectable if axis wrong; must handle type switch |
| Mixture / WTA (MultiPath++, Wayformer, HiVT, aWTA) | K hypotheses (+ scale) | K x T x D | queries | Gaussian/Laplace NLL, WTA/aWTA | aWTA: MR 0.30 -> 0.19 with 6 queries | dead hypotheses, conditional-mean collapse at high temperature |
| Diffusion / flow matching (HumanMAC, MotionDiffuser, MoFlow) | full trajectory in DCT/PCA/raw space | K x D or T x D | denoiser | L2 denoising / CFM | no anchors, SOTA multimodal | 10-100x inference cost unless distilled |

---

## 7. Ranked recommendations for SegAffordance

Context: we already predict joint type, axis direction and hinge origin. Physically, the interaction point's trajectory is a circular arc about that axis (revolute) or a line along that direction (prismatic); the only remaining continuous unknowns are the *extent* (angle theta or distance l) and, secondarily, the *timing profile* (how theta(t) ramps over the 20 samples) and a small deviation field (hand slip, non-ideal hinge). Every recommendation below is a one-shot head (no AR), supervised in point space through a fixed differentiable decoder, following BezierLaneNet/SIMPL/siMLPe.

**1. Analytic screw decoder + low-DoF residual (FlowBot++ / DKM pattern). Highest priority.**
   - Head predicts: extent (theta or l; positive scalar via softplus), a monotone timing profile s(t) in [0,1] (e.g. K=4 DCT or Bezier coefficients of a cumulative profile, or a softmax over 19 increments then cumsum — this is the one place a cumsum is safe because it is bounded and monotone), and an optional residual field of K'=3 DCT coefficients per axis with a strong L2 penalty / late-start schedule.
   - Decoder: p(t) = R(omega, theta s(t)) (p0 - o) + o - p0 for revolute; p(t) = l s(t) d for prismatic; select by the type head (soft-mix during training with type probabilities, or teacher-force GT type). Scale by 1/depth-of-first-point exactly as now.
   - Loss: existing L2/normalised trajectory loss on the decoded 20x3 (SF3D), existing 2D projection loss on hand datasets, plus a direct loss on extent when GT extent is available (derive it from the GT trajectory on SF3D). The axis-consistency loss becomes unnecessary (consistency is structural).
   - Why it addresses our failures: noise — the curve is a rigid arc by construction (roughness -> 0); OOD — the trajectory can only fail if the axis/type fail, and we observe those *stay plausible* OOD, so the trajectory inherits their robustness (this is exactly FlowBot++'s argument: "inherits the generalization properties" of the dense axis prediction; 0.73 -> 0.18 on unseen categories). The residual field is limited to K'=3 coefficients so it cannot regenerate squiggles.
   - Cost: trivial (a few hundred parameters, Rodrigues in PyTorch); requires deriving GT extent/profile from SF3D trajectories (least-squares fit of theta(t) given GT axis).
   - Evidence: FlowBot++ (Table 1), DKM (0 % infeasible, better heading, same ADE), MultiPath++ Control (same ADE as raw, 0 % infeasible), TNT (given the target the completion is unimodal, an MLP suffices).

**2. Bezier (Bernstein) control-point head with sampling loss and tangent loss — as a type-agnostic fallback/ensemble member.**
   - Predict n+1 = 5-7 control points per trajectory (degree 4-6, matching the AIC-optimal degree 3-6 in arXiv:2211.01696), first control point fixed at 0; trajectory = B P with a fixed basis matrix (same cost as IDCT). Add a cosine loss on the analytic tangent B' P against GT finite-difference tangents (SIMPL: heading error 0.134 -> 0.055) and against the predicted axis (cross-product constraint for revolute, parallel for prismatic).
   - Why: convex-hull property bounds the curve inside the control polygon, so the head cannot produce squiggles beyond degree-n wiggles; control points live in the same metric space as points (well-conditioned, unlike monomials: 1.738 vs 1.457 minFDE), and EP's cross-dataset study is the only direct evidence that this representation *degrades least OOD* (+24.7 % vs +61.5 % minADE6). Versus our DCT head: comparable smoothness, but end-point and tangent semantics that let us tie it to the axis heads.
   - Cost: none beyond the current head; direct drop-in replacement for the 6-DCT head. Keep K small (5-7 control points).

**3. Residual-to-anchor readout with a physics anchor + velocity loss (siMLPe lessons applied to whatever head is used).**
   - Build an *anchor trajectory* from the predicted (or, during training, GT) axis/type with a default extent (e.g. 45 deg / 0.15 depth-units) and let the head predict residual DCT/Bezier coefficients on top; add a velocity (first-difference) L2 term.
   - Evidence: residual-to-anchor is the single largest ablation effect in siMLPe (12.4 -> 9.6 mm at 80 ms; larger than DCT itself), MultiPath (anchors + offsets) and MTR (intention anchors +4.3 mAP); velocity loss helps long-horizon (111.3 -> 109.4). This is essentially design 1 with a fixed extent, and is a cheap intermediate experiment: if residuals shrink toward zero on OOD data the model has learned to "trust the physics".
   - Cost: trivial. Risk: if the anchor is wrong the residual must do all the work; keep the residual K low.

**4. Small-K winner-takes-all over extent/profile with annealed assignment (aWTA), Laplace NLL on extent.**
   - Our remaining ambiguity is scalar (how far the drawer/door goes) and roughly bimodal on hand videos (partial vs full open). Predict K=3 (extent, profile) hypotheses + logits under design 1's decoder; train with annealed WTA (Xu et al. 2024) to avoid dead hypotheses; report the argmax at inference.
   - Why: avoids the conditional-mean collapse that a single L2 regression of extent would show (Martinez 2017; aWTA sec. III-B) and gives calibrated extent uncertainty for downstream use. Multimodality over *shape* is not needed given the axis, so no full GMM over 20x3.
   - Cost: K x (1 + K_profile) extra outputs; negligible.
   - Evidence: aWTA MR 0.30 -> 0.19 with only 6 queries on MTR; TNT unimodality-given-target; HiVT/QCNet Laplace > Gaussian.

**5. One step of anchor-based refinement with trajectory-sampled features (MTR / QCNet / SmartRefine pattern).**
   - After decoding the design-1 trajectory, project the 20 3D points into the image, bilinearly sample the fused feature map at those pixels (plus the mask-pooled feature), and predict a small offset (residual K'=3 DCT per axis, or a delta-extent). One layer only.
   - Why: our current head sees only a pooled 512-d vector; the trajectory has no chance to check whether it passes through plausible pixels. Deformable-DETR-style refinement is exactly what lifts MTR (mAP 0.306 -> 0.323) and QCNet's refinement stage, and it addresses the OOD case where the pooled feature is off-distribution but local appearance along the arc is informative.
   - Cost: one grid_sample + small MLP; needs the trajectory to be in the image (true for our camera-frame outputs). Moderate implementation effort.

Not recommended for now: autoregressive GRU/transformer decoders (Martinez/NAT/DKM UM-LSTM show no gain and add exposure bias), full T x D per-point heads (our own noise result plus MultiPath++'s infeasibility), diffusion/flow heads (cost >> benefit for a nearly deterministic 20-point output), and monomial polynomial coefficients (uniformly worst in BezierLaneNet, SIMPL, DKM).

Suggested experiment order: (3) as a 1-day sanity check -> (1) -> (2) as an ablation partner (same K) -> (4) on top of (1) -> (5) if OOD is still the bottleneck. Metrics to add beyond L2: roughness (mean second difference), axis-consistency error, ID vs OOD *delta* per Yao et al., and "feasibility" = residual norm relative to the analytic part.

---

## 8. References (year, venue, id)

- Mao, Liu, Salzmann, Li. Learning Trajectory Dependencies for Human Motion Prediction. ICCV 2019. arXiv:1908.05436. https://github.com/wei-mao-2019/LearnTrajDep
- Mao, Liu, Salzmann. History Repeats Itself: Human Motion Prediction via Motion Attention. ECCV 2020. arXiv:2007.11755
- Guo, Du, Wang, Alameda-Pineda, Moreno-Noguer. Back to MLP: A Simple Baseline for Human Motion Prediction (siMLPe). WACV 2023. arXiv:2207.01567
- Chen et al. HumanMAC: Masked Motion Completion for Human Motion Prediction. ICCV 2023. arXiv:2302.03665
- Towards Accurate Human Motion Prediction via Iterative Refinement (FreqMRN). 2023. arXiv:2305.04443
- Wei et al. NeRMo: Learning Implicit Neural Representations for 3D Human Motion Prediction. ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06076.pdf
- Li, Tian, Zhang, Feng, Li. Multitask Non-Autoregressive Model for Human Motion Prediction (NAT). 2020. arXiv:2007.06426
- Martinez, Black, Romero. On Human Motion Prediction Using Recurrent Neural Networks. CVPR 2017. arXiv:1705.02445
- Yuan, Kitani. DLow: Diversifying Latent Flows for Diverse Human Motion Prediction. ECCV 2020. arXiv:2003.08386
- Barquero, Escalera, Palmero. BeLFusion. ICCV 2023. arXiv:2211.14304
- Wong et al. View Vertically: A Hierarchical Network for Trajectory Prediction via Fourier Spectrums (V2-Net). ECCV 2022. arXiv:2110.07288; follow-up arXiv:2304.05106
- PatchTraj: Unified Time-Frequency Representation Learning via Dynamic Patches for Trajectory Prediction. 2025. arXiv:2507.19119
- Learning Velocity and Acceleration: Self-Supervised Motion Consistency for Pedestrian Trajectory Prediction. 2025. arXiv:2503.24272
- Fu et al. MoFlow: One-Step Flow Matching for Human Trajectory Forecasting. CVPR 2025. arXiv:2503.09950
- Reichardt. An Empirical Bayes Analysis of Object Trajectory Representation Models. 2022. arXiv:2211.01696
- Yao, Yan, Goehring, Burgard, Reichardt. Improving Out-of-Distribution Generalization of Trajectory Prediction for Autonomous Driving via Polynomial Representations. 2024 (v3 Jan 2025). arXiv:2407.13431; dissertation: Long-term Traffic Scene Prediction via Polynomial Representations. arXiv:2608.03330
- Zhang, Sun, Wang, Liu. SIMPL: A Simple and Efficient Multi-agent Motion Prediction Baseline. RA-L 2024. arXiv:2402.02519
- Varadarajan et al. MultiPath++. ICRA 2022. arXiv:2111.14973; Chai et al. MultiPath. CoRL 2019. arXiv:1910.05449
- Shi, Jiang, Dai, Schiele. Motion Transformer with Global Intention Localization and Local Movement Refinement (MTR). NeurIPS 2022. arXiv:2209.13508
- Zhou, Wang, Li, Huang. Query-Centric Trajectory Prediction (QCNet). CVPR 2023. https://github.com/ZikangZhou/QCNet ; HiVT CVPR 2022 arXiv:2205.09753
- Nayakanti et al. Wayformer. ICRA 2023. arXiv:2207.05844
- Seff et al. MotionLM: Multi-Agent Motion Forecasting as Language Modeling. ICCV 2023. arXiv:2309.16534
- Jiang et al. MotionDiffuser. CVPR 2023. arXiv:2306.03083
- Zhao et al. TNT: Target-driveN Trajectory Prediction. CoRL 2020. arXiv:2008.08294; Gu et al. DenseTNT. ICCV 2021. arXiv:2108.09640
- Gilles et al. HOME. ITSC 2021. arXiv:2105.10968; GOHOME. ICRA 2022. arXiv:2109.01827
- SmartRefine. CVPR 2024. arXiv:2403.11492; R-Pred. ICCV 2023. arXiv:2211.08609
- Cui et al. Deep Kinematic Models for Kinematically Feasible Vehicle Trajectory Predictions. ICRA 2020. arXiv:1908.00219
- Buhet et al. PLOP: Probabilistic poLynomial Objects trajectory Planning. CoRL 2020. arXiv:2003.08744
- Weiss, Behl. DeepRacing: Parameterized Trajectories for Autonomous Racing. 2020. arXiv:2005.05178
- Feng et al. Rethinking Efficient Lane Detection via Curve Modeling (BezierLaneNet). CVPR 2022. arXiv:2203.02431
- Liu et al. End-to-end Lane Shape Prediction with Transformers (LSTR). WACV 2021. arXiv:2011.04233
- Rupprecht et al. Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses. ICCV 2017. arXiv:1612.00197
- Makansi, Ilg, Cicek, Brox. Overcoming Limitations of Mixture Density Networks (EWTA). CVPR 2019. arXiv:1906.03631
- Xu et al. Annealed Winner-Takes-All for Motion Forecasting. 2024. arXiv:2409.11172. https://github.com/valeoai/MF_aWTA
- Zhang, Eisner, Held. FlowBot++: Learning Generalized Articulated Objects Manipulation via Articulation Projection. CoRL 2023. arXiv:2306.12893
- Wu et al. VAT-Mart. ICLR 2022. arXiv:2106.14440
- Bareddy et al. ScrewMimic. RSS 2024. arXiv:2405.03666
