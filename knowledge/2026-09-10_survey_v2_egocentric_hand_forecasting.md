# Survey v2 (2026-09-10): trajectory-head structures in egocentric hand / contact-point forecasting

Clean-slate literature survey (web-only, no repo knowledge used). Scope: methods that predict a
FUTURE 2D/3D trajectory of a hand or contact point from a single frame or a short clip, with or
without language, and specifically WHAT READOUT they use (parameterisation, decoder, loss) and
whether they show evidence of beating a plain MLP regressor. Written for the SegAffordance
trajectory head (pooled 512-d fused feature -> tiny MLP -> 20x3 "plain" or 6 DCT coeffs/axis).

## 0. TL;DR

1. Nobody in this field uses a DCT/basis parameterisation. Every method emits per-step waypoints
   (4 to 256 of them), anchored to the first or last observed frame, and regularises SMOOTHNESS
   through the LOSS (velocity/displacement consistency, angle/length terms) or through a
   generative decoder (CVAE / diffusion / flow matching), not through a truncated basis.
2. The strongest, most repeatable evidence against "pooled vector -> MLP" is:
   (a) OCT Table 5: MLP 0.21/0.16 -> CVAE 0.12/0.11 ADE/FDE (min-of-20 sampling caveat);
   (b) ForeHand4D Table 3 (single image -> 3D): transformer regressor 15.3/30.0/29.2 vs
       diffusion 14.8/27.9/24.0 (in-domain / AssemblyHands / zero-shot EgoExo4D) - the gain is
       largest OUT OF DOMAIN;
   (c) EgoMAN Table 3: bare regression 0.273 m ADE -> flow-matching decoder 0.162 -> +explicit
       start/contact/end waypoint tokens 0.151, and a 27% ADE gain holds on OOD HOT3D;
   (d) MADiff Fig. 12: a 0-block (MLP) denoiser is "substantially worse" than 4-6 sequence blocks;
   (e) USST Table 3: a velocity-consistency loss barely helps in-domain (0.189 -> 0.183 3D ADE)
       but cuts UNSEEN-scene error 0.168 -> 0.120 (-29%). This is the cheapest OOD lever found.
3. Coarse-to-fine (predict anchor waypoints first, then fill the trajectory conditioned on them)
   is the structural idea that shows up in every 2025-26 SOTA that reports OOD numbers
   (EgoMAN, VidBot, LatentAct/CoherentHand) and matches our situation exactly: our mask/type/axis
   stay plausible OOD while the trajectory collapses, i.e. the anchors are there, the fill-in is not.

## 1. Comparison table

ADE/FDE are in normalised image units for 2D methods, metres (or cm where noted) for 3D.
"Head" = how the fused feature becomes a trajectory.

| Method (venue, arXiv) | Input | Output representation | Head | Loss | Horizon | Head-structure evidence / OOD |
|---|---|---|---|---|---|---|
| FHOI, Liu et al. (ECCV'20, 1911.10967) | 32-frame clip, I3D/CSN | 2D "motor attention": T_m x H_m x W_m softmax heat-tensor in last frame (32x/8x downsampled) | 3D conv + per-slice softmax, Gumbel-softmax stochastic units | KL(p(M|x) || Gaussian around GT fingertip) + action CE | 0.5 s (EGTEA), 1 s (EK) every 8 frames | EGTEA ADE/FDE: KF 0.32/0.48, GPR 0.29/0.37, LSTM 0.22/0.35, FHOI 0.23/0.36; stochastic > deterministic (JointDet) on hotspots F1 +2.4 |
| OCT (CVPR'22, 2204.01696) | 10 frames @4 fps, TSN RoI hand/obj/global tokens, no language | 2D px of both hand centres projected into LAST observed frame, F=4 | Transformer enc (6) + AUTOREGRESSIVE dec (4) fed previous predicted locations -> C-VAE (1-layer MLP enc/dec), min-of-20 at test | L2 recon + KL; L = L_H + 0.1 L_O | 1 s / 4 pts | Table 5 (EK100): MLP 0.21/0.16, bivariate-Gaussian 0.19/0.14, C-VAE 0.12/0.11. EK->EGTEA cross-dataset: Seq2Seq-LSTM 0.24/0.19 vs OCT 0.16/0.13 |
| Ego4D FHP baseline (CVPR'22, github EGO4D/forecasting) | 60 frames before PRE-1.5s | 2D px (x_l,y_l,x_r,y_r) at 5 key frames | I3D-R50 8x8 + fully-connected regression | regression (not specified in README) | ~1.5 s / 5 pts | EMAG reports this "I3D+Regression" loses 21.1% cross-dataset vs 9.6% for a transformer with ego-motion tokens |
| EgoPAT3D (CVPR'22, 2203.13116) | RGB-D point cloud + IMU per frame | 3D action TARGET point in the current camera frame (one point, updated every frame) | per-frame feats -> MLP -> 2-layer LSTM/GRU -> per-axis confidence over a 1024-bin grid -> expectation | truncated weighted regression (TWRLoss) vs NLL | whole clip | Table 2 (cm): NLL 21.63 vs TWRLoss 18.61; GRU 20.07 vs LSTM 18.61 |
| EgoPAT3Dv2 (ICRA'24, 2403.05046) | RGB only + MediaPipe landmarks | same as above | ConvNeXt-T + MLP -> 2-layer LSTM -> grid | TWR + aux hand-position + time losses | whole clip | 16.70 -> 14.97 cm; aux hand loss is the main gain |
| USST (ICCV'23, 2307.08243) | 60% of a 64-frame clip: 64x64 RGB + past 3D points | 3D GLOBAL coords (world frame of first frame), normalised to [-1,1] | Transformer encoders (256-d) -> attention-based AUTOREGRESSIVE state transition (z 16-d) -> emission MLP([z_t; o_{t-1}]) giving mean + log-var; separate velocity MLP | Huber heteroscedastic NLL (depth-decoupled, depth-stability weights) + velocity constraint (per-step + cumulative-sum consistency) | ~26 future steps | Table 3 (3D ADE seen/unseen): ProTran-det 0.201/0.195, SST 0.190/0.174, full 0.183/0.120, w/o L_velo 0.189/0.168, w/o NLL 0.292/0.267; Table 5: local-camera targets 0.202/0.174 vs global 0.183/0.120 |
| EMAG (2024, 2405.20030) | hand/obj box tracks + RGB + flow + homographies | 2D px both hands, 256-px normalised, F=4 | transformer enc + two AUTOREGRESSIVE decoders (hand, ego-motion) + MLP heads | self-adjusting smooth-L1 + L2 on homography | 1 s / 4 pts | EK55 ADE 48.78 vs OCT 53.85 px; EK->Ego4D 53.67 vs 57.74; ego-motion tokens are the most transferable input |
| BOT (2024, 2405.05552) | 2 s clip, flow (TSN) + last frame (Segformer) | 2D waypoints as HEATMAPS (argmax), F=4; hotspot heatmap | bidirectional progressive cross-attention between trajectory and hotspot branches | MSE on heatmaps + C-VAE hotspot | 1 s / 4 pts | Ego4D: OCT 0.12/0.12 -> BOT 0.09/0.08; ablation individual 0.14 -> bi-progressive 0.09 |
| Diff-IP2D (IROS'25, 2405.04370) | 10 frames @4 fps + homographies | 2D px both hands on last frame, F=4 (+10 contact points) | LATENT DIFFUSION over (N_p+N_f) x 512 sequence, 6-layer transformer denoiser, 1000 steps, NON-autoregressive; MLP trajectory head on denoised future latents; C-VAE affordance head | VLB + L2 traj + KL-aff (0.1) + latent reg (0.2) | 1 s / 4 pts | EK55 WDE/FDE 0.411/0.181 vs OCT 0.446/0.208, USST 0.458/0.210; temporal-enhanced variant +5.6% FDE ("smoother"); EGTEA zero-shot +7-15% |
| MADiff (TPAMI'25, 2409.02638) | frames + GLIP vision-language feats (prompt "hand") + homographies | 2D waypoints (rounded px), F=4 | Mamba denoiser (6 blocks) with motion-driven selective scan; continuous-discrete-continuous rounding each step; MLP decode | VLB + displacement L2 + ANGLE (cosine) + LENGTH losses | 1 s / 4 pts | Fig 12: 0 blocks (=MLP) "substantially worse", 4-6 best, transformer denoiser worse than Mamba; Table VI: waypoints-only 0.079 -> +visual 0.070 -> +text 0.065 ADE |
| MMTwin (2025, 2504.07375) | RGB + point cloud + past 3D pts + text | 3D waypoints in camera frame of FIRST input frame, 60/40 split @30 fps | twin diffusion: ego-motion Mamba + hand branch hybrid Mamba-Transformer; MLP decode | 2x VLB + displacement + reg + angle | ~1 s | EgoPAT3D-DT seen ADE/FDE 0.170/0.336 vs MADiff3D 0.183/0.363, USST 0.183/0.341; unseen 0.118/0.189; w/o ego-motion diffusion 0.186/0.363 |
| Uni-Hand (2025-26, 2511.12878) | RGB(+D) + past pts + text | 2D and 3D waypoints, multi-joint, + contact state | dual-branch diffusion (as MMTwin), MLP decoders | 2x VLB + dis + angle + reg + BCE contact | ~1 s | EgoPAT3D-DT unseen 0.118/0.189 vs MADiff3D 0.139/0.224 |
| HandsOnVLM (2024, 2412.13187) | 10 frames + LANGUAGE | 2D centres both hands in last frame, F=4 | LLaVA-style VLM emits <HAND> tokens; each hidden state -> C-VAE decoder | text CE + recon + KL | 1 s / 4 pts | EK55 ADE/FDE/WDE 0.136/0.106/0.062 vs LLaVA-Traj 0.126/0.142/0.073; qualitative zero-shot to H2O/FPHA |
| EgoMAN "Flowing from Reasoning to Motion" (2025, 2512.16907) | SINGLE RGB frame + past wrist track + intent text | 6-DoF wrists (3D pos + 6D rot) @10 fps in camera frame | Qwen2.5-VL emits <ACT>,<START>,<CONTACT>,<END> tokens -> timestamp + 3D pos + rot per waypoint; encoder-decoder transformer FLOW-MATCHING motion expert with waypoint tokens placed at predicted timestamps and future queries | LM + action-semantic (cosine/InfoNCE) + Huber waypoint loss with Gaussian time windows; FM MSE | up to T steps (multi-second) | Table 1 (unseen, K=10 ADE/FDE m): USST 0.233/0.394, MMTwin 0.206/0.256, HandsOnVLM 0.171/0.228, FM-base 0.160/0.229, EgoMAN 0.124/0.179. Table 3 (K=1): none 0.273, reasoning+waypoints no FM 0.215, FM only 0.162, full 0.151; OOD HOT3D -27.3% vs HandsOnVLM; 3.45 FPS |
| SFHand (2025-26, 2511.18127) | streaming frames + language + previous hand state | next-frame hand type, 2D box, MANO pose, 3D world trajectory; 16 frames/3 s | DETR-style 4-layer decoder with learnable QUERIES over memory-augmented embeddings | CE + L1/GIoU + L1 pose + L1 traj, Hungarian matching | 3 s / 16 pts | ADE/FDE: USST 49.15/55.20, EgoH4 22.56/22.87, SFHand 12.65/13.08; w/o memory 15.56/18.38 |
| EgoH4 (2025, 2504.08654) | 20 frames @10 fps + camera poses + 2D hands + ViT-S feats | 57 joints (body + 2x21 hand) canonicalised to first camera, F=10 | diffusion transformer (4 layers, 512) | L1 joints + BCE visibility + 2D reprojection | 1 s / 10 pts | Table 3: baseline 0.295 -> full 0.261 m ADE; MPJVE reported (0.763 m/s) |
| EggHand (2026, 2605.07642) | 20 frames (2 s) + past 21-joint poses + text | 10 future 21-joint frames in normalised egocentric canonical frame anchored at first obs | GR00T-N1.5-3B flow-matching ACTION HEAD fine-tuned deterministically | L1 abs (0.6) + wrist-relative L1 (0.2) + pairwise L2 (0.2) | 1 s / 10 pts | EgoH4 0.267/0.333/0.116 vs EggHand 0.271/0.271/0.076 (ADE/FDE/MPJPE); random-init decoder 0.321 vs pretrained 0.271 |
| Exo2EgoPose (2026, 2607.15890) | frames + language + past 42x3 poses | future 42x3 joints | adaLN + cross-attn modulation blocks -> MLP | smooth-L1 + BCE validity | - | vanilla transformer +1.07 MPJPE, removing modulation +4.74 |
| ForeHand4D (2025, 2510.06145) | SINGLE RGB image | both hands MANO (16 joints x 6D rot + 3D transl), 256 steps, camera frame at t=0, normalised translation | MDM-style conditional DIFFUSION, 16-layer transformer encoder, 1024-d, ViT image tokens | diffusion loss on 3D labels; 2D weak labels via a lifting model | 256 steps | Table 3: transformer REGRESSOR 15.3/30.0/29.2 vs diffusion 14.8/27.9/24.0 (in-domain / AssemblyHands / zero-shot EgoExo4D); adding 2D via aux regression head 26.8/25.9, aux diffusion head 31.9/24.8, imputed labels 20.3/18.8 |
| LatentAct "How Do I Do That?" (2025, 2504.12284) | single RGB + text + 3D contact point (16^3 Gaussian voxel) | MANO sequence T=30/16 + 778-d contact maps; translation/rotation RELATIVE TO FIRST FRAME | VQ-VAE interaction CODEBOOK (6 quantisers, K=512) + learned indexer + transformer decoder | smooth-L1 MANO, L1 rel. translation, L1 6D rot, BCE contact | 16-30 steps | Table 4 (MPJPE): transformer 7.52 -> +codebook 6.72; diffusion 8.30 -> +codebook 6.53 |
| CoherentHand (CVPR'26, Boote et al.) | single object image + action prompt + initial 3D contact point | continuous 3D hand trajectory | VLM-guided codebook + FLOW-based decoder | (abstract only; PDF blocked) | - | claims better temporal consistency than codebook/diffusion baselines on HoloAssist/ARCTIC |
| VidBot (CVPR'25, 2503.07135) | single RGB-D frame + language | N_c x 3 contact points + H x 3 trajectory in observation camera frame | COARSE: Perceiver goal-heatmap + depth and contact-heatmap; FINE: 1D U-Net DIFFUSION over the trajectory conditioned on coarse outputs + TSDF; test-time guidance (goal, collision, normal) | BCE heatmaps + L2 depth; diffusion MSE | H waypoints | Table 2 (robot success): full 85.6%, w/o coarse goal 57.8%, w/o multi-goal guidance 73.3%, w/o normal 76.7%, w/o collision 77.8% |
| What Happens Next? (2025, 2509.21592) | SINGLE image | dense 2D trajectory grid (H/s x W/s x T x 2), T in {16,24,30} | VAE latent + rectified FLOW MATCHING spatio-temporal transformer, DINOv2 cross-attn | Huber + KL (VAE); RF loss | 16-30 pts | LIBERO-90 MSE: ATM regression 23.07/67.37 vs generative mean 16.70/52.70, min-of-8 10.99/32.01 |
| MotionForesight (2026, 2607.16192) | 7 frames + pointmaps | 3D trajectory field of object points, T=15, last-camera frame | frozen video DiT (Wan2.1) + LoRA-32 predicting residual track latents through a frozen tracker decoder | MSE (obs 0.25, future 1.0) | 15 pts | deterministic; SSv2 ADE 4.47; authors note it under-predicts motion magnitude (ratio 0.72) |
| FIction (2024, 2412.00932) | video + SMPL + 3D scene voxels | future interaction voxels + pose | linear voxel decoder + C-VAE pose | BCE + L2/L1 + KL | 5-180 s | not a trajectory head; listed for completeness |

## 2. Per-family notes

### 2.1 Heat-tensor / attention heads (FHOI 2020, BOT 2024)
Liu et al. (ECCV 2020) never regress coordinates: the "motor attention" is a T x H x W probability
tensor produced by one 3D conv + per-slice softmax on early I3D features, supervised with a KL to a
Gaussian around the annotated fingertip and sampled with Gumbel-softmax. Their trajectory ADE is
slightly worse than an LSTM that is given the observed hand coordinate (0.23 vs 0.22) but works
without any hand detection. BOT (2405.05552) keeps the heat-map readout (waypoints = argmax) and
adds bidirectional progressive refinement between trajectory and hotspot branches; progressive
refinement alone moves Ego4D ADE 0.14 -> 0.09. Relevance to us: a heatmap head is dense and
spatially grounded, which is a plausible reason it degrades gracefully, but it is 2D-only and
would need an extra depth channel for our 3D output.

### 2.2 Autoregressive transformer + latent stochastic head (OCT 2022, EMAG 2024, HandsOnVLM 2024)
OCT is the reference architecture for 2D egocentric HOI forecasting: encoder over hand/object/global
tokens, an autoregressive decoder that consumes previously predicted hand locations, and a
C-VAE readout. The head ablation (Table 5, EK100) is the clearest MLP-vs-alternative number in the
field: MLP 0.21/0.16, bivariate Gaussian 0.19/0.14, C-VAE 0.12/0.11. Two caveats: (i) evaluation is
min-of-20 samples, so part of the CVAE gain is oracle selection, (ii) OCT transfers poorly to
3D (USST adapted it and got 0.252 ADE on H2O vs 0.031 for USST, blamed on KL vanishing).
EMAG keeps the autoregressive decoder but adds an ego-motion decoder; its cross-dataset drop is
9.6% vs 21.1% for the Ego4D I3D+regression baseline, and ego-motion tokens are the most
transferable input. HandsOnVLM puts the same C-VAE readout on <HAND> tokens of a VLM; its win is
language-conditioned reasoning rather than the head.

### 2.3 Recurrent / state-space with uncertainty and velocity (EgoPAT3D 2022, USST 2023)
EgoPAT3D shows that for 3D targets, a classification-over-grid readout with a truncated weighted
regression loss beats NLL (18.61 vs 21.63 cm) and LSTM beats GRU. USST is the most instructive
paper for our loss design. Its emission head is only MLP([z_t; o_{t-1}]) with a 16-d latent, yet:
(a) heteroscedastic NLL with a depth-specific variance is essential (removing it: 0.183 -> 0.292);
(b) a velocity head + consistency loss (per-step velocity supervised by first differences, and the
cumulative sum of predicted velocities from the last observed point tied to the predicted
position) gives almost nothing in-domain (0.189 -> 0.183) but cuts unseen-scene 3D ADE from 0.168
to 0.120, which the authors attribute to injecting a physical rule; (c) predicting in a GLOBAL
frame anchored at the first frame beats per-frame camera coordinates (0.174 -> 0.120 unseen).
USST is autoregressive and its authors list that as a limitation.

### 2.4 Non-autoregressive latent diffusion (Diff-IP2D, MADiff, MMTwin, Uni-Hand)
This family (all IRMV Lab, SJTU) argues explicitly that autoregressive decoders accumulate error
and lack bidirectional constraints, and instead denoises the whole (past + future) latent
sequence in parallel, with an MLP reading waypoints off the denoised future latents. Gains over
OCT/USST are consistent but modest in 2D (WDE 0.411 vs 0.446) and larger in 3D (MMTwin 0.170 vs
USST 0.183 seen). Two head-structure findings matter to us: MADiff Fig. 12 shows that a
denoiser with zero sequence blocks (a pure MLP) is substantially worse, and that 4-6 blocks
saturate; and every paper from MADiff on adds an ANGLE loss (cosine between predicted and GT
displacement vectors) and a LENGTH loss on displacement norms, arguing that displacement-only L2
cannot distinguish a physically plausible from an implausible trajectory of equal error. Diff-IP2D
also reports a temporally-enhanced variant that improves FDE by 5.6% and is described as smoother.
Cost: 1000 denoising steps at inference (they report it is still real-time-ish for 4 waypoints),
6 x 512-d blocks.

### 2.5 VLM token interface + flow-matching motion expert (EgoMAN 2025, EggHand 2026, SFHand)
EgoMAN is the closest published analogue to our problem: a SINGLE RGB frame plus text yields a
multi-second 6-DoF wrist trajectory in the camera frame. Its structure is two-stage: the VLM emits
four special tokens (<ACT>, <START>, <CONTACT>, <END>) whose hidden states regress a timestamp,
3D position and 6D rotation each; a separate encoder-decoder transformer trained with flow
matching then fills in the dense trajectory, with the waypoint tokens inserted at their predicted
timestamps and learned future queries for the remaining steps. The ablation isolates the two
ideas: flow-matching decoder alone takes single-sample ADE from 0.273 to 0.162 m; adding the
explicit waypoints gives 0.151; the full model beats HandsOnVLM by 27% on unseen splits and by a
matching 27% on OOD HOT3D. The waypoint tokens also make the model 3.45 FPS versus <0.05 for
affordance baselines. EggHand instead reuses a robot VLA action head (GR00T flow matching) and
fine-tunes it deterministically with L1 losses (absolute + wrist-relative + pairwise), reporting
that the pretrained decoder prior is worth 0.05 m ADE over random init. SFHand shows that a
DETR-style query decoder (learnable queries cross-attending to memory) is enough for streaming
3D forecasting and outperforms USST by ~4x on its benchmark.

### 2.6 Single-image 3D hand motion (ForeHand4D, LatentAct, CoherentHand, VidBot)
ForeHand4D gives the cleanest "same backbone, regression vs diffusion" comparison in the
single-image setting: in-domain the gap is small (15.3 vs 14.8), but on the held-out
AssemblyHands (30.0 vs 27.9) and zero-shot EgoExo4D (29.2 vs 24.0) the diffusion head is
clearly better, i.e. the generative head is mostly an OOD lever. It also finds that injecting
weak 2D supervision through an auxiliary 2D regression or 2D diffusion head does NOT help
(26.8/25.9 and 31.9/24.8 vs 27.9/24.0 baseline), whereas lifting 2D labels to imputed 3D labels
does (20.3/18.8). This is a warning sign for our 2D-projection loss on HOI4D/EPIC/ARCTIC: the
one paper that tested this exact idea found aux 2D heads neutral-to-harmful and label imputation
much better. LatentAct conditions on a 3D contact point and a text prompt (exactly our
interaction-point + instruction) and shows that a discrete VQ codebook of interaction motions
improves both a transformer decoder (7.52 -> 6.72 MPJPE) and a diffusion decoder (8.30 -> 6.53);
it parameterises translation and rotation RELATIVE TO THE FIRST FRAME, as we do. CoherentHand
(CVPR 2026) extends it with VLM-guided codebooks and a flow decoder for temporal consistency
(PDF not retrievable; abstract only). VidBot is coarse-to-fine in the robot-affordance sense:
contact and goal heatmaps first, then a 1D U-Net diffusion over the 3D trajectory conditioned on
them, with differentiable test-time costs; removing the coarse goal drops success from 85.6% to
57.8%, more than removing any guidance term.

### 2.7 Dense trajectory fields from one image (What Happens Next?, MotionForesight)
Both predict trajectories for many points rather than one, but the head lessons transfer.
"What Happens Next?" shows a rectified-flow generator over a VAE latent of the trajectory grid
beats the ATM regression head by ~28% in mean and ~52% in min-of-8 MSE. MotionForesight is
deterministic and explicitly reports under-prediction of motion magnitude (ratio 0.72), the
classic mean-regression symptom that also produces "squiggles" when the conditional is multimodal.

## 3. Cross-cutting findings for our failure modes

Output parameterisation. All methods emit waypoints; nobody in this field uses DCT or any other
basis (DCT is standard in third-person body-motion forecasting, not here). Coordinates are
anchored either to the last observed frame (2D family) or the first frame (3D family: USST, MMTwin,
EggHand, LatentAct, ForeHand4D), and USST shows anchoring in a stable global/first frame is worth
0.05 m OOD. Several 3D methods normalise translation by dataset statistics; EggHand adds a
wrist-RELATIVE loss on top of the absolute one, which is the closest analogue to our
delta / depth-of-first-point convention.

Smoothness. No paper reports a roughness metric (EgoH4 reports MPJVE for body joints only).
Smoothness is obtained via (i) velocity/displacement heads with consistency terms (USST),
(ii) angle + length losses on successive displacements (MADiff, MMTwin, Uni-Hand), (iii) generative
decoders whose samples are individually smooth even when the mean is not (Diff-IP2D, ForeHand4D,
EgoMAN), (iv) discrete codebooks of whole-motion primitives (LatentAct, CoherentHand). Our
observation that the plain head is jittery while DCT is smooth-but-inexpressive maps onto
(i)-(ii) as the cheap fix and (iii)-(iv) as the structural fix.

OOD collapse. The explicit OOD numbers all point the same way: velocity constraint (USST unseen
-29%), ego-motion features (EMAG), generative decoder (ForeHand4D zero-shot -18%, EgoMAN -27% on
HOT3D), and explicit anchor waypoints (EgoMAN, VidBot). Mean-regression heads on pooled vectors
are exactly the component every one of these papers replaced.

Evidence that the head, not the backbone, matters. Same-backbone ablations: OCT Table 5,
ForeHand4D Table 3, EgoMAN Table 3, LatentAct Tables 3-4, MADiff Fig. 12, USST Table 3,
"What Happens Next?" Table 1. These are the numbers to cite when justifying a head change.

## 4. Ranked candidate head designs for SegAffordance

1. Anchor-then-fill ("waypoint-conditioned residual decoder"), from EgoMAN + VidBot + LatentAct.
   Design: predict K=3 anchor points (start = interaction point, mid/contact, end) with their
   normalised times from the fused feature (3 small MLPs or 3 learned tokens), then a 2-4 layer
   transformer decoder with 20 learned time queries cross-attending to the fused feature MAP
   (not the pooled vector) and to the anchor embeddings, outputting residuals around the linear
   interpolation of the anchors. Condition the anchors on the axis/type heads (revolute: end
   point lies on the circle about the predicted hinge; prismatic: along the axis) so the
   part of the model that stays plausible OOD constrains the part that collapses.
   Why it helps: OOD squiggles become at worst a straight/arc path between well-predicted anchors;
   EgoMAN's waypoint tokens add 7% on top of its generative decoder and enable a 27% OOD gain;
   VidBot loses 28 pp without the coarse goal. Cost: ~1-3 M params, one forward pass, no sampling.
   Keep DCT as the residual basis if desired (residual = IDCT of 6 coefficients) so smoothness is
   retained while anchors carry the geometry.

2. Velocity parameterisation with USST-style consistency plus MADiff angle/length losses.
   Design: head outputs 19 velocity vectors (or keeps DCT but decodes velocities); positions are
   the cumulative sum; loss = L2 on positions + L2 on velocities vs. GT first differences + cosine
   (angle) loss on successive displacements + L2 on displacement norms; optionally a
   heteroscedastic NLL with a separate depth variance (USST: -37% ADE from uncertainty alone on
   noisy-depth data, relevant to SF3D LMDB depth noise). Why: zero parameters, directly targets
   jitter (velocity supervision) and OOD (USST unseen 0.168 -> 0.120 from the velocity constraint
   alone). Cost: none at inference. Compatible with every other option here; do this first.

3. Non-autoregressive query decoder over the feature map (SFHand / Diff-IP2D-MADT without the
   diffusion). Design: 20 learned temporal queries + text/axis tokens, 4 layers x 256-d, cross-attn
   to the fused image features, per-query 3-d output (or 6 DCT coefficients per axis from a
   pooled query set). Why: MADiff's 0-block ablation and SFHand's results show sequence blocks
   with per-timestep queries beat a pooled-vector MLP; queries see spatial context that pooling
   destroys, which is the most likely cause of the OOD collapse while the mask head (which sees
   the map) stays fine. Cost: 3-8 M params, one pass; ~2x the current head latency.

4. Conditional flow-matching / diffusion trajectory head (ForeHand4D / EgoMAN motion expert /
   VidBot 1D U-Net). Design: MDM-style 4-layer transformer denoiser over the 20x3 trajectory
   (or over its 18 DCT coefficients), conditioned on the fused vector + anchors + text via
   adaLN; rectified-flow objective; 10-20 Euler steps at test; single sample or K=5 with
   min/median selection. Why: the only head family with same-backbone evidence of a large
   zero-shot gain (ForeHand4D 29.2 -> 24.0; EgoMAN 0.273 -> 0.162 single-sample), and samples
   are individually smooth. Cost: training ~1.5-2x slower, inference 10-20 passes of a ~5 M
   param head; evaluation must decide single-sample vs min-of-K; risks under-training on our
   small SF3D set (ForeHand4D trained on five lab datasets plus imputed in-the-wild labels).

5. C-VAE readout (OCT / HandsOnVLM). Design: encode GT trajectory + fused feature to a 32-d
   latent, decode with a 2-layer MLP, KL to N(0,1), sample at test. Why: cheapest stochastic
   head; OCT's MLP 0.21 -> CVAE 0.12 ADE is the largest reported MLP-vs-head gap. Caveat: the gap
   is measured min-of-20; OCT-style CVAEs collapsed when USST moved them to 3D (0.252 vs 0.031),
   and USST's authors deliberately avoided the ELBO for that reason. Ranked last because the
   deterministic mean of a CVAE is not obviously better than an MLP, and our failure mode
   (OOD collapse) is not addressed by latent stochasticity alone.

Suggested order: 2 (free) -> 1 (small, targets OOD) -> 3 or 4 (structural), with the 2D
projection loss re-examined in light of ForeHand4D Table 5 (auxiliary 2D heads hurt; imputed 3D
labels help), e.g. by lifting HOI4D/EPIC/ARCTIC 2D tracks to pseudo-3D with a depth model before
they touch the trajectory head.

## 5. Sources

- Liu, Tang, Li, Rehg. Forecasting Human-Object Interaction: Joint Prediction of Motor Attention and Actions in First Person Video. ECCV 2020. arXiv:1911.10967. https://arxiv.org/abs/1911.10967
- Liu, Tripathi, Majumdar, Wang. Joint Hand Motion and Interaction Hotspots Prediction from Egocentric Videos (OCT). CVPR 2022. arXiv:2204.01696. https://arxiv.org/abs/2204.01696
- Ego4D Future Hand Prediction baseline. https://github.com/EGO4D/forecasting/blob/main/Ego4D-Future-Hand-Prediction/README.md
- Li et al. Egocentric Prediction of Action Target in 3D (EgoPAT3D). CVPR 2022. arXiv:2203.13116. https://arxiv.org/abs/2203.13116
- EgoPAT3Dv2. ICRA 2024. arXiv:2403.05046. https://arxiv.org/abs/2403.05046
- Bao et al. Uncertainty-aware State Space Transformer for Egocentric 3D Hand Trajectory Forecasting (USST). ICCV 2023. arXiv:2307.08243. https://arxiv.org/abs/2307.08243
- EMAG: Ego-motion Aware and Generalizable 2D Hand Forecasting. 2024. arXiv:2405.20030. https://arxiv.org/abs/2405.20030
- BOT: Bidirectional Progressive Transformer for Interaction Intention Anticipation. 2024. arXiv:2405.05552. https://arxiv.org/abs/2405.05552
- Ma, Xu, Chen, Wang. Diff-IP2D. IROS 2025. arXiv:2405.04370. https://arxiv.org/abs/2405.04370
- MADiff: Motion-Aware Mamba Diffusion Models for Hand Trajectory Prediction. TPAMI 2025. arXiv:2409.02638. https://arxiv.org/abs/2409.02638
- MMTwin: Novel Diffusion Models for Multimodal 3D Hand Trajectory Prediction. 2025. arXiv:2504.07375. https://arxiv.org/abs/2504.07375
- Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views. 2025/26. arXiv:2511.12878. https://arxiv.org/abs/2511.12878
- Bao, Xu, Wang, Gupta, Bharadhwaj. HandsOnVLM. 2024. arXiv:2412.13187. https://arxiv.org/abs/2412.13187
- Flowing from Reasoning to Motion (EgoMAN). 2025. arXiv:2512.16907. https://arxiv.org/abs/2512.16907
- SFHand: Streaming Egocentric 3D Hand Forecasting. 2025/26. arXiv:2511.18127. https://arxiv.org/abs/2511.18127
- The Invisible EgoHand (EgoH4). 2025. arXiv:2504.08654. https://arxiv.org/abs/2504.08654
- EggHand: A Multimodal Foundation Model for Egocentric Hand Pose Forecasting. 2026. arXiv:2605.07642. https://arxiv.org/abs/2605.07642
- Exo2EgoPose. 2026. arXiv:2607.15890. https://arxiv.org/abs/2607.15890
- Bimanual 3D Hand Motion and Articulation Forecasting in Everyday Images (ForeHand4D). 2025. arXiv:2510.06145. https://arxiv.org/abs/2510.06145
- How Do I Do That? Synthesizing 3D Hand Motion and Contacts (LatentAct). 2025. arXiv:2504.12284. https://arxiv.org/abs/2504.12284
- Boote, Kim, Kara, Lee, Rehg. CoherentHand. CVPR 2026. https://openaccess.thecvf.com/content/CVPR2026F/papers/Boote_CoherentHand_Temporally_Consistent_3D_Hand_Trajectory_Synthesis_with_Semantic_Motion_CVPRF_2026_paper.pdf
- VidBot. CVPR 2025. arXiv:2503.07135. https://arxiv.org/abs/2503.07135
- What Happens Next? Anticipating Future Motion by Generating Point Trajectories. 2025. arXiv:2509.21592. https://arxiv.org/abs/2509.21592
- MotionForesight. 2026. arXiv:2607.16192. https://arxiv.org/abs/2607.16192
- FIction: 4D Future Interaction Prediction from Video. 2024. arXiv:2412.00932. https://arxiv.org/abs/2412.00932

Not found / not applicable: "EgoTraj" (no egocentric hand-trajectory paper by that name surfaced;
the hits were pedestrian-trajectory work), "LAPA" (Latent Action Pretraining is a robot-policy
latent-action method, no hand trajectory head), "HOI4D forecasting" (no dedicated HOI4D trajectory
forecasting benchmark found; HOI4D appears only as a data source in the works above).
