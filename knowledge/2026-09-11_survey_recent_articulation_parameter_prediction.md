# Survey — recent models for articulation-parameter prediction (axis / origin / type), with SceneFun3D coverage

Date: 2026-09-11. Scope: papers from ~Sept 2025 to Sept 2026 that output joint
parameters, plus every paper found that reports articulation numbers on
SceneFun3D (SF3D). Complements (does not repeat) the loss-level survey in
`../../knowledge/articulation-loss-survey.md` (25 papers, 2026-08-29) and the
trajectory-head survey `2026-09-10_survey_v2_articulation_affordance_trajectories.md`.
Sources: web search, arXiv pages, OpenAlex + Semantic Scholar citation graphs of
SceneFun3D / Articulate3D / OPDMulti, the SF3D EvalAI API, and the Articulate3D
challenge page.

## 0. TL;DR

- **SF3D has no public motion-estimation leaderboard.** The EvalAI challenge
  (id 2466) exposes only Functionality Segmentation and Task-Driven Affordance
  Grounding phases (val + test). Every SF3D articulation number in the
  literature is a paper-internal evaluation on the val split or a hand-picked
  scene subset.
- Only **four** papers report articulation numbers on SF3D: the SF3D baseline
  (Mask3D-FM, CVPR'24), **USDNet** (ICCV'25), **AffordBot** (NeurIPS'25) and
  **PhysGraph** (arXiv 2606.08655). They use three mutually incompatible
  protocols (see §1). USDNet's 30.5 AP50+axis+origin is still the only learned,
  scene-level, 3D-input number to beat.
- Closest public scene-level leaderboard is the **Articulate3D challenge**
  (ScanNet++): the winning entry "pico-mr" scores 0.41 MAP_AXIS_ORIGIN vs
  USDNet's 0.26 — a 58% relative gain with **no paper or report** published.
- The recent field has split into (a) feed-forward object-level regressors
  trained at 10⁴–10⁵ object scale (Particulate → Instruct-Particulate, ART,
  ArtSplat, ArtLLM), (b) video/egocentric scene pipelines that fit axes from
  tracks (FunRec, EgoFun3D, Pandora, MoMa-SG, PokeNet, REACT3D), and (c)
  VLM-prompting zero-shot scene pipelines (PhysGraph, DRAWER, NeoWorld-Pro).
  None of (a)–(c) trains a scene-level regressor on SF3D; nothing since USDNet
  does.
- Design consensus across the strongest new regressors: **per-point /
  per-pixel over-parameterised axis predictions (direction + closest point on
  the axis) aggregated by geometric fitting**, rather than one global 6-vector.
  This matches our closed-form per-point screw supervision.

## 1. Everything with SceneFun3D articulation numbers

| Paper (venue) | Input | How axis/origin/type are produced | Protocol on SF3D | Numbers | Code |
|---|---|---|---|---|---|
| SceneFun3D baseline "Mask3D-FM" (CVPR'24) | scene point cloud | Mask3D queries → FC heads; CE type + smooth-L1 axis + smooth-L1 origin | AP25 on val with cumulative constraints (+type, +axis 15°, +origin 0.25 m) | AP25 26.6 → 7.9 with type+axis+origin | yes |
| **USDNet / Articulate3D** (ICCV'25, 2412.01398) | scene point cloud (Mask3D backbone) | dense per-point axis + origin-offset, averaged over the instance mask; 1−cos axis loss, point-to-line origin loss | AP50 + axis (1−cos15°) + origin (0.25 m); val split | **30.5** (Mask3D† 22.4, SoftGroup† 12.8). Same metric on Articulate3D: 25.0 (+axis) / 34.6 (+origin), MultiScan 26.0 | yes + weights |
| **AffordBot** (NeurIPS'25, 2511.10017) | scene point cloud + rendered views + instruction | Mask3D (ScanNet200-pretrained, fine-tuned on SF3D) for masks; Qwen2.5-VL-72B chain-of-thought emits motion **type** + a **discretised direction** (translation: horiz-in/out, vert-in/out; rotation: horiz/vert). No continuous axis, no origin | AP25 +T (type) and +TD (type + discretised direction), 200 train / 30 val scenes, instruction-conditioned | AP25 23.3, **+T 18.3, +TD 10.8** (Fun3DU+motion: 18.7 / 11.5 / 4.0) | no |
| **PhysGraph** (arXiv 2606.08655) | RGB-D key frames (multi-view) | MobileSAMv2 part masks → GPT-5 visual prompting picks hinge start/end vertices (right-hand rule) → lifted to 3D and aggregated over key frames; prismatic direction from geometric priors + segmentation edges. Zero-shot | "MultiScan protocol" on **30 hand-picked articulation-rich SF3D scenes**: joint-type accuracy, min. distance between predicted/GT joint lines (revolute), orientation error | **type acc 96.0%, MD 6.33 cm, OE 5.84°** (DRAWER 68.3% / 18.3 cm / 2.07°; URDFormer 61.5% / 1.86 cm / –; 3DOI 33.6%) | no |
| AFUN (2606.02551) — for completeness | RGB-D + language | predicts a 3D Bézier trajectory, **not** axis/origin/type | SF3D test set (721 examples), trajectory metrics vs A0 / VRB / VidBot / General Flow | not articulation numbers | no |

Caveats for comparing against our runs:
- USDNet is the only one whose metric (AP50 + 15° + 0.25 m, whole val split)
  is a superset-style detection metric; our MA / matched-axis / origin numbers
  are per-GT-part conditional metrics. To claim an SF3D SOTA we would need to
  run the USDNet evaluator (public) on our predictions, or re-run USDNet's
  released weights under our per-part protocol.
- PhysGraph's 5.84° orientation error is computed only over parts its
  pipeline detected and type-classified, on 30 chosen scenes, with the
  matching rule unspecified. Not comparable to a whole-val axis error.
- AffordBot's +TD is a 4-/2-way direction classification, so its "axis" is
  coarser than any regression head; it is nevertheless the only SF3D
  articulation result from an MLLM.

## 2. Public leaderboards that exist (and don't)

- **SceneFun3D EvalAI (challenge 2466)** — phases: 6118/6119 Functionality
  Segmentation val/test, 6120/6121 Affordance Grounding val/test. **No motion
  phase.** For reference, test-split leaders as of 2026-09-11: functionality
  segmentation AP 29.0 / AP50 43.2 / AP25 49.6 (team TnG, 2026-09-04);
  affordance grounding AP50 25.9 / AP25 36.5 (OpenDL, 2026-08-24). Fun3DU is
  at AP50 3.6 on grounding test.
- **Articulate3D challenge (OpenSUN3D @ ICCV'25, ScanNet++)** — metric
  MAP_AXIS_ORIGIN, plus AP@50 / MA (axis) / MO (origin) / MAO-ST. Leaderboard:
  pico-mr **0.41** (winner), astar 0.29, Uni Stuttgart 0.27, ww 0.26, USDNet
  baseline 0.26. No method descriptions or reports are linked; searches for a
  pico-mr / PICO technical report found nothing (PICO-MR is the ByteDance
  PICO mixed-reality team that also won the MDEC depth challenge). The same
  team submitted to SF3D grounding (AP50 11.3, 2025-10-07).
- OPDMulti / OPDReal: no new leaderboard entries; the last method-level
  improvement is Locate n' Rotate / MOPD (ACCV'24 oral, 2412.13173, code
  public) which adds perceptual-grouping and geometric-prior foundation
  features to OPDFormer.

## 3. Recent articulation-parameter predictors WITHOUT SF3D numbers (Sept 2025 → Sept 2026)

### 3a. Feed-forward object-level regressors (the strongest axis heads right now)

| Paper | arXiv | Input → output | Axis/origin head | Data / benchmark | Headline numbers | Code |
|---|---|---|---|---|---|---|
| Particulate (CVPR'26) | 2512.11798 | static 3D mesh/points → parts + joints | per-point Plücker direction + per-point foot point, L1 each; fitted to one axis | PartNet-Mobility + procedural | (in loss survey) | train + ckpt |
| **Instruct-Particulate** | 2606.14699 | mesh + kinematic spec (part text, connectivity, joint types, optional point prompts) → part seg + joints | same over-parameterised per-query (local axis dir, closest axis point, positions at joint limits) → geometric fit to shared axis + limits | **~150k objects** (27k VLM-pseudo-labelled synthetic, 120k captioned part-segmented, 10k agent-generated, PartNet-Mobility, GRScenes); Lightwheel benchmark | part P/R 97.3/95.9; **axis 9.5°, location 0.015** (Particulate 20.9° / 0.040; best other 13.8° / 0.043); image→3D via HY3D-3.1 for real photos | not stated |
| ART: Articulated Reconstruction Transformer | 2512.14671 | sparse multi-state RGB → per-part geometry/texture + joints | MLP per part-slot: type CE; unit axis (L2-norm) MSE; origin = 2r·σ(·)−r MSE; per-stage angle/translation | ~12.3k objects (PartNet-Mobility 300, procedural 2k, StorageFurniture 10k); PartNet-Mobility + StorageFurniture tests | reports part gIoU / centroid / Chamfer only (no axis-error table) | no |
| ArtSplat | 2605.24304 | sparse uncalibrated multi-view, two states → cameras, depth, Gaussians, joints in one pass | **per-pixel joint map** on a VGGT-style transformer (DINOv2 tokens, cross-state attention) | 68 PartNet-Mobility objects | "competitive" joint estimation, 400× faster than optimisation baselines | no |
| ArtLLM | 2603.01142 | complete point cloud → autoregressive parts + joints (3D MLLM) | joints emitted as tokens | curated + procedural set; PartNet-Mobility | claims SOTA part layout + joint prediction; no numbers in abstract | no |
| SCAPO | 2606.01940 | single RGB-D, self-supervised, category-level | SE(3)-equivariant VN autoencoder + joint-aware blend skinning → pivots, axes, states | synthetic + real category datasets | beats self-supervised baselines | no |
| MonoArt (ECCV'26), DICArt (CVPR'26), Artic-O, DailyArt | see loss survey | single image → axis + pivot | covered in `articulation-loss-survey.md` | — | — | — |

### 3b. Scene-level from video / egocentric interaction (axes fitted from tracks or hands; all real-world, all optimisation)

| Paper | arXiv / venue | Input | Axis estimation | Benchmarks | Numbers | Code |
|---|---|---|---|---|---|---|
| **FunRec** (CVPR'26, SF3D authors) | 2604.05621 | egocentric RGB-D interaction video | per-track line/circle hypotheses → joint pose-graph + articulation refinement in Ceres (manifold opt.) | new **RealFun4D** (351 interactions, 60 apartments), **OmniFun4D** (127, OmniGibson), HOI4D subset. **SF3D not used** | axis dir error 5.3° / 12.4° / 5.6° (OmniFun4D / HOI4D / RealFun4D), position 0.03–0.06 m; BundleSDF 26–38°, MonST3R 47–58° | not stated (functionalscenes.github.io) |
| **EgoFun3D** | 2604.11038 | egocentric video → parts + articulation + function templates | benchmarks ArtiPoint and iTACO as articulation estimators on upstream masks | new dataset of 271 egocentric videos; SF3D only cited | ArtiPoint: axis 1.057 rad, origin 0.346 m, type 74.2%, 46% failures; iTACO: 1.022 rad, 0.665 m, type 26.8% — "tracking-based articulation prediction is unreliable" | dataset promised |
| Pandora | 2603.28732 | Aria egocentric video | hand-trajectory arc/line fit picks type; refine with one-sided Chamfer (object returns to pre-state) + hand-consistency cost | Blender sim + 2 annotated Aria kitchens | prismatic 0.19°, revolute 0.62°, pivot 0.004 (bbox-diag normalised); Ditto 10.8° / 1.26° / 0.26 | no |
| MoMa-SG (Articulated 3D Scene Graphs for Open-World Mobile Manipulation) | 2602.16356 | RGB-D sequence, mobile robot | occlusion-robust point tracking → **unified twist estimation** solving revolute + prismatic in one pass | new **Arti4D-Semantic** (62 seq., 600 interactions) + Arti4D | ablations only in abstract | code + data (momasg.cs.uni-freiburg.de) |
| PokeNet | 2602.02741 | point-cloud sequence of one human demo | PointNet++ + transformer encoder/decoder with learnable joint slots → confidence, type, unit axis, 3D anchor point (learned, end-to-end) | sim + real | +27% axis/state accuracy vs prior | not stated |
| REACT3D (RA-L) | 2510.11340 | static scan + detected parts | covered in trajectory survey; evaluates on ScanNet++ with Articulate3D GT | — | — | — |
| Articulation in Prime | 2605.18645 | single casual monocular video | primitive fitting + joint optimisation of segmentation and joint params | new AiP-synth / AiP-real | no axis numbers in abstract | not stated |
| Articulation in Motion (ICLR'26) | 2603.02910 | start-state 3DGS scan + interaction video | dual-Gaussian dynamic/static disentanglement + sequential RANSAC, no part-count prior | object-level | — | project page |
| ArtPro, ArtMesh, FreeArtGS, PAOLI, VideoArtGS | 2602.22666, 2605.16582, 2603.22102, 2509.04276, 2509.17647 | multi-view / video object captures | GS/mesh optimisation with motion-consistency; axes recovered from part poses | PartNet-Mobility-style, ArtMesh introduces Articulate-100 | object-level only | mixed |

### 3c. Zero-shot VLM-prompting scene pipelines (RGB/RGB-D in, URDF/USD out)

| Paper | arXiv | Articulation mechanism | Evaluation | Notes |
|---|---|---|---|---|
| **PhysGraph** | 2606.08655 | GPT-5 visual prompting over part masks (§1) | 30 SF3D scenes (§1) | the only zero-shot VLM method with SF3D numbers |
| DRAWER (CVPR'25) | 2504.15278 | 3DOI hinge/affordance prediction + GPT-4o second opinion + VLM arbitration + 3D grounding | 6 self-captured kitchens with fishing-wire GT videos: precision 97.2 / recall 84.0 (URDFormer 93.5 / 46.4); EA-score 0.994 vs 3DOI 0.861 | baseline inside PhysGraph's SF3D table |
| NeoWorld-Pro | 2608.24212 | MLLM code synthesis of geometry + articulation + physics, refined physics-in-the-loop | monocular RGB; no public benchmark numbers in abstract | cites USDNet |
| UNITE (Unified Semantic Transformer) | 2512.14364 | VGGT-1B + DPT head: per-pixel movable-part existence + 3D motion vector (rotation approximated as 90° displacement), focal + ℓ2, multi-view consistency | **MultiScan only**: IoU 70.3, movable-part F1 10.0, motion-type F1 6.9 (OPDPN 2.1/1.7, S2M 9.4/6.3) | RGB-only feed-forward scene model; no axis/origin regression |
| GSAM | 2605.30740 | vision perceiver + fine-tuned VLM CoT refiner for kinematic params | 50 hinge manipulation tasks | robotics-oriented |

### 3d. New datasets relevant to axis supervision

- **Artiverse** (2605.24403): 5.4k human-authored objects, 88 categories,
  multi-DoF joints, metric scale/mass; benchmarks part mobility analysis.
- **Hoi!** (2512.04884): 3,048 sequences, 381 articulated objects, 38
  environments, force/tactile; articulation annotations (details in paper).
- **RealFun4D / OmniFun4D** (FunRec), **EgoFun3D** (271 videos),
  **Arti4D-Semantic** (MoMa-SG), **AiP-real** — all egocentric interaction
  video with axis GT; none overlap SF3D scenes.

## 4. What this means for SegAffordance

1. **No new learned SF3D competitor since USDNet.** The AffordBot number
   (AP25+TD 10.8, discretised direction) is far below what a regression head
   gives; PhysGraph's numbers are on a curated subset with a different
   protocol. If we want an external comparison, the cheapest credible one is
   the USDNet evaluator (AP50 + 15° + 0.25 m) on the SF3D val split with USDNet's
   released weights re-run — everything else needs re-implementation.
2. **Headroom is real.** pico-mr's 0.41 vs 0.26 on Articulate3D (same
   Mask3D-style task) says ~60% relative gains over USDNet-style dense
   averaging are achievable on real scans; the recipe is unpublished.
3. **Head design converging on ours.** Instruct-Particulate (9.5° axis on
   Lightwheel, trained on 150k objects), USDNet (dense per-point averaged over
   mask) and ArtSplat (per-pixel joint map on a VGGT backbone) all predict
   per-point direction + closest-point-on-axis and fit afterwards. Our
   closed-form per-point screw quadratics are the loss-side version of this;
   a per-pixel joint-map readout on the DINOv3 features (à la ArtSplat/UNITE)
   is the untested architectural counterpart.
4. **Pseudo-labelling source for 2D datasets.** Instruct-Particulate /
   Particulate (public ckpt) can label axes on generated meshes from single
   images; PhysGraph/DRAWER show GPT-5/4o hinge prompting is reliable for
   *type* (96%) but weak on origin (6–18 cm). Neither is a training-time
   dependency we would want, but both are viable for auditing HOI4D/EPIC
   pseudo-axes.
5. **Video-fit axes are not a free lunch.** EgoFun3D measured ArtiPoint and
   iTACO at ~1 rad axis error with 5–46% failures on real egocentric video;
   FunRec gets 5–12° only with its full pose-graph + Ceres refinement. Our
   HOI4D trajectory labels derived from object poses are likely cleaner than
   any off-the-shelf tracker fit.

## 5. Sources (arXiv ids / URLs)

SceneFun3D CVPR'24 (scenefun3d.github.io; EvalAI challenge 2466) · Articulate3D/USDNet 2412.01398 (challenge: insait-institute.github.io/articulate3d.github.io/challenge.html) · AffordBot 2511.10017 · PhysGraph 2606.08655 · AFUN 2606.02551 · FunRec 2604.05621 · EgoFun3D 2604.11038 · Pandora 2603.28732 · MoMa-SG 2602.16356 · PokeNet 2602.02741 · REACT3D 2510.11340 · Articulation in Prime 2605.18645 · Articulation in Motion 2603.02910 · ArtPro 2602.22666 · ArtMesh 2605.16582 · FreeArtGS 2603.22102 · Particulate 2512.11798 · Instruct-Particulate 2606.14699 · ART 2512.14671 · ArtSplat 2605.24304 · ArtLLM 2603.01142 · SCAPO 2606.01940 · DRAWER 2504.15278 · NeoWorld-Pro 2608.24212 · UNITE 2512.14364 · GSAM 2605.30740 · Locate n' Rotate 2412.13173 · Artiverse 2605.24403 · Hoi! 2512.04884 · ArtiPoint/Arti4D 2509.01708 · iTACO 2506.08334 · Fun3DU 2411.16310 · OPDMulti 2303.14087 · 3DOI 2305.09664
