# ARTHUR: Learning 3D Articulation Affordances from Human Video — draft outline (v0, 2026-09-13)

Method name: ARTHUR = ARTiculation affordances from HUman Recordings (chosen 2026-09-13).

CANONICAL DRAFT: `~/Research/AA3D/paper/draft.md` (IEEEtran project; written 2026-09-13). This repo copy is
the outline snapshot and is no longer updated. Status: high-level outline only. Numbers
are placeholders keyed to experiment IDs in `experiments/INDEX.md`. FINAL MODEL (fixed 2026-09-13):
`20260913_joint4_decoder_l2anchor_dense_off` — CRIS trunk + dense hinge voting + per-pixel offset loss +
analytic decoder on the L2 2pi + axis recipe (SF3D MA 46.0, ARCTIC hinge offset 0.029). The field model
(all-fields redesign) is the research line, reported as an ablation. TODO = open.

## 0. The story in one paragraph

Prior articulation-estimation work predicts joint parameters from curated 3D input (point clouds,
RGB-D scans, CAD). A robot in the wild has an image, sometimes with depth. It needs a model that goes
from an image and an instruction to the full 3D articulation affordance of the part it should act on:
the part mask, the interaction point, the motion type, the axis, the origin and the sweep. Curated 3D
annotation is expensive, so 3D-only training does not generalise; the scale that exists is 2D human
video. We build on a frozen foundation trunk (DINOv3), express the articulation as per-pixel fields on
one decoded map, and train 3D-annotated data and 2D human video jointly through an analytic decoder
that renders the trajectory from the articulation parameters, so a 2D hand track and a 3D sweep pull
on the same loss landscape. Result: state of the art on SceneFun3D by a wide margin, hinge placement
that transfers to unseen hand-video objects, and one language-conditioned model, RGB-only or RGB-D,
that also segments and localises the part.

## 1. Introduction

- Robots need articulation (type, axis, origin, extent) to act; prior methods assume curated 3D input.
- Deployment reality: an image (+ optional depth) and an instruction. The output must name the part
  (referring expression -> mask), the contact (point) and the motion (type, axis, origin, sweep).
- 3D annotation is scarce; generalisation requires scale; scale is in human video, which has no 3D
  labels but shows how objects move.
- Our approach in one sentence (trunk / fields / analytic decoder / joint training).
- Contributions:
  1. A language-conditioned model producing the full articulation affordance tuple on a frozen
     foundation trunk. RGB-only training and inference is a first-class mode because that is what
     human-video data has; depth is used as an input when available.
  2. Joint 2D/3D training through an analytic trajectory decoder and matched loss landscapes: the 3D
     closed-form loss is the continuous limit of the 2D trajectory loss.
  3. A spatial articulation readout (per-pixel hinge voting + an offset loss) that replaces pooled
     vectors and MLP heads; it sets the SceneFun3D records and is the first readout whose hinge
     placement transfers to hand video.
  4. Three 2D articulation-affordance datasets mined from human video (HOI4D, EPIC-KITCHENS, ARCTIC)
     with VLM / rule / object-model motion-type labels, and an analysis of what 2D supervision can
     and cannot determine (the axis-sign / hinge-side ambiguity).

## 2. Related work

- Articulation from 3D input: OPD, OPDFormer / MOPD, USDNet, voting designs (ANCSH, Shape2Motion),
  generative (SINGAPO, CAGE).
- Affordance and interaction hotspots from egocentric video; hand-object forecasting; point tracks as
  supervision.
- Language-conditioned segmentation (CRIS lineage); frozen foundation backbones and dense adapters.
- Positioning: first to output the full tuple from an image with language, and first to use 2D human
  video as articulation supervision through a differentiable trajectory renderer.

## 3. Problem and data

- Task: (image, optional depth, instruction) -> (mask, point, type, axis n, origin q, sweep). Camera
  frame; the scale-free frame (metres / z_p) for RGB-only.
- 3D data: SceneFun3D (processed LMDB, key filter, 5,088-sample test split); depth available.
- 2D data (ours): HOI4D, EPIC-KITCHENS/VISOR, ARCTIC -> per-window part mask, contact point, hand track,
  motion type (VLM for EPIC, rules for HOI4D, object models for ARCTIC). Counts, scene-level splits.
  TODO figure: examples from the three sources.
- What 2D supervises: a short arc fixes the tangent and, with the type, the direction; but (n, hinge on
  one side) and (-n, hinge on the other) project to the same arc. Stated here, analysed in 5.6.

## 4. Method

4.1 Trunk. Frozen DINOv3 ViT-L (+ dino.txt text), multi-tap pyramid adapter, text-gated FPN,
    transformer decoder over word tokens -> one decoded map; dynamic-kernel projector -> mask, point
    and origin heatmaps. Depth input: optional encoder fused into the FPN (the depth arms); RGB-only is
    the default mode. TODO: confirm the RGB-D variant of the final model.
4.2 Dense hinge voting (the final model). A 1.8M conv head on the decoded map predicts, per pixel, a
    rot axis, a trans direction, type logits and a 2D offset to the hinge; the part-weighted means
    (GT mask in training, predicted mask at test) give the axis, the type and the hinge location
    (origin_uv). A per-pixel offset loss pulls every part pixel's vote to the projected GT hinge on
    3D data. The point comes from the point heatmap's soft-argmax; the depths z_p / z_q and the arc
    length come from small MLPs on the pooled condition vector (point depth with a local feature
    sample); lifts with intrinsics. Ablation: the all-fields variant (depth and length as fields, no
    pooled vector, predicted-mask weighting, `model/field_model.py`).
4.3 Analytic trajectory decoder. Rot: the arc about the axis line through the origin with angle L / r;
    trans: L d. The trajectory is a function of the parameters, never a free head.
4.4 Losses and the matched landscape. 2D: projection loss of the decoded arc onto the hand track
    (unit anchor). 3D: closed-form position quadratic (the N -> inf limit of the trajectory loss over a
    sweep Theta) + the 1-cos axis term; the master-formula view in the appendix. Per-pixel offset loss
    toward the projected hinge; depth-field loss (3D only); type CE; mask DiceBCE; point / origin
    heatmap BCE.
4.5 Joint training. Source-homogeneous batches, per-source loss profiles, hand sources x10 augmented;
    2D batches train the fields but not the trunk through them. TODO: final balance.

## 5. Experiments

5.1 Setup. SceneFun3D protocol (MA @10 deg on type-matched rows, signed variants, flip rate, origin /
    line distance, radius, mIoU, PDet); ARCTIC 3D hinge probe (held-out objects, GT hinges from object
    models: axis error, hinge-line offset); hand-video masks / PDet. Seeds on headline rows (TODO).
5.2 Main results on SceneFun3D vs baselines (OPDFormer C/P, MOPD, USDNet retrained on SF3D —
    `experiments/baselines_sf3d/`). Ours: dense voting 44-46 MA vs previous best 36.7
    (`20260913_joint4_decoder_l2anchor_dense{,_seed7,_off}`; field model TODO).
5.3 Does 2D human video help 3D articulation? SF3D-only vs joint on the final model (TODO run), and
    chain (pretrain -> post-train) vs joint (`20260912_sf3d_decoder_l2anchor_ft_multi3dec` vs the joint
    arms).
5.4 Transfer to hand video: ARCTIC probe (hinge offset 0.029 with the offset loss vs 0.07-0.16),
    qualitative HOI4D / ARCTIC panels (`viz/20260913_readout_arms_hand_panels`,
    `viz/20260912_hoi4d_laptop_decoder_probe`).
5.5 Ablations: readout (pooled MLP / attention pooling / queries / dense voting; capacity control), the
    3D-side loss family (L2 2pi vs cf_frame; the anchor), the offset loss, trunk detach, depth field,
    RGB-only vs RGB-D (TODO run). Seed noise stated.
5.6 Analysis: the sign / hinge-side ambiguity (13 % vs 72 % flips across seeds at equal axis error),
    what breaks it (ARCTIC GT axes under the closed-form loss — TODO run); decoded vs free trajectories
    (roughness).

## 6. Limitations

- The axis sign on 2D-only objects is undetermined without 3D labels or a geometric prior.
- Hand-source masks vs 3D articulation trade in the joint recipe (final model's numbers TODO).
- Hand-video depth is unsupervised (scale-free frame); metric scale needs depth or intrinsics.
- Single image in, single motion out; no multi-part or sequential reasoning.

## 7. Conclusion

## Appendix (planned)

- A. The closed-form loss: derivation of the N -> inf limit, the master formula, sweep and leak
  terms, the anchor as the p = 0 case (`docs/slides/2026-09-11_l2_quadratic_from_scratch.html`,
  `knowledge/2026-09-11_revolute_loss_with_axis_term.md`).
- B. Dataset construction (VLM prompt, rules, ARCTIC object-model hinges, filters).
- C. Metric definitions and the seed-noise calibration.
- D. Architecture details and parameter counts.

## Paper-blocking TODO

- [x] Final model: dense + offset loss (`20260913_joint4_decoder_l2anchor_dense_off`); the field model is the research line.
- [ ] SF3D-only control on the final model.
- [ ] Seeds on headline rows.
- [ ] RGB-D variant of the final model.
- [ ] ARCTIC GT axes under the closed-form loss.
- [ ] Baselines table from the baselines session.
- [ ] Figures: teaser, method diagram, dataset examples, qualitative panels, ablation plots.
