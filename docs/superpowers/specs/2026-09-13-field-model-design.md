# Field model: one map, many fields (2026-09-13)

## Why

The readout study of 2026-09-12/13 (experiments `20260913_joint4_decoder_l2anchor_*`, STATE.md
"NIGHT 2 RESULTS") ranked the ways the articulation heads read the decoded map: mask-mean pooling +
MLPs (base) < attention pooling < learned queries < per-pixel voting (dense: SF3D MA 44-45 over two
seeds, every SF3D articulation record). Only the spatial readouts bought type, sign, masks and hand-video
transfer; the pooled condition vector is the design's weak link, and the scalar depth heads still read
it. The per-pixel offset loss (`dense_off`) was the first thing to move hand-video hinge PLACEMENT
(ARCTIC hinge-line offset 0.029 vs 0.07-0.16). The user asked (2026-09-13 ~11:45) for the clean version
to be built and run now, without touching legacy experiments or the running ones.

## Design (`model/field_model.py`, `train_field_better.py`)

Trunk = CRIS's, RGB only: frozen DINOv3 + multi-tap adapter -> text-gated FPN -> transformer decoder
over word tokens -> decoded map fq (B, 512, H/16, W/16) -> dynamic-kernel projector -> mask / point /
origin heatmaps at H/4 (the origin channel is an auxiliary target only).

`FieldHead` (2 x conv3x3 + GN + GELU, 1x1 -> 12 channels) on fq: rot axis 3 | trans direction 3 | type
logits 2 | offset-to-hinge 2 | log-depth 1 | arc length 1. Readouts:

| prediction | readout |
|---|---|
| rot / trans axis, type, arc length | part-weighted mean of the field (tanh axes; softplus length) |
| origin_uv | part-weighted mean of (pixel uv + offset) — the votes |
| point_uv | soft-argmax of the point heatmap |
| z_p, z_q | exp(log-depth field) bilinearly sampled at point_uv / origin_uv (locations detached) |
| point_3d, origin_3d | lifts with the batch intrinsics (gen-7 convention) |
| trajectory | analytic decoder (2026-09-11) in the scale-free frame |

Part weights = the DETACHED predicted mask probability (train and test; no GT-mask teacher forcing).
No pooled condition vector, no MLP heads, no legacy modes. Trainable parameters 55M (CRIS 71M).

Losses: the SF3D trainer's, unchanged (mask DiceBCE, point heatmap + coord, origin heatmap BCE, type CE,
routed 1-cos axis anchor, closed-form L2 2pi on SF3D, projection loss on hand video, per-pixel offset loss
`dense_offset_weight`) plus one new term, `loss_params.depth_field_weight`: L1 on log-depth between the
field and the batch depth map pooled to the field resolution (valid-aware; SF3D only through the loss
profile). 2D rule: `dense_trunk_detach` in the 2d profile — hand-video batches train the fields, never
the trunk through them.

Implementation rule (user): NEW FILES ONLY. CRIS, the legacy trainers and every existing config are
untouched; `model/outputs.py` gains one defaulted field (`depth_field`), `config/opd_train.py` one
defaulted loss weight. `FieldTrainingModule` subclasses `SF3DTrainingModule`, swaps the model (an
uncompiled `_Recorder` shell around the compiled `FieldModel` keeps the last outputs for the extra term
and forwards the trainer's per-batch flag), and adds the depth-field loss after the parent's step.
Checkpoint keys: `model.core.*`. Tests: `tests/test_field_model.py`.

## Experiment

`20260913_field_joint4_l2anchor` (`config/joint4_decoder_field.yaml`, `run_joint4dec_field_chain.sh`):
the joint4 data recipe with `load_depth: true`, the l2anchor loss recipe, `dense_offset_weight` 0.5,
`depth_field_weight` 0.5 (0 on the 2d profile), `dense_trunk_detach` on the 2d profile, 20 epochs, seed
42. Judged against `dense` / `dense_off` (SF3D MA, origin, masks) and on hand video (masks, ARCTIC
hinge-line offset) — the question is whether the clean design keeps the SF3D record AND the hand masks.
