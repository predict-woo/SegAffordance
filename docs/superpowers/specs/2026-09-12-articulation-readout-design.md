# Articulation readout: learned-query attention over the decoded map (2026-09-12)

## Problem

Every articulation head (type + rot/trans axis MLP, point depth z_p, origin depth z_q, arc length)
reads ONE condition vector: the mask-mean pooled decoded map (512), the global image token (1024),
the text state (2048), point_uv and origin_uv (2 + 2). The heads are 2-layer MLPs of width 256.
Mask-mean pooling destroys where the hinge sits relative to the part; the local evidence is a
minority of the vector. The 2026-09-12 hand-video study (viz/20260912_hand2d_articulation_100)
shows hinge PLACEMENT (origin, radius) as the failing quantity while type and axis direction
transfer — consistent with an information bottleneck at the readout rather than a capacity limit
(the same architecture moved MA 26 -> 36 by loss changes alone). DINOv3 stays frozen (user).

## Design (ModelParams.articulation_readout)

- `mlp` (default): unchanged classical path.
- `attnpool` (option 1): one learned query, one masked cross-attention over the stride-16 decoded
  map fq (B, 1024 tokens, 512). Attention logits get `log(mask + eps)` added per token (mask = GT
  mask in teacher-forced training, predicted mask otherwise — the same source the mean pooling
  used), so part tokens dominate and the surroundings (seam, frame) stay reachable. Output replaces
  the pooled slot; heads unchanged. ~1M params.
- `query` (option 2): `readout_queries` (4) learned queries through `readout_layers` (2) pre-norm
  layers: self-attention among queries, masked cross-attention over fq with 2D sine positions
  added to the keys, FFN (`readout_dim_ffn` 1024), LayerNorm out (6.3M params at d 512). Query 0 feeds the type/axis
  MLP, 1 the point depth, 2 the origin depth, 3 the arc length — each head's condition vector
  has ITS query in the pooled slot and the shared globals after it.
- `mlp1024` control: `vae_hidden_dim` 1024 + `trajectory_length_hidden` 1024, classical pooling —
  separates "not enough capacity" from "not enough information".

Everything downstream is unchanged: same output tensors, analytic decoder, closed-form loss on
SF3D, projection loss on the 2D sources, test configs with per-arm `--model.model_params`
overrides. Implementation: `model/layers.py` (`ArticulationReadout`, `_MaskedCrossAttention`,
`sine_positions_2d`), `model/segmenter.py` (build; `_head_condition`), `config/opd_train.py`.
Tests: `tests/test_articulation_readout.py`.

## Experiment

Three arms on the final recipe `config/joint4_decoder_l2anchor.yaml` (L2 2pi + axis 0.5 on SF3D,
joint4 data, 20 ep, seed 42), one PRO 6000 pod each (~4.5 h, ~$10 each):
`20260913_joint4_decoder_l2anchor_{query,attnpool,mlp1024}`. Judged against
`20260912_joint4_decoder_l2anchor` by SF3D origin / rot flips / MA / masks and the ARCTIC hinge probe
(axis error, flips, hinge-line offset). Single seed; read with +-1 MA / +-2 cm / +-0.01 mIoU noise.
