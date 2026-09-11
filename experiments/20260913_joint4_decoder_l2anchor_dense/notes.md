# 20260913_joint4_decoder_l2anchor_dense — articulation readout arm `dense`

**Goal.** DENSE HINGE VOTING (per-pixel rot/trans axis, type logits and 2D offset-to-hinge fields on the decoded map; part-mask-weighted means give axis / type / origin_uv; heatmap kept as auxiliary; scalar heads classical). The fourth arm of the head-bottleneck test (option 3 of the 2026-09-12 discussion):
the origin is LOCATED by the part's pixels (each votes an offset to the hinge) instead of being read
out of one pooled vector. Supervision reaches the fields only through the averaged outputs (origin_pred
via the lift, the closed-form loss on SF3D, the projection loss on the 2D sources) — no dense per-pixel
loss yet (candidate follow-up: mask-weighted |uv + offset - q*_uv| on SF3D).
Spec: docs/superpowers/specs/2026-09-12-articulation-readout-design.md.

**Comparison row.** `20260912_joint4_decoder_l2anchor` (MA 31.05 / 30.27, rot flips 19.0, origin 0.294,
mIoU 0.241 / PDet 18.5, HOI4D 0.576; ARCTIC probe: axis 48.2 deg, flips 26.4 %, hinge-line offset 0.070)
and the three readout arms of the same date (query / attnpool / mlp1024).

**Setup.** `config/joint4_decoder_l2anchor_dense.yaml`, `run_joint4dec_l2anchor_dense_chain.sh`, pod jdec-dense.

**Result.** (pending)
