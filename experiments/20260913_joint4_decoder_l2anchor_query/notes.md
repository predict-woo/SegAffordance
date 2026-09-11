# 20260913_joint4_decoder_l2anchor_query — articulation readout arm `query`

**Goal.** articulation readout = QUERY (4 learned queries, 2 pre-norm layers, mask-biased cross-attention over the decoded map; one query per head). Tests the head-bottleneck hypothesis (2026-09-12): the articulation heads read one
mask-mean pooled vector through a 256-wide MLP, which keeps nothing of WHERE the hinge sits.
Three arms in parallel on the user's final recipe (`config/joint4_decoder_l2anchor.yaml`, L2 2pi +
axis 0.5 on SF3D, analytic decoder, joint4 data): `query` (option 2), `attnpool` (option 1),
`mlp1024` (capacity control). Spec: docs/superpowers/specs/2026-09-12-articulation-readout-design.md.

**Comparison row.** `20260912_joint4_decoder_l2anchor` (MA 31.05 / 30.27, rot flips 19.0, origin 0.294,
mIoU 0.241 / PDet 18.5, HOI4D 0.576; ARCTIC probe: axis 48.2 deg, flips 26.4 %, hinge-line offset 0.070).

**Setup.** `config/joint4_decoder_l2anchor_query.yaml`, `run_joint4dec_l2anchor_query_chain.sh` (stage,
train 20 ep, best ckpt, SF3D test head + writer length, per-source tests), pod jdec-query.
Judged by SF3D origin / rot flips / MA and the ARCTIC hinge probe (`tools/arctic_axis_probe.py`).

**Result.** (pending)
