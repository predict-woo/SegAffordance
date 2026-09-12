# 20260913_joint4_decoder_l2anchor_query_l4 — joint decoder arm `l2anchor_query_l4`

**Goal.** query readout with 4 layers instead of 2 (depth of the spatial readout)

**Setup.** Derived from `config/joint4_decoder_l2anchor_query.yaml` (exp `20260913_joint4_decoder_l2anchor_query`) with overrides `model_params.readout_layers=4`; `config/joint4_decoder_l2anchor_query_l4.yaml`, `run_joint4dec_l2anchor_query_l4_chain.sh`, pod jdec-l2anchor_query_l4.

**Result.** (pending)
