# 20260913_joint4_decoder_l2anchor_query_eps01 — joint decoder arm `l2anchor_query_eps01`

**Goal.** query readout with mask bias floor eps 0.1 (log 0.1 = -2.3 vs -4.6: the surroundings of the part — seam, frame — reachable for the origin query; placement hypothesis)

**Setup.** Derived from `config/joint4_decoder_l2anchor_query.yaml` (exp `20260913_joint4_decoder_l2anchor_query`) with overrides `model_params.readout_mask_eps=0.1`; `config/joint4_decoder_l2anchor_query_eps01.yaml`, `run_joint4dec_l2anchor_query_eps01_chain.sh`, pod jdec-l2anchor_query_eps01.

**Result.** (pending)
