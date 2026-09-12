# 20260913_joint4_decoder_l2anchor_query_w1024 — joint decoder arm `l2anchor_query_w1024`

**Goal.** query readout + head width 1024 (spatial readout x capacity: the two effects that each bought MA)

**Setup.** Derived from `config/joint4_decoder_l2anchor_query.yaml` (exp `20260913_joint4_decoder_l2anchor_query`) with overrides `model_params.vae_hidden_dim=1024 model_params.trajectory_length_hidden=1024`; `config/joint4_decoder_l2anchor_query_w1024.yaml`, `run_joint4dec_l2anchor_query_w1024_chain.sh`, pod jdec-l2anchor_query_w1024.

**Result.** (pending)
