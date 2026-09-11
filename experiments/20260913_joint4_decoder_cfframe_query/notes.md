# 20260913_joint4_decoder_cfframe_query — joint decoder arm `cfframe_query`

**Goal.** query readout (4 queries x 2 layers) on the MA-record cf_frame 2:1 SF3D side — does the readout help the record recipe and fix its hand-video sign?

**Setup.** Derived from `config/joint4_decoder_cfframe.yaml` (exp `20260912_joint4_decoder_cfframe`) with overrides `model_params.articulation_readout=query model_params.readout_queries=4 model_params.readout_layers=2 model_params.readout_dim_ffn=1024 model_params.readout_mask_eps=0.01`; `config/joint4_decoder_cfframe_query.yaml`, `run_joint4dec_cfframe_query_chain.sh`, pod jdec-cfframe_query.

**Result.** (pending)
