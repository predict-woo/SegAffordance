# SceneFun3D validation model zoo: 27 checkpoints on the same 20 random val frames

`runpod/sf3d_zoo.sh` (dev pod, 2026-09-14): `tools/sf3d_vis_val.py --num 20 --seed 3` once per checkpoint
(the same 20 frames as `20260914_dense_final_sf3d_val_panels`), one directory per model with the
GT | prediction strips, then `grid_<n>_<rot|trans>_val<idx>.jpg` = GT panel + every model's prediction
panel, 7 columns, labelled (the burnt-in header of each panel carries type, p_rev, radius, z_p and the
SIGNED axis error). Models, in grid order: joint4_dct, joint4_dctv2, dct_chain (DCT-head era);
dec_base, dec_seed7, l2anchor, h1anchor, cfframe, cfframe_seed7, cfframe_a3, cfframe_query (decoder,
pooled readout, loss variants); attnpool, query, query_seed7, query_l4, query_eps01, query_pos,
query_w1024, mlp1024 (readout variants); dense, dense_seed7, dense_off, dense_d2d (dense voting);
sf3d_only (control), directloss, sampledtraj (loss ablations); field (all-fields model). Checkpoints =
each experiment's INDEX best; configs as in `runpod/sf3d_zoo.sh`.

**Baselines added (2026-09-15).** `baselines/` = the same 20 frames rendered by
`tools/sf3d_vis_baseline_vs_ours.py` from the baselines session's `preds.jsonl` files (oracle-matched
instance per element; drawn like the scorer scored them): OPDFormer-C (256 RGB-D), OPDFormer-P RGB-D,
OPDFormer-P RGB, OPDFormer-P 512 (resolution-matched), MOPD, USDNet, A3VLM text-only chain, A3VLM given the
GT box. The grids now end with these eight panels, labelled `BL <name>` in blue; "no overlapping
instance" marks elements the detector never covered. A3VLM's mask is its predicted 3D box hull.
