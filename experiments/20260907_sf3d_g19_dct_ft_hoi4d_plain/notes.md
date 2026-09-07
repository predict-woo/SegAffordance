# 20260907_sf3d_g19_dct_ft_hoi4d_plain — SF3D 3D-DCT post-training from HOI4D v2 baseline

**Question:** the first two post-training runs (20260907_sf3d_g19_dct_ft_hoi4d from dct_baseline: MA 29.76 / PDet 18.3; _ft_hoi4d_tf from teacher_forcing: MA 31.13 / PDet 23.27, record) showed the HOI4D init raises MA and that the init checkpoint matters. Do the PLAIN-head HOI4D arms transfer the same way?

**Recipe:** identical to 20260821_sf3d_g19_dct / the two ft runs (30 ep, lr 1e-5, save_top_k 1) + `finetune_from_path` = HOI4D v2 `20260907_hoi4d_2d_v2_baseline` best checkpoint (baseline (plain head, detached anchor); held-out mIoU/PDet/shape 0.716/88.2/0.0398). The plain 20-point trajectory head has no DCT counterpart: `load_finetune_weights` loads every tensor whose name and shape match, so the trunk, depth encoder, mask/point/z_p heads and the trajectory MLP body transfer and only the head's output projection re-initializes. Comparison: scratch g19_dct (val 0.9652; MA 25.98, mIoU 0.2685, PDet 21.72) and the two DCT-init runs above.

**Result:** (pending)
