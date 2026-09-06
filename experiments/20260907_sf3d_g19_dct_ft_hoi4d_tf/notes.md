# 20260907_sf3d_g19_dct_ft_hoi4d_tf — SF3D 3D-DCT post-training from the HOI4D teacher_forcing arm

**Question:** same as 20260907_sf3d_g19_dct_ft_hoi4d, but initialized from `20260907_hoi4d_2d_v2_teacher_forcing` (GT-anchored projection loss) instead of `dct_baseline` — run regardless of which HOI4D arm scores higher (user, 2026-09-06). The launched config.yaml records the checkpoint path (patched at launch).

**Recipe:** identical to 20260821_sf3d_g19_dct (30 ep, lr 1e-5) + `finetune_from_path`.

**Result:** (pending)
