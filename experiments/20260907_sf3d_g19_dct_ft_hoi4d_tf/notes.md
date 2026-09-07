# 20260907_sf3d_g19_dct_ft_hoi4d_tf — SF3D 3D-DCT post-training from the HOI4D teacher_forcing arm

**Question:** same as 20260907_sf3d_g19_dct_ft_hoi4d, but initialized from `20260907_hoi4d_2d_v2_teacher_forcing` (GT-anchored projection loss) instead of `dct_baseline` — run regardless of which HOI4D arm scores higher (user, 2026-09-06). The launched config.yaml records the checkpoint path (patched at launch).

**Recipe:** identical to 20260821_sf3d_g19_dct (30 ep, lr 1e-5) + `finetune_from_path`.

**Result:** best val 0.9794 (ep 25; scratch g19_dct 0.9652). Test (same protocol, 5,088): **MA 31.13 / signed 30.80** (scratch 25.98 / 25.83; previous ALL-TIME record cf_h1only 30.64 / 30.11) — a new MA record on the plain g19_dct recipe; PDet **23.27** (scratch 21.72; record g21_dct_dir 23.21), mIoU 0.2664 (0.2685), point 0.1003 (0.1048), roughness **0.0079** (0.0090). Costs: axis err all 28.0° / matched 19.7° (25.3 / 18.2), flips 12.4 (10.0), type pass_rate_m 91.9 (95.1), origin 0.293 (0.276), traj_dir 92.5 (94.5). Reading: real-hand-video pretraining shifts the model toward better mask/point/type-agnostic motion agreement (MA) and smoother curves, while the fine axis/origin geometry gets slightly worse — single seed, unconfirmed. Pod A deleted after the run; checkpoint best-epoch25-valloss0.9794.ckpt on the volume.
