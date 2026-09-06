# 20260907_sf3d_g19_dct_ft_hoi4d — SF3D 3D-DCT post-training from the HOI4D v2 winner

**Question:** does pretraining on real hand video (HOI4D v2, 3,075 windows, VLM masks + knuckle tracks; 20260906_hoi4d_2d_v2_e100_lr3e5) help the full-SF3D 3D pipeline? Same architecture (split axis heads, type head, DCT trajectory head), so the init is a straight state-dict load.

**Recipe:** identical to 20260821_sf3d_g19_dct (30 ep, lr 1e-5, milestones 24/28, batch as g19) + `finetune_from_path` = the HOI4D checkpoint. Comparison baseline: g19_dct from scratch (val 0.9652 @ ep 20; mIoU 0.2685, PDet 21.72, roughness 0.0090).

**Result:** (pending)
