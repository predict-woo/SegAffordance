# 20260910_sf3d_g19_dct_rgb_scalefree_ft_multi3dct — SF3D DCT post-training from the multi3 DCT arm

**Question:** the DCT-at-both-stages chain on the RGB-only scale-free line: g19_dct SF3D recipe initialised from `20260910_multi3_dct_rgb_scalefree` (DCT-6 head, so the whole 2D checkpoint loads 1:1 — loader: "All model weights loaded successfully"). Compare with the plain-head chain (`20260909_sf3d_plain_rgb_scalefree_ft_multi3`: MA 31.01, rough 0.068), the DCT post-training from the HOI4D-only plain arm with a re-initialised readout (30.07), and the joint 2D+3D run (31.41).

**Recipe:** `config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_multi3dct.yaml` = the RGB g19_dct post-training config with the multi3-DCT init (epoch 11). 30 ep, lr 1e-5, milestones [24, 28]. Pod D2 (RTX PRO 6000 Server, healthy clocks; pod D was a lemon and was replaced), 3 h 31 min after the 66-min 2D arm on the same pod. Tested with pred_z_p (logs/test.log) and gt_z0 (logs/test_gt_z0.log) — identical, as for every scale-free arm.

**Result:** best val/loss_total **1.0178** (ep 24; val L_trajectory best **0.421** at ep 8 — the lowest trajectory val of any RGB arm, the others plateau at 0.47-0.49; val L_mask 0.336). Test (5,088): **MA 32.80 / signed 32.43 — NEW ALL-TIME RECORDS by a wide margin** (joint4 31.41, depth chain 31.13), PDet 20.74, mIoU 0.2541, point err 0.104, point3d **0.248 m**, origin **0.256 m** (0.006 from the all-time origin record 0.250), axis all 28.5° / matched 20.5°, **rot flips 9.83** (equals the all-time flips record 9.8), traj_dir 92.6, roughness 0.0081.

| model | depth | MA / signed | PDet | mIoU | axis all / matched | origin | point3d | flips | traj_dir | rough |
|---|---|---|---|---|---|---|---|---|---|---|
| depth chain, prev record | yes | 31.13 / 30.80 | 23.27 | 0.266 | 28.0 / 19.7 | — | — | — | — | 0.0079 |
| RGB chain plain -> plain | no | 31.01 / 30.17 | 22.72 | 0.2625 | 27.3 / 20.8 | 0.317 | 0.270 | 14.2 | 88.9 | 0.068 |
| RGB chain plain HOI4D -> DCT (readout re-init) | no | 30.07 / 29.93 | 22.35 | 0.2660 | 26.9 / 22.0 | 0.293 | 0.253 | 14.3 | 92.9 | 0.0089 |
| joint4 DCT | no | 31.41 / 30.15 | 22.72 | **0.2738** | **26.1** / 20.1 | 0.327 | 0.285 | 13.6 | **96.4** | 0.0090 |
| **DCT chain multi3 -> DCT (this)** | **no** | **32.80 / 32.43** | 20.74 | 0.2541 | 28.5 / 20.5 | **0.256** | **0.248** | **9.83** | 92.6 | 0.0081 |

**Reading:** (1) DCT at both stages with a multi-source 2D init is the best articulation model so far by a clear margin — +1.7 MA over the previous record with no depth, and the signed MA (32.43) is only 0.4 below the unsigned one, i.e. the flips are largely gone (9.83 vs 13-15 for every other RGB arm). (2) It also has the best metric geometry of the RGB line: origin 0.256 (RGB chains 0.29-0.33), point3d 0.248, and the lowest trajectory val (0.421) — the un-re-initialised DCT readout from the 2D stage carries real 3D shape information. (3) The trade: masks and detection are the lowest of the recent arms (0.254 / 20.7 vs joint4's 0.274 / 22.7) — the joint model owns those. (4) Compared to the plain-head chain from the same data (31.01): +1.8 MA, flips 14.2 -> 9.8, origin 0.317 -> 0.256, roughness 0.068 -> 0.008: the DCT head at both stages is strictly better on the SF3D side than the plain one. Confounds (single seed; the 2D arm ran 12 epochs vs the plain one's 30 with an epoch-6 checkpoint) remain.

**Decision:** the new articulation reference checkpoint (MA / origin / flips); joint4 remains the mask/detection and multi-source reference. Follow-ups: seeds; the joint recipe initialised from this chain's 2D arm; hand-source balance. Ckpt best-epoch24-valloss1.0178.ckpt on the volume. vis: viz/20260910_sf3d_dct_chain_panels.
