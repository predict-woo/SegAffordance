# 20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d — SF3D 3D-DCT post-training from the RGB-only scale-free HOI4D arm

**Question:** the RGB-only counterpart of `20260907_sf3d_g19_dct_ft_hoi4d_tf_plain` (MA 30.62 / PDet 20.03 / mIoU 0.2555, depth input, metric head): with NO depth anywhere and the scale-free head (metres = GT z0 · Δ̃ at train, z_p · Δ̃ at test), how much of the HOI4D→SF3D transfer survives, and what does depth buy? Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md.

**Recipe:** `config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml` = the g19_dct recipe (30 ep, lr 1e-5, DCT-6 head) with `use_depth false`, `load_depth false`, `trajectory_scale_free true`, `trajectory_scale_source "gt_z0"`, `test_trajectory_scale "pred_z_p"`; `finetune_from_path` = `20260909_hoi4d_2d_v2_rgb_scalefree` best-epoch83 (plain head → DCT head: everything but `trajectory_predictor.{trajectory_head.*, idct_m}` loaded, as in the tf_plain run). Pod A (RTX PRO 6000 Server), 3 h 35 min after the 48-min HOI4D arm on the same pod. Tested twice: pred_z_p (logs/test.log) and gt_z0 oracle scale (logs/test_gt_z0.log) — identical on every reported metric (none of them reads the trajectory scale; see the scratch arm's notes).

**Result:** best val/loss_total **1.0483** (ep 24). Test (5,088): MA **30.07** / signed 29.93, PDet **22.35**, mIoU **0.2660**, point err 0.1031, point3d 0.253 m, origin 0.293 m, axis all **26.9°** / matched 22.0°, rot flips 14.3, traj_dir 92.9, roughness 0.0089.

| init | depth | MA | PDet | mIoU | axis all / matched | origin | point3d |
|---|---|---|---|---|---|---|---|
| scratch g19_dct | yes | 25.98 | 21.72 | 0.2685 | 25.3 / 18.2 | 0.276 | 0.259 |
| scratch rgb_scalefree | **no** | 24.88 | 17.39 | 0.2425 | 29.6 / 22.0 | 0.324 | 0.269 |
| HOI4D tf_plain | yes | 30.62 | 20.03 | 0.2555 | 28.0 / 20.0 | 0.286 | 0.224 |
| **HOI4D rgb_scalefree (this)** | **no** | **30.07** | **22.35** | **0.2660** | **26.9** / 22.0 | 0.293 | 0.253 |
| HOI4D tf (DCT), record | yes | 31.13 | 23.27 | 0.266 | 28.0 / 19.7 | — | — |

**Reading:** (1) The HOI4D→SF3D transfer survives the removal of depth completely: +5.2 MA, +5.0 PDet, +0.024 mIoU over the RGB scratch arm (the depth pair gained +4.6 / −1.7 / −0.013), so the RGB-only init is, if anything, a cleaner transfer (masks and detection improve too, which they did not with depth). (2) Against its depth counterpart it is −0.55 MA (single-seed noise level), +2.3 PDet, +0.01 mIoU, −1.1° axis-all, and 0.7 cm / 3 cm worse on origin / point3d. (3) The one clear cost of no depth is the trajectory term: val L_trajectory (normalized, on GT z0 · Δ̃) plateaus at 0.48 from epoch 3 while the depth arm keeps improving to 0.35 — both reach train 0.009, so this is generalization of the metric curve shape, not fitting. The trajectory-derived test metrics (direction 92.9, roughness 0.0089, proj2d) do not see it, and MA does not read the trajectory at all. (4) The depth cost measured on scratch (−1.1 MA / −4.3 PDet) largely disappears once the model is HOI4D-initialized; the 2D pretraining is a substitute for the depth channel on the mask / detection side.

**Decision:** RGB-only scale-free is now the model for the whole line (user decision) at essentially no cost on SF3D's headline metrics; the trajectory val gap (0.48 vs 0.35) is the thing to watch — candidates: the DCT teacher-forcing HOI4D arm as init (depth counterpart is the 31.13 record; expect the same +0.5 MA / +3 PDet head bonus), longer post-training, and EPIC/ARCTIC in the pretraining mix (ARCTIC has exact z0 → could also feed `gt_z0` at train). Single seed. Ckpt best-epoch24-valloss1.0483.ckpt on the volume. vis: viz/20260909_sf3d_rgb_scalefree_vs_depth_panels.
