# 20260910_joint4_dct_rgb_scalefree — JOINT 2D+3D training: SF3D + HOI4D + EPIC + ARCTIC, one stream

**Question:** user design 2026-09-10. Instead of 2D pretraining then SF3D post-training (the chain: MA 31.01 plain / 31.13 depth record), train the RGB-only scale-free DCT-6 model ONCE on all four sources in a seeded stream of source-homogeneous batches: SF3D batches scored with the 3D recipe (normalized trajectory with scale gt_z0, point-3D, origin, axis, type, consistency), hand-video batches with the 2D teacher-forcing recipe (projection 0.5 on the unit anchor, consistency, p_rev prior, no 3D-GT terms). Does joint training match the chain's SF3D numbers while keeping the 2D sources (the chain forgets HOI4D: mIoU 0.727 -> 0.12)?

**Recipe:** `config/joint4_dct_rgb_scalefree.yaml`. SF3D 54,086 raw train records, no augmentation; hand sources augmented x10 (HOI4D 26,250 / EPIC 3,060 / ARCTIC 22,300 views, HOI4D never mirrored) -> 105,696 samples/epoch, 1,651 steps at batch 64. lr 2e-5 (first guess), 20 epochs, milestones [16, 19]. Checkpoint monitor `val/sf3d/loss_total`. Pod E (RTX PRO 6000 Workstation, healthy clocks), 4 h 45 min incl. tests. Tested: SF3D via the scratch DCT config (pred_z_p + gt_z0, identical as always), hand sources via the single-source configs (dct 6).

**Result:** best `val/sf3d/loss_total` **0.9688 at epoch 12** — the lowest SF3D val loss of ANY arm, RGB or depth (depth post-train tf_plain 0.9768, depth scratch g19_dct 0.9652 is the only lower one; RGB chain arms 1.048-1.095). SF3D test (5,088): **MA 31.41 / signed 30.15 — NEW ALL-TIME MA RECORD** (prev 31.13, depth DCT chain), PDet 22.72, **mIoU 0.2738 (record; prev 0.2685)**, axis all **26.1°** / matched 20.1°, origin 0.327 m, point3d 0.285 m, rot flips 13.6, **traj_dir 96.4 (ties the g19_fdiff record 96.1)**, roughness 0.0090.

| model | depth | MA | PDet | mIoU | axis all / matched | origin | point3d | traj_dir | rough |
|---|---|---|---|---|---|---|---|---|---|
| depth chain record (HOI4D tf DCT -> DCT) | yes | 31.13 | 23.27 | 0.266 | 28.0 / 19.7 | — | — | — | 0.0079 |
| RGB chain, plain multi3 -> plain | no | 31.01 | 22.72 | 0.2625 | 27.3 / 20.8 | 0.317 | 0.270 | 88.9 | 0.068 |
| RGB chain, plain HOI4D -> DCT | no | 30.07 | 22.35 | 0.2660 | 26.9 / 22.0 | 0.293 | 0.253 | 92.9 | 0.0089 |
| RGB scratch DCT | no | 24.88 | 17.39 | 0.2425 | 29.6 / 22.0 | 0.324 | 0.269 | 92.0 | 0.011 |
| **joint4 DCT (this)** | **no** | **31.41** | 22.72 | **0.2738** | **26.1** / 20.1 | 0.327 | 0.285 | **96.4** | 0.0090 |

Hand sources with the SAME checkpoint (held-out splits; reference = the 2D-only DCT arm `20260910_multi3_dct_rgb_scalefree`):

| split | joint4 mIoU / PDet / shape | 2D-only DCT arm | chain post-trained models |
|---|---|---|---|
| HOI4D | 0.676 / 85.2 / 0.0343 | 0.753 / 89.8 / 0.0337 | ~0.12 mIoU (forgotten) |
| EPIC (53) | 0.305 / 15.1 / 0.091 | 0.520 / 58.5 / 0.090 | — |
| ARCTIC | 0.618 / 68.7 / 0.083 | 0.694 / 79.6 / 0.085 | — |

Val curves: SF3D descends steadily to 0.969 (ep 12) and plateaus ~1.0 after the lr step; HOI4D val improves to 0.285-0.31; EPIC (0.80 -> 1.2) and ARCTIC (0.60 -> 0.82) RISE from epoch ~3 — the two small sources overfit at x10 while SF3D is still learning.

**Reading:** (1) Joint training beats the sequential chain on SF3D on the headline metrics (MA +0.4 over the depth record, mask record, direction record) — the 2D sources act as a regulariser/pretraining for SF3D without a hand-over, and there is no init to re-initialise. (2) It KEEPS the hand ability the chain loses: HOI4D 0.676 mIoU / 85 PDet from one model that also holds the SF3D record, vs 0.12 after post-training. The cost vs a dedicated 2D model is 0.08 mIoU / 5 PDet on HOI4D, more on the small sources (EPIC collapses to 0.30 on 53 records; ARCTIC -0.08 / -11), which is exactly what their rising val curves predicted: x10 views is too many for EPIC/ARCTIC at 20 epochs. (3) Geometry (origin, point3d) is at the RGB-chain level, still behind depth. (4) lr 2e-5 worked on the first try; the tuning to do is the BALANCE (fewer views for the small sources, e.g. x4-x6, or a per-source early checkpoint), not the lr.

**Decision:** joint4 = the new reference model (MA record, single model for all four sources). Next: balance sweep (hand repeat 4-6), a second seed, and lr 1e-5 / 3e-5 as the user asked for lr tuning. Ckpt best-epoch12-sf3dval0.9688.ckpt on the volume. vis: viz/20260910_sf3d_joint4_panels.
