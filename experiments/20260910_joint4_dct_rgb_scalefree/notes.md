# 20260910_joint4_dct_rgb_scalefree — JOINT 2D+3D training: SF3D + HOI4D + EPIC + ARCTIC, one stream

**Question:** user design 2026-09-10. Instead of 2D pretraining then SF3D post-training (the chain that gave MA 31.01 / 31.13), train the RGB-only scale-free DCT-6 model ONCE on all four sources in a seeded stream of source-homogeneous batches: SF3D batches scored with the 3D recipe (normalized trajectory with scale gt_z0, point-3D, origin, axis, type, consistency), hand-video batches with the 2D teacher-forcing recipe (projection 0.5 on the unit anchor, consistency, p_rev prior, no 3D-GT terms). Does joint training match the chain's SF3D numbers while keeping the 2D sources (the chain forgets HOI4D: mIoU 0.727 -> 0.12)?

**Recipe:** `config/joint4_dct_rgb_scalefree.yaml`. Balance: SF3D ~53k raw train records, no augmentation; hand sources augmented x10 (5,161 x 10 = 51,610 views/epoch; HOI4D never mirrored) -> ~105k samples/epoch, ~1,640 steps at batch 64. lr 2e-5 (first guess between the 2D arms' 3e-5 and the post-training's 1e-5 — to be tuned), 20 epochs, milestones [16, 19]. Checkpoint monitor `val/sf3d/loss_total`. Per-source val losses logged as `val/<source>/loss_total`. Datamodule: datasets/multisource_datamodule.py (SourceBatchSampler); trainer: loss_profiles / source_profiles in train_OPDReal_better.py.

**Comparison rows:** chain plain multi3 -> plain SF3D: MA 31.01 / PDet 22.72 / mIoU 0.2625; chain DCT multi3 -> DCT SF3D: (pod D, pending); RGB scratch DCT 24.88 / 17.39 / 0.2425; HOI4D held-out of the multi3 2D arm 0.754 / 91.9.

**Result:** (pending)

**Decision:** (pending)
