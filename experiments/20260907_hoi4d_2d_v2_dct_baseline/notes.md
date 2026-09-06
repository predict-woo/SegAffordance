# 20260907_hoi4d_2d_v2_dct_baseline

**Goal:** reference arm. DCT trajectory head (6 coeffs), normalized projection loss, detached predicted anchor. All three arms: HOI4D v2 depth-complete data (3,084 one-per-window records, every record with sensor depth so the projection term covers all 13 categories), lr 3e-5, 100 epochs, batch 64, milestones 80/92 — the sweep winner's schedule. The sweep's own e100_lr3e5 had its trajectory term active on furniture windows only (depth-less rows skipped).

**Result:** best val/loss_total **0.3662** (epoch 90); held-out (110 objects): mIoU **0.720**, PDet **86.5**, point err 0.0149, traj proj-2D shape 0.0379 (all categories now), traj_dir acc 48.1 (chance). reference on depth-complete data; = the sweep recipe with the projection term now on ALL categories. Three-way: dct_baseline 0.720/86.5/0.0379, baseline 0.716/88.2/0.0398, teacher_forcing 0.727/88.0/0.0377 — statistically tied.
