# 20260907_hoi4d_2d_v2_teacher_forcing

**Goal:** does a clean GT anchor (teacher forcing) beat the detached predicted anchor? `trajectory_proj_anchor: gt_point`, `depth_anchor_source: gt_point`; DCT head kept; otherwise = dct_baseline. All three arms: HOI4D v2 depth-complete data (3,084 one-per-window records, every record with sensor depth so the projection term covers all 13 categories), lr 3e-5, 100 epochs, batch 64, milestones 80/92 — the sweep winner's schedule. The sweep's own e100_lr3e5 had its trajectory term active on furniture windows only (depth-less rows skipped).

**Result:** (pending)
