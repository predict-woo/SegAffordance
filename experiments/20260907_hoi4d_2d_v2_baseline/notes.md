# 20260907_hoi4d_2d_v2_baseline

**Goal:** does the DCT basis help on real hand tracks? Plain per-point 2D head (`trajectory_dct_coeffs: 0`), everything else = dct_baseline (normalized loss KEPT, detach KEPT). All three arms: HOI4D v2 depth-complete data (3,084 one-per-window records, every record with sensor depth so the projection term covers all 13 categories), lr 3e-5, 100 epochs, batch 64, milestones 80/92 — the sweep winner's schedule. The sweep's own e100_lr3e5 had its trajectory term active on furniture windows only (depth-less rows skipped).

**Result:** (pending)
