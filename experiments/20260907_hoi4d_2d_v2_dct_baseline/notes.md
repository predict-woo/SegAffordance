# 20260907_hoi4d_2d_v2_dct_baseline

**Goal:** reference arm. DCT trajectory head (6 coeffs), normalized projection loss, detached predicted anchor. All three arms: HOI4D v2 depth-complete data (3,084 one-per-window records, every record with sensor depth so the projection term covers all 13 categories), lr 3e-5, 100 epochs, batch 64, milestones 80/92 — the sweep winner's schedule. The sweep's own e100_lr3e5 had its trajectory term active on furniture windows only (depth-less rows skipped).

**Result:** (pending)
