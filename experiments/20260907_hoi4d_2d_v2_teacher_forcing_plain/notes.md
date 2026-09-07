# 20260907_hoi4d_2d_v2_teacher_forcing_plain

**Goal:** plain per-point 2D head (`trajectory_dct_coeffs: 0`) with the teacher-forced anchor (`trajectory_proj_anchor: gt_point`, `depth_anchor_source: gt_point`); normalized loss, lr 3e-5, 100 ep, depth-complete v2 data — completes the {DCT, plain} x {detach, teacher forcing} grid with dct_baseline (0.720/86.5/0.0379), baseline (0.716/88.2/0.0398), teacher_forcing (0.727/88.0/0.0377).

**Result:** (pending)
