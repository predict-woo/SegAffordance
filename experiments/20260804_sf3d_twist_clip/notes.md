# 20260804_sf3d_twist_clip

**Goal:** the twist arm retrained after the 2026-08-03/04 fix round:
no CVAE (MotionMLP), delta-cumsum trajectory heads (point 0 == 0, the
interaction point anchors), sign-SENSITIVE twist loss (stored sign is
canonical — verified 12/12 in viz/20260804_sf3d_gt_twist_check), plus the
speed stack (fast_pipeline, /dev/shm LMDBs, channels_last, compile).
NOT comparable to 20260728_sf3d_twist (architecture + losses changed).

**Setup:** config.yaml (= config/sf3d_train_runpod_twist.yaml @ launch).
Counterpart of 20260804_sf3d_twist_dinov3 — backbone is the only diff.

**Result:** (pending)

**Decision:** (pending)
