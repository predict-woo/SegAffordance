# 20260804_sf3d_twist_clip

**Goal:** the twist arm retrained after the 2026-08-03/04 fix round:
no CVAE (MotionMLP), delta-cumsum trajectory heads (point 0 == 0, the
interaction point anchors), sign-SENSITIVE twist loss (stored sign is
canonical — verified 12/12 in viz/20260804_sf3d_gt_twist_check), plus the
speed stack (fast_pipeline, /dev/shm LMDBs, channels_last, compile).
NOT comparable to 20260728_sf3d_twist (architecture + losses changed).

**Setup:** config.yaml (= config/sf3d_train_runpod_twist.yaml @ launch).
Counterpart of 20260804_sf3d_twist_dinov3 — backbone is the only diff.

**Result:** 16/16 epochs (epochs 0-2 on a power-throttled pod at 261
samples/s — migrated via last.ckpt resume to a healthy pod, ~428
samples/s; metrics.csv = the resumed portion, epochs 3+). Val improved
essentially to the end (best 1.1830 @ ep13, vs the old arm's overfit
after ep4). Hint-free eval on 43,870 val samples (best-ep13 / last-ep15,
near-identical): twist axis err 39.6/39.6 deg, twist type-from-|omega|
67.4/67.1%, twist_pass_rate_ma 23.0/22.8% (DINOv3 arm: 7.2/8.1%), axis
head err (matched) 14.4/13.7 deg, MA pass 40.5/39.7%, twist_dir_acc
64.4/64.8%, traj_dir_cos 0.43, traj_dir_acc 74.9/75.2%, mIoU
0.097/0.100, PDet 3.9/4.1%. Direction is now clearly learned (vs
chance-level in DINOv3), with the consistency terms still sin^2 in this
run — the 1-cos generation should push dir further. Eval logs:
logs/eval_best_tail.log, logs/eval_last_tail.log.

**Decision:** the fix round (no CVAE, delta-cumsum, sign-sensitive
twist) is validated — val no longer collapses early and every twist
metric beats the old generation. Use last.ckpt as the reference
checkpoint. Next lever: retrain with the 1-cos consistency (committed
after this launch) to push direction accuracy.

vis: viz/20260804_sf3d_gen2_panels + viz/20260804_sf3d_gen2_traj_points (3-arm comparison at last.ckpt)
