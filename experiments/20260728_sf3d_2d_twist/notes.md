# 20260728_sf3d_2d_twist

**Goal:** validate the prediction-anchored 2D supervision path on SF3D, where
full 3D GT exists to judge it — before any video mining is built on it. Adds
the 2D track head and the screw loss's track term (predicted twist's orbit
through the model's OWN anchor, projected, vs the observed 2D track;
no anchor_depth from the data pipeline).

**Setup:** config.yaml (= config/sf3d_train_runpod_2d_twist.yaml). Same as
20260728_sf3d_twist plus use_2d_trajectory_head, trajectory_2d_weight, and
return_trajectory_2d (activates L_screw_track). screw_omega_shrink stays 0
here (twist L2 pins omega); video pretraining must set it > 0.

**Compare against:** 20260728_sf3d_twist (does 2D supervision help or hurt
the 3D metrics?) and 20260728_sf3d_2d (legacy data-anchored 2D arm, not yet
run).

**Update 2026-08-02:** relaunched with `use_motion_type_input: true`
(GT type as auxiliary input, 50% conditioning dropout, val/test always
hint-free). The earlier partial run predates this architecture change
and is not comparable (its 6-epoch checkpoints live in
`checkpoints_pre_typein/`).

**Result:** best val 1.0906 at epoch 15 — improved EVERY epoch, no overfit
bend; undertrained at 16 epochs (opposite of the twist arm). Test (43,870
samples, hint-free): type 95.1%, MA 40.7%, axis 26.9° (16.2° matched), mIoU
0.093 / PDet 3.7%. Twist head decoded: type-from-|ω| 75.4%, axis 34.9°,
GT-origin→axis-line 2.63 m — **beats the twist arm on every twist metric**
(+7.2pp type, −4° axis, −37% line distance) and slightly on masks. The
prediction-anchored 2D-track supervision helps 3D articulation.

**Decision:** the 2D arm is the stronger recipe — extend it (resume from
last.ckpt or retrain longer) before any video-mining investment; profile its
~2× step cost if it becomes the default. Val-loss numbers are NOT comparable
across arms (different loss terms); compare on test/twist_* only.
