# 20260728_sf3d_twist

**Goal:** first training run of the unified screw-motion (se(3) twist)
parameterisation on the rebuilt SF3D dataset (sf3d_processed_v2). One
6-vector (omega, v) covers both motion types; motion type is emergent from
|omega|; the interaction point is retargeted to the element centroid
(point_source="element") since the twist carries the hinge as a LINE.

**Setup:** config.yaml (= config/sf3d_train_runpod_twist.yaml). Twist head +
sign-agnostic twist L2 + "screw" geometric consistency (GT-trajectory term +
prediction-anchored self term). Baseline heads (axis/type/CVAE) train
alongside for comparability. RTX PRO 6000 training pod, batch 64.

**Compare against:** 20260726_sf3d_geo_crossgt (baseline parameterisation,
not yet run) and 20260728_sf3d_2d_twist. Twist metrics: test/twist_type_acc,
test/twist_axis_err_deg, test/twist_pass_rate_ma, test/twist_axis_line_dist_m.
NOTE test/mean_point_error and mean_origin_error_m are NOT comparable with
motion_origin runs (different point GT).

**Update 2026-08-02:** relaunched with `use_motion_type_input: true`
(GT type as auxiliary input, 50% conditioning dropout, val/test always
hint-free). The earlier partial run predates this architecture change
and is not comparable (its 6-epoch checkpoints live in
`checkpoints_pre_typein/`).

**Result:** best val 0.9891 at epoch 4, then clear overfit (peak 1.094 ep8;
LR drop at 13 stabilised ~1.02 without reclaiming the minimum). Test (43,870
val samples, hint-free): type 95.1%, MA 40.0%, axis 26.2° (18.9° matched),
mIoU 0.083 / PDet 2.9% (masks are tiny at 256² — median GT ~14 px; needs a
baseline arm before reading much into it). Twist head decoded: type-from-|ω|
68.2%, axis 38.9°, GT-origin→axis-line 4.18 m — the twist head underperforms
the legacy axis/type heads in this recipe.

**Decision:** future SF3D arms should drop the LR milestone much earlier
(~ep4-6) or add early stopping; run the geo_crossgt baseline for mask/axis
context; twist-head loss weights deserve a sweep before judging the
parameterisation itself.

vis: viz/20260803_sf3d_twist_vs_2d_twist_panels (GT vs both arms, 16 val samples)
