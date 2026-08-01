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

**Result:** (pending)

**Decision:** (pending)
