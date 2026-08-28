# 20260828_sf3d_cf_h1only — the closed-form fdiff: H1 derivative only (IN FLIGHT)

**Question:** the fdiff family's continuous limit is Sobolev H1 — the
velocity term closes to the derivative quadratic; angle/length have no
closed form. How much of the closed form's gain (MA 29.19, origin 0.250)
is the derivative term alone? Does dropping the position quadratic (the
cf pair's only absolute-lever anchor) cost origin/MA, or sharpen matched
axis the way fdiff-style supervision has before?

**Recipe:** exactly 20260828_sf3d_closedform with
`closed_form_trajectory_weight 0.5 -> 0.0` and
`closed_form_velocity_weight 0.5 -> 1.0` (= the fdiff family's velocity
weight). Axis loss ON (vae_weight 0.5). No trajectory head, no L_pp,
30 epochs, seed 42. `L_cf_position` still logs unweighted.

**Watch vs closedform:** MA 29.19 / origin 0.250 / matched 22.3° /
flips 10.7/13.7. And vs cf_noaxis (27.71 / 0.253 / 17.6° / 10.5/15.9).

Result: PENDING
