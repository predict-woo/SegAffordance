# 20260828_sf3d_cf_noaxis — closed form minus the direct axis loss (IN FLIGHT)

**Question:** the identifiability proof (2026-08-28 discussion) says the
closed-form Gram quadratics + 3D origin/point losses have their unique
global zero at the correct articulation WITHOUT the 1-cos axis term.
Does the landscape cooperate — or does removing the scale-free axis
anchor slow/roughen revolute axis learning (lever-scaled, origin-coupled
gradients only; antipodal saddle with no second term's noise to kick
off it)?

**Recipe:** exactly 20260828_sf3d_closedform (MA 29.19, origin 0.250)
with `vae_weight: 0.5 -> 0.0`. Everything else identical: no trajectory
head, no L_pp, cf position 0.5 + derivative 0.5, 30 epochs, seed 42.
`L_vae_total` still logs unweighted — passive axis-error watch.

**Watch:** rot flips + matched-axis vs closedform (10.7/13.7, 22.3°);
trans rows should be ~unchanged (cf trans term IS 2(1-cos), effective
weight 2.5x -> 2x).

Result: PENDING
