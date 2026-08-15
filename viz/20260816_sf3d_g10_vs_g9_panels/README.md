# 20260816_sf3d_g10_vs_g9_panels — normalized-L_pp before/after

16 val samples (seed 42421, same picks as every gen-9-family batch), rows:
GT | g9joint (best-epoch23) | g10norm (best-epoch24). Regen: the ablation-
panels command with these two --model triplets (see manifest.yaml).

Interpretation: gen-10's trajectories track their own predicted orbits —
08_rot_val1684 (the motivating inconsistency sample) now follows the yellow
orbit and the cyan GT arc instead of curling away; 05_rot_val93 curls along
the orbit but with a shorter sweep (the under-sweep trade documented in the
experiment notes).
