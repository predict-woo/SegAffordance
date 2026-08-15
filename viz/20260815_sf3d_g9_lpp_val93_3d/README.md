# 20260815_sf3d_g9_lpp_val93_3d — L_pp inconsistency in 3D (val93, oven door)

`val93_3d.png`: camera-frame 3D plot of the gen-9 joint model's predicted
trajectory (magenta) vs the orbit implied by its OWN articulation heads
(orange 90° arc / dashed full circle about the red axis through q̂), plus GT
trajectory (cyan) and GT axis (green). Gray spokes = per-point radius to the
predicted axis. Sample = viz panel `..._ablation_panels_b2/05_rot_val93.jpg`,
L_pp 0.0223 (val p97; ref distribution in tools/diag_lpp_samples.py output).

Data: `val93.npz`, dumped by
`tools/diag_lpp_samples.py --indices 93 --dump-npz <this dir>` (g9 ckpt
best-epoch23). Rendered locally (matplotlib not on pod venv) by the
scratchpad script plot_lpp_val93.py (twistenv).

Interpretation: the trajectory falls nearly straight down from the anchor,
cutting INSIDE the predicted circle — per-point radius shrinks 0.286 m ->
0.118 m before recovering to 0.163 m (radial RMS 0.124 m, the dominant L_pp
term), while axial drift stays small (0.041 m). The articulation heads
sketch a plausible hinge sweep; the trajectory head draws a plausible
door-closing drop; nothing forced them to agree — L_pp's weighted
contribution (~0.011 even on this p97 sample) is ~2 orders below the direct
terms.
