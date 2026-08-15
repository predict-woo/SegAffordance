# 20260816_sf3d_g10_lpp_val93_3d — val93 consistency in 3D, gen-10

Same construction as viz/20260815_sf3d_g9_lpp_val93_3d (predicted
trajectory vs the orbit implied by the model's own articulation heads),
re-rendered with the gen-10 checkpoint (best-epoch24). npz dumps for val93
AND val1684 from tools/diag_lpp_samples.py --dump-npz.

Interpretation: radial RMS 0.124 -> 0.050 m — the trajectory hugs the
predicted circle (per-point radius 0.32 -> 0.24 m vs r_hat 0.32) instead of
spiraling inward; raw L_pp 0.0223 -> 0.0042. Visible trade: predicted sweep
length halved (net extent 0.41 -> 0.19 m) — part of the consistency gain is
under-sweeping.
