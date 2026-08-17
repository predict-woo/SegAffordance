# 20260817_sf3d_g16_vs_g13_panels — the rot-collapse fix, before/after

16 val samples (seed 42421, family-standard picks), rows: GT | g13res512
(best-epoch27) | g16trajnorm (best-epoch21), 512-px inputs, v3 GT.

Interpretation: g13's rot panels show the collapse (magenta clumps of a
few cm at the anchor); g16's magenta arcs sweep the full extent, riding
both the cyan GT track and the model's own yellow orbit (08_rot_val1684:
full arc + ax 16°→7°). Trans rays essentially unchanged (never collapsed).
Matches the metrics: traj_dir 86.2→94.5, sweeps 0.04-0.07→0.44-0.79 m.
