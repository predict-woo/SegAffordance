# 20260910_multi3_dctv2_rgb_scalefree — multi-source 2D DCT arm with the DCT readout conventions v2

**Goal.** Same recipe as `20260910_multi3_dct_rgb_scalefree` (HOI4D v2 + EPIC v1 + ARCTIC v1,
RGB-only scale-free, DCT-6 head, x8 augmentation, 12 ep) with the four readout fixes from the
2026-09-10 literature sweep (`knowledge/2026-09-10_trajectory_head_synthesis_v2.md`, #2):
pinned first point (AC-only coefficients), shape/scale split (unit-path-length curve x softplus
scale), the uv-space first-difference trio at half the g19 weights (velocity 0.5 / angle 0.25 /
length 0.25) and a log path-length loss (0.5). Stage 1 of the v2 chain; stage 2 =
`20260910_sf3d_g19_dctv2_rgb_scalefree_ft_multi3dctv2`.

**Comparison row.** multi3 DCT best-epoch11 val 0.4161; per-source held-out HOI4D / EPIC / ARCTIC
in its notes.

**Status.** launched 2026-09-10 night (pod M, `run_multi3dctv2_chain.sh`, log `multi3dctv2_chain.log`).
