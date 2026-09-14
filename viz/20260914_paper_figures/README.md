# Composed qualitative figures for the paper (Figs. 4-6)

Built on the dev pod from the three batches `20260914_fig4_sf3d_gt_baseline_ours`,
`20260914_handvideo_control_vs_dense`, `20260914_iphone_control_vs_dense` (headers cropped, rows resized,
column labels added; the compose code is in STATE.md's night-3 log / `tools/compose_panels.py` variant):
`fig_sf3d_qual.png` (2 x [GT | OPDFormer-C | ARTHUR] x 2 rows), `fig_handvideo_qual.png` (rows HOI4D / EPIC /
ARCTIC, 2 examples each, GT | SceneFun3D only | ARTHUR), `fig_wild_qual.png` (door, laptop: photo |
SceneFun3D only | ARTHUR). Copied to the Overleaf project as `figures/fig_*_qual.png` (2026-09-14).

**Fig. 4 replaced (2026-09-14 evening):** `fig_sf3d_qual_v3.png` = the user's picks val 1684 / 113 / 3726 / 403 in the
Fig. 3 style, rendered from prediction dumps by `tools/viz_fig4_panels.py` (batch `20260914_fig4v3_style`, baseline
OPDFormer-C at 512), now the Overleaf `figures/fig_sf3d_qual.png`. `fig_sf3d_qual.png` (old samples, old style) and
`fig_sf3d_qual_v2.png` (new samples, old style) are kept for reference.

**Fig. 4 again (2026-09-14 late):** `fig_sf3d_qual_v4.png` = one frame per row, eight columns (GT, OPDFormer-C 512,
OPDFormer-P 512, MOPD, USDNet, A3VLM chain, 3DOI, ARTHUR; MOPD / USDNet / 3DOI grey placeholders until their runs
land), ARTHUR's motion over the GT extent; batch `20260914_fig4v4_all_baselines`. Now the Overleaf
`figures/fig_sf3d_qual.png` (commit b1b1b03).
