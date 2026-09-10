# 20260910_sf3d_g19_dctv2_rgb_scalefree_ft_multi3dctv2 — SF3D post-training from the multi3 DCT v2 arm

**Goal.** Same recipe as `20260910_sf3d_g19_dct_rgb_scalefree_ft_multi3dct` (30 ep, lr 1e-5,
RGB-only scale-free, DCT-6 at both stages, nothing re-initialised) with the DCT readout
conventions v2 at both stages: pinned start, shape/scale split, the 3D first-difference trio at half
the g19_fdiff weights (0.5 / 0.25 / 0.25 — the never-run gen-20 candidate "DCT head + fdiff") and a
log path-length loss (0.5). Init = the v2 2D arm's best checkpoint.

**Comparison rows.** DCT chain best-epoch24: MA 32.80 / signed 32.43, PDet 20.7, mIoU 0.254,
origin 0.256, roughness 0.0081. Joint4: MA 31.41, mIoU 0.2738.

**Status.** stage 2 of pod M's chain (`run_multi3dctv2_chain.sh`).
