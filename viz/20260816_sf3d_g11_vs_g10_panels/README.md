# 20260816_sf3d_g11_vs_g10_panels — origin local sample + v3 sweeps

16 val samples (seed 42421, the family-standard picks), rows: GT |
g10norm (best-epoch24) | g11origin (best-epoch16). GT drawn from
sf3d_processed_v3 (0.7 m trans sweeps), so trans GT tracks extend far
beyond the old 0.1 m stubs — g10's predictions still sweep ~0.1 m (its
training GT) while g11's match the 0.7 m scale (e.g. 09_trans_val1353).
Regen: family viz command with --data-root /workspace/datasets/sf3d_processed_v3
(see manifest.yaml).

Interpretation: g11 trans trajectories adopt the v3 scale cleanly;
origin rings/axes comparable to g10 with slightly better hinge placement
on rot rows (matches the -0.7 cm origin_err). Mask overlays visibly
thinner on some g11 panels (the -1.1 pt mIoU).
