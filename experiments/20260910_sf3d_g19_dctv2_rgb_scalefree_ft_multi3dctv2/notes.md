# 20260910_sf3d_g19_dctv2_rgb_scalefree_ft_multi3dctv2 — SF3D post-training from the multi3 DCT v2 arm

**Goal.** Same recipe as `20260910_sf3d_g19_dct_rgb_scalefree_ft_multi3dct` (30 ep, lr 1e-5,
RGB-only scale-free, DCT-6 at both stages, nothing re-initialised) with the DCT readout
conventions v2 at both stages: pinned start, shape/scale split, the 3D first-difference trio at half
the g19_fdiff weights (0.5 / 0.25 / 0.25 — the never-run gen-20 candidate "DCT head + fdiff") and a
log path-length loss (0.5). Init = the v2 2D arm's best checkpoint.

**Comparison rows.** DCT chain best-epoch24: MA 32.80 / signed 32.43, PDet 20.7, mIoU 0.254,
origin 0.256, roughness 0.0081. Joint4: MA 31.41, mIoU 0.2738.

**Result (2026-09-10 07:05 local, pod M `segaffordance-dctv2-m`, 3 h 30 min train + tests, pod
deleted — verified).** Best `val/loss_total` **1.2020 at epoch 15** of 30 (chain: 1.0178 at ep 24 —
not comparable, extra terms; val L_trajectory reached 0.407 at ep 4 vs the chain's best 0.421).
Init = `20260910_multi3_dctv2_rgb_scalefree` best-epoch07 (loaded 1:1, nothing re-initialised).

SF3D test (pred_z_p; the gt_z0 pass is identical on every metric):

| metric | DCT chain (ep 24) | **DCT v2 chain (ep 15)** | joint v2 (ep 13) |
|---|---|---|---|
| MA / signed | **32.80 / 32.43** | 32.31 / 31.60 | 32.94 / 32.08 |
| matched axis | 20.52° | 18.91° | **16.42°** |
| all-axis / signed-all | 28.51 / 34.17 | 27.37 / 33.92 | **25.24 / 32.01** |
| flips all / rot | 9.89 / **9.83** | **9.69** / 10.67 | 10.83 / 15.17 |
| origin / line | **0.256 / 0.230** | 0.301 / 0.268 | 0.355 / 0.320 |
| point 3D | **0.248** | 0.254 | 0.279 |
| mIoU / PDet | **0.254 / 20.7** | 0.233 / 15.7 | 0.250 / 19.1 |
| traj_dir acc / cos | 92.57 / 0.776 | 95.74 / 0.823 | **95.99 / 0.828** |
| roughness | 0.0081 | **0.0078** | 0.0096 |

**Reading.** (1) Trajectory quality improves where the new terms act: traj_dir +3.2 points, the
smoothest sweeps yet (0.0078), matched axis -1.6°, all-axis -1.1°. (2) The headline MA drops 0.5
(signed -0.8) — inside single-seed noise but not an improvement — and origin worsens by 4.5 cm.
(3) Masks / detection regress again (mIoU -0.021, PDet -5): the third arm in a row (2D v2 arm, joint
v2, this) showing the g19_fdiff signature — derivative supervision on the trajectory sharpens the
axis and pulls the trunk off the mask / heatmap channels. (4) Best epoch moved 24 -> 15.

**Verdict.** On the chain recipe the v2 conventions are NOT a net win: better direction/smoothness/
axis sharpness, worse MA, origin and masks. On the joint recipe they gave the best MA of any arm
(32.94) at the same mask cost. Since four ingredients changed at once, the next step is the
ablation the fdiff precedent suggests: v2 head (pin + scale split + log-length) with the derivative
trio OFF, and the trio at 0.25/0.1/0.1 — on the joint recipe, which is where v2 paid.

