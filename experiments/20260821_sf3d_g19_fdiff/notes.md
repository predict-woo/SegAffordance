# 20260821_sf3d_g19_fdiff — ARM 2: first-difference trajectory losses

**Recipe:** g17 + segment-vector losses on the UNCHANGED head: velocity
(siMLPe, 1.0), angle (MADiff 1−cos, 0.5), length (MADiff L1, 0.5). Spec
2026-08-21. 30 epochs, best = epoch 29 (val 1.1780 — not comparable, new
loss terms in the total).

## The three-way answer (test, 5,088; GT roughness floor 0.0032)

| metric | g17 | g19-dct | g19-fdiff |
|---|---|---|---|
| traj ROUGHNESS (m) | 0.0915 | **0.0090** | 0.0509 |
| traj_dir acc / cos | 94.91 / 0.811 | 94.52 / 0.800 | **96.11 / 0.819 — project bests** |
| MA / MA_signed | 25.86 / 25.75 | 25.98 | **29.91 / 29.56 — records** |
| matched axis | 16.93° | 18.21° | 17.25° |
| 3D point (m) | 0.2315 | 0.2592 | **0.2270 — best** |
| 2D point | 0.0952 | 0.1048 | 0.0953 |
| mIoU / PDet | 0.2654 / 20.56 | **0.2685 / 21.72** | 0.2504 / 18.00 |
| type | 95.34 | 95.15 | 92.35 |
| origin q* / radius | **0.2569 / 0.1215** | 0.2764 / 0.1410 | 0.2961 / 0.1564 |

## Where the improvements came from

- **Smoothness is architectural, not loss-driven:** the DCT basis removes
  10× of the roughness (jitter is unrepresentable); the first-difference
  losses only 1.8× (they penalize noise but the head can still express
  it).
- **Direction quality is loss-driven:** fdiff's angle term pushed
  traj_dir to project bests (96.1/0.819) and — through the shared trunk
  and L_pp — MA to 29.9, a +4 record. The DCT bottleneck alone doesn't
  teach direction.
- Each arm has a tax: DCT pays small geometry dips (point/origin/radius);
  fdiff pays mask/type dips (mIoU −1.5, type −3.0 — three extra loss
  terms competing for the trunk).

**Recommendation recorded:** the arms are COMPLEMENTARY — DCT head +
fdiff losses is the natural gen-20 (smooth basis AND supervised segment
directions); if a single winner must ship today, g19-fdiff's checkpoint
is the best OVERALL articulation model (MA 29.9, traj_dir 96.1) and
g19-dct the best visual/smoothness model.
