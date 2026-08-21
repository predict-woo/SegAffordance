# 20260822_sf3d_g17_2d_dct — 2D arm + truncated-DCT trajectory head

**Recipe:** the arm-A-detach 2D-only config
(`20260820_sf3d_g17_2d_detach`) + the g19 DCT trajectory head
(`trajectory_dct_coeffs: 6` — the head emits 6 DCT coefficients per axis,
decoded to 20 points through a fixed orthonormal IDCT). Question: does the
architectural smoothness that won on 3D (g19_dct) transfer to the 2D-only
path? 30 epochs, best = epoch 19 (val 1.3702; val plateau 17–19).

## Test (5,088) vs arm A (2D baseline) / +uv-fdiff / g17 (3D reference)

| metric | arm A (detach) | +uv-fdiff | **+DCT** | g17 (3D) |
|---|---|---|---|---|
| proj2d total / anchor / shape | 0.175 / 0.129 / 0.0997 | 0.170 / 0.134 / 0.107 | **0.165 / 0.109 / 0.0947** | 0.155 / 0.09 / 0.11 |
| mIoU / PDet | 0.242 / 16.1 | 0.237 / 15.8 | **0.2655 / 19.4** | 0.265 / 20.6 |
| 2D point | 0.117 | 0.108 | **0.0938** | 0.095 |
| traj_dir acc / cos | 87.6 / 0.524 | 89.9 / 0.563 | **91.3 / 0.626** | 94.9 / 0.811 |
| traj roughness (m) | — | 0.177 | **0.0319** | 0.091 |
| axis (all) | 63.5° | 62.1° | 59.3° | 25.8° |

## Reading

**Best 2D arm on every trunk metric — the new 2D recipe.** Shape 0.0947
beats even the 3D-supervised g17 (0.11); masks reach the 3D level (mIoU
0.2655 vs 0.265, PDet within 1.2); 2D point matches (0.0938 vs 0.095);
roughness is 5.5× better than the fdiff arm and ~3× better than the 3D
baseline. The DCT basis helps the 2D path MORE than it helped 3D,
plausibly because the projection loss is noisier supervision than 3D
points and the low-dimensional basis absorbs that noise instead of
letting it reach the trajectory.

Caveat: `origin_err_m` / `radius_err_m` explode (41 m / 19.6 m vs ~1.4 /
0.44 in the other 2D arms). Articulation is unsupervised-and-deadlocked
in ALL 2D arms (those numbers were already garbage); the smooth
trajectories appear to let the L_pp circle fit drift to near-degenerate
huge-radius solutions. Irrelevant for pretraining — articulation heads
relearn under 3D supervision — but do not read those columns as signal.

**Adopted as the p90 pretrain recipe** for the label-efficiency
experiments (spec 2026-08-22-2d-pretrain-label-efficiency-design.md);
uv-fdiff is not adopted (see 20260822_sf3d_g17_2d_fdiff2d).

test pass: `test.log` (ckpt best-epoch19-valloss1.3702)
spec: docs/superpowers/specs/2026-08-21-smooth-trajectory-g19-design.md (port section)
