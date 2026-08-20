# 20260820_sf3d_g17_2d_detach — ARM A: anchor detached, normalization kept

**Recipe:** the g17-2d config + `trajectory_proj_detach_anchor: true` ONLY
(2026-08-20 post-mortem; slide 2026-08-20_g17_2d_proj_anchor_issue.html).
Gradients stop at the projection loss's anchor (point_uv + sampled depth);
the normalized per-row projection ratio is unchanged. 30 epochs, best =
epoch 24 (val 1.5182).

## Test (5,088) vs the broken g17-2d baseline / g17 (3D reference)

| metric | g17-2d (broken) | **ARM A** | g17 (3D) |
|---|---|---|---|
| proj2d total / anchor / shape | 0.235 / 0.19 / 0.16 | **0.175 / 0.129 / 0.0997** | 0.155 / 0.09 / 0.11 |
| mIoU / PDet | 0.057 / 1.1 | **0.242 / 16.1** | 0.265 / 20.6 |
| 2D point | 0.307 | **0.117** | 0.095 |
| type / axis(all) | 75.2 / 58.7° | 75.4 / 63.5° | 95.3 / 25.8° |

## Reading

The detach alone un-collapsed the model: masks back to within ~0.02 mIoU
of the 3D arm, and the trajectory SHAPE error (0.0997) now MATCHES the
fully-3D-supervised g17 (0.10–0.12) with zero 3D labels — the projection
loss teaches shape properly once its gradients stop routing through the
depth-edge anchor. Articulation (axis/type/origin) unchanged-bad, as
expected: that is the separate L_pp-is-too-weak problem.

vis: (three-way table in the fullfix arm's notes)
