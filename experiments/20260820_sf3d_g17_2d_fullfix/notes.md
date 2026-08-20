# 20260820_sf3d_g17_2d_fullfix — ARM B: anchor detached + UNNORMALIZED MSE

**Recipe:** arm A + `trajectory_proj_normalized: false` (plain masked uv²
MSE, the twist-era form). 30 epochs, best = epoch 12 (val 0.3621 — not
comparable across arms, different loss composition).

## The three-way answer (test, 5,088 samples)

| metric | g17-2d (broken) | ARM A (detach only) | ARM B (full fix) | g17 (3D) |
|---|---|---|---|---|
| proj2d TOTAL | 0.235 | 0.175 | **0.174** | 0.155 |
| proj2d anchor | 0.19 | 0.129 | **0.108** | 0.09 |
| proj2d SHAPE | 0.16 | **0.0997** | 0.128 | 0.11 |
| mIoU / PDet | 0.057 / 1.1 | 0.242 / 16.1 | **0.260 / 19.6** | 0.265 / 20.6 |
| 2D point | 0.307 | 0.117 | **0.100** | 0.095 |
| type acc | 75.2 | 75.4 | **79.0** | 95.3 |

## Where the improvement came from

1. **The collapse fix is ENTIRELY the detach** (arm A already recovers
   masks/point/anchor; the diagnosis was right: the depth-edge anchor
   gradient was the destroyer).
2. **The normalization choice is a genuine trade-off, not a fix:**
   - KEEPING it (arm A) gives the best trajectory SHAPE (0.0997 —
     matches the 3D-supervised g17): per-row equalization gives
     small-motion rows a fair gradient share.
   - DROPPING it (arm B) further calms the trunk → best masks (mIoU
     0.260, nearly g17), best point (0.100), best anchor, best label-free
     type (79.0) — but shape degrades to 0.128 (big-motion rows dominate
     the plain MSE).
   - proj2d TOTAL is a wash (0.175 vs 0.174) — the two arms spend the
     same budget on different components.
3. Articulation stays weak in both (axis ~60°, origin dark) — the L_pp
   limitation is orthogonal to this fix pair.

**Recommendation recorded:** if 2D-only continues, the natural combination
is detach + normalized-with-cap (or a normalization warmup) to get arm A's
shape with arm B's trunk health. As a PRETRAINING candidate, arm B is now
plausible (masks nearly intact + best point) where the original g17-2d was
poison.
