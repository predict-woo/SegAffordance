# Label efficiency v2: DCT head + screw-direction term — design

**Date:** 2026-08-23
**Status:** commissioned by the user ("run the pretraining experiment again
with this loss fix + the dct head... we have to run the baseline too").

## What changed since v1 (2026-08-22 spec)

1. **New loss term:** `pred_pred_art_dir_weight` — midpoint screw-direction
   consistency in L_pp (sign-aware; unit-tested; GT right-hand convention
   verified in tests/test_gt_sign_convention.py). v1 ran without it.
2. **DCT head everywhere:** v1's C baseline (g17) had the plain trajectory
   head while ft10 kept the DCT head — a recipe mismatch. v2 puts
   `trajectory_dct_coeffs: 6` in ALL arms.
3. **Pretrain undertraining confound killed:** the v1 p90 early-stopped at
   epoch 6 with its trunk 14% below the full-data 2D arm. v2's pretrain
   runs a FIXED 30 epochs (the schedule the full-data DCT arm validated),
   no early stopping.

## The common recipe: "g21" = g17 + trajectory_dct_coeffs 6 + dir term

`pred_pred_art_dir_weight: 0.1` — same scale as `pred_pred_art_weight`
(both dimensionless; the dir term joins L_pp as its oriented complement).
In 3D arms it routes trajectory sign into the axis heads next to the GT
axis loss; in the 2D pretrain (where L_pp at 0.1 is the sole articulation
teacher) it is the first sign teacher the 2D line has ever had.

## Arms (all on the g21 recipe, standard val/test, same 90/10 scene
partition as v1 — seed 4242, verified identical split)

| arm | experiment | data | epochs |
|---|---|---|---|
| **C' — new 3D baseline** | 20260823_sf3d_g21_dct_dir | 100% | 30 fixed (g19_dct schedule) |
| **B'1 — 2D pretrain** | 20260823_sf3d_p90_2d_dir | 90%, 2D-DCT losses + dir | 30 fixed |
| **B'2 — 3D finetune** | 20260823_sf3d_ft10_3d_dir | 10%, init = B'1 best | early stop p5, cap 60 |
| **A' — scratch control** | 20260823_sf3d_s10_3d_dir | 10% | early stop p5, cap 60 |

C' doubles as THE dir-term experiment on 3D: vs g19_dct (identical but for
the term) it isolates the flip-rate effect — headline metrics
axis_flip_rate_rot, err_adir_signed_all_deg. Comparisons: label-efficiency
A'/B'/C' all-new (no cross-version mixing); dir-term effect C' vs g19_dct;
2D sign emergence B'1 vs g17_2d_dct (axis metrics, previously ~random).

## Execution

Two PRO 6000 pods (stock-poll creates via train_pod.sh; Monitor-wrapped).
Pod 1: C' → A' → test passes → delete. Pod 2: B'1 → B'2 → test passes →
delete. No dev pod (host lost); code reaches the volume by scp through
the first training pod that lands. ft10 config is written when B'1's best
checkpoint exists (exact ckpt path in the config, as v1). Cost ≈ 2 pods ×
~5h ≈ $20.

## Out of scope

Wiring beyond `pred_pred_art_dir_weight` (floors stay at their defaults);
v1-vs-v2 cross-comparisons other than the three named above.
