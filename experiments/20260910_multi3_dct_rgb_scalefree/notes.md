# 20260910_multi3_dct_rgb_scalefree — the multi-source 2D arm with the DCT-6 head (12 epochs)

**Question:** user decision 2026-09-10 ("let's just go with DCT from now on", after the literature sweeps in knowledge/2026-09-10_*): the multi3 recipe (`20260909_multi3_rgb_scalefree`: HOI4D + EPIC + ARCTIC, RGB-only scale-free teacher-forcing recipe, augmentation x8, no mirror on HOI4D) with the DCT-6 trajectory head instead of the plain head, and the epoch budget cut to 12 (milestones [9, 11]) because the plain run's val projection loss bottomed at epoch 6 of 30. Does the smooth basis fix the jittery sweeps at equal mask/point/shape numbers?

**Recipe:** `config/multi3_dct_rgb_scalefree.yaml` = multi3_rgb_scalefree with `trajectory_dct_coeffs 6`, `max_epochs 12`, `scheduler_milestones [9, 11]`. Pod D2 (RTX PRO 6000 Server; pod D was a power-capped lemon, replaced), 66 min. Tested on the union and per source (single-source configs with `--model.model_params.trajectory_dct_coeffs 6`).

**Result:** best val/loss_total **0.4161 at epoch 11** (the last epoch: 0.492 -> 0.416, still drifting down; val projection 0.49-0.53 throughout, no overfit within 12 epochs — unlike the plain 30-epoch run whose projection val climbed after epoch 8).

| held-out split | n | plain head (ep 6/30) mIoU / PDet / point / shape / rough | **DCT head (ep 11/12)** mIoU / PDet / point / shape / rough |
|---|---|---|---|
| HOI4D | 459 | 0.754 / 91.9 / 0.0141 / 0.0355 / ~0.07 | 0.753 / 89.8 / 0.0137 / **0.0337** / **0.0072** |
| EPIC | 53 | 0.591 / 69.8 / 0.053 / 0.092 | 0.520 / 58.5 / 0.053 / 0.090 / 0.0062 |
| ARCTIC | 329 | 0.702 / 83.3 / 0.052 / 0.084 | 0.694 / 79.6 / 0.062 / 0.085 / 0.0053 |
| union | 841 | 0.724 / 87.2 / 0.031 / 0.058 | 0.715 / 83.8 / 0.035 / 0.057 / 0.0064 |

**Reading:** (1) Roughness drops 10x (0.07 -> 0.005-0.007): the smooth sweeps are back, as on the 3D line. (2) HOI4D is unchanged on mask/point and slightly better on 2D shape (0.0337, the best any arm has produced); PDet -2. (3) The two small/new sources lose a little detection (EPIC -11 PDet on 53 records, ARCTIC -4) and EPIC loses 0.07 mIoU — 53 records is noise territory, ARCTIC's -0.01 is real but small. (4) The budget: 12 epochs at x8 did not overfit and the last epoch was the best, so the DCT head tolerates (or wants) a few more epochs than the plain head. Panels: viz/20260910_multi3dct_val_panels.

**Decision:** the DCT-6 2D arm is the 2D-stage model from here (user). Its epoch-11 checkpoint feeds `20260910_sf3d_g19_dct_rgb_scalefree_ft_multi3dct` (DCT at both stages, all weights loaded). Single seed. Ckpt best-epoch11-valloss0.4161.ckpt on the volume.
