# 20260910_multi3_dct_rgb_scalefree — the multi-source 2D arm with the DCT-6 head (12 epochs)

**Question:** user decision 2026-09-10 ("let's just go with DCT from now on", after the literature sweeps in knowledge/2026-09-10_*): the multi3 recipe (`20260909_multi3_rgb_scalefree`: HOI4D + EPIC + ARCTIC, RGB-only scale-free teacher-forcing recipe, augmentation x8, no mirror on HOI4D) with the DCT-6 trajectory head instead of the plain head, and the epoch budget cut to 12 (milestones [9, 11]) because the plain run's val projection loss bottomed at epoch 6 of 30. Does the smooth basis fix the jittery 2D/ARCTIC/EPIC sweeps at equal or better mask/point/shape numbers?

**Recipe:** `config/multi3_dct_rgb_scalefree.yaml` = multi3_rgb_scalefree with `trajectory_dct_coeffs 6`, `max_epochs 12`, `scheduler_milestones [9, 11]`. Tested on the union and per source (the per-source single-source configs are plain-headed, so the tests pass `--model.model_params.trajectory_dct_coeffs 6`).

**Comparison row (plain head, ep-6 of 30):** HOI4D 0.754 / 91.9 / shape 0.0355, EPIC 0.591 / 69.8 / 0.092, ARCTIC 0.702 / 83.3 / 0.084; union 0.724 / 87.2.

**Result:** (pending)

**Decision:** (pending)
