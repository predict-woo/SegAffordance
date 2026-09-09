# 20260909_multi3_rgb_scalefree — HOI4D + EPIC + ARCTIC in one seeded shuffle, RGB-only scale-free plain-TF recipe, augmented x8

**Question:** the first multi-source 2D run. Does the RGB-only scale-free 2D recipe (`20260909_hoi4d_2d_v2_rgb_scalefree`: plain head, unit anchor, no depth) hold up when HOI4D (2,625 train), EPIC (306) and ARCTIC (2,230) are mixed in one seeded stream with geometry-consistent augmentation (photometric, must-keep scale+translate crop, horizontal flip on EPIC/ARCTIC only) at 8 augmented views per record per epoch (~41k samples/epoch, user: "40000 samples")? Val/test = the union of the three per-source scene splits (841 records). Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md; datamodule datasets/multisource_datamodule.py; augmentation datasets/augment.py (viz/20260909_augment_check).

**Recipe:** `config/multi3_rgb_scalefree.yaml` = the HOI4D rgb_scalefree loss recipe (projection 0.5 on the unit anchor, L_pp 0.1 normalized with radius floor 0.15, p_rev prior 0.5, no depth tether, no 3D-GT losses), lr 3e-5, batch 64, 30 epochs, milestones [24, 28], `epoch_multiplier 8`, per-source `hflip_p` (HOI4D 0.0, EPIC/ARCTIC 0.5), `flip_text skip`. LMDBs staged into /dev/shm on the pod.

**Comparison rows:** HOI4D-only rgb_scalefree on its own held-out split: mIoU 0.733 / PDet 88.7 / point 0.0147 / 2D shape 0.0378 (not directly comparable: this run's test set is the three-source union).

**Result:** (pending)

**Decision:** (pending)
