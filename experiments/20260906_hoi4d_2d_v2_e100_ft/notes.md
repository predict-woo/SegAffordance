# 20260906_hoi4d_2d_v2_e100_ft — HOI4D v2 sweep arm `e100_ft`

**Goal:** hyper-parameter retune on the v2 dataset (3,075 one-per-window records, 13 categories, VLM masks + descriptions, middle-knuckle full-window trajectories). Arm: epochs 100, lr 1e-05, milestones [80, 92], batch 64, initialized from the SF3D 2D-DCT best checkpoint (strict=False).

**Recipe:** otherwise identical to 20260901_hoi4d_2d_dct (config.yaml here).

**Result:** (pending)
