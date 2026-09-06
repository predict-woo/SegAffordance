# 20260906_hoi4d_2d_v2_base — HOI4D v2 sweep arm `base`

**Goal:** hyper-parameter retune on the v2 dataset (3,075 one-per-window records, 13 categories, VLM masks + descriptions, middle-knuckle full-window trajectories). Arm: epochs 30, lr 1e-05, milestones [24, 28], batch 64, from scratch.

**Recipe:** otherwise identical to 20260901_hoi4d_2d_dct (config.yaml here).

**Result:** best val/loss_total **0.3913** (epoch 27); test pass on the 110 held-out objects (= val split): mIoU **0.551**, PDet **68.0**, point err 0.0240, traj proj-2D shape 0.0369, traj_dir acc 50.2 (chance — as in v1, direction does not emerge from hand tracks). v1 schedule is too short for a higher-LR optimum but balanced (val≈train at ep 24-29); control. Checkpoint: the single best-*.ckpt (others pruned; volume-quota incident 2026-09-06). Sweep table: STATE.md / INDEX.md.
