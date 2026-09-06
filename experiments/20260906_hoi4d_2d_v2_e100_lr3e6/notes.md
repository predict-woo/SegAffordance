# 20260906_hoi4d_2d_v2_e100_lr3e6 — HOI4D v2 sweep arm `e100_lr3e6`

**Goal:** hyper-parameter retune on the v2 dataset (3,075 one-per-window records, 13 categories, VLM masks + descriptions, middle-knuckle full-window trajectories). Arm: epochs 100, lr 3e-06, milestones [80, 92], batch 64, from scratch.

**Recipe:** otherwise identical to 20260901_hoi4d_2d_dct (config.yaml here).

**Result:** best val/loss_total **0.4384** (epoch 70); test pass on the 110 held-out objects (= val split): mIoU **0.521**, PDet **61.3**, point err 0.0268, traj proj-2D shape 0.0379, traj_dir acc 50.2 (chance — as in v1, direction does not emerge from hand tracks). lower LR is strictly worse. Checkpoint: the single best-*.ckpt (others pruned; volume-quota incident 2026-09-06). Sweep table: STATE.md / INDEX.md.
