# 20260906_hoi4d_2d_v2_e100_lr6e5 — HOI4D v2 sweep arm `e100_lr6e5`

**Goal:** hyper-parameter retune on the v2 dataset (3,075 one-per-window records, 13 categories, VLM masks + descriptions, middle-knuckle full-window trajectories). Arm: epochs 100, lr 6e-05, milestones [80, 92], batch 64, from scratch.

**Recipe:** otherwise identical to 20260901_hoi4d_2d_dct (config.yaml here).

**Result:** best val/loss_total **0.3547** (epoch 22); test pass on the 110 held-out objects (= val split): mIoU **0.674**, PDet **84.8**, point err 0.0161, traj proj-2D shape 0.0345, traj_dir acc 47.7 (chance — as in v1, direction does not emerge from hand tracks). same val floor reached by ep 22; slightly worse test metrics than 3e-5. Checkpoint: the single best-*.ckpt (others pruned; volume-quota incident 2026-09-06). Sweep table: STATE.md / INDEX.md.
