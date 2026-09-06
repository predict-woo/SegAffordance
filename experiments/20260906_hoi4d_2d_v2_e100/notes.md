# 20260906_hoi4d_2d_v2_e100 — HOI4D v2 sweep arm `e100`

**Goal:** hyper-parameter retune on the v2 dataset (3,075 one-per-window records, 13 categories, VLM masks + descriptions, middle-knuckle full-window trajectories). Arm: epochs 100, lr 1e-05, milestones [80, 92], batch 64, from scratch.

**Recipe:** otherwise identical to 20260901_hoi4d_2d_dct (config.yaml here).

**Result:** best val/loss_total **0.3860** (epoch 58); test pass on the 110 held-out objects (= val split): mIoU **0.626**, PDet **78.7**, point err 0.0185, traj proj-2D shape 0.0346, traj_dir acc 47.3 (chance — as in v1, direction does not emerge from hand tracks). longer at lr 1e-5 = mild overfit (train 0.15 vs val 0.40); tiny gain. Checkpoint: the single best-*.ckpt (others pruned; volume-quota incident 2026-09-06). Sweep table: STATE.md / INDEX.md.
