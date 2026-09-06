# 20260906_hoi4d_2d_v2_e100_lr1e4 — HOI4D v2 sweep arm `e100_lr1e4`

**Goal:** hyper-parameter retune on the v2 dataset (3,075 one-per-window records, 13 categories, VLM masks + descriptions, middle-knuckle full-window trajectories). Arm: epochs 100, lr 0.0001, milestones [80, 92], batch 64, from scratch.

**Recipe:** otherwise identical to 20260901_hoi4d_2d_dct (config.yaml here).

**Result:** best val/loss_total **0.3535** (epoch 41); test pass on the 110 held-out objects (= val split): mIoU **0.702**, PDet **86.7**, point err 0.0156, traj proj-2D shape 0.0331, traj_dir acc 47.5 (chance — as in v1, direction does not emerge from hand tracks). same floor by ep 41; ties lr3e5 on mIoU/PDet, slightly worse point/shape. Checkpoint: the single best-*.ckpt (others pruned; volume-quota incident 2026-09-06). Sweep table: STATE.md / INDEX.md.
