# 20260906_hoi4d_2d_v2_e100_bs32 — HOI4D v2 sweep arm `e100_bs32`

**Goal:** hyper-parameter retune on the v2 dataset (3,075 one-per-window records, 13 categories, VLM masks + descriptions, middle-knuckle full-window trajectories). Arm: epochs 100, lr 1e-05, milestones [80, 92], batch 32, from scratch.

**Recipe:** otherwise identical to 20260901_hoi4d_2d_dct (config.yaml here).

**Result:** best val/loss_total **0.3581** (epoch 99); test pass on the 110 held-out objects (= val split): mIoU **0.688**, PDet **86.3**, point err 0.0153, traj proj-2D shape 0.0332, traj_dir acc 47.1 (chance — as in v1, direction does not emerge from hand tracks). 2x updates/epoch ≈ higher LR; runner-up, still improving at ep 99. Checkpoint: the single best-*.ckpt (others pruned; volume-quota incident 2026-09-06). Sweep table: STATE.md / INDEX.md.
