# 20260909_multi3_rgb_scalefree — HOI4D + EPIC + ARCTIC in one seeded shuffle, RGB-only scale-free plain-TF recipe, augmented x8

**Question:** the first multi-source 2D run. Does the RGB-only scale-free 2D recipe (`20260909_hoi4d_2d_v2_rgb_scalefree`: plain head, unit anchor, no depth) hold up when HOI4D (2,625 train), EPIC (306) and ARCTIC (2,230) are mixed in one seeded stream with geometry-consistent augmentation (photometric, must-keep scale+translate crop, horizontal flip on EPIC/ARCTIC only) at 8 augmented views per record per epoch (41,288 samples/epoch, user: "40000 samples")? Spec: docs/superpowers/specs/2026-09-09-rgb-only-scale-free-trajectory-design.md; datamodule datasets/multisource_datamodule.py; augmentation datasets/augment.py (viz/20260909_augment_check).

**Recipe:** `config/multi3_rgb_scalefree.yaml` = the HOI4D rgb_scalefree loss recipe (projection 0.5 on the unit anchor, L_pp 0.1 normalized with radius floor 0.15, p_rev prior 0.5, no depth tether, no 3D-GT losses), lr 3e-5, batch 64, 30 epochs (645 steps each), milestones [24, 28], `epoch_multiplier 8`, per-source `hflip_p` (HOI4D 0.0, EPIC/ARCTIC 0.5), `flip_text skip`. Pod C (RTX PRO 6000 Workstation), 2 h 46 min; frame caches in /dev/shm.

**Result:** best val/loss_total **0.3901 at epoch 6** — the val projection loss bottoms at 0.47 (ep 6) and climbs to 0.65 by ep 28 while the train projection loss falls to 0.02: the trajectory term OVERFITS the multi-source set from epoch ~8 despite the augmentation; val mask keeps improving to 0.100 (ep 17). Test on the 841-record union (logs/test.log): mIoU 0.724 / PDet 87.2 / point 0.031 / 2D shape 0.058. Per source with the same epoch-6 checkpoint (logs/test_<src>.log, the identical per-source scene splits):

| held-out split | n | mIoU | PDet | point err | 2D shape | traj_dir | single-source reference |
|---|---|---|---|---|---|---|---|
| HOI4D | 459 | **0.754** | **91.9** | **0.0141** | **0.0355** | 48.4 | HOI4D-only rgb_scalefree: 0.733 / 88.7 / 0.0147 / 0.0378 |
| EPIC | 53 | 0.591 | 69.8 | 0.053 | 0.092 | 77.4 | first numbers on EPIC |
| ARCTIC | 329 | 0.702 | 83.3 | 0.052 | 0.084 | 70.8 | first numbers on ARCTIC |

Panels (viz/20260909_multi3_val_panels, 12 per source): HOI4D masks/points/tracks as good as the single-source arm; EPIC masks on the right fixture in most panels with some spill onto counters and hands, and the TYPE GATE SELF-ORGANISED on EPIC (p_rev 0.84-0.98 on fridge/oven/cupboard doors, ~0.4 on drawers — on HOI4D-only arms p_rev never left the batch prior); ARCTIC masks on the moving part in 12/12 (blades, lever, waffle lid, notebook cover, laptop screen, microwave door). Predicted trajectories on EPIC/ARCTIC are short and jittery next to the long GT sweeps — the 2D shape 0.08-0.09 vs 0.036 on HOI4D. ARCTIC's real axes give axis-all 62.8° (L_pp is the only axis teacher; expected).

**Reading:** (1) Mixing in EPIC and ARCTIC with augmentation IMPROVES HOI4D itself on every metric (+0.02 mIoU, +3.2 PDet, shape 0.0378 -> 0.0355). (2) The recipe transfers to the two new sources at the mask/point level on the first try; their trajectories are the weak part (short predicted sweeps). (3) The early val minimum (epoch 6) says 8 views x 30 epochs is far past the useful budget for the trajectory term — for the next multi run use ~10 epochs at x8 or a stronger regulariser; the epoch-6 checkpoint is what goes into SF3D post-training.

**Decision:** epoch-6 checkpoint = the 2D init for `20260909_sf3d_plain_rgb_scalefree_ft_multi3` (plain-head SF3D post-training, nothing re-initialised). Single seed. Ckpt best-epoch06-valloss0.3901.ckpt on the volume.
