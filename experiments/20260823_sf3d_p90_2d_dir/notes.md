# 20260823_sf3d_p90_2d_dir — label-efficiency v2 arm B'1: 2D-DCT + dir term on 90%

**Recipe:** the 2D-DCT pretrain recipe + `pred_pred_art_dir_weight: 0.1`
(the 2D line's first axis-sign teacher) on the ~90% scene partition,
FIXED 30 epochs (v1's undertraining confound killed). Spec
2026-08-23-label-efficiency-v2-dir-term-design.md. Val-best epoch 10
(1.4297), but the finetune init is **epoch 26** — picked by testing both
candidates (v1 lesson: trunk improves past the val plateau; ep26 wins
mIoU 0.2358 vs 0.2258, PDet 17.1 vs 15.3, shape 0.103 vs 0.113).
CSV logger version_1.

## Test (5,088), epoch-26 ckpt, vs v1 p90 and the full-data 2D arm

| metric | p90 v1 (no dir, early-stopped) | **B'1 ep26** | g17_2d_dct (100% 2D) |
|---|---|---|---|
| mIoU / PDet | 0.228 / 11.2 | **0.236 / 17.1** | 0.2655 / 19.4 |
| proj2d shape / anchor | 0.122 / 0.123 | **0.103 / 0.129** | 0.0947 / 0.109 |
| 2D point | 0.108 | 0.107 | 0.0938 |
| axis all / rot flips | — (~random) | 57.1° / 45.3 | 59.3° / 52.5 |
| traj_dir acc / cos | 84.1 / 0.484 | **64.2 / 0.255** | 91.3 / 0.626 |
| roughness (m) | 0.022 | 0.040 | 0.032 |

## Reading

- **The 30-epoch fix worked for the trunk:** PDet 11.2→17.1, shape
  0.122→0.103 — most of the v1 undertraining gap to the full-data arm is
  closed. Good pretrain material on the mask/detection side.
- **The dir term made axes self-consistent, not correct.** Its val value
  fell 0.64→0.07 (axes now agree with the model's own trajectory signs)
  and rot flips vs GT improved a little over the no-dir 2D arm (45.3 vs
  52.5), axis-all edged better (57.1° vs 59.3°) — but absolute axis
  quality remains near-random. The consistency propagated in the WRONG
  direction for trajectories: traj_dir crashed 84.1→64.2 (v1
  comparison) — same mechanism as the 3D finding in g21_dct_dir (the
  term's two-way gradient lets unsupervised, wrong axes drag the
  trajectory), and worse in 2D where the trajectory's own anchor (the
  projection loss) is weaker than 3D GT.
- Net for the dir term at weight 0.1: **harmful in both domains as
  implemented.** The parked fix (detach the trajectory inside the term,
  making it teach axes only) now has two independent motivations.

Checkpoint `best-epoch26-valloss1.4539.ckpt` seeds
`20260823_sf3d_ft10_3d_dir` (B'2) — the finetune's 3D GT may still
rescue the articulation side; its trunk start is the best 10%-free one
we have. test passes: logs/test_best-epoch{10,26}-*.log on the volume.
