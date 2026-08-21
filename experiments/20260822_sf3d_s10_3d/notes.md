# 20260822_sf3d_s10_3d — label-efficiency ARM A: scratch 3D on 10%

**Recipe:** the exact g17 split-axis config trained FROM SCRATCH on the
~10% scene-level "finetune" partition (19/202 train scenes, 5,502/54,086
samples = 10.2%, seed 4242; spec
2026-08-22-2d-pretrain-label-efficiency-design.md). The control arm: how
far does 10% of the 3D annotation get you alone. Early-stopped at epoch
25 (patience 5), best = epoch 20 (val 1.8192). Val/test = standard splits.

## Test (5,088) vs g17 (100% 3D)

| metric | **scratch-10** | g17 (100%) |
|---|---|---|
| mIoU / PDet | **0.021 / 0.37** | 0.265 / 20.6 |
| matched axis / axis all | 25.0° / 44.4° | 16.9° / 25.8° |
| MA / M pass rate | 4.9 / 96.0 | 25.9 / — |
| origin / radius (m) | 0.555 / 0.253 | 0.257 / 0.122 |
| 3D point | 0.400 | 0.232 |
| traj_dir acc / cos | 80.3 / 0.477 | 94.9 / 0.811 |
| traj roughness (m) | 0.263 | 0.091 |

## Reading

**10% alone destroys the trunk, not the articulation heads.** Masks and
grounding collapse to near-zero (mIoU 0.021, PDet 0.4 — an order of
magnitude below even the broken-2D arm's recovery level), which drags MA
down with it (4.9: the axis can't pass on parts that are never detected).
The articulation heads themselves degrade far more gracefully — matched
axis 25.0° vs 16.9°, origin 0.55 vs 0.26 — consistent with them being
small MLPs on pooled features while segmentation/grounding needs data
volume.

This is exactly the complementarity the pretrain+finetune arm (B) bets
on: the 2D-DCT pretrain delivers a trunk at the full-3D level (mIoU
0.2655) with zero 3D labels, and the 10% 3D only has to teach the small
articulation heads. If B ≈ g17, the label-efficiency claim holds.

test pass: logs/test.log on the volume (ckpt best-epoch20-valloss1.8192);
pod deleted after the pass.
