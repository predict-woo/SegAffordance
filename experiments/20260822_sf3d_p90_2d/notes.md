# 20260822_sf3d_p90_2d — label-efficiency ARM B pretrain: 2D-DCT on 90%

**Recipe:** the adopted 2D-DCT recipe (`20260822_sf3d_g17_2d_dct`) on the
~90% scene-level "pretrain" partition (183/202 train scenes, 48,584
samples; exact complement of the 10% finetune set, seed 4242). Spec:
2026-08-22-2d-pretrain-label-efficiency-design.md. Early-stopped at epoch
11 (patience 5), best = epoch 6 (val 1.3746 — statistically the full-data
DCT arm's final 1.3702, reached in a third of the steps, then flat for 5
epochs).

## Test (5,088) vs the full-data DCT arm / scratch-10

| metric | **p90 (90% 2D)** | g17_2d_dct (100% 2D) | s10 (10% 3D scratch) |
|---|---|---|---|
| mIoU / PDet | 0.228 / 11.2 | 0.2655 / 19.4 | 0.021 / 0.4 |
| proj2d shape / anchor | 0.122 / 0.123 | 0.0947 / 0.109 | — |
| 2D point | 0.108 | 0.0938 | — |
| traj_dir acc / cos | 84.1 / 0.484 | 91.3 / 0.626 | 80.3 / 0.477 |
| roughness (m) | 0.022 | 0.032 | 0.263 |

## Reading

A solid trunk for the finetune — an order of magnitude above the
scratch-10 control on masks — but NOT at the full-data DCT arm's level
(mIoU −14%, PDet −42%, shape −29%). **Caveat for interpreting arm B:**
val loss said converged (5 flat epochs at the full-run's final value) but
the test metrics suggest the early stop at epoch 6 undertrained the trunk
relative to the 30-epoch full-data run — test-metric gains evidently
continue past the val-loss plateau on this recipe. If ft10 lands short of
g17, rerunning the pretrain with more patience (or a fixed 30-epoch
budget) is the first confound to eliminate. Articulation columns are the
usual unsupervised-2D garbage, by design.

Checkpoint `best-epoch06-valloss1.3746.ckpt` seeds
`20260822_sf3d_ft10_3d` via `model.finetune_from_path`.
test pass: logs/test.log on the volume.
