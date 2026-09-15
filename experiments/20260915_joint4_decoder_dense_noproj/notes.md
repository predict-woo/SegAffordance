# 20260915_joint4_decoder_dense_noproj — joint training WITHOUT the hand-video trajectory loss

**Goal (supervisor's ablation, 2026-09-15).** Fill the middle row of

| Training | ARCTIC axis (unsigned, deg) |
|---|---|
| SceneFun3D only | 56.6 |
| + video mask / type only | **57.1** (this run) |
| + video mask / type / trajectory | 45.8 |

i.e. does human video help 3D articulation through its masks and types alone, or is the projected trajectory loss
what carries articulation across? Answer: the trajectory loss. Without it the joint model's ARCTIC axes are at chance
(a random direction scores 57.3), exactly like the SceneFun3D-only model, even though it segments and types the
hand-video parts well.

**Setup.** `config/joint4_decoder_dense_noproj.yaml` = the final recipe (`config/joint4_decoder_l2anchor_dense.yaml`)
with `loss_profiles.2d.trajectory_proj_weight: 0.0`; hand-video batches keep the mask, type and point-heatmap terms.
Seed 42, 20 epochs, pod jdec-noproj (RTX PRO 6000 Server Edition, 2.07 it/s, ~4.3 h + tests, ~$10). Best checkpoint
epoch 18 (`best-epoch18-sf3dval1.0299.ckpt`). Chain `run_joint4dec_dense_noproj_chain.sh`.

**Results (paper protocol: PDet IoU >= 0.25, hinge 10 cm, axis unsigned on detected frames; per-sample CSV here).**

| | MA | axis | origin | mIoU | PDet | +M | +MA | +MAO | MAO | HOI4D mIoU | EPIC mIoU | ARCTIC mIoU | ARCTIC axis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| no trajectory loss (this) | 47.0 | 14.2 | 0.232 | 0.291 | 52.5 | 49.0 | 30.7 | 27.5 | 40.5 | 0.69 | 0.30 | 0.65 | 57.1 |
| final (with trajectory loss) | 42.7 | 13.9 | 0.248 | 0.277 | 51.9 | 49.9 | 29.1 | 24.8 | 35.9 | 0.51 | 0.19 | 0.51 | 45.8 |
| SceneFun3D only | 43.8 | 16.5 | 0.262 | 0.263 | 48.8 | | | | | 0.01 | 0.01 | 0.03 | 56.6 |

Trainer test logs (IoU 0.5 protocol): SF3D signed MA 46.95, PDet 27.1, origin 0.232, matched axis 13.4; HOI4D PDet 84.7,
EPIC 17.0, ARCTIC 79.0; ARCTIC err_adir_all 57.1, flips 48 %. ARCTIC probe: `arctic_axis_probe.csv` (329 strokes,
unsigned mean 57.1, median 56.5, 0.3 % under 10 deg, type revolute 100 %).

**Reading.** (1) The trajectory loss is the only channel through which hand video teaches 3D articulation: masks and
types alone leave ARCTIC axes at chance. (2) Dropping it makes the SF3D side slightly BETTER (MA 47.0 vs 42.7, masks
0.291 vs 0.277) and the hand-video masks much better (HOI4D 0.69 vs 0.51, EPIC 0.30 vs 0.19): the projection loss
competes with the mask/type objectives for the shared trunk. So the final model pays ~4 MA on SF3D and some mask quality
for articulation transfer to unseen objects. (3) Type transfers without the trajectory (100 % revolute on ARCTIC), as it
did in the final model.

**Decision.** Row added to the paper's Table III ("+ video, no trajectory loss") with a sentence in Sec. IV-C.
Vis: none yet (dumps not made for this arm; `tools/hoi4d_predict_articulation.py --dump` if needed).
- HOI4D official-pose axis probe (2026-09-15): unsigned 42.9 (final) vs 51.4 (noproj) vs 53.4 (sf3d_only) on 325 articulated test records; CSV experiments/20260913_joint4_decoder_l2anchor_dense/hoi4d_axis_probe.csv
