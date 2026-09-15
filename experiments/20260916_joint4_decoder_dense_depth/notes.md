# 20260916_joint4_decoder_dense_depth — ablation: the final recipe WITH the depth input

**Goal (user, 2026-09-16).** Does a depth input help the final model a lot? Row for the ablation table.

**Setup.** `config/joint4_decoder_dense_depth.yaml` = `config/joint4_decoder_l2anchor_dense.yaml` (the paper's final
recipe: dense voting, closed-form L2 2pi + axis 0.5 on SF3D, unit-anchor projection loss on hand video, seed 42, 20 epochs,
lr 2e-5, batch 64) with exactly two switches: `model_params.use_depth: true` (DepthEncoder built, the first two pyramid
levels widened by 128 / 256 channels, depth features concatenated with the DINOv3 maps at H/8 and H/16) and
`data.load_depth: true` (the frame caches' depth PNGs decoded to metres instead of a zero map). Losses, profiles and the
depth anchor (0.0 on hand video) untouched; the trajectory head stays scale-free. Test calls carry
`--model.model_params.use_depth true --data.load_depth true`.

**What each source feeds the depth encoder** (checked in the frame caches 2026-09-16):
SF3D ARKit sensor depth (mm, ~72 % valid); HOI4D sensor depth (mm, ~73 % valid); ARCTIC an object-only z-buffer render
from the MoCap object pose (mm, zero elsewhere; `tools/arctic_process_2d.py`); EPIC all zeros (no depth exists).
Raw metres, no normalisation (as in the earlier RGB-D arms). At test time the same maps.

**Smoke (dev pod, RTX PRO 4000 24 GB).** `fast_dev_run 4`, batch 8: model built with the depth encoder, 4 train + 4 val
batches ran; the run then died on `No space left on device` (the dev pod's 20 GB overlay is 98 % full) while writing
Lightning's end-of-run files — not a model problem.

**Run.** Pod jdec-depth (RTX PRO 6000 Server Edition), chain `run_joint4dec_dense_depth_chain.sh`, expected ~4.5 h + tests.

**Result.** (pending)
