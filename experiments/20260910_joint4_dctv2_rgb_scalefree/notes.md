# 20260910_joint4_dctv2_rgb_scalefree — joint 2D+3D training with the DCT readout conventions v2

**Goal.** The joint4 recipe unchanged (SF3D + HOI4D + EPIC + ARCTIC, source-homogeneous batches,
hand sources x10, lr 2e-5, 20 ep, monitor `val/sf3d/loss_total`) with the DCT readout conventions
v2 (`knowledge/2026-09-10_trajectory_head_synthesis_v2.md`, #2): pinned start, shape/scale split;
SF3D batches get the 3D first-difference trio (0.5 / 0.25 / 0.25) + log path-length 0.5, the 2D
batches the uv-space trio + log path-length via the `2d` loss profile.

**Comparison row.** `20260910_joint4_dct_rgb_scalefree` best-epoch12: MA 31.41 / signed 30.15,
mIoU 0.2738, PDet 22.72, traj_dir 96.4, roughness 0.0090; HOI4D 0.676 / 85.2.

**Status.** launched 2026-09-10 night (pod J, `run_joint4v2_chain.sh`, log `joint4v2_chain.log`).
