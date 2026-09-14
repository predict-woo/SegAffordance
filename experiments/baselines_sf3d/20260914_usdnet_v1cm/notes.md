# 20260914_usdnet_v1cm — USDNet on SF3D scenes at 1 cm voxels

**Goal.** `20260912_usdnet` (2 cm voxels, as in their Articulate3D setup) gave SF3D functional
elements a median of 11 points (97 % under 50) and AP50 0.000; PDet 0.0 on our protocol. This run
halves the voxel size so elements have ~8x the points. Requested by the paper session, approved
2026-09-14.

**Setup.** Same as `20260912_usdnet` (USDNet @ 0ba303d, `scripts/train_mov.sh` recipe from the
Mask3D scannet200 backbone, 200 epochs, val every 20, best `val_mean_ap_50` checkpoint, crop 5.5 m /
min 75k points, c2f radius 0.1) except `VOXEL=0.01`: converter `sf3d_to_usdnet.py --voxel 0.01` →
`data/usdnet_sf3d_v1cm`, `data.voxel_size=0.01` (`runpod/baselines/usdnet/chain.sh`). The conversion
(CPU) was 38 % done from Sep 12 and was finished on the dev pod (16 workers) while the training pod
came up; the chain waits for `.done_convert` (`CONVERT_HERE=0`).

**Run.** Pod `bl-usdnet1cm` (launch polling from 00:21 UTC 2026-09-14). Run dir `runs/usdnet_v1cm`,
results `results/usdnet_v1cm/`.

**Result.** pending.
