# 20260914_opdformer_c_rgbd_512 — OPDFormer-C (RGB-D) at 512x384, retrained on SF3D

**Goal.** Resolution-matched rerun of `20260912_opdformer_c_rgbd`: SF3D functional elements are ~12 px
at the recipe's 256x192, which the Sep-12 results suggested was the binding constraint (their own
test AP50 4.8). Requested by the paper session, approved by the user 2026-09-14 00:1x UTC.

**Setup.** Identical to `20260912_opdformer_c_rgbd` (OPDMulti `configs/opd_c_real.yaml`, R50
Mask2Former, batch 16, AdamW 1e-4, 60k iters, steps 36k/48k, COCO init, 8 SF3D classes, per-dataset
pixel stats) except the data: `data/opd_sf3d_512` from `sf3d_to_opd.py --size 512 384` (recentred
object poses, same splits: 32,171 train frames / 48,560 annotations). Their `MotionDatasetMapper`
applies no resize (flip / brightness / contrast only), so frames are consumed at the stored h5
size; `INPUT.*SIZE*` (256 in `opd_base.yaml`) are lifted to 512 for anything that reads them
(`IMG_SIZE=512` in `runpod/baselines/opd/chain.sh`). Pixel stats recomputed on the 512 h5.

**Run.** Pod `bl-opd512-c` (6w98k0sr3v4cma, A100-SXM4-80GB, $1.59/h, main volume EU-RO-1), chain
started 00:26 UTC 2026-09-14; detectron2 ETA at iter 560: 14:50 h (~0.9 s/iter, ~$24 incl. validation passes). Run dir
`/workspace/datasets/baselines/runs/opd512_c_rgbd`, results copied to `results/opd512_c_rgbd/`.

**Result.** pending.
