# 20260914_dense_sf3d_val_200 — final dense model on 200 random SF3D val frames, with a GPU-free prediction dump

**What.** 200 held-out SceneFun3D frames (100 prismatic + 100 revolute, stratified draw, seed 5) rendered as
`[GT | dense]` panels at 2x (1024 px per panel) by `tools/sf3d_vis_val.py`, for the final model
`20260913_joint4_decoder_l2anchor_dense` (`checkpoints/best-epoch13-sf3dval0.9789.ckpt`, test config
`config/sf3d_test_decoder_rgb_scalefree_dense.yaml`). Files `NNN_<rot|trans>_val<j>.png`; `j` is the val index
(same split and filters as every other SF3D viz batch, `split_dataset_by_scene(ds, 0.1, 42)`).

**The dump (new).** `preds.jsonl` holds one record per frame with the raw prediction (`tools/sf3d_preds_io.py`):
predicted mask (COCO RLE at the logits' resolution), type logits, lifted interaction point and its uv, axis
direction, hinge point and its uv, and the decoded trajectory relative to the point (after the scale-free
rescale). `tools/sf3d_render_preds.py --preds preds.jsonl --out viz/<new batch> [--idx ...]` redraws any subset
of these frames on CPU with the same drawing code (frame + GT come from the LMDB on the volume, no checkpoint,
no GPU). Verified on three frames: the re-render differs from the live panels on ~0.01 % of the pixels, all on
the predicted mask boundary (the RLE stores the thresholded mask, so bilinear upsampling of the hard mask
moves the contour by a pixel), nothing else.

**Overlays.** GT panel: mask green, 2D track cyan, interaction point white ring, axis green (hinge line through
the GT origin for rot, direction ray from the track start for trans). Prediction panel: mask red, point white
ring, decoded trajectory light green, axis red, 90-degree orbit yellow, hinge-heatmap uv small red circle;
text: type + p_rev + lever radius, axis direction, predicted depth, axis error vs GT and type OK/WRONG.

**Regen (pod).**
```
python tools/sf3d_vis_val.py --model dense config/sf3d_test_decoder_rgb_scalefree_dense.yaml \
  experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt \
  --out viz/20260914_dense_sf3d_val_200 --num 200 --seed 5 --dump viz/20260914_dense_sf3d_val_200/preds.jsonl
# GPU-free re-render of a subset:
CUDA_VISIBLE_DEVICES= python tools/sf3d_render_preds.py --preds viz/20260914_dense_sf3d_val_200/preds.jsonl \
  --out viz/<new batch> --idx 1696 1773
```
The PNGs are gitignored (mirror only); `preds.jsonl` (~200 records) is tracked so the frames can be redrawn
from a fresh clone plus the volume.
