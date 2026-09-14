# 20260914_fig4v3_style — paper Fig. 4 panels in the Fig. 3 style (GT | OPDFormer-C 512 | ARTHUR), from dumps

**What.** The four SF3D val frames the user picked for Fig. 4 on 2026-09-14 from `20260914_dense_sf3d_val_200`:
val 1684 (door, revolute), 113 (oven door, revolute), 3726 and 403 (drawers, prismatic). Rendered by the new
`tools/viz_fig4_panels.py` in the style of the dataset figure (`tools/viz_data_samples.py`): native aspect cropped to
4:3 around the interaction point, translucent red mask with outline, yellow axis (hinge line + foot dot for revolute,
direction ray for prismatic), green motion track with arrowhead (GT: annotated 2D track; ARTHUR: decoded trajectory),
white interaction point, type badge, and an axis-error label on prediction panels. Baseline panels show the
oracle-matched instance the scorer used (mask + axis, no point, no track: the detector predicts neither).

**No GPU.** ARTHUR comes from `viz/20260914_dense_sf3d_val_200/preds.jsonl` (the dump of
`20260913_joint4_decoder_l2anchor_dense` best-epoch13), the baseline from the peer's
`/workspace/datasets/baselines/results/opd512_c_rgbd/preds.jsonl` (`20260914_opdformer_c_rgbd_512`); frames and
GT from the LMDB on CPU.

**Files.** `NN_<rot|trans>_valJ.png` = strip of the three panels; `..._gt.png`, `..._OPDFormer-C.png`, `..._dense.png`
= single panels. Composed 2 x 2 (two frames per row) with column labels into
`viz/20260914_paper_figures/fig_sf3d_qual_v3.png` = Overleaf `figures/fig_sf3d_qual.png` (commit 2e2bc7c).
Axis errors: OPDFormer-C 5 / 28 / 15 / 24 deg, ARTHUR 5 / 4 / 8 / 7 deg.

**Regen (pod, CPU).**
```
CUDA_VISIBLE_DEVICES= python tools/viz_fig4_panels.py --ours dense viz/20260914_dense_sf3d_val_200/preds.jsonl \
  --baseline OPDFormer-C /workspace/datasets/baselines/results/opd512_c_rgbd/preds.jsonl \
  --idx 1684 113 3726 403 --out viz/20260914_fig4v3_style
# rows = hstack of two strips each, then
python tools/compose_panels.py --out viz/20260914_paper_figures/fig_sf3d_qual_v3.png --panel-width 1024 --row-height 300 \
  --gap 6 --labels "Ground truth,OPDFormer-C,ARTHUR,Ground truth,OPDFormer-C,ARTHUR" row1.png row2.png
```
Style scale `--k 2.0` is chosen for print: each panel is ~3 cm wide in the two-column figure, so the badge text lands
at ~4 pt like Fig. 3's (`--k 1.5` looked cleaner on screen but printed at ~2.5 pt).
