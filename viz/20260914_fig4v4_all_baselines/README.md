# 20260914_fig4v4_all_baselines — paper Fig. 4: one frame per row, every baseline as a column, from dumps only

**What.** The user's four Fig. 4 frames (val 1684 door, 113 oven door, 3726 and 403 drawers; picked from
`20260914_dense_sf3d_val_200`) as 8-panel rows: GT | OPDFormer-C 512 | OPDFormer-P 512 | MOPD 512 | USDNet 1 cm |
A3VLM (text chain) | 3DOI | ARTHUR. Rendered CPU-only by `tools/viz_fig4_panels.py` in the Fig. 3 style (4:3 crop
around the interaction point, red mask + outline, yellow axis with hinge foot, green motion with arrowhead, white
interaction point, type badge, axis-error label). MOPD, USDNet and 3DOI were still training on 2026-09-14 evening and
are grey `(pending)` placeholders; "no instance" = the oracle matcher found no overlapping detection (OPDFormer-P on
the oven door).

**Motion on ARTHUR's panels = GT extent through the predicted parameters** (`--extent gt`, the default now): the arc
around the predicted hinge / the segment along the predicted direction, starting at the predicted interaction point,
over the GT track's sweep angle or travel length (90 deg / 0.70 m on these frames). The model's own decoded
trajectory (`--extent decoded`) has a learned extent that is meaningless on SF3D, where only the articulation
parameters are supervised; it made the predictions look "too short". Baselines predict neither a point nor a motion,
so their panels show mask + axis only.

**Raw predictions saved here** so the figure can be re-rendered in any style without the models:
`preds_ours.jsonl` (our dump records, `tools/sf3d_preds_io.py` schema), `preds_OPDFormer-C.jsonl`,
`preds_OPDFormer-P.jsonl`, `preds_A3VLM.jsonl` (the peer's oracle-matched records for these keys: mask RLE at native
size, type, axis_cam, origin_cam). All git-tracked. Full-split sources: `viz/20260914_dense_sf3d_val_200/preds.jsonl`
and `/workspace/datasets/baselines/results/{opd512_c_rgbd,opd512_p_rgb}/preds.jsonl`, `a3vlm/preds_chain.jsonl`.

**Update (same evening).** (1) 3DOI landed -> real column from `$R/3doi_runpod/preds.jsonl` (point-prompted; on the
door it answers "free-form", so its panel shows the SAM mask with "no joint predicted"; axis errors 14 / 108 / 10 deg
on the other three). (2) "no instance" cells replaced: `tools/baselines_sf3d/opd_nearest_instance.py` re-reads the
detector's raw `instances_predictions.pth` and exports, per figure key, the oracle instance if any overlaps the element,
else the NEAREST one (mask centroid to GT centroid) flagged `fallback: nearest` -> `preds_nearest_opd512_{p_rgb,c_rgbd}.jsonl`
here; the renderer labels such panels "nearest instance, axis error N". OPDFormer-P on the oven door: 100 detections
in the frame, nearest 37 px away at score 0.00, axis error 83 deg. Composite now `fig_sf3d_qual_v5.png` (Overleaf f88d2c8).

**Update (2026-09-15 ~06:00 UTC).** USDNet 1 cm landed -> real column from `$R/usdnet_v1cm/preds.jsonl` (its scene
instances projected; no instance on the oven door). Composite `fig_sf3d_qual_v6.png` (Overleaf 09d0332) now also carries
the Baselines / Ours column-group header (`compose_panels --col-groups "Baselines:1-6,Ours:7-7"`). Strips are now named
`NN_sf3d_<rot|trans>_J.png` (renderer prefixes the source). Only MOPD 512 is still a placeholder.

**Files.** `NN_<rot|trans>_valJ.png` = 8-panel strip; `..._<column>.png` single panels. Composite with column labels:
`viz/20260914_paper_figures/fig_sf3d_qual_v4.png` = Overleaf `figures/fig_sf3d_qual.png` (commit b1b1b03).

Axis errors (deg): OPDFormer-C 5 / 28 / 15 / 24; OPDFormer-P 5 / none / 21 / 15; A3VLM 2 / 5 / 19 / 6; ARTHUR 5 / 4 / 8 / 7.

**Regen (pod, CPU; when MOPD / USDNet / 3DOI land, replace `pending` by their preds.jsonl).**
```
uv pip install --python /opt/venv/bin/python matplotlib
R=/workspace/datasets/baselines/results; S=viz/20260914_fig4v4_all_baselines
CUDA_VISIBLE_DEVICES= python tools/viz_fig4_panels.py --ours dense viz/20260914_dense_sf3d_val_200/preds.jsonl \
  --baseline OPDFormer-C $R/opd512_c_rgbd/preds.jsonl --baseline OPDFormer-P $R/opd512_p_rgb/preds.jsonl \
  --baseline MOPD pending --baseline USDNet pending --baseline A3VLM $R/a3vlm/preds_chain.jsonl --baseline 3DOI pending \
  --idx 1684 113 3726 403 --out $S
python tools/compose_panels.py --out viz/20260914_paper_figures/fig_sf3d_qual_v4.png --panel-width 1024 --row-height 240 \
  --gap 6 --labels "Ground truth,OPDFormer-C,OPDFormer-P,MOPD,USDNet,A3VLM,3DOI,ARTHUR" $S/00_*.png $S/01_*.png $S/02_*.png $S/03_*.png
```

**Update (2026-09-15 ~08:00 UTC).** Column order changed (user): GT | AFUN | baselines. Composite `fig_sf3d_qual_v8.png`
(Overleaf, Fig. 4):
```
python tools/compose_panels.py --out viz/20260914_paper_figures/fig_sf3d_qual_v8.png --panel-aspect 1.33333 --cols 0,7,1,2,3,4,5,6 \
  --row-height 240 --gap 6 --labels "Ground truth,AFUN,OPDFormer-C,OPDFormer-P,MOPD,USDNet,A3VLM,3DOI" --col-groups "Ours:1-1,Baselines:2-7" \
  $S/00_sf3d_rot_1684.png $S/01_sf3d_rot_113.png $S/02_sf3d_trans_3726.png $S/03_sf3d_trans_403.png
```

**Update (2026-09-16).** MOPD 512 landed (`$R/mopd512_rgb/preds.jsonl`, oracle instance; all four frames matched, axis
errors 5 / 44 / 13 / 16 deg) -> strips re-rendered with every column real; composite `fig_sf3d_qual_v10.png` (label EgoArt,
Overleaf d5b5ddf). Same render command as above with `--baseline MOPD $R/mopd512_rgb/preds.jsonl`, `--baseline USDNet
$R/usdnet_v1cm/preds.jsonl`, `--baseline 3DOI $R/3doi_runpod/preds.jsonl` and the `preds_nearest_opd512_*` files for the OPDFormers.
