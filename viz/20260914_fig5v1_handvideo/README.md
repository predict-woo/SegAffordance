# 20260914_fig5v1_handvideo — paper Fig. 5: five picked hand-video frames, every model as a column, from dumps only

**Rows (user's picks from `20260914_fig5_handvideo_50`, 2026-09-14 late), with the in/out-of-distribution label
(= whether SceneFun3D contains the category):** ARCTIC laptop lid (dataset idx 270, out), ARCTIC microwave door (1224, in),
EPIC drawer (226, in), HOI4D toy car (2366, out), HOI4D drawer (633, in).
**Columns:** GT | OPDFormer-C 512 | OPDFormer-P 512 | MOPD 512 (pending) | USDNet (not applicable: needs a scan) |
A3VLM text chain | 3DOI | SceneFun3D-only control | ARTHUR.

Rendered CPU-only by `tools/viz_fig4_panels.py --source {arctic,epic,hoi4d}` in the Fig. 3 style; motion on our panels =
the decoded trajectory (`--extent decoded`, the hand track supervises its extent here); GT motion = the hand track; GT
hinge only on ARCTIC (object model); axis-error labels (unsigned) only on ARCTIC. Composite with column and row labels
(`tools/compose_panels.py --panel-aspect 1.33333 --row-labels ...`): `viz/20260914_paper_figures/fig_handvideo_qual_v2.png`
= Overleaf `figures/fig_handvideo_qual.png` (commit e834e51).

**Baseline protocol on hand video (peer session, 2026-09-14 22:30-23:30 UTC; inputs = `/workspace/datasets/handvideo_fig5_samples`):**
- OPDFormer-C / -P 512: frames letterboxed to 512x384, oracle best-IoU instance vs the GT mask; on these five frames the
  overlap is near zero (IoU 0.001-0.12) except OPD-C on the toy car (0.57); where nothing overlaps, the NEAREST detection
  (mask centroid) is drawn and labelled (`handvideo_preds_nearest.jsonl`). OPD-C needs depth: EPIC row = "needs depth".
- 3DOI: prompted with the GT-mask centroid snapped into the mask (Table II protocol; `handvideo_preds_centroid.jsonl`);
  the hand-contact-point variant (`handvideo_preds.jsonl`) gives tiny masks because that point lies inside the part in
  only 14/150 samples. Axis lifted with depth on HOI4D/ARCTIC, 2D line on EPIC; "freeform" answers = no joint.
- A3VLM: our instruction -> REC box -> joint; mask = box hull; EPIC rows unprojected with a nominal depth (2D faithful).
- MOPD 512: pending (training ends ~Sep 15 23:30 UTC) -> rerun the command below with its files.

**Records saved here (tracked):** `preds_<source>_ours.jsonl` (both our models), `preds_<source>_<baseline>.jsonl` (the
exact peer records used), so the figure can be re-rendered in any style from this directory + the LMDBs.

**Regen (pod, CPU).**
```
uv pip install --python /opt/venv/bin/python matplotlib
D=viz/20260914_fig5_handvideo_50; S=viz/20260914_fig5v1_handvideo; R=/workspace/datasets/baselines/results
COMMON="--baseline OPDFormer-P $R/opd512_p_rgb/handvideo_preds_nearest.jsonl --baseline MOPD pending --baseline USDNet na \
  --baseline A3VLM $R/a3vlm/handvideo_preds.jsonl --baseline 3DOI $R/3doi_runpod/handvideo_preds_centroid.jsonl"
for spec in "arctic 270 1224" "epic 226" "hoi4d 2366 633"; do set -- $spec; src=$1; shift
  CUDA_VISIBLE_DEVICES= python tools/viz_fig4_panels.py --source $src --baseline OPDFormer-C $R/opd512_c_rgbd/handvideo_preds_nearest.jsonl \
    $COMMON --ours sf3d_only $D/$src/preds.jsonl --ours dense $D/$src/preds.jsonl --idx $@ --out $S; done
# USDNet column (index 4 in the strips) dropped from the paper figure (user, 2026-09-15): needs a scan, said in the caption
python tools/compose_panels.py --out viz/20260914_paper_figures/fig_handvideo_qual_v2.png --panel-aspect 1.33333 --cols 0,1,2,3,5,6,7,8 --row-height 220 --gap 6 \
  --labels "Ground truth,OPDFormer-C,OPDFormer-P,MOPD,A3VLM,3DOI,SceneFun3D only,ARTHUR" \
  --row-labels "out of distribution,in distribution,in distribution,out of distribution,in distribution" \
  --col-groups "Baselines:1-5,Ours:6-7" --row-groups "ARCTIC:0-1,EPIC:2,HOI4D:3-4" \
  $S/00_arctic_rot_270.png $S/01_arctic_rot_1224.png $S/00_epic_trans_226.png $S/00_hoi4d_trans_2366.png $S/01_hoi4d_trans_633.png
```

**Update (2026-09-16).** MOPD 512 landed -> `--baseline MOPD $R/mopd512_rgb/handvideo_preds_nearest.jsonl` (nearest-centroid
fallback on the microwave door and both HOI4D rows, labelled "nearest instance"); composite `fig_handvideo_qual_v5.png`
(label EgoArt, Overleaf d5b5ddf). No column is pending any more.

**Update (2026-09-16, user): Ours columns first.** `fig_handvideo_qual_v5.png` -> `fig_handvideo_qual_v6.png`: column order
GT | SceneFun3D only | EgoArt | OPDFormer-C | OPDFormer-P | MOPD | A3VLM | 3DOI (`--cols 0,7,8,1,2,3,5,6`,
`--col-groups "Ours:1-2,Baselines:3-7"`), everything else as v5.
