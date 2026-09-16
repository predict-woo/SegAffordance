# Teaser, option A ("failure contrast"): 2D and 3D sources in, baselines fail, EgoArt succeeds

Paper Fig. 1 candidate built from the mockup the user approved on 2026-09-16 (Taein's brief: make the 2D
source look 2D, the 3D source look 3D, show prior methods failing on a new object, keep it unlike Fig. 2).
Message: 3D articulation on an object class that appears only in 2D human video.

**Top row, what the model learns from.**
- 2D card: six hand-video records drawn with the hand track only (no mask, no badge), scattered like
  prints: HOI4D `ZY20210800001_H1_C4_N38/S21_s05_T2_w0_f046` (right cabinet door) and
  `ZY20210800003_H3_C4_N05/S50_s01_T3_w1_f086` (upper drawer); EPIC `P20_03/P20_03_15` (fridge) and
  `P02_121/P02_121_58` (drawer); ARCTIC `s04_laptop/laptop_use_01_s01_f00080` (a laptop, a different
  subject than the test record, so the reader sees the laptop was met in 2D) and
  `s05_box/box_use_02_s04_f00492` (box lid). `tools/viz_data_samples.py --no-caption --no-badge --no-mask`
  (the `--no-mask` switch was added for this figure), renders in `panels/{hoi4d,epic,hoi4d}`.
- 3D card: SceneFun3D val record 213 ("Open the bottom door of the oven", a kitchen) as an oblique
  point cloud back-projected from the record's metric depth, with the annotated hinge axis (yellow),
  hinge (yellow dot), interaction point (white ring) and sweep arc (green) in 3D. Depth + 3D ground truth
  dumped by `viz/20260914_arthur_method_figure/dump_sample.py` (extended to write `depth.npy` and
  `meta3d.json`) into `panels/sf3d_213`; rendered on the Mac by `render_pointcloud.py`
  (`--zoom 1.75 --yaw -30 --pitch 14`). Records 1773, 4233, 529 and 1246 were dumped and tried too
  (`panels/sf3d*`); the kitchen is the only one wide enough to read as a scan.

**Bottom row, a new object seen only in human video.** ARCTIC laptop record idx 270 (Fig. 5 row 1,
"open the laptop screen"): the raw frame with the instruction; the prior methods as a scattered pile of
five smaller cards, each with its own cross and a method-name chip (3DOI in front at the lower right, its blob mask and sideways
line an obvious miss; A3VLM, OPDFormer-C, MOPD, OPDFormer-P tossed behind at their own offsets and
angles, all "prismatic" with the wrong part); EgoArt (`20260913_joint4_decoder_l2anchor_dense`, best-epoch13) under a check: revolute, full
lid, hinge on the lid edge. Rendered by `tools/viz_fig4_panels.py --no-error --save-frame` (both switches
added for this figure: badges kept, axis-error labels dropped, raw crop written) from the Fig. 5 records in
`viz/20260914_fig5v1_handvideo/preds_arctic_*.jsonl` and the dump in `viz/20260914_fig5_handvideo_50`.

Files: `fig.svg`, `fig.pdf` (paper include), `preview.png`, `build.py`, `render_pointcloud.py`, `src/`
(downscaled inputs), `panels/` (pod renders + manifests).

Regenerate:

```sh
# pod: panels (see the commands recorded in panels/*/manifest.yaml), then
python3 viz/20260916_teaser_option_a/render_pointcloud.py --src viz/20260916_teaser_option_a/panels/sf3d_213 --out viz/20260916_teaser_option_a/src/pointcloud.png --zoom 1.75 --yaw -30 --pitch 14
python3 viz/20260916_teaser_option_a/build.py
rsvg-convert -w 2000 -o viz/20260916_teaser_option_a/preview.png viz/20260916_teaser_option_a/fig.svg
rsvg-convert -f pdf -o viz/20260916_teaser_option_a/fig.pdf viz/20260916_teaser_option_a/fig.svg
```
