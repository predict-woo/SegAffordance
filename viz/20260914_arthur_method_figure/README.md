# ARTHUR method figure (paper Fig. `fig:method`)

Architecture schematic of the FINAL model, `20260913_joint4_decoder_l2anchor_dense` (seed 42;
`config/joint4_decoder_l2anchor_dense.yaml`, `model/segmenter.py`, `model/layers.py`
`DenseArticulationHead`, `model/losses/geometric.py` `analytic_decode_curves`). Architecture figure,
not checkpoint inference: no model runs. The example is SceneFun3D validation record 1773
("Open the left door of the closet near the window"), dumped on the dev pod by `dump_sample.py`
(`sample/frame.png`, `sample/depth.png` inferno-coloured, `sample/mask.png`, `sample/meta.json`); the output panel shows its GROUND-TRUTH
mask (magenta with a dark outline, so a handle-sized mask stays visible), hinge axis (red),
interaction point (yellow ring) and sweep arc (yellow) projected with the record's intrinsics.
`sample/mask_overlay.png` is derived from `sample/mask.png` by the ImageMagick line in `build.py`'s
docstring.

Style follows the MAE / SAM / ViT / BLIP-2 / DiT figure conventions the user pointed at: flat
pastel rounded blocks with thin outlines, left-to-right data flow, frozen towers in blue with a
snowflake, feature maps as small grids, one pastel colour per functional group (gray trunk, orange
heatmap head, green voting head, purple scalar heads, yellow analytic decoder), Helvetica labels
and Times math with the paper's notation (bold vectors `I, T, s, w, F, F_q, u_p, ĥ, n, d, p, q, K`;
italic scalars `M, c, z_p, z_q, L, γ(s)`).

Three dashed groups match the method subsections: language-conditioned feature encoder (frozen
DINOv3 ViT-L/16 + dino.txt, pyramid adapter to strides 8/16/32, the text-gated FPN drawn EXPLODED as a
sub-panel: conv 3x3 per level, the /32 level gated by the sentence state, /32 up x2 and /8 pool x2 into
a concat bar at /16, conv 1x1 aggregation, CoordConv -> F, 3-layer
transformer decoder cross-attending to the word tokens, decoded map 512x32x32; the OPTIONAL depth
branch of the RGB-D variant is drawn dashed: depth map -> small conv depth encoder -> features
concatenated to the /8 and /16 pyramid levels before fusion, `model/segmenter.py` `use_depth`); dense hinge voting
and readouts (dynamic-kernel projector to two cell grids, one with a bar of dark cells at the handle (mask) and one shaded by distance from the point (heatmap), with soft-argmax, dense voting head
with per-pixel axis / direction / type / hinge-offset fields averaged under the mask, part-pooled
MLPs for z_p, z_q, L); analytic trajectory generator (lift with K, render γ(s)) with the two
supervision paths (2D projection loss on hand tracks, 3D closed-form loss on SceneFun3D sweeps).

Fact drawn as implemented, not as the current paper text says: the pyramid adapter reads the
ViT's FINAL-layer patch tokens only (`SimpleFeaturePyramid`; `dinov3_multilayer_taps` is false in
the final config). The four-tap DPT-style adapter exists in the code but was only used in the g14
/ g15 configs, so the "four intermediate taps" paragraph of `03_method.tex` needs correcting.

Sizing: 3210 x 1145 px canvas (2.8:1), fonts scaled 1.45x on 2026-09-16 (FS in `build.py`). At IEEE
`\textwidth` (7.16 in) block labels and group titles print at ~6.5 pt, annotations at ~5.4 pt.

Files: `model.svg` (editable vector), `model.pdf` (paper include), `preview.png` (review copy),
`sample/` (SF3D frame, mask, GT geometry), `dump_sample.py` (pod-side dump), `build.py` (generator),
`manifest.yaml`.

Regenerate (the sample dump only needs re-running for a different record):

```sh
bash runpod/dev.sh run "python viz/20260914_arthur_method_figure/dump_sample.py --idx 1773 --out viz/20260914_arthur_method_figure/sample"
python3 viz/20260914_arthur_method_figure/build.py
rsvg-convert -w 2200 -o viz/20260914_arthur_method_figure/preview.png viz/20260914_arthur_method_figure/model.svg
rsvg-convert -f pdf -o viz/20260914_arthur_method_figure/model.pdf viz/20260914_arthur_method_figure/model.svg
```

LaTeX: `\includegraphics[width=\textwidth]{method_overview.pdf}` after copying `model.pdf` to the
Overleaf clone's `figures/method_overview.pdf`.
