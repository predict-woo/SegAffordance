# Generalisation figure: 2D human video in, 3D articulation on an unseen category out

Paper figure (user's sketch, 2026-09-15): three real hand-video records, one per 2D source, feed the
model on the left; on the right the model's 3D articulation prediction on an object class that the 3D
training data (SceneFun3D) does not contain. Teaser style (user, 2026-09-15): no type badges, no axis-error label, no legend, no chips inside the
panels. Built as SVG in the method-figure style (`viz/20260914_arthur_method_figure`). The four panels
were RE-RENDERED clean on the dev pod into `panels/` (`tools/viz_data_samples.py --no-badge --no-caption`
for the three records; `tools/viz_fig4_panels.py --clean`, a switch added for this figure, for the
prediction) from the same records as the existing renders:

| panel | source render | record |
|---|---|---|
| HOI4D | `viz/20260913_fig2_data_samples/picks_nocap/hoi4d/00_ZY20210800001_H1_C4_N67__S391_s02_T2_w1_f122.png` | "Close the lower cabinet door", revolute (Fig. 2 pick) |
| EPIC-KITCHENS | `viz/20260913_fig2_data_samples/picks_nocap/epic/00_P02_09__P02_09_20.png` | "open drawer", prismatic (Fig. 2 pick) |
| ARCTIC | `viz/20260913_fig2_data_samples/picks_nocap/arctic/00_s02_laptop__laptop_use_02_s03_f00443.png` | "close the laptop screen", revolute (Fig. 2 pick) |
| SceneFun3D | `src/sf3d_frame.jpg` + `src/sf3d_mask.png` + `src/sf3d_meta.json` (copied from `viz/20260914_arthur_method_figure/sample`, val 1773 "Open the left door of the closet near the window"); the GT mask, hinge axis + origin and sweep arc are drawn in SVG from the meta in the teaser colours | the 3D training source, entering the model from below |
| output | `panels/arctic_pred/00_arctic_rot_270_dense.png` (same record and dump as Fig. 5 row 1: `viz/20260914_fig5_handvideo_50/arctic/preds.jsonl`, idx 270) | ARCTIC laptop lid, out of distribution; EgoArt = `20260913_joint4_decoder_l2anchor_dense` best-epoch13; overlays: red mask, yellow hinge axis + origin, green decoded trajectory from the white interaction point |

The three 2D panels are cropped to one 3:2 size (ARCTIC top-aligned so its track stays in frame) with the
source name under each; the SceneFun3D panel sits under the model block with its bottom edge level with
the stack. Arrows: the three records -> "train" -> the model
block; the model -> "test" -> the prediction panel. Captions: "2D human video, hand tracks, no 3D
labels" and "3D articulation on an unseen object category". The overlay colours are explained in the
LaTeX caption.

Files: `fig.svg` (editable), `fig.pdf` (paper include, ~2 MB: rsvg re-encodes the embedded JPEGs losslessly), `preview.png`,
`src/` (downscaled inputs), `panels/` (clean pod renders + manifests), `build.py`.

Regenerate:

```sh
python3 viz/20260915_generalization_figure/build.py
rsvg-convert -w 2000 -o viz/20260915_generalization_figure/preview.png viz/20260915_generalization_figure/fig.svg
rsvg-convert -f pdf -o viz/20260915_generalization_figure/fig.pdf viz/20260915_generalization_figure/fig.svg
```

To swap a panel, drop a new PNG through `magick <png> -resize 720x -quality 90 src/<name>.jpg` and edit
the `rows` list / output path in `build.py`.
