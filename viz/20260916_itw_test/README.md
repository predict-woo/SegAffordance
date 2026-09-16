# 20260916_itw_test — in-the-wild phone photos, 46 HEIC (IMG_1018..1082), 88 image-prompt cases, two models

**Inputs (NOT in the mirror).** The user's `~/Downloads/test/*.HEIC` (46, portrait 4:3, f35 = 26 mm), converted on the Mac
with `sips -s format jpeg -Z 1600` and copied to the pod: `/workspace/datasets/itw/20260916_test/inputs/` (+ `cases.txt`,
one `IMG\tprompt` line per case). Prompts by number: 1018-1019 close/open the window; 1020 open the door; 1021 open the
left/right closet; 1041-1059 close the laptop; 1060-1062 move the mouse forward; 1063-1071 pick up the cup / the umbrella /
move the mouse forward / pick up the scissors / the key (all five per photo); 1072-1078 close the laptop; 1079 left/right
closet; 1080 open the door; 1081-1082 open/close the window.

**Models.** `sf3d_only` = `20260913_sf3d_decoder_l2anchor_dense` best-epoch13; `dense` (EgoArt, final) =
`20260913_joint4_decoder_l2anchor_dense` best-epoch13; config `config/sf3d_test_decoder_rgb_scalefree_dense.yaml`.
Panels photo | sf3d_only | dense at 1200x1600 each: `/workspace/datasets/itw/20260916_test/out/NN_IMG_xxxx.png` (88, on the
volume only). Contact sheets (4 cases each, JPEG): `sheet_00..21.jpg` here.

**Regen (dev pod).**
```
D=/workspace/datasets/itw/20260916_test; DENSE=config/sf3d_test_decoder_rgb_scalefree_dense.yaml
ARGS=$(awk -F"\t" -v d=$D/inputs '{printf "--case %s/%s \"%s\" ", d, $1, $2}' $D/inputs/cases.txt)
eval python tools/predict_image.py --model sf3d_only $DENSE experiments/20260913_sf3d_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval1.1144.ckpt \
  --model dense $DENSE experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt $ARGS --out $D/out --f35 26
```

**Read (2026-09-16).** Windows (1018/1019/1081/1082): both models find the handle and say revolute; the control's hinge line
is a diagonal through the glass, EgoArt's is near-vertical along the sash edge (the correct hinge side), same for open and
close. Door (1020/1080): both correct, near-vertical hinge at the far edge. Closets (1021/1079): both pick the handle of the
named door; EgoArt's hinge is on the outer edge of the left door, the control's is off the cabinet. Laptops (1041-1059,
1072-1078): the control says prismatic and segments keyboard patches in every frame; EgoArt says revolute (p_rev ~0.95),
segments (part of) the screen, and its hinge line runs along the screen base in about half the frames, elsewhere tilted
across the keyboard or above the screen. Mouse (1060-1062): both prismatic, mask on the mouse, direction along the desk.
Pick-up prompts (1063-1071): the control ignores the noun and marks the mouse for every prompt; EgoArt segments the cup, the
spoon, the key and the umbrella for the matching prompts, with an upward/along-desk translation, but misses the scissors and
sometimes the cup rim. Same story as Fig. 6: hand video widens the object set and the prompt grounding, not the hinge
accuracy.

**Paper picks (user, 2026-09-16): Fig. 6 v3.** IMG_1068 "move the mouse forward", IMG_1075 "close the laptop", IMG_1079
"open the right closet", IMG_1081 "close the window" (cases 56 / 77 / 82 / 85 = sheet 14 row 1, sheet 19 row 2, sheet 20
row 3, sheet 21 row 2). Rendered in the Fig. 4 style by the new `tools/viz_photo_panels.py` from a prediction dump
(`predict_image.py --dump`, new flag), square crops around EgoArt's point, prompts under each row
(`compose_panels.py --row-captions`, new flag):
```
B=viz/20260916_itw_test; D=/workspace/datasets/itw/20260916_test
python tools/predict_image.py --model sf3d_only ... --model dense ... --case $D/inputs/IMG_1068.jpg "move the mouse forward" \
  --case $D/inputs/IMG_1075.jpg "close the laptop" --case $D/inputs/IMG_1079.jpg "open the right closet" \
  --case $D/inputs/IMG_1081.jpg "close the window" --out $D/picks_raw --f35 26 --dump $B/picks/preds.jsonl
CUDA_VISIBLE_DEVICES= python tools/viz_photo_panels.py --dump $B/picks/preds.jsonl --models sf3d_only,dense --idx 0 1 2 3 --out $B/picks/panels --aspect 1.0
python tools/compose_panels.py --out viz/20260914_paper_figures/fig_wild_qual_v3.png --panel-aspect 1.0 --row-height 300 --gap 6 \
  --labels "Photo,SceneFun3D only,EgoArt" --row-captions '"move the mouse forward"|"close the laptop"|"open the right closet"|"close the window"' \
  $B/picks/panels/00_00_IMG_1068.png $B/picks/panels/01_01_IMG_1075.png $B/picks/panels/02_02_IMG_1079.png $B/picks/panels/03_03_IMG_1081.png
```
`picks/preds.jsonl` is tracked (the exact records behind the figure). Overleaf `figures/fig_wild_qual.jpg` = v3.
