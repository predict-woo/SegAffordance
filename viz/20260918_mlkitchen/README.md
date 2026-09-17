# 20260918_mlkitchen — ML-hall showroom kitchen, 8 frames x 18 prompts, SceneFun3D-only vs EgoArt

**What.** The user's `ml-hall-kitchen` photo set (39 frames, 4080x3060, Samsung, no EXIF focal length) of the
mock kitchen in the ML hall: wooden cabinet doors and drawer stacks, a built-in oven, a fridge, a microwave,
a white dishwasher and a sink. This is squarely inside \sfd's distribution (kitchen fixtures), so it tests
prompt grounding more than category generalisation, unlike `viz/20260916_itw_test`.

**Inputs (not in the mirror).** Eight frames chosen for variety (two wide counter views, two corner views,
two close-ups, one fridge, one dishwasher), resized to 1600 px and copied to the pod:
`/workspace/datasets/itw/20260918_mlkitchen/inputs/` with `cases.txt` (one `IMG<TAB>prompt` line per case).
18 cases: 3 x 182638 (oven / cabinet under the sink / drawer next to the oven), 3 x 182653 (left cabinet door /
top drawer / oven), 2 x 182654 (middle drawer / oven), 2 x 184240 (fridge / cabinet door), 2 x 184247
(top drawer / cabinet door on the right), 3 x 184258 (oven / cabinet door / bottom drawer), 2 x 182702
(oven / dishwasher), 1 x 184211 (drawer).

**Models.** `sf3d_only` = `20260913_sf3d_decoder_l2anchor_dense` best-epoch13; `dense` (EgoArt, final) =
`20260913_joint4_decoder_l2anchor_dense` best-epoch13; config `config/sf3d_test_decoder_rgb_scalefree_dense.yaml`,
`--f35 26`. Raw 3-panel outputs `/workspace/datasets/itw/20260918_mlkitchen/out/` (18), paper-style panels
`panels/` (`tools/viz_photo_panels.py`, 4:3 crops), review sheets `sheet_0..3.jpg`, dump `preds.jsonl` (36 records).

**Regen (dev pod).**
```
D=/workspace/datasets/itw/20260918_mlkitchen; B=viz/20260918_mlkitchen; DENSE=config/sf3d_test_decoder_rgb_scalefree_dense.yaml
ARGS=$(awk -F"\t" -v d=$D/inputs '{printf "--case %s/%s \"%s\" ", d, $1, $2}' $D/inputs/cases.txt)
eval python tools/predict_image.py --model sf3d_only $DENSE experiments/20260913_sf3d_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval1.1144.ckpt \
  --model dense $DENSE experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt $ARGS --out $D/out --f35 26 --dump $B/preds.jsonl
CUDA_VISIBLE_DEVICES= python tools/viz_photo_panels.py --dump $B/preds.jsonl --models sf3d_only,dense --idx $(seq 0 17) --out $B/panels --aspect 1.33333
```

**Read (2026-09-18).** Both models call the motion type correctly on all 18 cases (ovens, cabinet doors, the
fridge and the dishwasher revolute at p_rev 0.92-0.97; every drawer prismatic at p_rev 0.04-0.08), and their
axes agree within 20 deg on 12 of 18. The difference is prompt grounding, and it is consistent: on the three
cases with a spatial qualifier or a competing part in view ("open the left cabinet door", "open the cabinet
door on the right", "open the cabinet door" beside an oven) the \sfd-only model lands on a neighbouring drawer
or on the oven trim, while EgoArt masks the named door's handle and puts a near-vertical hinge on its outer
edge; the contact points differ by 0.10-0.33 of the image on exactly those cases and by <0.05 where the prompt
is unambiguous. Hinge quality is the other way round on ovens: the \sfd-only model draws the horizontal
bottom-edge hinge more often (cases 0, 5, 16), EgoArt sometimes tilts it. On the fridge both put the hinge
line inside the door rather than on its far edge. No case is a failure; this scene is in distribution for both.
