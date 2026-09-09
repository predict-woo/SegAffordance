# 20260911_iphone_probe — the joint model on four phone photos (in the wild)

Four iPhone photos taken by the user on 2026-09-09 (apartment: wardrobe,
entrance door, chair, laptop; 26 mm-equivalent lens, portrait 4:3, resized
to 1152x1536 in `inputs/` — the photos are NOT committed, only this README
and the manifest), run through `20260910_joint4_dct_rgb_scalefree`
best-epoch12 (RGB-only scale-free DCT-6, joint 2D+3D) with the user's
prompts. Intrinsics from the 35 mm-equivalent focal length (f = 26/36 x
1536 = 1109 px, principal point at the centre); model input = the photo
stretched to 512x512, as the training frames were.

Panels: left = photo + prompt, right = prediction: mask (red), point (white
ring), projected trajectory (light green, z_p-scaled), predicted axis (red:
hinge line for rot, 0.5 m direction ray for trans), 90-deg orbit (yellow)
for rot, origin-heatmap uv (small red circle); text = type, p_rev, axis, z_p.

Regen (dev pod): `python tools/predict_image.py --model joint4_dct
config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml
experiments/20260910_joint4_dct_rgb_scalefree/checkpoints/best-epoch12-sf3dval0.9688.ckpt
--case viz/20260911_iphone_probe/inputs/IMG_0877.jpg "open closet" --case
.../IMG_0876.jpg "open door" --case .../IMG_0875.jpg "push chair forward"
--case .../IMG_0874.jpg "close laptop" --out viz/20260911_iphone_probe --f35 26`.

| photo | prompt | what the model did |
|---|---|---|
| IMG_0876 | open door | **Works.** Mask on the handle, point on the handle, type ROT (p_rev 0.94), the hinge axis drawn as a vertical line on the RIGHT edge of the door — where the real hinges are — radius 0.64 m, z_p 1.43 m (plausible). The projected trajectory is a short squiggle near the handle rather than the swing. |
| IMG_0874 | close laptop | Mask on the whole screen, point on the screen's left edge, z_p 0.88 m (plausible); but type trans (p_rev 0.02) with the axis pointing into the screen — the laptop failure of the HOI4D probes, reproduced in the wild. |
| IMG_0877 | open closet | Mask only on the right handle, point on the handle, p_rev 0.48 (undecided), z_p 1.44 m; direction ray down-right. The door leaf is not segmented. |
| IMG_0875 | push chair forward | Fails: no mask on the chair, point floating above it near the bag, a looping trajectory, trans 0.07. No dataset contains chairs or "push forward". |

Reading: on the one category SF3D covers well (a room door) the RGB-only
joint model transfers to a phone photo of a different apartment with the
right type, the right hinge side and a plausible depth. Wardrobe doors it
localises by the handle but does not segment; laptops keep the known
"trans" failure; chairs are out of every training distribution. Prompts
were short ("open door") vs the datasets' fuller phrasing ("open the left
door of the closet"), which may cost the wardrobe case.
