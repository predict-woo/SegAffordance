# SF3D-only control vs the final model on held-out hand video (paper Fig. 5 source)

`tools/hoi4d_predict_articulation.py --dataset {hoi4d,epic,arctic} --num 12 --per-seq 1 --scale 2`, models
`sf3d_only` = `20260913_sf3d_decoder_l2anchor_dense` best-epoch13, `dense` = `20260913_joint4_decoder_l2anchor_dense`
best-epoch13 (same recipe, hand sources removed vs present). Panels: GT (green mask, cyan track) |
sf3d_only | dense (red mask, hinge line, yellow decoded arc). `contact_<ds>_{0,1}.jpg` = 6 rows each.

**Read (2026-09-14).** The control finds almost no part (tiny or wrong red masks: keyboard instead of
screen, wall instead of drawer) and calls laptop screens / lids "trans"; dense segments the part on
11/12 HOI4D, 9/12 EPIC (spills on the fridge), 12/12 ARCTIC, types right, arcs along the track. Dense hinge
lines: on the door edge for HOI4D cabinets and near the fridge hinge, but ACROSS laptop / notebook / phone
parts on ARCTIC with too-large radii (probe r_med 0.71 m) — placement does not transfer. Rows used in the
paper: hoi4d 00 (laptop), 02 (cabinet door); epic 01 (fridge), 02 (drawer); arctic 02 (laptop), 03 (notebook).
