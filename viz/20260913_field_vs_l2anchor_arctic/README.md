# 20260913_field_vs_l2anchor_arctic — l2anchor (CRIS, pooled MLP heads) vs the field model on the same 20 ARCTIC strokes

Same 20 held-out strokes as `20260913_field_arctic_panels` (seed 3). Panels: GT (mask green, track cyan,
GT hinge green) | `l2anchor` = `20260912_joint4_decoder_l2anchor` best-epoch17 | `field` =
`20260913_field_joint4_l2anchor` best-epoch14 (`--field-names field`). `sheets/` = 4-row contact sheets.

What it shows (user's reading, confirmed): on HAND-VIDEO HINGE GEOMETRY the l2anchor model is visibly
better — laptop hinge lines at the top edge of the lid (08, 13) where the field model draws them across
the middle; the notebook spine (10) found by l2anchor, missed by field; phone fold (09), box crease (12),
microwave hinge edge (01) closer for l2anchor; plausible radii (0.3-1.4 m) vs the field model's
collapsed 2-30 cm and sign-flipped 3D axes on the lids. The field model wins masks (waffle lid 18,
laptop 14, box 12, microwave 16 are missed or spilled by l2anchor), type and SceneFun3D.
Metric lesson: the ARCTIC hinge-line OFFSET (point-to-line) let the field model's "parallel line across
the lid" score well (0.048 vs 0.070); the mean axis error (48 vs 55 deg) had the ranking right. A
stricter placement metric (projected line angle + offset, radius / part size) is needed before any
"best hand-video model" claim. Field-model defect to fix: radius collapse on hand video (2D votes off the
trunk + no 2D depth supervision). Status: l2anchor stays the hand-video hinge model; the field model is
the SceneFun3D + segmentation model.
