# 20260913_iphone_model_zoo — 24 checkpoints on the user's four phone photos

Same four iPhone photos as `20260911_iphone_probe` (NOT committed), prompts "open door" / "close laptop"
/ "open closet" / "push chair forward", `tools/predict_image.py --f35 26` per model into `<tag>/`, then
`grid_<photo>.jpg` = the prediction half of every model's panel, 6 columns, labelled. Models (row order):
joint4_dct, joint4_dctv2, dct_chain, dec_base, dec_seed7, l2anchor | h1anchor, cfframe, cfframe_seed7,
cfframe_a3, cfframe_query, attnpool | query, query_seed7, query_l4, query_eps01, query_pos, query_w1024 |
mlp1024, dense, dense_seed7, dense_off, dense_d2d, field (checkpoints = each experiment's INDEX best).

What it shows (24 models x 4 photos, no ground truth — a visual ranking only):
- **Door** (true hinge: vertical, right edge). Every model calls rot and finds the handle. The most
  VERTICAL hinge lines closest to the right edge come from the DENSE family (dense, dense_seed7,
  dense_off, dense_d2d, field) and the DCT-era joint4 models (joint4_dct, dctv2, dct_chain, cfframe_query,
  query_w1024); the pooled-MLP decoder arms (dec_base, l2anchor, cfframe*, attnpool, mlp1024) and h1anchor
  tilt the line 20-70 deg across the door; query_l4 / query_eps01 draw it near-horizontal. None puts the
  line ON the hinge edge; dense and dense_seed7 are nearest.
- **Laptop** (true hinge: horizontal at the screen's bottom edge). All but dct_chain call rot. Only
  query, query_l4 and dec_base draw a near-horizontal line, and at the screen's middle / top rather than
  its base; l2anchor is vertical through the keyboard, h1anchor / cfframe_a3 / dense_seed7 / field
  diagonal. Masks: joint4_dct, dec_base, cfframe_query, query and dense_off segment the screen; l2anchor,
  h1anchor, cfframe, query_seed7, query_eps01, mlp1024 and field put mask on the external keyboard, mouse
  or vacuum instead.
- **Closet** (true hinge: vertical, right edge of the right door). Everyone finds a handle and says rot;
  lines are vertical-ish for cfframe_query, query_seed7, dense_off, dense_d2d, dense (near the seam,
  not the edge) and tilted for the rest; h1anchor and attnpool are near-horizontal.
- **Chair**: nobody segments the chair; trans rays on the bed / floor. Out of distribution for all.

Reading. In the wild no checkpoint places a hinge on a door edge, and the laptop hinge is missed by all
24; the differences are in axis DIRECTION and mask discipline. The dense family generalises the axis
direction best on the door (vertical), the DCT joint models are the most stable across the four photos
(vertical door / closet axes, clean screen mask), and the pooled-MLP decoder arms — including l2anchor,
the ARCTIC favourite — tilt the door hinge the most here. Four photos are anecdotes; a labelled
in-the-wild set is needed before any of this is a claim.
