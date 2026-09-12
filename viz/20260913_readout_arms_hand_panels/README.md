# 20260913_readout_arms_hand_panels — l2anchor base vs query readout vs dense voting

Same three checkpoints on 12 SF3D val samples (`sf3d/`, `tools/sf3d_vis_val.py`, seed 7), 12 held-out
ARCTIC strokes (`arctic/`, GT hinge green) and 12 held-out HOI4D windows (`hoi4d/`)
(`tools/hoi4d_predict_articulation.py --per-seq 2 --seed 1`); 2x sharp style, 0.5 m rays; `sheets/`
= 4-row contact sheets. Panels: GT | `l2anchor` = `20260912_joint4_decoder_l2anchor` best-epoch17 |
`query` = `20260913_joint4_decoder_l2anchor_query` best-epoch16 | `dense` = `20260913_joint4_decoder_l2anchor_dense`
best-epoch13 (test configs `config/sf3d_test_decoder_rgb_scalefree{,_query,_dense}.yaml`). Metrics in
the experiments' notes (dense: SF3D MA 44.0 record; query: MA 35.75, best hand-video sign transfer).

What it shows. SF3D: on the revolute samples the dense model's hinge line sits on the door / washing
machine edge with the smallest axis error of the three (nightstand door 5 deg vs 2 / 14; washing
machine 11 vs 12 / 25), its 90-deg orbit follows the GT arc, and prismatic rays are along the drawer
pull for all three; the dense masks are the tightest (drawer knob, keyboard). Hand video: all three
call the type right; on ARCTIC the query model's hinge lines are the ones closest to the green GT on
the scissors and the phone, the dense model puts its lines through the object at a wrong angle more
often (espresso lever, notebook) — consistent with its worse ARCTIC probe (offset 0.112, flips 30 %)
— and its hand-video masks are visibly thinner (phone, notebook: the collapse in the metrics). HOI4D:
trash-can lid hinge (row 01) is placed at the lid's back edge by the query model and drawn as a
long lever by the dense model; drawers and the toy car are prismatic rays for all three.
Reading: dense voting is the SF3D model; the query readout is the hand-video model; the wave-4 arm
`dense_d2d` (2D votes computed off a detached trunk) tests whether the two can be one model.
