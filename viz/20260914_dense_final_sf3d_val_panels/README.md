# Final model (plain dense voting) on 20 random SceneFun3D validation frames

Checkpoint `experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt`
(the paper's final model), `tools/sf3d_vis_val.py --num 20 --seed 3 --scale 2`. Panels: left = GT (green
mask, green axis line through the annotated origin, cyan GT arc, white ring = interaction point), right =
prediction (red mask, red axis line through the predicted hinge, yellow decoded arc; header = type,
p_rev, lever radius, axis error, type verdict). 10 revolute + 10 prismatic picks; `contact_rot.jpg` /
`contact_trans.jpg` are 2-column sheets.

Regen (dev pod): `python tools/sf3d_vis_val.py --model dense config/sf3d_test_decoder_rgb_scalefree_dense.yaml
experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt
--out viz/20260914_dense_final_sf3d_val_panels --num 20 --seed 3`

**Read (2026-09-14).** Revolute: 9/10 axes within 24 deg, 8/10 within 13 deg (closet / bathroom / cabinet
doors, glass door); the miss is the oven's bottom door (65 deg: a horizontal hinge predicted vertical, the
vertical-door prior). Hinge lines land on the door edge for the closets and the cabinet, but on the two
close-up bathroom doors the red line runs through the door panel ~0.3 door widths from the hinge edge.
Prismatic: 9/10 within 23 deg (drawers 2-20 deg, amp knobs / metronome 7-23 deg, drum pedal 20 deg); the
miss is "open the dishwasher" (annotated trans, predicted rot with a 93 deg error: type WRONG). Masks: the
predicted handle mask is on the right element in 18/20; small handles are found even in cluttered frames
(amp, drum pedal). Types: 19/20. Candidate frames for paper Fig. 4 (ours row): val1696, val1773, val529
(closet doors), val4233 (cabinet), val283 / val501 (drawers), val1140 (metronome).
