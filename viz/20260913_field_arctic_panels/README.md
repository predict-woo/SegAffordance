# 20260913_field_arctic_panels — the field model on 20 random held-out ARCTIC strokes

20 held-out ARCTIC strokes (val split = the joint4 datamodule's, seed 3, up to 2 per sequence),
`tools/hoi4d_predict_articulation.py --field --dataset arctic`, 2x sharp style, 0.5 m rays. Panels:
GT (mask green, knuckle track cyan, GT hinge line green from the object model) | `field` =
`20260913_field_joint4_l2anchor` best-epoch14 (config `config/joint4_decoder_field.yaml`): mask red,
point ring, hinge line red through the voted origin, 90-deg orbit yellow, decoded trajectory light
green; text = type / p_rev / radius / axis / z_p. `sheets/` = 4-row contact sheets.

What it shows. Masks are on the moving part in 20/20 (phone flip, laptop lid, notebook cover, box
lid, waffle-iron lid, capsule / espresso levers, scissor blade), usually with the grasping fingers
included — the ARCTIC mIoU 0.652 record. Type is revolute in 19/20; the one miss is the microwave
window hidden under the forearm (row 16, p_rev 0.19). Hinge placement: the voted origin lands on or
next to the true hinge for the box lid (12), the phone folds (09, 15, 19), the scissor pivots (02, 07,
17) and the capsule lever (05); on the laptop lids the line is parallel to the true hinge but runs
across the lid rather than along its edge (08, 13, 14); the notebook spine (06, 10) and the waffle-iron
hinge (18) are missed. The predicted hinge RADIUS is small on hand video (0.02-0.31 m; median 0.10 in
the probe), so the yellow orbits are short arcs hugging the point rather than sweeps around the true
hinge, and the axis SIGN is the known coin flip (red lines are drawn sign-agnostically). Consistent
with the ARCTIC probe of this checkpoint: hinge-line offset 0.048 of the image width (dense_off 0.029,
other arms 0.07-0.16), mean axis error 55 deg, flips 41 %.
Regen: manifest.yaml.
