# 20260912_hoi4d_laptop_decoder_probe — the four joint decoder models on HOI4D laptops

10 held-out HOI4D laptop windows (C3, val split, seed 0), `tools/hoi4d_predict_articulation.py`, 2x sharp
style, 0.5 m rays: GT (mask green, 2D knuckle track cyan) | `L2_2pi` = `20260911_joint4_decoder_rgb_scalefree`
best-epoch16 | `L2_2pi_axis` = `20260912_joint4_decoder_l2anchor` best-epoch17 | `H1_axis` =
`20260912_joint4_decoder_h1anchor` best-epoch16 | `cf_frame` = `20260912_joint4_decoder_cfframe`
best-epoch17 (all through `config/sf3d_test_decoder_rgb_scalefree.yaml`; predicted type at test).
Per panel: mask red, point ring, hinge line red (through the predicted origin), 90-deg orbit yellow,
decoded trajectory light green (mostly under the orbit), text = type / p_rev / radius / axis / z_p.

What it shows. (1) TYPE now transfers: every model calls every laptop revolute at p_rev 0.93-0.95
(the earlier joint4 probe called laptops "trans"); the joint recipe trains the type head on HOI4D's
category labels, and laptops are C3 = rot. (2) Masks are on the lid, the point on the lid edge. (3) The
HINGE does not transfer: no model puts the axis along the lid/keyboard seam. L2_2pi draws the hinge
diagonally across the lid with a 1-3 m radius; L2_2pi_axis and cf_frame put the origin far away (r 0.8-2.3
m, orbits sweeping off-screen); H1_axis collapses the radius (0.09-0.21 m) with the hinge line crossing
the lid at random. z_p is unsupervised on hand video and ranges 0.13-3.8 m for the same laptop, so the
absolute scale is arbitrary; the relative geometry (where the hinge sits vs the lid) is what is wrong.
(4) The decoded trajectories follow their orbits — smooth, but around the wrong hinge.

Reading: HOI4D only supervises a 2D hand track, which fixes the arc's tangent at the knuckle but not
the hinge placement; SF3D has no laptops. The one source with laptop hinge GT (ARCTIC: laptops,
notebooks, boxes) sits on the 2D side in these runs. Putting ARCTIC under the closed-form loss is the
direct fix. Regen: manifest.yaml.
