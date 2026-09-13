# Final dense model vs the l2anchor pooled model on the same 20 SF3D validation frames

Same 20 picks as `20260914_dense_final_sf3d_val_panels` (seed 3), three columns: GT | `dense` =
`20260913_joint4_decoder_l2anchor_dense` best-epoch13 | `l2anchor` = `20260912_joint4_decoder_l2anchor`
best-epoch17 (the pooled-readout ancestor: same losses, data and decoder, MLP readout). Header "axis err"
is the SIGNED angle (0-180 deg), so ~170 means the right line with the wrong sense of rotation.
`contact_rot.jpg` / `contact_trans.jpg` = one column of 10.

Regen (dev pod): `python tools/sf3d_vis_val.py --model dense config/sf3d_test_decoder_rgb_scalefree_dense.yaml
experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt --model l2anchor
config/sf3d_test_decoder_rgb_scalefree.yaml experiments/20260912_joint4_decoder_l2anchor/checkpoints/best-epoch17-sf3dval1.1495.ckpt
--out viz/20260914_dense_vs_l2anchor_sf3d_val --num 20 --seed 3`

**Read (2026-09-14).** The difference is the SIGN. On the doors l2anchor draws a hinge line in roughly the
right place but with the axis pointing the wrong way (bathroom door 171 deg, glass door 168, closet 147),
so its decoded arcs swing away from the GT arc; dense gets the same doors at 8 / 24 / 9 deg with arcs
along the GT. Exceptions: the oven's bottom door, where l2anchor is right (6 deg, horizontal hinge) and
dense predicts a vertical hinge (65 deg); the dishwasher, which both call "rot" (GT trans). Prismatic:
l2anchor also flips directions (top-left drawer 104 deg vs dense 18). Masks and points are similar for
both (small handles found). Consistent with the SF3D numbers (signed MA 42.7 vs 30.3; matched axis 11.4
vs 18.9 deg): dense voting's gain on this benchmark is largely resolving the sense of rotation, i.e. the
sign that a pooled vector cannot read from the seam.
