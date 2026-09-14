# Final dense model vs the field model on the same 20 SF3D validation frames

Same 20 picks as `20260914_dense_final_sf3d_val_panels` (seed 3): GT | `dense` = `20260913_joint4_decoder_l2anchor_dense`
best-epoch13 | `field` = `20260913_field_joint4_l2anchor` best-epoch14 (`model/field_model.py`, loaded via
`--field-names field`, config `config/joint4_decoder_field.yaml`). Header "axis err" = signed angle.
`contact_{rot,trans}_{0,1}.jpg` = 5 rows each.

Regen (dev pod): `python tools/sf3d_vis_val.py --model dense config/sf3d_test_decoder_rgb_scalefree_dense.yaml
experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt --model field
config/joint4_decoder_field.yaml experiments/20260913_field_joint4_l2anchor/checkpoints/best-epoch14-sf3dval1.0536.ckpt
--field-names field --out viz/20260914_dense_vs_field_sf3d_val --num 20 --seed 3`

**Read (2026-09-14).** Axis errors, dense vs field: oven door 65 vs 32 (field better), cabinet door 9 vs 29,
closet doors 10/11/9/13 vs 1/11/7/19, glass door 24 vs 84, bathroom doors 9/8/10 vs 16/33/14; drawers and
knobs within a few degrees of each other (2-20 vs 4-35); dishwasher wrong type for both. Field wins 3 of 20,
loses 8 by more than 5 deg, the rest tie — consistent with its SF3D signed MA 42.9 vs 42.7 being a tie with a
different error distribution. Masks: field's part masks are sometimes larger / messier (a blob across the
cabinet front on val4233, the metronome panel) where dense stays on the handle; hinge lines land in similar
places (door edges for closets, inside the panel for the close-up bathroom doors). Nothing here argues for
switching the final model.
