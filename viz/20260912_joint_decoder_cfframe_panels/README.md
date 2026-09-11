# 20260912_joint_decoder_cfframe_panels — the MA-record model next to the base decoder

12 SF3D val samples (6 trans + 6 rot, seed 7 — the same picks as `viz/20260912_joint_decoder_panels`),
sharp 2x style, `tools/sf3d_vis_val.py`: GT | `cfframe` = `20260912_joint4_decoder_cfframe` best-epoch17
(MA 36.36 record; SF3D side = the shape-designed closed-form loss) | `dec_base` = `20260911_joint4_decoder_
rgb_scalefree` best-epoch16 (SF3D side = L2 at 2pi). Both through `config/sf3d_test_decoder_rgb_scalefree.yaml`.

What it shows: on the oven door (08) the record model's hinge is 12° from GT and the arc swings the door
DOWN like the GT track, where the base has a 29° axis error; on most doors the two agree to a few
degrees. Origins sit visibly further from the door edge in a few cfframe panels (the metric says +4 cm),
which is the trade the numbers show. Regen: manifest.yaml.
