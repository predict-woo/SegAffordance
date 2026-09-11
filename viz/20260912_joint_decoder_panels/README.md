# 20260912_joint_decoder_panels — the analytic decoder's first joint run, rendered

`sf3d/`: 12 SF3D val samples (6 trans + 6 rot, seed 7, sharp 2x style, `tools/sf3d_vis_val.py`):
GT | `decoder` = `20260911_joint4_decoder_rgb_scalefree` best-epoch16 (via `config/sf3d_test_decoder_
rgb_scalefree.yaml`) | `joint4_dct` = `20260910_joint4_dct_rgb_scalefree` best-epoch12. `hoi4d_v2/`,
`epic_v1/`, `arctic_v1/`: 8 held-out records each from the decoder checkpoint (`tools/hoi4d_vis_2d_panels.py
--set use_trajectory_head=false --set trajectory_decoder=analytic`; the rot/trans tag in the EPIC/ARCTIC
file names is the tool's HOI4D-key heuristic, not a label).

What it shows: the decoder's trajectory is by construction a segment of the decoded orbit (rot) or of the
direction ray (trans) — no jitter anywhere. Closet door (11): hinge line on the door's right edge, 6° axis
error, arc along the orbit; joint4's head on the same sample draws a looping jitter. Oven door (08): clean
arc but a 29° axis error (joint4 9°) — the curve is only as right as the axis. Drum pedal (02): the decoder
localises the pedal, joint4 the drum. Hand sources: masks on the right parts, p_rev right, the projected
decoded path short and along the GT track.

Regen: see manifest.yaml in each subdir.
