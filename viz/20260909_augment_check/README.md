# 20260909_augment_check — geometry-consistent augmentation on real records

8 random TRAIN records of the multi-source config (`config/multi3_rgb_scalefree.yaml`:
HOI4D v2 + EPIC v1 + ARCTIC v1), each as `[raw | aug 1 | aug 2 | aug 3]` with
the augmentation of `datasets/augment.py` at the config's settings
(photometric jitter, horizontal flip with left/right text swap, scale +
translate crop that keeps the whole mask and the point in frame). Every
overlay is drawn FROM THE TRANSFORMED FIELDS: mask green, bbox orange,
point white ring, 2D track cyan, and — where the record's 3D track is
metric (ARCTIC) — `project(K, traj3d)` in magenta, which must coincide with
the cyan track if the intrinsics and the 3D fields were transformed
consistently.

Regen (dev pod): `python tools/augment_check.py --config config/multi3_rgb_scalefree.yaml
--out viz/20260909_augment_check --num 8` (manifest.yaml has argv).

What it shows: on all three sources the mask, box, point and track move
together under zoom, flip and colour changes (HOI4D truck 01, EPIC drawer
03, ARCTIC scissors 00); on the ARCTIC waffle iron (07) the magenta
projection of the 3D track lies exactly on the cyan 2D track in the raw
frame, the zoomed frame and both flipped frames — the intrinsics and the
camera-frame 3D transform consistently. The crop never cuts the element:
the whole mask and the point are inside every window. Unit tests:
`tests/test_augment.py` (projection identity under flip and crop, flip is
an involution, axis convention under reflection, must-keep box, text swap,
determinism, RGB-only photometric).
