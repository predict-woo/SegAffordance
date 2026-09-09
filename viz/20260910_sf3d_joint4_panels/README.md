# 20260910_sf3d_joint4_panels — the joint 2D+3D model on SF3D val, vs the plain chain model and the depth model

16 stratified SF3D validation samples (seed 42421, g19 filters — the same
picks as `viz/20260909_sf3d_plain_ft_multi3_panels`), four panels each:
`GT | joint4_dct | plain_ft_multi3 | depth_tf_plain`.

- `joint4_dct` = `20260910_joint4_dct_rgb_scalefree` best-epoch12-sf3dval0.9688 (RGB-only scale-free DCT-6, trained jointly on SF3D + HOI4D + EPIC + ARCTIC) — MA **31.41** (record) / PDet 22.72 / mIoU 0.2738 (record) / traj_dir 96.4.
- `plain_ft_multi3` = `20260909_sf3d_plain_rgb_scalefree_ft_multi3` best-epoch19 (RGB chain, plain head) — 31.01 / 22.72 / 0.2625.
- `depth_tf_plain` = `20260907_sf3d_g19_dct_ft_hoi4d_tf_plain` best-epoch25 (depth input, DCT head) — 30.62 / 20.03 / 0.2555.

Overlays as in every `sf3d_vis_predictions.py` batch (magenta = projected
trajectory head, z_p-scaled for the scale-free models; cyan = GT track; red
/ green = predicted / GT axis; yellow = decoded sweep; text = type call +
axis error). `contact_sheet.jpg` = all 16 at 0.4 scale.

Regen (dev pod): `python tools/sf3d_vis_predictions.py --model joint4_dct
config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml <ckpt> --model plain_ft_multi3
config/sf3d_train_runpod_plain_rgb_scalefree_ft_multi3.yaml <ckpt> --model depth_tf_plain
config/sf3d_train_runpod_g19_dct_ft_hoi4d_tf_plain.yaml <ckpt> --data-root
/workspace/datasets/sf3d_processed_v3 --frame-cache-path /workspace/datasets/sf3d_frames_512.lmdb
--input-size 512 --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl
--min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 --num 16
--seed 42421 --out viz/20260910_sf3d_joint4_panels` (manifest.yaml has the exact argv).

What it shows: the joint model's sweeps are smooth (DCT head) AND run the
full length of the GT track (the plain chain model also runs the track but
zigzags; the depth model's arcs are smooth but shorter on some doors) —
the visual counterpart of traj_dir 96.4. On the fridge (13), the hard case
where every earlier model failed (depth 147°, plain 36°), the joint model
places the axis at 26° with the mask on the fridge door. Its remaining
weakness is the axis SIGN: on the closet door (03) the hinge line is on the
right edge with the direction flipped (179°), i.e. the rot flip rate (13.6%)
rather than placement; the plain chain model gets 4° there. Masks are the
best of the three (mIoU 0.2738), consistent across the sheet.
