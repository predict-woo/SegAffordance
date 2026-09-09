# 20260910_sf3d_dct_chain_panels — the DCT chain model (MA 32.80) vs the joint model vs depth, SF3D val

16 stratified SF3D validation samples (seed 42421, g19 filters — the same
picks as the other 2026-09-09/10 SF3D batches), four panels each:
`GT | dct_chain | joint4_dct | depth_tf_plain`.

- `dct_chain` = `20260910_sf3d_g19_dct_rgb_scalefree_ft_multi3dct` best-epoch24-valloss1.0178 (RGB-only scale-free DCT-6, SF3D post-training from the multi3 DCT 2D arm, nothing re-initialised) — MA **32.80** / signed 32.43 (records), origin 0.256, flips 9.83.
- `joint4_dct` = `20260910_joint4_dct_rgb_scalefree` best-epoch12 (joint 2D+3D) — MA 31.41, mIoU 0.2738 (record).
- `depth_tf_plain` = `20260907_sf3d_g19_dct_ft_hoi4d_tf_plain` best-epoch25 (depth input) — 30.62.

Overlays as in every `sf3d_vis_predictions.py` batch (magenta = projected
trajectory, cyan = GT track, red / green = predicted / GT axis, yellow =
decoded sweep, text = type + axis error). `contact_sheet.jpg` = all 16.

Regen (dev pod): `python tools/sf3d_vis_predictions.py --model dct_chain
config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_multi3dct.yaml <ckpt> --model joint4_dct
config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml <ckpt> --model depth_tf_plain
config/sf3d_train_runpod_g19_dct_ft_hoi4d_tf_plain.yaml <ckpt> --data-root
/workspace/datasets/sf3d_processed_v3 --frame-cache-path /workspace/datasets/sf3d_frames_512.lmdb
--input-size 512 --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl
--min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 --num 16
--seed 42421 --out viz/20260910_sf3d_dct_chain_panels` (manifest.yaml has the argv).

What it shows: the DCT chain model puts the hinge ON the GT hinge line with
the right sign — closet door (03): red axis on the green one, 5° (the joint
model has it flipped at 179°, depth 8°); fridge (13), the case every earlier
model failed: 7° with the axis along the fridge edge (joint 26°, depth
147°). Sweeps are smooth and follow the track. This is the visual side of
the origin 0.256 / flips 9.83 / signed-MA 32.43 numbers. Masks are thinner
than the joint model's (mIoU 0.254 vs 0.274), consistent with the numbers.
