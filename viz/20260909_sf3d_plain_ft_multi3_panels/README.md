# 20260909_sf3d_plain_ft_multi3_panels — the plain-head RGB post-trained model vs the DCT RGB arm vs depth

16 stratified SF3D validation samples (seed 42421, g19 filters), four panels
each: `GT | plain_ft_multi3 | dct_ft_hoi4d_rgb | depth_tf_plain`.

- `plain_ft_multi3` = `20260909_sf3d_plain_rgb_scalefree_ft_multi3` best-epoch19-valloss1.0953 (RGB-only scale-free, PLAIN head kept from the multi-source 2D arm, nothing re-initialised) — MA 31.01 / PDet 22.72 / mIoU 0.2625, roughness 0.068.
- `dct_ft_hoi4d_rgb` = `20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d` best-epoch24 (RGB-only, DCT head re-initialised at the hand-over from the HOI4D-only arm) — 30.07 / 22.35 / 0.266, roughness 0.009.
- `depth_tf_plain` = `20260907_sf3d_g19_dct_ft_hoi4d_tf_plain` best-epoch25 (depth input, DCT head) — 30.62 / 20.03 / 0.2555.

Overlays as in every `sf3d_vis_predictions.py` batch: magenta = the 3D
trajectory head projected (z_p · Δ̃ for the scale-free heads), cyan = GT
track, red = predicted axis, green = GT axis, yellow = decoded sweep, text
= type call + axis error. `contact_sheet.jpg` = all 16 at 0.4 scale.

Regen (dev pod): `python tools/sf3d_vis_predictions.py --model plain_ft_multi3
config/sf3d_train_runpod_plain_rgb_scalefree_ft_multi3.yaml <ckpt> --model
dct_ft_hoi4d_rgb config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml
<ckpt> --model depth_tf_plain config/sf3d_train_runpod_g19_dct_ft_hoi4d_tf_plain.yaml
<ckpt> --data-root /workspace/datasets/sf3d_processed_v3 --frame-cache-path
/workspace/datasets/sf3d_frames_512.lmdb --input-size 512 --key-cache
/workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl
--min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05
--num 16 --seed 42421 --out viz/20260909_sf3d_plain_ft_multi3_panels` (manifest.yaml has the exact argv).

What it shows: the plain arm's projected sweeps follow the GT track much
FURTHER than the DCT RGB arm's (closet door 03: the magenta dots run the
whole cyan track where the DCT arm stops near the handle) — the 2D-learned
readout that survived the hand-over — at the price of visible jitter
(roughness 0.068 vs 0.009). Axes are tight on the easy cases (closet door
4°, drawer 3°, the best of the three), but the hinge PLACEMENT is the weak
spot: on the closet door the red axis sits on the wrong door edge, the
visual face of the 2 cm worse origin error. Type calls match the other two
models. The depth model still has the longest, smoothest arcs.
