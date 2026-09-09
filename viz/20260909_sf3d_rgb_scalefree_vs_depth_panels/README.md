# 20260909_sf3d_rgb_scalefree_vs_depth_panels — SF3D val: RGB-only scale-free models vs the depth model

16 stratified SF3D validation samples (seed 42421, g19 filters), four panels
each: `GT | rgb_ft_hoi4d | rgb_scratch | depth_tf_plain`.

- `rgb_ft_hoi4d` = `20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d` best-epoch24-valloss1.0483 (no depth anywhere, scale-free head, init = the RGB-only HOI4D arm) — MA 30.07 / PDet 22.35 / mIoU 0.266.
- `rgb_scratch` = `20260909_sf3d_g19_dct_rgb_scalefree` best-epoch20-valloss1.0791 (same model from scratch) — MA 24.88 / PDet 17.39 / mIoU 0.2425.
- `depth_tf_plain` = `20260907_sf3d_g19_dct_ft_hoi4d_tf_plain` best-epoch25-valloss0.9768 (depth input, metric head, the depth counterpart of the first) — MA 30.62 / PDet 20.03 / mIoU 0.2555.

Overlays as in every `sf3d_vis_predictions.py` batch: magenta = the 3D
trajectory head projected (for the RGB models: z_p · Δ̃, exactly the test-
time convention), cyan = GT track, red = predicted axis, green = GT axis,
yellow = the decoded sweep, text = type call + axis error. `contact_sheet.jpg`
= all 16 at 0.4 scale.

Regen (dev pod):
```
python tools/sf3d_vis_predictions.py \
  --model rgb_ft_hoi4d config/sf3d_train_runpod_g19_dct_rgb_scalefree_ft_hoi4d.yaml experiments/20260909_sf3d_g19_dct_rgb_scalefree_ft_hoi4d/checkpoints/best-epoch24-valloss1.0483.ckpt \
  --model rgb_scratch config/sf3d_train_runpod_g19_dct_rgb_scalefree.yaml experiments/20260909_sf3d_g19_dct_rgb_scalefree/checkpoints/best-epoch20-valloss1.0791.ckpt \
  --model depth_tf_plain config/sf3d_train_runpod_g19_dct_ft_hoi4d_tf_plain.yaml experiments/20260907_sf3d_g19_dct_ft_hoi4d_tf_plain/checkpoints/best-epoch25-valloss0.9768.ckpt \
  --data-root /workspace/datasets/sf3d_processed_v3 --frame-cache-path /workspace/datasets/sf3d_frames_512.lmdb --input-size 512 \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl \
  --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 \
  --num 16 --seed 42421 --out viz/20260909_sf3d_rgb_scalefree_vs_depth_panels
```

What it shows: the RGB-only post-trained model is visually on par with the
depth model — the same type calls on the easy cases (toilet flush, closet
door, drawer), axes within a few degrees of each other (closet door: rgb_ft
4° vs depth 8°; drawer: 7° vs 4°), the point on the right handle, and the
projected trajectories running along the GT track in the right direction.
Where they differ: the RGB trajectories tend to be SHORTER than the depth
model's along the track (the closet door arc stops near the handle where
the depth model's follows the cyan track further) — the visual face of the
trajectory val-loss gap (0.48 vs 0.35). The hard cases fail for all three
(the fridge: axes 112° / 32° / 147°). The scratch RGB model is the weakest
of the three on axes and sweeps, as its numbers say.
