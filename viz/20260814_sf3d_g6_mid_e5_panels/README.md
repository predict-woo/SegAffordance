# 20260814_sf3d_g6_mid_e5_panels

**MID-TRAIN snapshot** of the gen-6 split-heads arm — epoch 5 of 16
(best-epoch05-valloss1.0279.ckpt), rendered while the run was still in
flight on the training pod (~epoch 9). NOT final weights; the finished
run gets its own batch.

16 stratified val samples, seed 3, FILTERED split (min_revolute_radius
0.10) — the SAME sample draw as `20260813_sf3d_g4_vs_g5_panels`, so
panels can be flipped between folders for a cross-generation eyeball.

First render of a split-arm checkpoint: the 2D marker is the PROJECTED
predicted 3D interaction point (no heatmap exists on this arm), the
trajectory is anchored at that predicted 3D point, and the RED axis is
drawn from the predicted origin q̂ + direction d̂ (no twist decode).
GREEN = GT axis, `ax=` = signed direction error, `cls=` from the type
head's logits.

Early-epoch reading (val: total 1.0279, traj 0.0134, origin 0.898,
point3d 0.208): type and direction are already strong — e.g.
`13_rot_val14122` "Close the door" decodes rot at ax=19deg,
`05_trans_val7950` trans at ax=5deg with the ray parallel to GT. What
is still coarse at epoch 5: the predicted origin/hinge placement (the
red axis often sits off the true hinge edge — hinge-side ambiguity is
exactly where a K=1 head hedges), the interaction-point anchor (~0.4 m
RMS, e.g. one drawer off), patchy masks, and short/wiggly trajectories.

Regenerate (dev pod, repo root):

```
python tools/sf3d_vis_predictions.py \
  --model g6e5 config/sf3d_train_runpod_split.yaml experiments/20260813_sf3d_split_g6/checkpoints/best-epoch05-valloss1.0279.ckpt \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010.pkl --min-revolute-radius 0.10 \
  --out viz/20260814_sf3d_g6_mid_e5_panels --num 16 --seed 3
```
