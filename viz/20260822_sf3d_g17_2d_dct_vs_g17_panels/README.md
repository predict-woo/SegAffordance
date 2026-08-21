# 20260822_sf3d_g17_2d_dct_vs_g17_panels — GT | g17 (3D) | 2D-DCT

16 val samples (seed 42421, family-standard picks), columns: GT |
g17 split-axis (best-epoch18, full 3D supervision) | g17-2d-dct
(best-epoch19, ZERO 3D GT: 2D-only arm-A recipe + truncated-DCT
trajectory head — the newly adopted best 2D arm and p90 pretrain recipe).

Rendered on training pod `g2ddct` (dev pod down); images scp'd to the Mac
mirror by hand.

Regenerate:

```
tools/sf3d_vis_predictions.py \
  --model g17 config/sf3d_train_runpod_g17_splitax.yaml \
    experiments/20260818_sf3d_g17_splitax/checkpoints/best-epoch18-valloss0.9272.ckpt \
  --model g17-2d-dct config/sf3d_train_runpod_g17_2d_dct.yaml \
    experiments/20260822_sf3d_g17_2d_dct/checkpoints/best-epoch19-valloss1.3702.ckpt \
  --data-root /workspace/datasets/sf3d_processed_v3 \
  --frame-cache-path /workspace/datasets/sf3d_frames_512.lmdb --input-size 512 \
  --key-cache /workspace/cache/sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl \
  --min-revolute-radius 0.10 --min-mask-area-frac 0.001 --edge-margin-frac 0.05 \
  --num 16 --seed 42421 --out viz/20260822_sf3d_g17_2d_dct_vs_g17_panels
```

Interpretation (matches the metrics — shape 0.0947 best of any arm, mIoU
0.2655 = 3D level, roughness 0.032; articulation unsupervised):

- **Trajectories are the story: smooth and glued to the GT track.**
  05_rot_val93 (oven door): the magenta sweep rides the cyan GT arc as a
  single clean curve — no jitter at all, visibly smoother than g17's
  trajectory in the same panel. 00_trans_val452 (toilet flush): clean
  straight downward push matching GT. 08_rot_val1684 (door): hugs the GT
  closing arc point-for-point. A slight late-trajectory rightward drift
  recurs in a few panels (also seen in the arm-A batch).
- **Masks/points match the 3D column.** The oven-door blob (05) is as
  crisp as g17's, arguably tighter; interaction points sit on
  handles/buttons in both columns.
- **Ignore the articulation overlays in the 2D-DCT column** — cls is
  often wrong (e.g. trans on the doors of 05/08) and the red predicted
  axis is wild (150° on the flush). The type/axis/origin heads are
  UNSUPERVISED in 2D arms; this is the known deadlock, not a regression,
  and these heads relearn under 3D supervision in the ft10 finetune.

Checkpoints: 20260818_sf3d_g17_splitax/best-epoch18-valloss0.9272,
20260822_sf3d_g17_2d_dct/best-epoch19-valloss1.3702.
