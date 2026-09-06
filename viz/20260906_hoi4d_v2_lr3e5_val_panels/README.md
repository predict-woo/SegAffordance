# HOI4D v2 sweep winner (e100_lr3e5) — validation prediction panels

16 panels from `experiments/20260906_hoi4d_2d_v2_e100_lr3e5/checkpoints/best-epoch77-valloss0.3518.ckpt`
(the from-scratch winner of the 2026-09-06 hyper-parameter sweep: 100 epochs,
lr 3e-5, batch 64, v1 recipe otherwise; held-out mIoU 0.694 / PDet 86.7 /
point 0.0148 / traj shape 0.0329). Left = GT (green mask, GT point, cyan
projected knuckle trajectory), right = prediction (red mask, predicted
point, magenta trajectory, p_rev readout).

Regen (on a training pod, config + data on the volume):
`python tools/hoi4d_vis_2d_panels.py --config config/hoi4d_v2_sweep_e100_lr3e5.yaml
--ckpt experiments/20260906_hoi4d_2d_v2_e100_lr3e5/checkpoints/best-epoch77-valloss0.3518.ckpt
--out viz/20260906_hoi4d_v2_lr3e5_val_panels --num 16` (manifest.yaml has argv).

Reading: masks land on the correct part (drawer fronts, safe doors) with
clean boundaries; the predicted point sits at the knuckle; predicted
trajectories follow the GT track's direction and extent, including door
swings. Caveat: the tool's rot/trans stratification samples C4/C6 only —
the other 11 categories in the v2 set are not shown here. p_rev sits at
0.24-0.31 for both drawers and doors (uncalibrated, as in v1).
