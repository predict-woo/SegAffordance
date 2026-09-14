# SF3D val frames: GT | OPDFormer-C (oracle-matched instance) | ours (paper Fig. 4 source)

`tools/sf3d_vis_baseline_vs_ours.py --val-idx 1696 1773 529 4233 283 501 1140 1246 4193 615` with ours =
`20260913_joint4_decoder_l2anchor_dense` best-epoch13 and the baseline column read from
`/workspace/datasets/baselines/results/opd_c_rgbd/preds.jsonl` (the 256x192 OPDFormer-C run, scored by the
baselines session with best-IoU instance matching). Header "axis err" is the signed angle.

**Read (2026-09-14).** OPDFormer-C's oracle instance lands on the handle (that is what the oracle picks) at
scores 0.00-0.01, but its axis is tilted across the door (33 deg closet, 74 / 72 deg drawer / metronome) or
flipped (176 deg closet), and 2/10 frames have no overlapping instance at all. Ours: 2-14 deg on 8/10,
24 deg glass door, dishwasher type wrong (93). Rows in the paper: 1696, 4233, 283, 501.
