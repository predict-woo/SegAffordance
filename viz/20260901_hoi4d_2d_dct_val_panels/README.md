# 20260901_hoi4d_2d_dct_val_panels — GT | hoi4d-2d on HOI4D val

10 val samples (seed 42421), columns GT | 20260901_hoi4d_2d_dct
(best-epoch28). 12 images (6 drawer / 6 safe), rendered on the dev pod, synced to the
Mac. manifest.yaml has the exact command.

Uses the training split (0.15, seed 42) via tools/hoi4d_vis_2d_panels.py.
Trajectory overlay = the trainer's exact projection. KEY OBSERVATION:
the GT masks are often the HAND (the data pipeline's moving-part bug);
the model reproduces them faithfully — and occasionally predicts the
drawer against a hand label (00_trans).
