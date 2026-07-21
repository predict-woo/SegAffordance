# 20260721_opdmulti_tune_lr2e6

**Goal:** LR sweep (round 1) for the OPDMulti full fine-tune.

**Setup:** = ft_full recipe, lr 2e-6, 10 epochs. Init from opdreal_base best.

**Result:** best val 0.4698 @ ep5. 300-sample: mIoU 0.599, **det 71.3%**,
type 96.7%, axis 16.6°. Confirmed on 1000 fresh samples (seed 1):
det 69.6% vs ft_lowlr's 67.9%. 24-sample vis: mIoU 0.607, 17/24.
Note val loss is HIGHER than ft_full's 0.4601 — val loss and detection rate
disagree; we select on detection rate.

**Decision:** NEW RECOMMENDED OPDMulti checkpoint:
`checkpoints/best-epoch05-valloss0.4698.ckpt`. Recipe promoted to
`config/opdmulti_train_runpod_tuned.yaml`.
