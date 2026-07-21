# 20260721_opdmulti_ft_lowlr

**Goal:** Smooth out 20260721_opdmulti_ft_full's instant overfit with a
gentler LR, hoping for a better minimum.

**Setup:** `config.yaml` (= `config/opdmulti_train_runpod_nofreeze_lowlr
.yaml`); init from 20260721_opdreal_base best; 8 epochs, lr 3e-6,
milestones [5,7], nothing frozen.

**Result:** best val 0.4654 @ epoch 2, then stable (no rot, unlike lr 1e-5).
300-sample eval: mIoU 0.587, **IoU>0.5 70.3%** (best), type 97.3%,
**axis 16.8°** (best).

**Decision:** RECOMMENDED OPDMulti checkpoint:
`checkpoints/best-epoch02-valloss0.4654.ckpt`. Val-loss differences vs ft_full
are within noise; this one wins on detection rate and axis error.
Takeaway: OPDMulti overfits within ~2 epochs of full fine-tuning — keep runs
short.
