# 20260721_opdreal_base

**Goal:** Retrain OPDReal from scratch to replace the lost Euler-era
OPDReal_v17 checkpoint, using the regenerated (Codex gpt-5.6-luna)
descriptions.

**Setup:** `config.yaml` (= `config/opdreal_train_runpod.yaml`); 30 epochs,
batch 64, lr 2e-5, MultiStepLR [25,27], everything trainable (CLIP included).
30,537 train / 8,746 val annotations. RTX PRO 6000 96GB, ~5 it/s, ~1.2 h.

**Result:** best `val/loss_total` **0.4069 @ epoch 15**; val plateaus after
(train keeps dropping — mild overfit, best-ckpt selection matters). On 24
val samples: mIoU 0.564, IoU>0.5 16/24, type acc 24/24, axis err 7.0°.
Failures are hard frames (target nearly out of frame / occluded), not
pathologies.

**Decision:** `checkpoints/best-epoch15-valloss0.4069.ckpt` is the pretrain
init for all OPDMulti fine-tunes.
