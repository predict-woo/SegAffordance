# 20260721_opdmulti_ft_full

**Goal:** Test whether full fine-tuning (nothing frozen) beats the heads-only
recipe on OPDMulti.

**Setup:** `config.yaml` (= `config/opdmulti_train_runpod_nofreeze.yaml`,
`train_only_heads: false`); init from 20260721_opdreal_base best; 30 epochs,
lr 1e-5.

**Result:** best val **0.4601 @ epoch 0** — a single epoch of full
fine-tuning beats the entire heads-only run, then val degrades monotonically
(clear overfitting on 9.5k samples). 300-sample eval (ep0 ckpt): mIoU 0.592,
IoU>0.5 68.0%, type 97.7%, axis 17.3°.

**Decision:** full fine-tuning wins but needs very short runs → motivated the
low-LR variant (20260721_opdmulti_ft_lowlr).
