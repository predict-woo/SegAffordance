# 20260721_opdmulti_headsonly

**Goal:** OPDMulti fine-tune with the original project's recipe: freeze
backbone (CLIP), depth encoder and FPN neck; train decoder + heads only.

**Setup:** `config.yaml` (= `config/opdmulti_train_runpod.yaml`,
`train_only_heads: true`); init from 20260721_opdreal_base best; 30 epochs,
lr 1e-5. 20M of 150M params trainable. Note: freezing was requires_grad-only
at the time — backbone BatchNorm running stats still updated (fixed later by
`freeze_backbone`'s eval-mode handling).

**Result:** best val 0.4917 @ epoch 8. 300-sample eval: mIoU 0.566,
IoU>0.5 65.7%, type 96.7%, axis 18.2°.

**Decision:** worst of the three fine-tune recipes — superseded by
20260721_opdmulti_ft_lowlr.
