# 20260721_opdmulti_frozenclip

**Goal:** Middle ground between headsonly (freeze CLIP+depth+neck) and
ft_full (freeze nothing): freeze ONLY the CLIP backbone, train neck +
decoder + heads. Does keeping CLIP fixed prevent the instant overfit?

**Setup:** `config.yaml`; init from 20260721_opdreal_base best; 12 epochs,
lr 1e-5, `freeze_backbone: true` (eval-mode CLIP), `train_only_heads: false`.

**Result:** best val **0.4683 @ epoch 0**, degrading monotonically after —
the instant-overfit pattern is NOT caused by CLIP fine-tuning; the
neck/decoder/heads overfit 9.5k samples on their own. 300-sample eval:
mIoU 0.581, IoU>0.5 67.3%, type 95.7%, axis 17.5°. 24-sample vis: mIoU
0.604, 16/24, type 21/24, axis 22.9°.

**Decision:** Ranks between headsonly (65.7%) and the full fine-tunes
(68.0% / 70.3%). Does not beat 20260721_opdmulti_ft_lowlr, which stays the
recommended checkpoint.
