# 20260721_opdreal_frozenclip

**Goal:** Quantify what fine-tuning CLIP actually buys on OPDReal by freezing
the CLIP backbone (visual + text; `freeze_backbone: true`, eval-mode so BN
stats don't drift) and keeping everything else identical to
20260721_opdreal_base.

**Setup:** `config.yaml`; 30 epochs, lr 2e-5, 46.2M trainable / 104M frozen.
~7.6 it/s vs base's ~5 (frozen backbone skips most of the backward) —
run took ~40 min instead of ~70.

**Result:** best val **0.4138 @ epoch 11** vs base 0.4069 @ ep15 (+0.007).
Late epochs are more stable than base (0.414–0.415 vs base's 0.417–0.419
drift). 24-sample vis: mIoU 0.525, IoU>0.5 13/24, type 24/24, axis 9.5°
(base: 0.564, 16/24, 24/24, 7.0°).

**Decision:** CLIP fine-tuning helps OPDReal only marginally; frozen CLIP is
a legitimate cheaper option (~40% less wall-clock, 1/3 the trainable params)
but the unfrozen base checkpoint remains the pretrain init of choice.
