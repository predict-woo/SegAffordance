# 20260726_opdreal_siglip2l

**Goal:** Test whether a modern vision-language backbone beats CLIP RN50 on
this architecture. SigLIP 2 Large replaces both towers, keeping image and text
in a *shared* embedding space — which the FPN's multiplicative gate
(`layers.py:527`) and the dynamic-kernel projector both depend on. Matched to
20260721_opdreal_frozenclip (frozen backbone, val 0.4138) so the backbone is
the only variable.

**Setup:** `config.yaml` = opdreal_train_runpod_frozenclip.yaml with
`backbone: siglip2`, `backbone_id: google/siglip2-large-patch16-256`,
`word_len: 64`. 30 epochs, lr 2e-5, batch 64, 256², frozen pretrained towers.
941.1M total params, **59.5M trainable** (vs 46.2M for frozen CLIP — the extra
is the ViT→pyramid adapter and the 1024→512 word projection). Peak 8.73 GiB at
batch 64. Ran on a dedicated RTX PRO 6000 Blackwell Server Edition pod.

Backbone-specific handling worth recording:
- SigLIP 2's vision tower is a plain ViT, so the /8-/16-/32 pyramid the FPN
  needs is rebuilt by a ViTDet-style `SimpleFeaturePyramid`.
- Tokenizer is Gemma sentencepiece capped at **64** tokens, not CLIP's 77-token
  BPE — hence `word_len: 64`.
- SigLIP 2 expects 0.5/0.5/0.5 normalisation; the dataloader emits ImageNet
  stats, so the backbone re-normalises internally
  (`model/backbones/base.py:_register_renorm`).
- The text tower is 564M params on its own, dominated by the 256k-vocab
  embedding table (262M). That, not the ViT, is why total is 941M.

**Result:** 30/30 epochs in ~87 min (~3 min/epoch on RTX PRO 6000, vs ~1.5 for
frozen CLIP RN50). Best val **0.4087 @ epoch 17**, against frozen CLIP's
0.4138 and unfrozen CLIP's 0.4069 — i.e. a frozen SigLIP 2 backbone beats
frozen CLIP by 0.005 and lands within 0.002 of *fine-tuned* CLIP.

Curve: 0.5536 → 0.4234 (ep7) → 0.4127 (ep10) → 0.4087 (ep17), then flat.
A wobble at ep22-23 (0.4154, 0.4190) is damped by the LR drop at milestone 25;
the last six epochs sit in a tight 0.4095–0.4113 band. No runaway overfitting,
unlike OPDMulti fine-tuning.

Metrics at the best-**val** checkpoint (ep17), 1000 fixed OPDReal-valid
samples, seed 0 — every model re-measured on the same draw, because the
24-sample numbers in INDEX.md for the OPDReal rows are far too noisy to
compare:

| model (frozen unless noted) | val | ckpt | mIoU | det@0.5 | type | axis |
|---|---|---|---|---|---|---|
| CLIP RN50, **unfrozen** | 0.4069 | ep15 | 0.578 | 67.9% | 98.2% | 11.9° |
| CLIP RN50, frozen | 0.4138 | ep11 | 0.566 | 66.6% | 98.1% | 11.5° |
| **SigLIP 2 Large** | 0.4087 | ep17 | 0.579 | **68.4%** | 98.2% | 12.5° |

vs the like-for-like frozen CLIP baseline: **+1.8pt det, +0.013 mIoU**, and
+0.005 val. Against *fine-tuned* CLIP it is +0.5pt det — a tie inside the
noise band, which is itself notable given SigLIP 2's backbone never trains.
Axis error is the one regression (12.5° vs 11.5°).

**Checkpoint-selection caveat (observation only — selection stays on val
loss, else we optimise the metric we report):** det@0.5 does not order the
same way as val loss. SigLIP 2 ep24 (val 0.4095, *worse*) scores det 69.2% /
mIoU 0.590, beating its own best-val ep17 by 0.8pt; ep14 (val 0.4093) scores
66.0%. The effect is bigger on the frozen-CLIP baseline: ep26 hits det 69.3%
vs ep11's 66.6% (**2.7pt**). So absolute det numbers here are contingent on
val-loss selection, and cross-model det gaps under ~1.5pt should not be read
as real (see knowledge/training-findings.md on eval noise).

**Decision:** SigLIP 2 Large is a genuine improvement over frozen CLIP RN50
and matches unfrozen CLIP while keeping its backbone frozen — but it does not
beat fine-tuned CLIP outright, and it costs 2× the epoch time and 6× the
parameters (941M vs 150M). Worth keeping as an option; not yet a reason to
replace the CLIP pretrain checkpoint. Compare against
20260726_opdreal_dinov3l before deciding.
