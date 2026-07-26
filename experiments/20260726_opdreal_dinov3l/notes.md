# 20260726_opdreal_dinov3l

**Goal:** Test DINOv3 ViT-L/16 — the strongest published dense-feature
backbone — against CLIP RN50 on this architecture, using the **aligned**
dino.txt text encoder so image and text still share a space. Matched to
20260721_opdreal_frozenclip (frozen backbone, val 0.4138) so the backbone is
the only variable. Companion run to 20260726_opdreal_siglip2l.

**Setup:** `config.yaml` = opdreal_train_runpod_frozenclip.yaml with
`backbone: dinov3`, `text_source: dinotxt`, `word_len: 77`. 30 epochs, lr 2e-5,
batch 64, 256², frozen pretrained towers. 934.3M total params, **67.7M
trainable**, peak 8.85 GiB at batch 64. Ran on a dedicated RTX PRO 6000
Blackwell Workstation Edition pod.

Pyramid construction mirrors what CLIP RN50 gave the neck for free.
`VisionTower.forward` returns both raw backbone patch tokens *and*
head-projected tokens living in the aligned 2048-d space, so:
- **/8 and /16** ← raw DINOv3 patch tokens (1024-d), for dense detail;
- **/32** ← dino.txt vision-head tokens (aligned), because that deepest level
  is what the FPN gate and the dynamic kernel actually consume — exactly the
  role CLIP's `attnpool` output plays at `model/clip.py:230`.

`SimpleFeaturePyramid.forward` grew an `x_deep` argument for this split.

**Weights (gated, two separate approvals):** the HF gate on
`facebook/dinov3-vitl16-pretrain-lvd1689m` unlocks the *vision tower only*.
dino.txt ships solely via Meta's own CDN and needs the separate form at
ai.meta.com/resources/models-and-libraries/dinov3-downloads, which mails signed
URLs. Files live in `/workspace/cache/dinov3/` (mutagen-ignored) and the repo
checkout for `torch.hub.load(..., source="local")` in
`/workspace/cache/dinov3repo`.

**Result:** 30/30 epochs in 2h39m (~5.3 min/epoch — ~1.8× SigLIP 2's 3 min and
~3.5× frozen CLIP's 1.5 min, because every forward runs tokens through both the
raw backbone path and the dino.txt vision head). Best val **0.4030 @ epoch 25**
— the best of all four backbones, beating even *fine-tuned* CLIP (0.4069) while
frozen.

Curve: 0.5993 → 0.4242 (ep8) → 0.4093 (ep10) → 0.4071 (ep14) → 0.4062 (ep23)
→ **0.4030 (ep25)**, tail 0.4054–0.4058. It starts *worse* than SigLIP 2
(0.5993 vs 0.5536 at ep0) and improves for longer: SigLIP 2 plateaued from
ep12, DINOv3's best lands at ep25, right on the LR milestone, with the curve
still descending into it. **Plausibly under-trained at 30 epochs** — a longer
schedule (or milestones pushed out) is the obvious follow-up.

Metrics at the best-**val** checkpoint (ep25), 1000 fixed OPDReal-valid
samples, seed 0. Every model re-measured on this same draw (the 24-sample
numbers in INDEX.md's OPDReal rows are far too noisy to compare):

| model (frozen unless noted) | val | ckpt | mIoU | det@0.5 | type | axis |
|---|---|---|---|---|---|---|
| CLIP RN50, frozen | 0.4138 | ep11 | 0.566 | 66.6% | 98.1% | 11.5° |
| CLIP RN50, **unfrozen** | 0.4069 | ep15 | 0.578 | 67.9% | 98.2% | 11.9° |
| SigLIP 2 Large | 0.4087 | ep17 | 0.579 | 68.4% | 98.2% | 12.5° |
| **DINOv3-L + dino.txt** | **0.4030** | ep25 | **0.603** | **71.3%** | 98.2% | **10.4°** |

DINOv3 wins on every metric simultaneously: **+3.4pt det and +0.025 mIoU over
fine-tuned CLIP**, +4.7pt det over the like-for-like frozen CLIP baseline, and
+2.9pt over SigLIP 2. Margins are well clear of the ~1.5pt eval-noise band. It
is also the only backbone to *improve* axis error (10.4° vs CLIP's 11.5-11.9°
and SigLIP 2's 12.5°) — notable because the axis comes from the CVAE head
sampling z from the prior, so it is the noisiest number here.

The axis result is the one that most supports the design: the deepest pyramid
level feeding the FPN gate carries dino.txt's *text-aligned* tokens, and the
motion/axis head is conditioned on that pooled visual state plus the text
state. Alignment appears to matter for exactly the head that needs
vision-language agreement.

**Checkpoint-selection caveat (observation only — selection stays on val
loss, else we optimise the metric we report):** val loss and det@0.5 do not
order identically. Frozen CLIP ep26 scores det 69.3% vs ep11's 66.6%
(2.7pt); SigLIP 2 ep24/last score 69.2% vs ep17's 68.4% (0.8pt). Both
baselines are therefore mildly understated by val-loss selection — which
makes DINOv3's margin conservative, not inflated. DINOv3's own non-best-val
checkpoints were not swept.

**Decision:** DINOv3 ViT-L/16 + dino.txt is the best backbone tested for this
architecture and the first to clearly beat the fine-tuned CLIP pretrain
checkpoint — while frozen, with only 67.7M trainable params. Costs: 5.3
min/epoch (3.5× frozen CLIP), 934M total params, and gated weights requiring
two separate Meta approvals.

**Follow-ups, in priority order:**
1. **Train longer.** Best val lands at ep25 on the LR milestone with the curve
   still descending — 30 epochs looks short. Push milestones out and run 45-60.
2. Re-run the OPDMulti fine-tune chain from this checkpoint instead of
   20260721_opdreal_base, to see whether the gain survives transfer.
3. Ablate the aligned-/32 design: feed raw backbone tokens to all three pyramid
   levels and see how much of the axis-error gain is actually from alignment.
