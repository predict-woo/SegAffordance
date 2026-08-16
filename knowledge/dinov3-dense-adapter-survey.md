# DINOv3 dense-feature + text-fusion survey (2026-08-16)

Two-agent literature sweep (2022–2026) prompted by the question: is our
custom DINOv3 backbone adapter (`model/backbones/dinov3.py` + `pyramid.py`)
standard practice? Short answer: no — three deviations, two of which the
literature says cost us.

## Verdict on our implementation

1. **256 px input (16×16 tokens) is below every published dense recipe.**
   DINOv2 seg probes: 512–640; Depth Anything: 518; DINOv3 linear probes:
   ~1024 patch tokens (512² @ /16). DINOv3 had high-res post-training
   (RoPE + coordinate jitter; stable 256→~4096 px), so raising resolution
   is drop-in — no positional-embedding surgery. Likely THE binding
   constraint before any adapter choice.
2. **Last-layer-only pyramid (our ViTDet-style SimpleFeaturePyramid) is a
   minority choice** borrowed from a fine-tuning regime (ViTDet = MAE,
   fine-tuned, 1024 px). Frozen-DINO practice: **multi-layer taps** —
   ViT-L layers [4, 11, 17, 23] (DINOv2 "lin-4", Depth Anything's DPT,
   SegDINO 2025) reassembled to /8, /16, /32. Our deconv-up /8 branch
   manufactures detail the final layer discarded; early layers hold it.
3. **The aligned-tokens-at-/32 trick is our invention with no precedent,
   and it under-uses dino.txt.** dino.txt (arXiv 2412.16334): frozen ViT +
   2 trainable vision blocks ψ; text aligned to `[CLS'; avg-pool(patch')]`
   — so the 2048-d text vector is TWO halves: first 1024 aligns with CLS
   (global), **only the second 1024 aligns with patches**. Dense inference
   = cosine(ψ patch token @ native /16, LOCAL half of text). Consequences:
   our FPN gate compares the full 2048-d state (half of it in a mismatched
   subspace), and we use aligned tokens only after downsampling to /32
   (native alignment is /16). dinov3.seg (2603.19531) uses BOTH half-
   similarities as complementary correlation features.
4. Register tokens: DINOv3 ships 4 (fixes the ICLR'24 high-norm-artifact
   problem); just strip them before grid reshape (we do — trailing-grid
   slice).

## Ranked upgrades (frozen towers kept)

Adapter side:
- **A1. Input resolution 448–512** (strongest evidence, zero architecture
  change; ~4× backbone FLOPs at 512; train-384/eval-512 is a middle
  ground).
- **A2. 4-layer DPT-style taps** ([4,11,17,23], per-tap 1×1, reassemble to
  /8-/16-/32) replacing the last-layer-only pyramid; feed the existing
  text-gated FPN. Small trainable cost.
- **A3. ViT-Adapter + Mask2Former-class heads** — best published frozen
  numbers (ADE20k 60.2 DINOv2 → 63.0 DINOv3) but heavy; only if A1+A2
  plateau.

Text side:
- **T1. Explicit patch-text cost map at /16**: cosine(ψ tokens, local-half
  text) [+ global-half, dinov3.seg-style], injected as FPN channels + as
  decoder conditioning (ZegCLIP "relationship descriptor", CAT-Seg cost
  volume). Cheapest, directly targets small-referent grounding.
- **T2. Earlier fusion + subspace fix**: aligned ψ tokens at /16, gate on
  the patch-aligned half only, 1–2 bidirectional fusion blocks pre-FPN
  (Grounding DINO / EVF-SAM / TALENT evidence).
- **T3. Target-centric contrastive aux loss** on the cost map (TALENT,
  CVPR'26 — frozen DINOv2 + frozen CLIP text, explicitly built for
  co-category distractors: our "middle drawer in the third row" failure).
  Also: the `text_source="clip"` ablation needs a Talk2DINO-style learned
  projection to be meaningful.

Caveat on cost maps: they light up ALL drawers; relational expressions
still need word-level cross-attention — the cost map is a candidate prior,
the decoder resolves the relation (TALENT's combo is the published fix).

## Key papers

dino.txt 2412.16334 (CVPR'25) · DINOv3 2508.10104 · DINOv2 2304.07193 ·
Registers 2309.16588 (ICLR'24) · SegDINO 2509.00833 · DPT/Depth Anything ·
ViTDet · dinov3.seg 2603.19531 · Talk2DINO 2411.19331 (ICCV'25) ·
ProxyCLIP 2408.04883 · CAT-Seg 2303.11797 · TALENT 2604.00609 (CVPR'26) ·
EVF-SAM 2406.20076 · ZegCLIP 2212.03588 · OOAL 2311.17776 (affordances:
frozen DINOv2 + CLIP text + multi-layer fusion).
