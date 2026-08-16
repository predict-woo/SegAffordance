# DINOv3 Research-Standard Stack (gens 13–15)

**Date:** 2026-08-16
**Status:** DRAFT — awaiting user review
**Basis:** `knowledge/dinov3-dense-adapter-survey.md` (two-agent literature
sweep). User decisions: one spec, three independent flag-gated stages run
as SEQUENTIAL experiments for clean attribution; resolution target **512**;
dinov3 path only (CLIP family stays at 256 as the established baseline).
**Baseline for all stages:** gen-12 (`20260816_sf3d_g12_dinov3`) — the
current custom adapter inside the modern recipe (v3 data, trajectory_weight
0.15, origin local sample, normalized L_pp). Each stage adds ONE flag to
its predecessor's config.

## Stage R — input resolution 512 (gen-13)

**Why first:** strongest evidence, zero architecture change. Every
published dense DINO recipe runs 512-class resolution; our 256 px gives a
16×16 token grid (each token = 6% of image width) — the plausible binding
constraint behind the close-up/mask ceiling. DINOv3's RoPE high-res
post-training makes 512 native (no positional-embedding surgery).

**Infra (one-time, before the run):** build a 512-target frame cache with
the existing parameterized builder:

```
python tools/sf3d_build_frame_cache.py \
  --data-root /workspace/datasets/sf3d_processed_v2 \
  --out /workspace/datasets/sf3d_frames_512.lmdb  [512-target params]
```

(~45 min pod job, FUSE-read-bound; est. 35–50 GB — volume has room. The
reader already validates cache depth size against `input_size`, so a
mismatched cache fails loudly. v3 symlinks it alongside the v2 one.)

**Config deltas (gen-13 = gen-12 +):**

- `config.input_size: [512, 512]`, `model_params.backbone_image_size: 512`
- `data.frame_cache_path: /workspace/datasets/sf3d_frames_512.lmdb`
  (launch staging copies THIS file; ~40 G in /dev/shm — needs a
  6000-class pod, which ViT-L wants anyway)
- `loss_params.point_sigma: 16.0` — the Gaussian GT sigma is in heatmap
  pixels; 8 px at 256 = the same physical extent as 16 px at 512. Without
  this the point/origin heatmap targets silently shrink 2×.
- `data.batch_size_train/val`: start **64** (activations ≈ 4×/sample;
  128 @ 256 fit a 24 GB card, so 64 @ 512 should fit 96 GB comfortably —
  bump back to 128 if headroom allows after the first epoch's memory
  high-water). `lr` unchanged (AdamW; note the batch change in the
  experiment record).
- Steps/epoch double at bs 64 (~925) — keep `max_epochs: 30`,
  `scheduler_milestones: [24, 28]` (epoch-based, unchanged).

**Watch/success:** mIoU + PDet (the resolution hypothesis's direct
target; gen-12 baseline TBD), 2D point err, small-mask samples in viz.
Trajectory/articulation metrics should hold or improve. Cost ~2–3× gen-12
wall-clock on the same GPU class.

## Stage T — 4-layer DPT-style taps (gen-14 = gen-13 + `dinov3_multilayer_taps`)

**Why:** frozen-DINO standard practice (DINOv2 lin-4 / Depth Anything /
SegDINO) taps intermediate layers; our /8 level is currently deconvolved
out of the FINAL layer, which no longer holds high-frequency spatial
detail. ViTDet's last-layer pyramid was validated on fine-tuned MAE — a
different regime.

**Change (`model/backbones/dinov3.py`, flag `dinov3_multilayer_taps:
false` default):** when true, pull ViT-L intermediate layers
**[4, 11, 17, 23]** via the dinov3 repo backbone's
`get_intermediate_layers` (available on the hub model's
`visual_model.backbone`), then reassemble:

- `/8` = tap 4 → 1×1 proj → deconv ×2 (ConvTranspose + BN + GELU) →
  `fpn_in[0]` channels
- `/16` = concat(tap 11, tap 17) → 1×1 → `fpn_in[1]`
- `/32` = tap 23 → stride-2 conv → `fpn_in[2]`, **still built from the
  dino.txt-ALIGNED head tokens when `text_source="dinotxt"`** (the
  aligned head consumes the final layer anyway) — i.e. `x_deep` behavior
  is preserved; only the /8 and /16 sources change. The gate-subspace
  question belongs to Stage C, not here.

`SimpleFeaturePyramid` stays for the default path (gen-12/13 reproduce
bit-identically with the flag off). New module `MultiTapPyramid` beside it
in `pyramid.py`. Registers/CLS stripped per tap exactly as today
(trailing-grid slice).

**Watch/success vs gen-13:** mask/point sharpness again (mIoU, PDet, 2D
point), plus depth-lift quality (`point_err_3d`, `origin_err`) — DPT
evidence says intermediate taps help metric depth most.

## Stage C — patch-text cost map + gate subspace fix (gen-15 = gen-14 + `text_cost_map`)

**Why:** dino.txt's text embedding is two halves — first 1024-d aligns
with CLS (global), only the second 1024-d with patches. Our FPN gate
currently compares the FULL 2048-d state (half in a mismatched subspace),
and we never use the patch-text similarity explicitly. ZegCLIP / CAT-Seg /
dinov3.seg / TALENT all show explicit similarity maps beat implicit fusion
for small/distractor-heavy referents ("the middle drawer in the third
row" = our dominant error).

**Change (flag `text_cost_map: false` default), dinotxt path only:**

1. Compute two cosine-similarity maps at native /16: ψ patch tokens vs
   the LOCAL half of the pooled text embedding, and vs the GLOBAL half
   (dinov3.seg: complementary).
2. Resize both maps to each FPN level and concatenate as 2 extra input
   channels per level (lateral convs take `fpn_in[i] + 2`; flag-off keeps
   today's dims).
3. Gate subspace fix, same flag: the FPN text gate consumes only the
   patch-aligned LOCAL half (1024-d) of the state instead of the full
   2048-d vector.

Decoder-side ZegCLIP-style relationship descriptor: OPTIONAL stretch,
only if the cost-map run leaves grounding on the table — kept out of the
committed scope.

**Watch/success vs gen-14:** grounding-tail metrics — PDet, the wrong-
element rate in viz batches, point 2D/3D tails; type/axis should be
untouched.

## Constants across stages

v3 dataset, gen-9 split (59,174 / same key cache), 30 epochs, milestones
[24, 28], seed 42, frozen towers, `trajectory_weight 0.15`, normalized
L_pp @ 0.1, origin local sample ON, 6000-class pods. Each stage's config
derives from its predecessor with ONLY the stage's flag(s) + experiment
paths changed; every flag defaults off so gen-12 reproduces bit-identically.

## Out of scope

- ViT-Adapter / Mask2Former-class heads (survey A3 — only if R+T plateau).
- TALENT-style target-centric contrastive loss (recorded as the follow-up
  if Stage C's cost map helps but distractors persist).
- CLIP-RN50 high-resolution arm; unfreezing either tower;
  `text_source="clip"` bridging (needs Talk2DINO-style projection — noted,
  not planned).
