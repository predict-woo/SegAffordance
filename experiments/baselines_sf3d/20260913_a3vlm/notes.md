# 20260913_a3vlm — A3VLM (CoRL'24) retrained on SF3D

**Goal.** The image + language articulation baseline: the closest published competitor to our
story, retrained on SF3D and scored with our protocol.
Plan: `docs/superpowers/plans/2026-09-12-sf3d-baselines-a3vlm-3doi.md`.

**Upstream.** github.com/changhaonan/A3VLM @ 436e715 overlaid on Alpha-VLLM/LLaMA2-Accessory
(the fork ships a partial `model/accessory`), SPHINX-1k 13B, recipe = `scripts/a3vlm_train.sh`:
llama_ens5, model_parallel 2, data_parallel sdp, bf16, 3 epochs, lr 2e-5, warmup 0.03, clip 8,
weight decay 0, max_words 2048, `--dialog --image_transform padded_resize`, batch 2 x accum 8 on
16 GPUs = effective batch 128. We keep the effective batch at 128 via accum = 128/(2*DP).

**Data.** `tools/baselines_sf3d/sf3d_to_a3vlm.py` — frames padded to a centred square with the CLIP
mean colour and saved at 448 px (exactly what their transform does), so all normalised coordinates
are the ones the model sees. Three of their 3D tasks: DET (all parts + 8-vertex boxes), REC
(description -> box) and REG-Joint (box -> type + axis). Boxes are axis-aligned camera-frame boxes
around each element's back-projected mask, serialised as their 8 projected vertices [u,v,d] with
d normalised per image by the valid depth range. Train 129,291 samples/epoch (32,171 det + 48,560
rec + 48,560 joint), test 5,088 elements over 3,313 frames.

**Encoding ceiling (measured).** Their answer format quantises (u,v,d) to 2 decimals and one depth
step is ~2.4 cm on SF3D, so the axis endpoints must be far apart or the format itself dominates the
error. With element-sized segments (mean 0.135 m) a perfect model would score only 89.5% within the
10 deg MA threshold (5.09 deg mean). Emitting the longest segment that still projects inside the
padded square (mean 0.50 m) raises that ceiling to **99.9% within 10 deg, 1.21 deg mean**. The
dataset was regenerated with the long segments before training.

**Two scored protocols, one trained model.**
- `preds_chain.jsonl` — language query: the REC answer to our description is fed into their
  REG-Joint question; the mask is the predicted box hull, so PDet is a box IoU.
- `preds_gtbox.jsonl` — their own REG-Joint protocol with the GT box given in the question;
  isolates the joint head (PDet not meaningful there).

**Where it runs.** RunPod, 8 x H100 in EU-FR-1 on volume `bl-eufr` (5pw4vigftc), which holds the
38 GB SPHINX-1k weights and the converted data. It cannot run on Euler: `sdp` shards the optimizer
over data-parallel ranks only, so 2 GPUs (the hard Euler per-user cap) give DP=1 and ~75 GB/GPU of
AdamW state — measured OOM at 74.87/79.19 GiB on 2 x H100.

**Status.** Approved 2026-09-13, waiting on 8- or 4-GPU stock in EU-FR-1 (the volume is locked to
that datacenter). Results below when it finishes.
