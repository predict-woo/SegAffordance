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

**Oracle ceiling (measured through the real exporter + scorer).** Feeding the GT answer strings
through `a3vlm_preds_to_jsonl.py` and scoring them is the ceiling A3VLM can reach on our protocol:

| | MA | type % | PDet | mIoU | axis all | axis matched | origin (m) |
|---|---|---|---|---|---|---|---|
| A3VLM oracle (GT answers) | 99.9 | 100.0 | 34.1 | 0.408 | 1.19 | 1.14 | 0.279 |

Two things to carry into the results table: the answer format costs essentially nothing on the
axis once long segments are used (1.19 deg mean, MA ceiling 99.9), but **PDet tops out at 34.1**
because A3VLM's only spatial output is a 3D box, whose 2D hull is a coarse mask for a thin
functional element. Their segmentation number must be read against that 34.1, not against 100.

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

**As actually run (2026-09-13).** Pod `bl-a3vlm`, **4 x H200 141 GB** in AP-JP-1 on volume
`bl-apjp` (700 GB), $18.36/h — no 8-GPU pod existed in any volume-capable RunPod datacenter (probed
by real create attempts). With model_parallel 2 that is DP=2, so `--accum_iter 32` keeps the
effective batch at 128 as in the recipe. Measured peak memory 79.6 GB/GPU (would not fit 80 GB
H100s). **2 epochs, not the recipe's 3**: the measured cost of 3 was ~$519 against an approved
~$230-300 and the user chose 2 (~$400 with the fixes below). Two operational deviations, neither
touching the training maths: `--save_iteration_interval 4000` instead of 500 (a 135 GB checkpoint
every ~11 min cost 0.40 s/step, 3.6 h of I/O per epoch; the flag counts micro-batches), applied by
stopping at micro-step ~12,500 and resuming from the complete checkpoint at 12,479 (~80 samples
seen twice). Steady-state 0.83 s per micro-step of 4 samples after the change.

**Status.** Training since 01:14 UTC 2026-09-13; expected to finish epoch 1 ~19:20 UTC, eval + export
~20:50 UTC. Results below when it finishes.
