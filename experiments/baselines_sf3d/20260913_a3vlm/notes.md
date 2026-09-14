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

**Training curve.** Causal-LM loss (`closs`, running average) 1.58 at the first steps -> 0.22 at
micro-step 16,480 -> 0.20 at 24,480 -> 0.15 at the end of epoch 0 (32,320) -> ~0.15 throughout
epoch 1 (fluctuating 0.14-0.17, grad norm ~0.3). **Learning-rate schedule — correction.** The
restart was meant to run with `--epochs 2`, but the chain's train branch hardcoded 3, so the run
used the recipe's 3-epoch cosine throughout (lr 1.5e-5 at the end of epoch 0, 0.5e-5 at the end of
epoch 1) and, unnoticed for ~1.5 h, continued into a third epoch (to micro-step 5,810, ~$27) before
being stopped. The evaluated model is therefore **the end-of-epoch-1 checkpoint of a 3-epoch
schedule, i.e. 2 of 3 epochs at the recipe's own learning-rate curve** — the "stopped early"
reading, not a re-annealed 2-epoch run. An earlier version of these notes claimed the latter; it
was wrong. The partial third-epoch checkpoint was deleted and is not used anywhere.

**Status.** Training 01:14-18:53 UTC 2026-09-13 (epoch 0 done 10:52, epoch 1 done 18:53; epoch 1
took 7:54). Evaluation + export from `epoch1` 20:20-23:09 UTC. Both question sets parsed 5,088/5,088.

**Results (our protocol, `score_predictions.py`, 5,088 test elements).**

| protocol | MA | MA signed | type % | axis all / matched | flips all / rot | origin_err_m | origin_line_err_m | PDet | mIoU |
|---|---|---|---|---|---|---|---|---|---|
| GT box -> REG-Joint (their protocol) | 47.4 | 45.7 | 96.5 | 18.7 / 20.8 deg | 9.4 / 8.6 | 0.361 | 0.198 | 34.1 | 0.408 |
| description -> REC -> REG-Joint (chained, text only) | 45.5 | 43.8 | 96.3 | 20.3 / 17.5 deg | 10.2 / 10.6 | 0.434 | 0.291 | 12.3 | 0.240 |
| oracle ceiling (GT answers) | 99.9 | | 100.0 | 1.19 / 1.14 | | 0.279 | | 34.1 | 0.408 |

The GT-box row's PDet/mIoU are the GT box hull by construction (identical to the ceiling): they say
nothing about detection and are kept only so the row is complete. Read the row as "articulation
with the part given": type 96.5 %, MA 47.4 vs 12.9-23.1 for the five cheaper baselines and 36.0 for
ours; axis 18.7 deg and origin 0.36 m are in the same band as ours (17.6 deg / 0.31 m).

The chained row is the text-only protocol (image + our description, nothing else). The joint head
is almost insensitive to whether the box is predicted or given (signed MA 43.8 vs 45.7, type 96.3
vs 96.5); text grounding costs +7 cm of origin error and shows mainly in localisation: the REC
boxes' hulls reach PDet 12.3 / mIoU 0.240 against the 34.1 / 0.408 that a perfect box would give
(627 of 5,088 elements matched at IoU > 0.5; predicted-box vs GT-box hull IoU 0.51 mean). Paper
convention: lead with the signed MA (43.8 text-only, 45.7 given the GT box).

**Chained protocol: first run invalid, regenerated (2026-09-13 23:44 - 2026-09-14 00:28 UTC).** The first chained export
scored MA 35.5 / type 95.8 but PDet 0.0 with every mask empty. Cause: the pod's copy of
`a3vlm_preds_to_jsonl.py` predates e68b5da, so `make-joint` parsed their dot-stripped REC answers
("021" -> 21.0) and the REG-Joint questions fed to the model carried boxes scaled x100
(`[[21.00,47.00,23.00],...]`), i.e. references the model never saw in training. The REC answers
themselves are good: 0/5,088 parse failures, predicted-box hull vs GT-box hull IoU 0.509 mean,
58.2 % > 0.5 (first 500). The joint questions were rebuilt from the same REC answers with the fixed
parser (`/workspace/tmp/a3vlm_fix/joint_pred_test.json`, boxes in [0,1]) and only the joint
generation was re-run on bl-a3vlm (`MODE=eval_joint_pred` in chain.sh, 41 min, GPUs otherwise idle
while the final model copied off the pod); export + scoring redone on the dev pod with
`runpod/baselines/a3vlm/export_chain_fix.sh`. The x100 question file is kept as
`eval/joint_pred_test.scaled100.json`; its answers are not used anywhere.

**Artefacts.** Main volume `results/a3vlm/`: `preds_gtbox.jsonl`, `preds_chain.jsonl`, `metrics_*.json`,
all question/answer JSONs (`vqa_logs/`), eval + train logs, tfevents, and `final_model/` (the
end-of-epoch-1 weights, 2 x 19.9 GB + tokenizer/config; the 63 GB optimizer state was dropped). The
invalid first chained pass is kept as `*.scaled100.*` for the record. Per-sample CSVs:
`per_sample_metrics_gtbox.csv`, `per_sample_metrics_chain.csv` (this dir). Pod bl-a3vlm deleted
00:39 UTC 2026-09-14; total A3VLM spend ~$440 (incl. the smoke pod and ~$27 of the unwanted third
epoch). Volume `bl-apjp` (700 GB, AP-JP-1) still holds the original copy until the user releases it.
