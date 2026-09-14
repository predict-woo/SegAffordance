# 20260913_3doi — 3DOI (ICCV'23) retrained on SF3D

**Goal.** Single-image, point-prompted articulation baseline: their SAM-backbone model
(`monoarti`) trained on our SF3D train split and scored with our protocol.
Plan: `docs/superpowers/plans/2026-09-12-sf3d-baselines-a3vlm-3doi.md`.

**Upstream.** github.com/JasonQSY/3DOI @ 9ff32ef, recipe = `monoarti/configs/sam.yaml`
(SAM ViT-B, 768x1024, batch 2/GPU, AdamW lr 1e-4 / backbone 1e-5, fp16, StepLR 500,
loss weights mask 2.0 / dice 1.0 / affordance 400 / bbox 5 / giou 2 / axis 2 / axis_offset 10 /
depth 1.0, clip 0.1, 200 epochs). Our copy: `config/baselines/3doi_sam_sf3d.yaml` — identical
except the dataset names, the basic (non-Slurm) launcher, checkpointing every epoch and
`validation_epoch_interval: 10`.

**Epoch count / optimisation budget.** Their released recipe is 200 epochs over the 3DOI train
set, which the paper sizes at "over 50K objects across 10K images" (Qian & Fouhey, ICCV'23), at
effective batch 8 (batch 2 x 4 GPUs) = 2.00M image-presentations = 250,000 optimizer steps. Our
train split is 32,171 frames, 3.2x theirs, so their literal 200 epochs would be 6.43M
presentations -- 3.2x their optimisation budget and ~15 days on 2 GPUs. We instead match the
budget: **62 epochs at effective batch 8** = 1.99M presentations = 250,000 steps. Euler caps us at
2 GPUs, so the effective batch is reproduced with batch 4 per GPU, which also preserves their
learning-rate-to-batch ratio. Everything else in `optimizer:` is theirs verbatim.

**Early stopping (user request).** Their loop trains a fixed number of epochs and exports the LAST
checkpoint; it also composes a validation loss per batch and throws it away (`losses = []` is never
appended to). We accumulate that loss, validate every 2 epochs, keep `checkpoint_best.pth` whenever
it improves, stop after 4 validations without improvement, and export from the best checkpoint.
This changes only when training stops and which checkpoint is scored -- no loss, optimiser or
hyper-parameter is touched -- and it can only reduce compute relative to the 62-epoch budget above.
Two fixes found in the RunPod run at epoch 2 (2026-09-13 18:33 UTC): (i) each DDP rank had been
computing the val loss over its own shard and deciding independently (0.5087 vs 0.4970 at epoch 2),
which could make one rank stop while the other hangs -- the loss is now all-reduced (sum/count) so
every rank sees one global number, and the best value + counter are persisted to
`checkpoints/best_val.json` so a resume/requeue keeps its early-stop state; (ii) their `train.py`
resume restores the model and epoch counter but has the optimizer-state load commented out (their
`test.py` loads it), so every requeue would have reset AdamW's moments -- restored. The RunPod run
was restarted from the end-of-epoch-2 checkpoint with these fixes; the epoch-0/2 validations before
the restart were shard-local numbers and the early-stop baseline starts fresh from epoch 4.

**Deviations (all documented, none touch the training math).**
- `SF3D_LIMIT_VAL_ITERS=200`: their validation walks the whole val split and fires at epoch 0
  (`epoch % interval == 0`), ~1 h per pass on our 3,495-frame bvalid split. Nothing selects a
  checkpoint from it (train.py just overwrites `checkpoint.pth`), so it is capped.
- `submitit` installed rather than import-guarded: absent, `submitit.JobEnvironment()` raises
  AttributeError instead of the RuntimeError their off-Slurm guard catches.
- depth validity: their loss asks `depth[:, 0, 0] > 0` to decide whether a frame has depth at all
  (fine for hole-free Taskonomy renders); SF3D sensor depth has holes, so the check becomes
  "any valid pixel". The per-pixel mask already excluded holes from the loss.

**Data.** `tools/baselines_sf3d/sf3d_to_3doi.py` — 38,979 frames at 1024x768 (portrait frames
rolled 90 deg CW), one instance per annotated element: keypoint/affordance = element point,
movable one_hand, rigid yes, kinematic rotation|translation, mask = one polygon (largest contour
of the 3 px-dilated splat mask), axis = the GT 3D axis projected and cut by the image rectangle.
Depth is supervised from our input depth through an omnidata-style tree, which their loader picks
up because the images are named `taskonomy_*`. Splits: train 32,171 frames / 48,560 instances,
val (our bvalid) 3,495 / 5,526, test 3,313 / 5,088. Every instance has a valid 2D axis line.

**Where it runs.** ETHZ Euler, 2 GPUs with `--gres=gpumem:80g` (A100 80 GB or RTX PRO 6000),
`euler/baselines/3doi.sbatch`. The per-user cap of 2 GPUs is enforced by a client-side cli_filter,
so 2 is the maximum; the job checkpoints every epoch, self-requeues on the 24 h wall clock and
resumes from `checkpoints/checkpoint.pth`. Jobs: smoke 13967514 (1 x A100 40 GB, passed incl.
resume); 13971778 was placed on 40 GB A100s despite asking for RTX PRO 6000 by type (Euler's submit
plugin rewrites the partition list) and OOMed at batch 4 after 7 min; **13993189** is the live
submission, constrained by GPU memory, with a VRAM guard in the chain that exits before training
on an under-sized card.

**Validation before the run** (RunPod dev pod, 30 iterations): train -> checkpoint -> resume ->
export -> score all pass. Export of 30 test frames gives 45 predictions, 43 of them above IoU 0.5,
matched-row axis error 15.2 deg — SAM is pretrained and prompted with the GT element point, so
strong masks are expected from the start and PDet should be the model's strongest number.

**RunPod run (user decision 2026-09-13 ~13:00 UTC).** Euler's queue estimate slipped to ~22:20 UTC and
the login node then became unreachable, so the user chose to run 3DOI on RunPod as well and keep the
free Euler job queued. Pod `bl-3doi`, **2 x H200** in AP-JP-1 on volume `bl-apjp` (no 2- or 4-GPU
A100 / RTX PRO 6000 stock in EU-RO-1 for ~40 min of polling; the 17 GB dataset was rsynced to the
Japan volume from the dev pod), $9.18/h. Per-GPU batch 4 x 2 GPUs = effective batch 8 (recipe),
62-epoch budget, validation every 2 epochs, early stopping patience 4, export from the best-val
checkpoint. Expected ~$170-400 depending on where early stopping triggers. Whichever of the two
runs finishes first is the one reported; the other is a free replicate.

**RunPod run, as it happened.** 13:45 UTC 2026-09-13 - 20:33 UTC 2026-09-14 (~31 h, ~$283; above
the $170-250 estimate because their training loop is CPU-bound — ~1.5 s per step of 8 images with
the GPUs mostly idle — and because of the restart at epoch 2 with the early-stopping fixes). Global
(all-reduced, 200-iteration) validation loss every 2 epochs: 0.5994 (epoch 4, best), 0.6039, 0.6034,
**0.5917 (epoch 10, best)**, 0.6979, 0.6703, 0.6831, 0.8950 → early stop after epoch 18 (4/4
validations without improvement; the loss was rising, not plateauing). Exported from
`checkpoint_best.pth` (epoch 10): 5,088 predictions, 4,486 with a lifted axis (602 answered
"freeform", i.e. no joint → counted as type-wrong / axis 90 deg), 350 s. Results, logs and the best
checkpoint are on the main volume under `results/3doi_runpod/`. Pod deleted 20:33 UTC.

**Our protocol (point-prompted: 3DOI receives the GT interaction point; 5,088 elements; signed MA first):**

| signed MA | MA (unsigned) | type % | axis all / matched | origin_err_m | origin_line_err_m | PDet | mIoU | flips all / rot |
|---|---|---|---|---|---|---|---|---|
| **23.2** | 35.7 | 84.4 | 30.4 / 31.3 deg | 0.811 | 0.720 | **72.2** | **0.595** | 33.1 / 21.4 |

**Reading.**
- *Masks:* the strongest segmentation number of any baseline by far (PDet 72.2, mIoU 0.595; 3,671 of
  5,088 elements above IoU 0.5) — a pretrained SAM prompted with the GT element point. This row is
  "given the part location", comparable to A3VLM's GT-box row, not to the detectors.
- *Type:* 84.4 % overall, but strongly biased to translation: 604 revolute predictions vs 1,068
  rotational GT (rotational rows: 45.5 % correct, 508 of them answered translation or freeform;
  prismatic rows: 94.7 %). 3DOI's kinematic head was trained on 3DOI's web images where most
  movables translate; on SF3D it under-calls hinges.
- *Axis — the signed/unsigned gap is structural.* 3DOI predicts an undirected 2D line (two image
  endpoints); the lifted 3D direction's sign is whichever endpoint order the model emitted, so 33 %
  of the axes point the wrong way (signed error > 90 deg; 37 % on prismatic, 41 % on rotational
  rows). Unsigned MA is 35.7 (40.5 over the 4,486 rows with a joint); the paper's signed convention
  gives 23.2. Both are reported; the paper should footnote that 3DOI's axis has no sign.
- *Origin:* 0.81 m mean (0.45 m median line distance over the 560 rotational rows with a joint) —
  the hinge position comes from lifting 2D endpoints through a RANSAC depth plane, which is
  ill-conditioned for hinges seen edge-on.
- Overall, on articulation 3DOI sits between the OPD detectors (signed MA 14-26) and A3VLM / ours
  (43.8 / 42.7), while dominating on masks it was handed the location of.

**Confidence-thresholded columns** do not apply: 3DOI emits one mask per prompt with no detection
score (`thresholded.json` records conf = 1 for every row). Per-sample CSV: `per_sample_metrics.csv`.

**Euler replicate.** Job 13993189 started 00:49 UTC 2026-09-14 (2 GPUs, ~3x faster per epoch than the
RunPod pod); the login node has been unreachable (VPN) since ~02:00 UTC — to be read out when it is
back. Its result is a free replicate; the RunPod run above is the one reported.
