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

**Status.** RunPod run launching 2026-09-13 ~13:50 UTC; Euler job 13993189 queued (login node
unreachable since ~13:00 UTC). Results below when one finishes.
