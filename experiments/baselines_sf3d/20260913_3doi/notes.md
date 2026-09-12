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

**Where it runs.** ETHZ Euler, 2 x RTX PRO 6000 (96 GB), `euler/baselines/3doi.sbatch`.
The per-user cap of 2 GPUs is enforced by a client-side cli_filter, so 2 is the maximum; the job
checkpoints every epoch, self-requeues on the 5-day wall clock and resumes from
`checkpoints/checkpoint.pth`. Jobs: smoke 13967514 (1 x A100), full 13968368.

**Validation before the run** (RunPod dev pod, 30 iterations): train -> checkpoint -> resume ->
export -> score all pass. Export of 30 test frames gives 45 predictions, 43 of them above IoU 0.5,
matched-row axis error 15.2 deg — SAM is pretrained and prompted with the GT element point, so
strong masks are expected from the start and PDet should be the model's strongest number.

**Status.** Queued 2026-09-13. Results below when it finishes.
