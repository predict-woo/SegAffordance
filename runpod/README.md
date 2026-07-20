# RunPod Runbook

Provisioned 2026-07-15. See `RUNPOD_PLAN.md` at the repo root for the original
rationale; this file reflects what actually exists.

## Provisioned resources

| Resource | Value |
| --- | --- |
| Network Volume | `segaffordance-data`, id `bckt1t9uuf`, 500GB, **EU-RO-1** (Romania) |
| Scratch Volume (**temporary**) | `segaffordance-scratch`, id `s3qha8tz50`, 500GB, EU-RO-1 — created 2026-07-16 to hold the raw SceneFun3D download while rebuilding the training LMDB (Euler access was revoked). Bills ~$35/mo prorated; **delete it as soon as the processed LMDB lands on the main volume** (`runpodctl network-volume delete s3qha8tz50`). |
| Scratch CPU pod (**temporary**) | `segaff-scratch-dl`, id `ngu9q4vy3wyzrr`, 2 vCPU/4GB, $0.06/hr, EU-RO-1, mounts the scratch volume at `/workspace`. SSH: `ssh -i ~/.runpod/ssh/runpodctl-ssh-key root@<ip> -p <port>` (get ip/port from `runpodctl pod get ngu9q4vy3wyzrr`; changes on restart). Running the SceneFun3D train_val download (started 2026-07-16): toolkit at `/workspace/scenefun3d-toolkit`, data → `/workspace/scenefun3d/train_val/`, log → `/workspace/download_train_val.log`. Assets: laser_scan_5mm, crop_mask, transform, hires_wide, hires_wide_intrinsics, hires_depth, hires_poses, annotations, descriptions, motions (no lowres, no videos). **Delete pod when download+processing is finished.** |
| Base image (all pods) | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` (CUDA 12.8, torch 2.9.1) |
| Dev pod | `segaffordance-dev` (RTX PRO 4000 Blackwell 24GB, $0.57/hr, Secure Cloud) |

### Why EU-RO-1, not US-CA-2 + H100 (deviation from RUNPOD_PLAN.md)

At provisioning time, no network-volume-capable datacenter had both an H100 SXM
and a cheap dev GPU actually available — the availability API's empty
`stockStatus` means *out of stock*, and every "in stock" GPU in US-CA-2,
EUR-IS-1, and EU-CZ-1 failed to provision. EU-RO-1 was the only DC that
provisioned dev pods, and it hosts the widest GPU variety:

- Dev tier: RTX PRO 4000 24GB ($0.57/hr), RTX PRO 4500 32GB ($0.74/hr),
  L4 24GB ($0.39/hr), RTX 4090 ($0.69/hr), RTX A6000 48GB ($0.49/hr)
- Training tier: **A100 PCIe/SXM 80GB ($1.39–1.49/hr)** — same class as the
  SLURM `a100_80gb` setup this project already trains on — plus B200 and
  RTX PRO 6000 96GB ($1.99/hr) when in stock. No H100 in this DC.

Network volumes cannot move between DCs. If H100 SXM becomes a hard
requirement, create a second volume in an H100 DC (US-CA-2, US-NE-1, EU-FR-1,
EUR-IS-3, EUR-NO-2, AP-JP-1) and sync data over.

## Volume layout

The volume mounts at `/workspace` on every pod that attaches it:

```text
/workspace
  datasets/      # lmdb datasets (SF3D, OPD*)
  checkpoints/   # ModelCheckpoint outputs
  runs/          # wandb + experiment logs
  models/        # reusable weights (e.g. CLIP RN50.pt)
  cache/         # HF_HOME / TORCH_HOME
  SegAffordance/ # git clone of this repo
```

Rules: `/workspace` is the source of truth for data and outputs. Pods are
disposable. Dev pod is for dev and smoke tests only; training pods get
stopped/deleted the moment a run finishes.

## Dev pod (RTX PRO 4000, $0.57/hr)

```bash
runpodctl pod create \
  --name segaffordance-dev \
  --cloud-type SECURE \
  --gpu-id "NVIDIA RTX PRO 4000 Blackwell" \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --network-volume-id bckt1t9uuf \
  --container-disk-in-gb 20 \
  --ports "22/tcp,8888/http"
```

Day-to-day control goes through `runpod/dev.sh` (run locally):

```bash
bash runpod/dev.sh status          # pod state + hourly cost
bash runpod/dev.sh stop            # stop when done (billing stops, /workspace persists)
bash runpod/dev.sh start           # start, wait for SSH, refresh ~/.ssh/config
bash runpod/dev.sh ssh             # interactive shell
bash runpod/dev.sh run <cmd...>    # run a command in /workspace/SegAffordance
```

`start` warns and exits nonzero if the pod comes back without a GPU (possible
when the host's stock was taken while stopped) — in that case delete and
recreate the pod with the command above.

**KNOWN ISSUE (2026-07-16): `dev.sh start` currently fails with "not enough
free GPUs on the host machine"** — the dev pod's host lost its GPU while
stopped. Delete + recreate the dev pod next time GPU work is needed.

## Local mirror of /workspace (mutagen)

`~/dev/ethz-workspace` on the Mac is a continuous two-way mutagen sync of the
pod's `/workspace` (session name `ethz-workspace`). Edit files there and they
land on the pod in ~1s — no manual copying. Heavy content is excluded from
sync (never downloaded): `datasets/`, `cache/`, `models/`, `checkpoints/`,
`runs/wandb/`, `*.pt`, `*.ckpt`, `.git/lfs`, plus `__pycache__`/`*.pyc`.
So the Mac mirror holds code only (~43MB). Weights are not in git at all:
the CLIP weights live at `/workspace/models/RN50.pt` on the volume, and
setup.sh symlinks `pretrain/RN50.pt` to them (downloading from the official
OpenAI CLIP URL if the volume copy is ever missing).

- `bash runpod/dev.sh sync` — session status
- `bash runpod/dev.sh sync-reset` — terminate + recreate the session (use
  after recreating the pod, or if sync gets stuck)
- Sync pauses automatically while the pod is off and reconciles on start.

Conventions: `~/dev/ethz-workspace/SegAffordance` is the canonical working
copy — run mutating git (commit/pull/push) only from there; commits reach the
pod through the sync itself, so the pod never needs `git pull`. `.git/index`
is sync-ignored (machine-specific), so the pod's `git status` can show phantom
staged changes — `git reset -q` on the pod clears them. `core.fileMode false`
is set in the clone because mutagen does not propagate executable bits.

## Training pod (A100 80GB, ~$1.39–1.49/hr)

Launch only when data is on the volume and the code smoke-tests on the dev pod:

```bash
runpodctl pod create \
  --name segaffordance-train-a100 \
  --cloud-type SECURE \
  --gpu-id "NVIDIA A100 80GB PCIe" \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --network-volume-id bckt1t9uuf \
  --container-disk-in-gb 20 \
  --ports "22/tcp"
```

(`"NVIDIA A100-SXM4-80GB"` for the SXM variant; `"NVIDIA B200"` when it's in
stock and you want a serious speedup.) Availability fluctuates — if creation
fails with "no instances available", retry later; the error is per-attempt,
nothing gets created on failure.

**Delete the training pod immediately after the run**
(`runpodctl pod delete <pod-id>`). Checkpoint often enough that losing the pod
never costs more than an hour.

## Connect + bootstrap

```bash
runpodctl pod list                 # find the pod id
runpodctl ssh info <pod-id>        # SSH command + key
```

From the local machine, `ssh segaff-dev` connects to the dev pod. The pod's
IP/port change on stop/start — refresh the `~/.ssh/config` entry with:

```bash
bash runpod/update-ssh-config.sh
```

On a fresh pod:

```bash
git clone https://github.com/predict-woo/SegAffordance /workspace/SegAffordance
cd /workspace/SegAffordance
bash runpod/setup.sh
```

`setup.sh` is idempotent: creates the `/workspace` layout, installs
`requirements.txt` on top of the base image's torch, and points HF/torch/wandb
caches at the volume.

## Data transfer (TODO — deferred)

**Euler access was revoked (2026-07-16), so datasets were re-downloaded from
their public sources** (a CPU pod on the volume did the work; see git history
of this file for the procedure). Current state:

- **OPD datasets: ON the main volume** under `/workspace/datasets/`:
  `MotionDataset_h5_6.11` (OPDSynth — see below, DROPPED),
  `MotionDataset_h5_real` (OPDReal), `OPDMulti/MotionDataset_h5` +
  `OPDMulti/obj_info.json`, plus `OPDMulti_raw` and `OPD_vis`
  (visualization-only). `annotations_bwdf/` regenerated with
  `datasets/filter_bad_annotations.py`. Sources: HuggingFace
  `Jianghanxiao/OPD` (dataset.tar.gz) and `3dlg-hcvc/OPDMulti`.
- **OPDSynth is DROPPED from training (2026-07-19, user decision)** — the
  Euler-era project had already excluded it because of widespread problematic
  data. Its files stay on the volume for reference, and
  `datasets/opd_intrinsics.py` retains fov/aspect support, but there is no
  training config and it is excluded from description regeneration.
- **OPD descriptions: REGENERATED (2026-07-20).** All six
  `annotations_bwdf/` files (OPDReal + OPDMulti, train/valid/test — 62,904
  annotations, 100% coverage) carry fresh image-conditioned descriptions
  generated by `tools/gen_descriptions.py` via Codex app-server
  (gpt-5.6-luna, medium effort; `info.description_source =
  "codex-gpt-5.6-luna-medium-v1"`). Style: short imperative naming WHICH
  part and the goal, with spatial/appearance qualifiers for disambiguation
  but NO motion mechanics or direction words (user decision — motion must be
  learned from the image, not the text). Pipeline: per-annotation render of
  (original, annotated) pair — all parts outlined, target filled, motion
  arrow — sent as two images per call; validation pass bans overlay/
  mechanics vocabulary; genuinely green/orange furniture rescued by a
  pixel-hue salvage check. Raw results + logs: `/workspace/desc_gen/`.
  The Euler-era OPDReal test descriptions were deliberately NOT reused
  (consistency; preserved in `.old-dont-run/MotionNet_test.json`). Audited
  quality: ~16/20 clean, ~5% state/qualifier noise (comparable to the
  Euler-era labels). Historical analysis and the placeholder era: see git
  history of this section; `tools/add_placeholder_descriptions.py` remains
  for reference.
- **OPD models RETRAINED (2026-07-21, RTX PRO 6000 pod)** with the
  regenerated descriptions; both CSV-logged under `/workspace/runs/csv/`:
  - OPDReal: `config/opdreal_train_runpod.yaml`, 30 epochs, best
    `/workspace/checkpoints/OPDReal_RUNPOD/best-epoch15-valloss0.4069.ckpt`
    (val-sample check: mIoU 0.56, type acc 24/24, axis err 7°). Replaces the
    lost OPDReal_v17 for fine-tuning.
  - OPDMulti (fine-tuned from that checkpoint; three variants compared on
    300 val samples): heads-only (`opdmulti_train_runpod.yaml`, best ep8,
    IoU>0.5 65.7%) < full-finetune 1 epoch (`…_nofreeze.yaml`, ep0, 68.0%)
    ≈ full-finetune lr 3e-6 (`…_nofreeze_lowlr.yaml`, ep2, **70.3%**,
    axis 16.8° — recommended): `/workspace/checkpoints/
    OPDMulti_RUNPOD_NOFREEZE_LOWLR/best-epoch02-valloss0.4654.ckpt`.
    OPDMulti overfits within ~2 epochs of full fine-tuning — keep runs
    short. Prediction visualizations: `/workspace/vis_out/` (synced).
- **Raw SceneFun3D train_val (302GB): on the SCRATCH volume** at
  `/workspace/scenefun3d/train_val` (230 scenes / 609 videos, hires assets,
  verified complete). Downloaded with the toolkit
  (`/workspace/scenefun3d-toolkit` on the scratch volume, plus a
  `parallel_download.py` driver).
- **SF3D LMDB: REBUILT and on the main volume** at
  `/workspace/datasets/sf3d_processed/{data.lmdb,images/,depth/}` (169GB;
  461,334 records, 229 scenes, 206,468 image+depth frames; every record
  validated against the reader contract and loss geometry). Built by
  `tools/sf3d_process.py` — a reconstruction of the lost Euler-era writer
  that adds the two fields the training reader requires: `depth_image_path`
  (16-bit mm PNGs in `depth/`) and `trajectory_3d_camera_coords` (GT
  trajectories verified to zero out the training losses; trans = 0.1m
  segment along motion dir, rot = 90° arc around the axis at the element's
  radius — the docstring documents the evidence chain and the two
  loss-unconstrained choices). Run it from the SceneFun3D toolkit root with
  `PYTHONPATH=.`.
- **Smoke-validated end-to-end (2026-07-17)**: `train_SF3D_better.py fit
  --config config/sf3d_train_runpod.yaml` ran 20 batches + val on the dev
  pod — losses finite and decreasing. That config is the RunPod variant of
  `sf3d_train.yaml` (volume paths, wandb off). NOTE:
  `datasets/scenefun3d.py:64` hardcodes `/dev/shm/data.lmdb` — copy
  `data.lmdb` (11.6GB) there before training, e.g.
  `cp -r /workspace/datasets/sf3d_processed/data.lmdb /dev/shm/`.
- **Scratch volume (`s3qha8tz50`, now 700GB after a mid-run resize) still
  holds the raw download + original LMDB copy.** Keep until a real training
  run validates the rebuilt trajectories, then delete
  (`runpodctl network-volume delete s3qha8tz50`) — it bills ~$49/mo.
- ~~The OPDReal_v17 checkpoint was lost with Euler access~~ — resolved
  2026-07-21: OPDReal retrained on RunPod (see above).

Note `train.sh`/configs still reference cluster paths — point
`train_data_dir` at `/workspace/datasets/...` for RunPod runs, and copy the
lmdb to `/dev/shm` first as `train.sh` already does.

## Budget

Balance was $50 at setup, spend limit $80/mo. Fixed cost: volume $35/mo
(~$1.17/day). Dev pod: $0.57/hr while running. A100 training: ~$1.49/hr (a 4h
run ≈ $6). Top up and consider raising the spend limit before long training
runs.
