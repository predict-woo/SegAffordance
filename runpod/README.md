# RunPod Runbook

Provisioned 2026-07-15. See `RUNPOD_PLAN.md` at the repo root for the original
rationale; this file reflects what actually exists.

## Provisioned resources

| Resource | Value |
| --- | --- |
| Network Volume | `segaffordance-data`, id `bckt1t9uuf`, 500GB, **EU-RO-1** (Romania) |
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
copy for pod work — run git from there (or on the pod), not from both sides
at once. `core.fileMode false` is set in that clone because mutagen does not
propagate executable bits.

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

Datasets currently live on the Euler cluster
(`/cluster/work/cvg/students/andrye/sf3d_processed`, plus OPD* data). Preferred
route once ready: SSH from the dev pod (or via the Mac as a jump host) and
`rsync` straight into `/workspace/datasets/`. Note `train.sh`/configs reference
cluster paths — point `train_data_dir` at `/workspace/datasets/...` for RunPod
runs, and copy the lmdb to `/dev/shm` first as `train.sh` already does.

## Budget

Balance was $50 at setup, spend limit $80/mo. Fixed cost: volume $35/mo
(~$1.17/day). Dev pod: $0.57/hr while running. A100 training: ~$1.49/hr (a 4h
run ≈ $6). Top up and consider raising the spend limit before long training
runs.
