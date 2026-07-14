# RunPod Runbook

Provisioned 2026-07-15. See `RUNPOD_PLAN.md` at the repo root for the rationale.

## Provisioned resources

| Resource | Value |
| --- | --- |
| Network Volume | `segaffordance-data`, id `er8u1eqoro`, 500GB, **US-CA-2** |
| Base image (both pods) | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` (CUDA 12.8, torch 2.9.1) |
| Dev pod | `segaffordance-dev-a5000` (RTX A5000, Secure Cloud, US-CA-2) |

All pods must be created in **US-CA-2** so they can mount the volume. The volume
mounts at `/workspace`:

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
disposable. The A5000 is for dev and smoke tests only; the H100 is for full
training only, and gets stopped/deleted the moment a run finishes.

## Dev pod (RTX A5000, ~$0.27/hr)

```bash
runpodctl pod create \
  --name segaffordance-dev-a5000 \
  --cloud-type SECURE \
  --gpu-id "NVIDIA RTX A5000" \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --network-volume-id er8u1eqoro \
  --container-disk-in-gb 20 \
  --ports "22/tcp,8888/http"
```

Stop it when you're done for the day (billing stops, `/workspace` persists):

```bash
runpodctl pod stop <pod-id>
runpodctl pod start <pod-id>   # next morning
```

## Training pod (H100 SXM, ~$3.29/hr)

Launch only when data is on the volume and the code smoke-tests on the A5000:

```bash
runpodctl pod create \
  --name segaffordance-train-h100 \
  --cloud-type SECURE \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --network-volume-id er8u1eqoro \
  --container-disk-in-gb 20 \
  --ports "22/tcp"
```

**Delete it immediately after the run** (`runpodctl pod delete <pod-id>`).
Checkpoint often enough that losing the pod never costs more than an hour.

## Connect + bootstrap

```bash
runpodctl pod list                 # find the pod id
runpodctl ssh info <pod-id>        # SSH command + key
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
(~$1.17/day). A5000 dev: $0.27/hr. H100: $3.29/hr (a 4h run ≈ $13).
Top up and consider raising the spend limit before the first real H100 run.
