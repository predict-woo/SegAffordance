# SegAffordance

Affordance segmentation research code (PyTorch Lightning). Training entry
points are `train_SF3D_better.py`, `train_OPDMulti_better.py`,
`train_OPDReal_better.py`, each driven by a YAML config in `config/`
(e.g. `python train_SF3D_better.py fit --config config/sf3d_train.yaml`).
Paper notes and related-work summaries live in `knowledge/`.

## Remote GPU environment (RunPod)

All GPU work happens on RunPod, not locally. Full runbook: `runpod/README.md`.
The infra that exists:

- **Network volume** `bckt1t9uuf` (500GB, datacenter **EU-RO-1**) — mounts at
  `/workspace` on every pod. Source of truth for datasets, checkpoints, logs.
- **Dev pod** `segaffordance-dev` (RTX PRO 4000 Blackwell 24GB, $0.57/hr,
  Secure Cloud) — for interactive dev and smoke tests only.
- **Training pods** are created on demand in EU-RO-1 (A100 80GB ~$1.49/hr; no
  H100 in this DC) and deleted immediately after a run. Create commands are in
  the runbook.

### Controlling the dev pod

Use `runpod/dev.sh` from the local machine — don't hand-roll runpodctl calls:

```bash
bash runpod/dev.sh status            # state + $/hr
bash runpod/dev.sh start             # start, wait for SSH, refresh ssh config
bash runpod/dev.sh stop              # stop when done — it bills while running
bash runpod/dev.sh run "<cmd>"       # run a command in /workspace/SegAffordance
bash runpod/dev.sh ssh               # interactive shell (alias: ssh segaff-dev)
```

The pod's IP/port change on every restart; `dev.sh start` refreshes the
`segaff-dev` entry in `~/.ssh/config` automatically (standalone:
`runpod/update-ssh-config.sh`). On a fresh/recreated pod, bootstrap with
`bash runpod/setup.sh` (idempotent; installs requirements.txt on top of the
image's torch).

### Editing pod files: use the mutagen mirror

**`~/dev/ethz-workspace` is a live two-way mutagen sync of the pod's
`/workspace`.** To change code on the pod, edit files under
`~/dev/ethz-workspace/SegAffordance/` with normal file tools — they propagate
to the pod in ~1s. Then execute with `dev.sh run "<cmd>"`. No need to scp,
rsync, or edit through SSH. Heavy dirs (`datasets/`, `cache/`, `models/`,
`checkpoints/`, `runs/wandb/`) are sync-ignored and exist only on the volume —
never try to read datasets locally. `dev.sh sync` shows session health;
`dev.sh sync-reset` recreates it.

Git rules: run mutating git (commit/pull/push/checkout) ONLY from the Mac side
(`~/dev/ethz-workspace/SegAffordance`) — commits reach the pod through the
sync itself (`.git` objects/refs sync; no `git pull` needed on the pod).
`.git/index` is machine-specific and sync-ignored, so the pod's `git status`
may show phantom staged changes after Mac-side commits — run `git reset -q`
on the pod to clear it; never trust or commit from the pod's index.
CLIP weights (`pretrain/RN50.pt`) are NOT in git: the real file is
`/workspace/models/RN50.pt`, symlinked into the repo by `setup.sh`.

### Cost etiquette

The account runs on a small prepaid balance. Stop the dev pod after finishing
work on it (ask the user first if they might be using it). Never leave a
training pod alive after its run. The volume bills $35/mo regardless.

### Sharp edges (learned the hard way)

- `runpodctl datacenter list`: an **empty** `stockStatus` means *out of
  stock*; "Low"/"Medium" means available — and even then only an actual
  `pod create` attempt is authoritative. Failed creates make nothing, so
  retrying is safe; but a `network-volume create` that errors with a 500 may
  still have created the volume — check `network-volume list` before retrying.
- Network volumes are locked to their datacenter forever (resize up only).
  Only some DCs support volumes; the volume-create error message lists them.
- The `runpod/pytorch:*-ubuntu2404` image enforces PEP 668 — pip needs
  `--break-system-packages` (setup.sh handles it).
- If `dev.sh start` warns the pod has no GPU, the host's stock was taken while
  stopped: delete the pod and recreate it from the runbook (state on
  `/workspace` survives).

## Experiment organization

Every training run gets a directory `experiments/YYYYMMDD_<dataset>_<variant>/`
(inside this repo). `experiments/INDEX.md` is the summary table — one row per
experiment, updated when a run finishes; `experiments/README.md` documents the
full workflow. Per experiment:

- **Tracked in git:** `notes.md` (goal/setup/result/decision), `config.yaml`
  (exact launch config), `metrics.csv` (CSV-logger copy), `vis_summary.txt`
  (from `tools/vis_predictions.py`).
- **Volume-only (gitignored):** `checkpoints/` (`*.ckpt` is also
  mutagen-ignored — checkpoints never leave the volume), `logs/` (raw CSV
  versions + train.log), `vis/` (prediction-panel PNGs).

New runs: copy a `config/*_runpod*.yaml`, point its ModelCheckpoint `dirpath`
and CSVLogger `save_dir` at the new experiment dir, launch on a training pod,
then fill in the tracked files and INDEX row and commit. The `config/` file
and the experiment's `config.yaml` should stay identical.

## Current state / TODO

- **Euler access was revoked (2026-07-16)**; all datasets were rebuilt from
  public sources and are now on the main volume under `/workspace/datasets/`:
  OPD* (incl. regenerated `annotations_bwdf/`) and `sf3d_processed/` (SF3D
  LMDB rebuilt by `tools/sf3d_process.py`, incl. reconstructed
  depth+trajectory fields — see its docstring for GT conventions).
  End-to-end smoke-validated 2026-07-17: SF3D, OPDReal, and OPDMulti train
  via the `config/*_train_runpod.yaml` variants (wandb off). **OPDSynth is
  dropped** (2026-07-19, as in the Euler-era project — too much problematic
  data); its files remain on the volume but it has no config and is excluded
  from description regeneration. The cluster configs (`sf3d_train.yaml`
  etc.) still reference dead Euler paths.
  OPD `description` texts were REGENERATED 2026-07-20 for all six
  OPDReal/OPDMulti annotation files (62,904 annotations, 100% coverage) by
  `tools/gen_descriptions.py` — image-conditioned Codex (gpt-5.6-luna)
  generations, `info.description_source = "codex-gpt-5.6-luna-medium-v1"`;
  see `runpod/README.md` for style rules and pipeline details.
  OPD models RETRAINED 2026-07-21 (replacing the lost OPDReal_v17) — see
  `experiments/INDEX.md`; best checkpoints live under
  `experiments/<id>/checkpoints/` on the volume (OPDReal:
  `20260721_opdreal_base`, OPDMulti recommended:
  `20260721_opdmulti_tune_lr2e6`, recipe in
  `config/opdmulti_train_runpod_tuned.yaml`).
  **Open item:** raw SceneFun3D (302GB) sits on the temporary scratch
  volume pending deletion after a real training run validates the rebuilt
  trajectories. See `runpod/README.md`.
- `RUNPOD_PLAN.md` is the original plan; where it disagrees with
  `runpod/README.md` (e.g. it promises an A5000 dev pod and H100 training in
  US-CA-2), the README reflects reality.
