# Experiments

One directory per training run, named `YYYYMMDD_<dataset>_<variant>`.
`INDEX.md` is the summary table — update it whenever an experiment finishes.

Tracked in git (small, text):

- `notes.md` — goal, setup, result, decision
- `config.yaml` — the exact config the run was launched with
- `metrics.csv` — the CSV logger output (copy of `logs/*/metrics.csv`)
- `vis_summary.txt` — aggregate line + per-sample list from
  `tools/vis_predictions.py`

Volume-only (gitignored, sync-ignored where noted):

- `checkpoints/` — ModelCheckpoint output (`*.ckpt` never leaves the volume)
- `logs/` — raw CSVLogger version dirs and the training stdout log
- `vis/` — prediction panel PNGs from `tools/vis_predictions.py`

Workflow for a new experiment:

1. Copy an existing `config/*_runpod*.yaml`, point `dirpath` at
   `/workspace/SegAffordance/experiments/<id>/checkpoints` and the CSVLogger
   `save_dir` at `/workspace/SegAffordance/experiments/<id>/logs`.
2. Run it on a training pod (see `runpod/README.md`), stdout to
   `experiments/<id>/logs/train.log`.
3. Afterwards: copy `logs/csv/version_0/metrics.csv` to `metrics.csv`, run
   `tools/vis_predictions.py --out .../vis`, copy its `summary.txt` to
   `vis_summary.txt`, write `notes.md`, add a row to `INDEX.md`, commit.
