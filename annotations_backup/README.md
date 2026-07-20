# OPD annotation backup (with regenerated descriptions)

Gzipped copies of the six `annotations_bwdf/MotionNet_{train,valid,test}.json`
files for OPDReal (`MotionDataset_h5_real`) and OPDMulti
(`OPDMulti/MotionDataset_h5`), plus the raw generation checkpoint
(`desc_gen_results.jsonl.gz`, one JSON line per annotation).

These exist in git because the `description` fields are **not recoverable
from public sources**: the published OPD datasets ship without descriptions,
and the first (Euler-era) set of descriptions was lost when cluster access
was revoked. Everything else about the datasets can be re-downloaded; the
descriptions cannot.

- Generated 2026-07-20 by `tools/gen_descriptions.py` (Codex app-server,
  gpt-5.6-luna, medium effort), 62,904 annotations, 100% coverage.
  `info.description_source = "codex-gpt-5.6-luna-medium-v1"`.
- Restore: `gunzip -kc opdreal/MotionNet_train.json.gz >
  /workspace/datasets/MotionDataset_h5_real/annotations_bwdf/MotionNet_train.json`
  (and likewise for the others).
- The results JSONL alone is also sufficient to rebuild the files from a
  fresh public download: re-run `datasets/filter_bad_annotations.py`, then
  `tools/gen_descriptions.py merge --results <path to unzipped jsonl>`.
