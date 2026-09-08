# EPIC/VISOR 2D LMDB v1 — 20 uniform random records, HALF-SPAN trajectories (current build)

Build: `/workspace/datasets/epic_processed_2d/` rebuilt 2026-09-08 ~04:40
local, `tools/epic_process_2d.py` defaults: trajectories keep the FIRST HALF
of the onset..span_end interval (+4 frames; `--traj-frac 0.5`, user: full
spans "still too long") — **353 records**, median 12 points (p10 7, p90 21).
Regen: `python tools/epic_lmdb_sample.py --lmdb /workspace/datasets/epic_processed_2d --out viz/20260908_epic_v1_random20_half --n 20 --seed 11 --uniform --work /workspace/datasets/epic_processed_2d/work`
Same layout as `20260908_epic_v1_random20` (records_sample.jpg + one QA
panel per record). Note the tile header bar hides the top ~85 px of the
1080p frame: P25_101_17's cupboard door sits there (see its panel) —
rendering artefact, the record is fine.

**Read:** 20/20 masks on the moving part (P20_03_54's fridge door has a
jagged SAM2 edge; P25_101_17 is a door mostly above the frame). Trajectories
are short strokes from the hand on the part in the opening/closing direction.
Build stats (478 items -> 353): |d| > 30: 56, area-ratio: 36, start far from
mask: 9, frame jump: 3, < 5 points after the half cut: 21.
