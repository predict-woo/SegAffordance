# EPIC/VISOR 2D LMDB v1 — 20 UNIFORM random records + their QA panels

Build: `/workspace/datasets/epic_processed_2d/` rebuilt 2026-09-08 ~04:10
local with `tools/epic_process_2d.py` (**359 records**; knuckle trajectories
now CUT at the narrated span end + 4 frames — the collaborator's tracks ran
90 frames past the action and included the hand leaving for the next task,
visible as loops/wander in the first sample; median 20 points, p10 11, p90 36).
Regen: `python tools/epic_lmdb_sample.py --lmdb /workspace/datasets/epic_processed_2d --out viz/20260908_epic_v1_random20 --n 20 --seed 11 --uniform --work /workspace/datasets/epic_processed_2d/work`

`records_sample.jpg`: the 20 stored records (512x512 frame, mask coords red,
knuckle track green, cyan = onset, magenta = end; header = noun / verb / hand
/ d / T / scale regime; line 2 = key + narration).
`panel_NN_<noun>_<narration_id>.jpg`: the matching QA panel from the
propagation step — LEFT the VISOR frame with its human polygon (red), RIGHT
the onset frame with the SAM2-propagated mask (green) and the full
(uncut) knuckle track (cyan). `sample_keys.json` = the keys.

**Read (this sample):** masks on the moving part in 19/20 (P35_105_96 is
VISOR's whole-cupboard label; P12_03_48's fridge door is a thin sliver at
the frame edge, a 720p video). After the span cut the trajectories are
short open/close strokes starting at the hand on the part.
Build stats (478 items -> 359): |d| > 30: 56, area-ratio: 36, start far
from mask: 14, frame jump: 9, short: 4.
