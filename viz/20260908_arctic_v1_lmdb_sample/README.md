# ARCTIC 2D LMDB v1 — 24 uniform random records (seed 5)

`records_sample.jpg`: 24 records from `/workspace/datasets/arctic_processed_2d/`
(2,559 stroke records from 238 ARCTIC "use" sequences, rebuilt 2026-09-08
~21:40 local by `tools/arctic_process_2d.py` after the stroke-splitter fix:
strokes now end at the last moving frame and pauses > 5 frames break them —
the first build (2,633 records) absorbed the rest phase after a close into
the stroke, so the hand wandered for up to ~90 frames; user spotted it on
s05_mixer_use_01_s06_f00329). Per tile: the stored 512x512 frame (direct
resize of the 2800x2000 ego frame), stored mask coords in red (the moving
part RENDERED from ARCTIC's per-part meshes with the mocap GT pose), the 2D
middle-knuckle track of the interacting hand over the stroke in green
(cyan = stroke start, magenta = end), orange dot = the GT hinge origin
projected into the image; header = object / open|close / hand / angle
start -> end / T; line 2 = key + template description.
Regen: `python tools/epic_lmdb_sample.py --lmdb /workspace/datasets/arctic_processed_2d --out viz/20260908_arctic_v1_lmdb_sample --n 24 --seed 5 --uniform`
(NOTE: the regen wipes the directory — re-add this README.)

**Read:** 24/24 masks are the moving part (notebook covers, espresso and
capsule-machine levers, mixer heads, waffle-iron lids, laptop screens,
microwave doors, scissors blade, box lid), hinge origins on the hinge
lines. Hands are NOT cut out of the masks (no MANO models). Trajectories
are short single-motion arcs starting at the hand on the part.
Build stats: 2,559 ok; dropped 130 (hand outside the frame at the start),
12 (no image: s01_box_use_01 not downloaded), 6 tiny masks, 5 far from
mask, 3 frame jumps. Stroke length median 22 frames (p10 14, p90 38).
Reader smoke test (`config/arctic_v1_smoke.yaml`, fast_dev_run) passed.
Companion: `viz/20260908_arctic_builder_test`, `viz/20260908_arctic_v1_random24b`.
