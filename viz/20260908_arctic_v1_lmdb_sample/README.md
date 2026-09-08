# ARCTIC 2D LMDB v1 — 24 uniform random records

`records_sample.jpg`: 24 records from `/workspace/datasets/arctic_processed_2d/`
(2,633 stroke records from 238 ARCTIC "use" sequences, built 2026-09-08
~20:00 local by `tools/arctic_process_2d.py`). Per tile: the stored 512x512
frame (direct resize of the 2800x2000 ego frame), stored mask coords in red
(the moving part RENDERED from ARCTIC's per-part meshes with the mocap GT
pose), the 2D middle-knuckle track of the interacting hand over the stroke
in green (cyan = stroke start, magenta = end), orange dot = the GT hinge
origin projected into the image; header = object / open|close / hand /
angle start -> end / T; line 2 = key + template description.
Regen: `python tools/epic_lmdb_sample.py --lmdb /workspace/datasets/arctic_processed_2d --out viz/20260908_arctic_v1_lmdb_sample --n 24 --seed 5 --uniform`

**Read:** 24/24 masks are the moving part (notebook covers, espresso and
capsule-machine levers, mixer heads, waffle-iron lids, laptop screens,
microwave doors, scissors blade, box lid), hinge origins on the hinge
lines. Hands are NOT cut out of the masks (no MANO models). Trajectories
start at the hand on the part; strokes that swing a lid toward the
head-mounted camera leave the frame (laptop close, T=168) — physically
right, the reader keeps in-front-of-camera points.
Build stats: 2,633 ok; dropped 118 (hand outside the frame at the start),
12 (no image: s01_box_use_01 not downloaded), 12 frame jumps, 15 far from
mask, 8 tiny masks. Reader smoke test (`config/arctic_v1_smoke.yaml`,
fast_dev_run) passed. Companion: `viz/20260908_arctic_builder_test`.
