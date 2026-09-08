# ARCTIC 2D LMDB v1 — a second draw of 24 uniform random records (seed 17)

Same build and layout as `20260908_arctic_v1_lmdb_sample` (2,559 stroke
records, stroke-splitter fix applied): stored 512x512 frame, rendered GT
moving-part mask in red, knuckle track in green (cyan = start, magenta =
end), orange = hinge origin; header = object / open|close / hand / angle
start -> end / T; line 2 = key + template description.
Regen: `python tools/epic_lmdb_sample.py --lmdb /workspace/datasets/arctic_processed_2d --out viz/20260908_arctic_v1_random24b --n 24 --seed 17 --uniform`

**Read:** 24/24 masks on the moving part (scissors = the moving blade,
phones = the flip, notebooks = the cover, mixer heads, waffle-iron and box
lids, laptop screens, microwave door, espresso/capsule levers, ketchup cap);
every trajectory is a single short arc (T 13-48 frames) that starts at the
hand on the part — no wandering tails after the fix.
