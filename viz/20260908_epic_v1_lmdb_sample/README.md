# EPIC/VISOR 2D LMDB v1 — 20 random records (stratified by fixture noun)

`records_sample.jpg`: 20 records from `/workspace/datasets/epic_processed_2d/`
(main volume; 326 records, built 2026-09-08 ~03:00 local by
`tools/epic_process_2d.py` from the 478 SAM2-propagated work items in
`epic_processed_2d/work/`). Per tile: the stored 512x512 frame (direct
resize of the 1920x1080 onset frame, HOI4D convention), stored mask coords
in red (thinned to the reader's gather grid), the 2D middle-knuckle track
in green (cyan = onset, magenta = end), header = noun / verb / hand /
d (VISOR frame offset from onset) / T / scale regime, second line = key +
EPIC narration (the description). `sample_keys.json` lists the keys.
Regen: `python tools/epic_lmdb_sample.py --lmdb /workspace/datasets/epic_processed_2d --out viz/20260908_epic_v1_lmdb_sample --n 20`

**Read:** 17/20 masks are the moving part (drawer fronts incl. pulled-out
boxes, fridge/freezer/oven/microwave/dishwasher/room doors). The 3 others
are VISOR semantics, not propagation errors: 2 cupboards labelled as the
whole carcass + door (P10_04_358, P18_06_17) and a window whose mask
spills onto the wall (P06_101_62). Trajectories start at the hand on the
part and follow the open/close motion; a few loop on rotation-only videos
(P24_09_23) — the collaborator's camera-compensation caveat.

**Build stats** (478 work items -> 326 records): dropped 56 with |d| > 30,
36 by area-ratio (0.35-2.5: the door-swing failure mode), 43 by the HOI4D
frame-jump rule (> 300 px between stride-2 frames), 15 start-far-from-mask,
2 short. Per noun: drawer 161, fridge 64, cupboard 49, oven 21, dishwasher
9, freezer 6, door 4, microwave 4, cabinet 3, misc 5. Depth = zeros (EPIC
has none; hand z at onset kept in `epic.hand_z_onset_m`). Reader smoke test
(`config/epic_v1_smoke.yaml`, fast_dev_run): 326 records read, scene split
by kitchen video, one train + val step OK.
