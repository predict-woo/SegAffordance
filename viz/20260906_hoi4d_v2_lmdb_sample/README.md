# HOI4D 2D LMDB v2 — 16 random records, one per category

`records_sample.jpg`: one random record per category from
`/workspace/hoi4d_processed_2d_v2/` (HOI4D volume; 128,163 records, built
2026-09-06 00:29 UTC by `tools/hoi4d_process_2d.py` from the gpt-5.6-terra
VLM selections in `/workspace/vlm_select_v2/selections.json`).
Per tile: the stored 512x512 frame, the stored mask coordinates in red
(thinned to the reader's gather grid, hence dotted), the 2D trajectory in
green (middle-knuckle track from the sample frame to the window end; cyan =
start, magenta = end), header = category / verb / hand / trajectory length,
second line = the VLM description.

Regenerate: scratchpad `records_sample.py` on the pod (reads both LMDBs).

Review panels (scratchpad `review_v2.py`, 2026-09-06 morning):
- `panel_verbs.jpg` — 2 random records per verb (12 verbs). Part-motion
  verbs all correct (safe door, drawer, lamp base for press/switch, toy car,
  bucket, scissors). Tool verbs are INCONSISTENT about tool vs material:
  one `binding` masks the paper, one `cut` masks the apple (the other of
  each masks the stapler / knife).
- `panel_multipart.jpg` — 12 whole-object answers (lamp 1,2,3; laptop;
  bucket + handle; pliers; scissors; trash can + lid): all correct.
- `panel_far.jpg` — records whose trajectory start is > 300 px from the
  mask (0.2% of 3,000 sampled; p50/p90/p99 = 47/74/141 px): WiLoR outlier
  detections (a knuckle jumping off-frame, a zig-zag between two positions)
  — candidates for a distance/jump filter in the builder.
- `panel_none_single.jpg` — single-candidate windows the VLM refused:
  73 of 77 are stapler sequences where the ONLY segmented object is the
  sheet of paper (stapler absent/too small) — NONE is the right call.
- 400 random descriptions: median 5 words, p95 8; 3 contain "left/right"
  as spatial qualifiers (allowed), none mention overlays or hands.

Reading: masks land on the intended part or whole object in all 16 — laptop
screen for "close the laptop screen", the top drawer, the safe's front
flap, the trash-can lid, the whole kettle for pouring, whole chair / pliers
/ scissors for pick-up and put-down. Descriptions are OPD-style
imperatives with qualifiers ("Put down the small bowl in the drawer",
"Pour from the white kettle into the small blue c..."). Trajectories are
short (median 16 frames) and local, as expected for pick/put/open/close
windows without `carry`.
