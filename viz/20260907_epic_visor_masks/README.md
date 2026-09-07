# EPIC-KITCHENS VISOR masks on our fixture interactions — 22 panels

Regen (dev pod): `python /workspace/visor_coverage.py && python /workspace/visor_viz.py 3`
(scripts in the session scratchpad; annotations at `/workspace/datasets/visor/`).
No checkpoints involved (dataset audit).

Per panel: the VISOR sparse-annotation frame nearest to the interaction's
contact onset (from `epic_hands_package/epic_interaction_index.csv`, 1,305
ok interactions), fixture masks (entity name matching the EPIC noun) filled
red, hands green, other entities blue outline, labels = VISOR entity names,
"[touch]" = VISOR's `in_contact_object` points at the fixture. Header:
narration_id, narration, noun, hand side, VISOR frame and its offset from
onset, span. Up to 3 per noun, ≤2 per video; 5 videos' frame zips failed
to download (transient), so 22 of 27 picks rendered. `contact_sheet.jpg`
= all 22; `picks.json` = the picks with their hit records.

Coverage (from `/workspace/datasets/visor/coverage.json`): 118 of our 340
videos are VISOR videos; 500 / 1,305 interactions have a fixture mask on a
VISOR frame inside the processing window (473 with the hand annotated in
contact with it); nearest such frame is at the onset itself for only 9,
within ±5 frames for 110, ±15 for 273, ±30 for 422, ±60 for 478.

Interpretation: VISOR annotates the ACTIVE object, which for drawers,
oven/dishwasher/fridge/freezer/microwave doors and room doors is the
moving part itself — mask quality is high (tight polygons, hands cut out).
"cupboard"/"cabinet" is inconsistent: sometimes the door panel, sometimes
the whole carcass + door (P18_06). The frames are near, not at, the onset,
so using them needs either VISOR's dense interpolations (mask at the exact
onset frame when it lies between two sparse frames) or re-anchoring the
record to the VISOR frame's camera.
