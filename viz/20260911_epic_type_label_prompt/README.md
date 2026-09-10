# 20260911_epic_type_label_prompt — VLM rot/trans labelling of the EPIC 2D LMDB

What the VLM (gpt-5.6-luna via codex, `tools/epic_vlm_label_types.py`) was sent per record: the
onset frame at 512 px with the part mask OUTLINED in red and the hand's 2D path in green (cyan
start, magenta end), full frame left + zoomed crop right, title = verb / noun / narration / key;
prompt in the tool. Four example composites here (drawer, fridge, cupboard, oven) and
`pilot20_sheet.jpg` = the 20-record stratified pilot (one to two per noun) with the VLM's
type / confidence / reason under each tile.

Result (full run, 359 records, ~2 min with 6 workers): 172 prismatic (170 drawer + 2 drawer:fridge),
187 revolute (fridge 73, cupboard 58, oven 24, dishwasher 9, freezer 6, microwave 5, cabinet 5,
door 4, window 1, oven:microwave 1, door:microwave 1); every answer high confidence; ZERO
disagreements with the noun rule (drawer -> trans, else -> rot). Reasons cite construction (hinge
edge, runners, handle placement) and the hand path. Applied to
`/workspace/datasets/epic_processed_2d/data.lmdb` (`motion_type` + `motion_type_source =
vlm-gpt-5.6-luna-v1`; backup `data.lmdb.bak_pre_vlm_types`); label file kept at
`docs/data/epic_vlm_type_labels_v1.json`. Only the onset frame exists (the videos were deleted after
the build), so no end-frame view was sent — user decision 2026-09-11: onset-only is accurate enough.
