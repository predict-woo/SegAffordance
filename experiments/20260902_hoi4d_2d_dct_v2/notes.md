# 20260902_hoi4d_2d_dct_v2 — VLM-selected masks (DATA REBUILD ONLY — training NOT commissioned)

**Change vs v1 (20260901_hoi4d_2d_dct):** the per-window moving-part
mask is chosen by gpt-5.6-luna (effort high) via Set-of-Mark composites
(tools/hoi4d_vlm_select_all.py) instead of the motion-energy heuristic
that picked the HAND in ~57% of v1's records. NONE/ERROR windows
dropped. Everything else identical: same recipe (g17_2d_dct losses),
same split (physical object, 15%), same 30-epoch budget, seed 42.

**Watch vs v1** (v1: mIoU 0.477 [hand-contaminated], point 0.0084,
shape 0.027, traj_dir 49% chance, p_rev AUC 0.657): mIoU on the now
~pure part-mask task (expect LOWER raw number but the RIGHT task);
whether part-mask supervision moves p_rev/traj_dir at all.

Status: user commissioned VLM selection + LMDB rebuild ONLY
(2026-09-02, "stop after lmdb rebuild"). Config staged for whenever
training is commissioned. Rebuilt data: hoi4d volume
/workspace/hoi4d_processed_2d_v2 (transfer to the main volume needed
before any launch).
