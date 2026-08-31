# HOI4D real-hand 2D training — design (2026-08-31)

**Goal:** train the `g17_2d_dct` arm (the 2D-only recipe: projection
loss + depth anchor + L_pp + p_rev prior, DCT-6 trajectory head, no 3D
GT) FROM SCRATCH on real HOI4D hand-interaction video, testing whether
mined-real hand data alone teaches the 2D pipeline what SF3D's derived
2D tracks taught it. User-approved init: from scratch (fine-tune
comparison deferred).

## Data sources (all already on our volumes)

- `hoi4d_official_drawer_package/` (main volume): WiLoR hands
  (joints_2d 21×2 px, 1-BASED frame numbers, ~99% right hand) + official
  metric camera poses + intrinsics for ALL 354 furniture sequences
  (187 StorageFurniture C4 + 167 Safe C6); action segments for 8 kitted
  sequences.
- Raw archives (`segaffordance-hoi4d` volume, /workspace/hoi4d_raw/):
  RGB videos (HOI4D_release.zip), depth videos (tar.gz0-6), 2D motion
  segmentation + action JSONs + 3D obj pose (HOI4D_annotations.zip).

## Sample construction (mirrors the SF3D reader semantics)

One sample per usable frame inside an interaction window:

| SF3D field | HOI4D source |
|---|---|
| RGB frame | decoded video frame, resized 512 |
| depth | decoded depth frame (mm→m), aligned, 512 |
| GT mask | 2D motion segmentation of the MOVING part (drawer front / safe door) |
| interaction point | WiLoR wrist joints_2d[0] at the sample frame |
| 2D trajectory | wrist track from sample frame → window end, resampled to 20 pts, RELATIVE (uv, normalized) |
| text | template: "open/close the drawer" (C4) / "open/close the safe" (C6) from the action label |
| intrinsics | official K, normalized |
| 3D GT fields | ABSENT (weights already 0 in the recipe; loader emits zeros + valid=False like the video path) |

Windows: action segments labeled open/close (≈2+2 per sequence → ~1,400
windows). Frames sampled at stride 2 inside each window with ≥5 frames
of remaining track → est. 25–45k samples. Frames without a hand
detection are skipped (80% coverage). is_right filter: keep right hand
only (99%).

Type/axis GT: none used at train (recipe has motion_type_weight=0,
vae_weight=0). For EVAL, open/close on C4 = prismatic, C6 = revolute —
so p_rev separability and traj_dir CAN be scored by category proxy.

## Pipeline steps

1. **Extract** (CPU pod on the HOI4D volume — no GPU needed): unpack
   RGB + depth videos + 2Dseg + action JSONs for the 354 sequences
   only; decode with the official HOI4D-Instructions conventions
   (depth is 16-bit video; RGB 1920×1080@15fps, 300 frames).
   Immediately downscale to 512-side and write per-sequence npz/jpg
   bundles — never keep full-res on disk (volume headroom ~330G).
2. **Process** (`tools/hoi4d_process_2d.py`, new): join hands package +
   extracted frames → SF3D-format LMDBs (data.lmdb records +
   frames_512.lmdb) so `SF3DDataset` reads it unmodified (fields absent
   → zeros, mirroring the 2D video path). Deterministic scene-level
   split: hold out ~15% of SEQUENCES (val), stratified C4/C6.
3. **Transfer** the two LMDBs (est. 8–15G) to the main volume
   (runpodctl send/receive or S3 API; they must live where training
   pods mount).
4. **Train** `20260831_hoi4d_2d_dct`: g17_2d_dct config with only data
   paths + experiment dirs changed; 30 epochs, seed 42; PRO 6000 pod.
5. **Eval**: (a) HOI4D val: mask IoU, point error, projection-loss
   metrics, traj_dir vs wrist GT, p_rev separation by C4/C6 proxy;
   (b) zero-shot 2D metrics on SF3D val (comparability to g17_2d_dct);
   (c) prediction panels (viz batch).

## Risks / open items

- Depth-video decode fidelity (16-bit packing) — validate against the
  known per-subject bias priors before trusting the anchor term.
- 2Dseg may label the whole object, not the moving part — check on the
  8 kitted seqs first; fall back to whole-object mask if part-level is
  absent (noted in results if so).
- 354 unique objects → redundancy; frozen trunk mitigates.
- Action JSONs for the 346 non-kitted seqs come from the annotations
  archive — verify coverage before committing to ~1,400 windows.
