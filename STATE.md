# SegAffordance — living project state

**The single source of truth for "where is this project right now".**
Update this document at every experiment wrap, decision, or infra change —
it is the first thing a fresh/compacted session should read. Keep entries
terse; details live in the linked specs/notes. Last update: 2026-08-22.

## Current best checkpoints (all on the volume, `experiments/<id>/checkpoints/`)

| role | experiment | checkpoint |
|---|---|---|
| **best articulation (3D)** | 20260821_sf3d_g19_fdiff | best-epoch29-valloss1.1780 — MA 29.9, traj_dir 96.1/0.819 (records) |
| **best smooth/visual (3D)** | 20260821_sf3d_g19_dct | best-epoch20-valloss0.9652 — roughness 0.0090 (10×), mIoU 0.2685 + PDet 21.72 (records) |
| previous overall best | 20260818_sf3d_g17_splitax | best-epoch18-valloss0.9272 |
| best 2D-only | 20260820_sf3d_g17_2d_detach | best-epoch24 — proj2d shape 0.0997 (= 3D level) |

## Generation lineage (full numbers in experiments/INDEX.md + notes.md)

g9 joint baseline (59,174-key split) → g10 normalized L_pp → g11 origin
local sample + v3 data (0.7m trans rays) → g11b w=0.15 (caused rot
collapse) → g12 dinov3+dino.txt → **g13 input 512 (the big jump: mIoU
+66%, PDet 4×)** → g14 taps (mixed, not default) → g15 cost map (geometry
helps, parked) → **g16 normalized trajectory loss (rot-collapse fix)** →
**g17 split axis heads (motion_head_rot/trans, GT-routed; type 95.3, MA
+2.8, origin records)** → g17-2d line (2D-only: projection loss + L_pp;
detach-anchor fix; direction emerges 81%, articulation doesn't) → **g19
smooth trajectories: DCT head (smoothness is architectural, 10×) + fdiff
losses (direction is loss-driven, MA 29.9)**. Gen-18 = renamed g17-2d
(never reuse the name). Refuted ideas: sigmoid-octant axis bug (rescale
exists, segmenter.py:636), emergent type from L_pp (majority baseline).

## In flight (2026-08-22)

- Training: `g2ddct` (arm A 2D base + DCT head) and `g2dfd` (arm A +
  uv-space fdiff port) on two PRO 6000 pods; combined monitor active.
  Wrap: test pass (proj2d + roughness columns), delete pods, compare vs
  arm A (shape 0.0997 / roughness / masks), notes/INDEX.
- The 3D next candidate (recorded, not commissioned): gen-20 = DCT head +
  fdiff losses combined.
- COMMISSIONED next (user, 2026-08-22): 2D-pretrain label-efficiency
  (spec: docs/superpowers/specs/2026-08-22-2d-pretrain-label-efficiency-design.md).
  Arms: A = g17 recipe scratch on 10% train scenes (config
  sf3d_train_runpod_s10_3d.yaml, READY — can launch on a freed pod after
  its test pass, no re-poll); B = best-2D-recipe pretrain on 90%
  (p90_2d — recipe decided by the in-flight 2D arms' results) → g17
  finetune on the 10% via model.finetune_from_path (ft10_3d — config
  written when p90's best ckpt exists); C = existing g17_splitax numbers.
  Machinery landed: data.train_scene_subset pretrain|finetune (scene-level
  greedy-by-sample-count partition, ratio 0.1 seed 4242, val/test
  untouched; partition_subset_by_scene + 9 tests, suite 226).

## Open threads / parked (user decision or next pick)

- Rot axis SIGN flips ~13% (sign-aware metrics 2026-08-18) — candidate:
  sign-consistency between predicted axis and predicted sweep.
- 2D-only articulation deadlock — candidates: track-curvature pseudo-type
  labels; analytic screw decode (survey option 3, routes 2D gradients
  into articulation params).
- Relational-grounding tail ("second drawer…" misses) — TALENT-style
  contrastive parked.
- Cost-map-without-taps on the current best base (g15's geometry gains).
- Finetune-from-2D vs scratch (needs a non-poison 2D ckpt; arm B qualifies).
- 700GB scratch volume deletion (~$49/mo, holds raw SceneFun3D; NEEDS
  USER APPROVAL).
- US mirror volume (~$7/mo, doubles pod-creation surface) — recorded
  option; US PRO 6000 creates verified working (probe 2026-08-22).

## Infra facts

- Volumes: main `bckt1t9uuf` 1TB EU-RO-1 (~$70/mo, MooseFS — silently
  truncates at quota, pause mutagen FIRST on any quota event); scratch
  700GB (pending deletion). Datasets/checkpoints are volume-only.
- Dev pod `segaffordance-dev` $0.57/hr. **Policy since 2026-08-22: stopped
  when idle; start on demand (`dev.sh start`, ~1 min), stop after.** The
  mutagen mirror routes through it — with it off, Mac↔volume syncs queue.
- Training pods: PRO 6000 class only (96GB; 4500-class can't hold the 512
  stack). Stock: poll creates every ~10 min via Monitor-wrapped script
  (~10-90 min to land); WK $1.89/hr, Server $2.09/hr. ALWAYS reconcile
  `pod list` after creates (orphans bill silently); delete pods right
  after their test pass.
- Launch: `bash runpod/train_pod.sh launch <name> <exp_id> <config>` —
  auto-selects trainer, stages LMDBs from the CONFIG's paths to /dev/shm.
  Detached jobs: `setsid nohup ... < /dev/null` (plain nohup died once).
  Local monitors/pollers get reaped by the harness — use the Monitor tool
  (persistent) with error-pattern + triple-GONE checks.
- Test pass: `train_SF3D_better.py test --config <cfg> --ckpt_path <ckpt>
  --trainer.logger=false --data.lmdb_path /dev/shm/data.lmdb
  --data.frame_cache_path /dev/shm/frames.lmdb` on the training pod
  before deleting it (~2 min). Never pass --trainer.enable_checkpointing.
- Data: sf3d_processed_v3 (458k entries; trans = 0.7m rays), frames_512
  LMDB 39G, key cache cutoff05_minrad010_maskfrac0010_edge05 = 59,174
  keys (22.5% rot). Standard eval: 5,088-sample test split; probes/viz
  need `--input-size 512 --frame-cache-path .../sf3d_frames_512.lmdb`.
- Co-author dataset access: RunPod S3 API (header auth only — presigned
  URLs unsupported); key `dataset-share` in user's console — REVOKE when
  co-author is done. Instructions: ~/Downloads/sf3d_coauthor_download.md.

## Conventions

- Workflow per experiment: spec in docs/superpowers/specs/ → implement +
  tests (suite currently 217; local venv
  scratchpad/twistenv/bin/python) → smoke on dev pod if model changed
  (SMOKE_ONLY=<tag> tools/smoke_dinov3_stack.py) → launch → monitor →
  test pass → delete pod → notes.md + INDEX row + viz batch (seed 42421,
  16 samples, `tools/sf3d_vis_predictions.py`) → commit/push → UPDATE
  THIS FILE.
- Metrics: sign-aware axis columns (flip rate = signed>90°); proj2d
  err/anchor/shape (uv); traj_rough_pred/gt (2nd-diff m, GT floor
  0.0032). Type/MA are NOT reported for arms with motion_type_weight 0
  (unsupervised head — harness skips them).
- Naming: experiments `YYYYMMDD_sf3d_<tag>`; slides docs/slides/; surveys
  knowledge/ (repo) — infra lessons live in ../knowledge/ (workspace).
- Git: Mac-side only for mutating ops; commit style ends with
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; keep pushed.
- User prefs: questions are read-only; no Artifacts (local files only);
  all subagents on the session model; cost-sensitive — reconcile pods,
  report spend.
