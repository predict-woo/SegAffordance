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
| best 2D-only | 20260822_sf3d_g17_2d_dct | best-epoch19-valloss1.3702 — shape 0.0947, mIoU 0.2655 (= 3D level), roughness 0.032 |

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

## In flight (2026-08-24)

- **Label-efficiency v2 RUNNING** (spec
  docs/superpowers/specs/2026-08-23-label-efficiency-v2-dir-term-design.md;
  all arms = "g21" recipe: g17 + DCT head + pred_pred_art_dir_weight 0.1):
  pods `segaffordance-g21` (C' 20260823_sf3d_g21_dct_dir, 100% 3D) and
  `segaffordance-p90d` (B'1 20260823_sf3d_p90_2d_dir, 90% 2D) both ~ep
  21/30, ~1h left as of 2026-08-24. Then: test passes; A'
  (s10_3d_dir) launches on the freed g21 pod, B'2 (ft10_3d_dir, config
  written then) on the p90d pod; delete pods after. NOTE: CSV logger is
  version_1 on both (a dead 4500 launch claimed version_0). B'1's val
  L_geo_pp_dir 0.64→0.07 = first-ever 2D axis-sign learning.
- **Supervision ablation v2 RUNNING** (spec
  docs/superpowers/specs/2026-08-24-supervision-ablation-v2-design.md;
  separate session, coordinated 2026-08-24): re-runs the 2026-08-15
  joint-vs-either ablation on the g21 recipe AND adds the deconfounding
  arm that spec deferred. Arm A (joint) is NOT retrained — it is the
  reused `20260823_sf3d_g21_dct_dir` (C') run above. New arms, all
  derived from `sf3d_train_runpod_g21_dct_dir.yaml`:
  `20260824_sf3d_supabl2_art` (B, trajectory head gone — L_pp dies with
  it structurally), `20260824_sf3d_supabl2_traj` (C, articulation heads
  gone; needs split_axis_heads + use_origin_local_feature off, which
  gen-9's arm C did not), `20260824_sf3d_supabl2_nolpp` (D, arm A's
  supervision with pred_pred_art_weight + dir_weight 0 — geometric_loss
  stays "pred_pred_art" so L_geo_pred_pred_art is still LOGGED as a
  diagnostic). D is TRAINING on pod `segaffordance-abl-nolpp`; B and C
  are waiting on PRO 6000 stock (exhausted 2026-08-24). Cleanup scope of
  that session = `abl-*` pods only.
- Dev pod still DOWN (host GPUs taken; delete+recreate when next
  needed). Mutagen mirror therefore down — the volume's code tree was
  synced to commit a991ae9 via `git archive | ssh tar -x` through a
  training pod; anyone changing code that must run on pods has to scp
  or re-archive it themselves.

## Earlier (2026-08-22)

- BOTH 2D smoothness arms DONE 2026-08-22: `g17_2d_dct` = best 2D arm
  everywhere (shape 0.0947, mIoU 0.2655 = 3D level; ADOPTED as p90
  pretrain recipe; origin/radius cols exploded = unsupervised garbage,
  ignore); `g17_2d_fdiff2d` = wash, NOT adopted. Notes + INDEX rows in.
- Label-efficiency: `s10_3d` DONE 2026-08-22 (early-stop ep25, best
  ep20) — trunk COLLAPSES on 10% scratch (mIoU 0.021, PDet 0.4) while
  articulation heads degrade gracefully (matched 25.0°). `p90_2d` DONE
  (best ep6; trunk mIoU 0.228, under full-data 0.2655 — early stop
  undertrained it, recorded confound). `ft10_3d` DONE (best ep19):
  **headline** — trunk transfers (mIoU 0.217 = 82% of full-3D, PDet
  10.9, roughness 0.0176 best ever), articulation only partial (MA 8.7
  vs 25.9). Verdict: 2D+10% ≫ 10% alone, short of full 3D on
  articulation. Full table in ft10_3d/notes.md. ALL PODS DELETED —
  nothing running or in flight.
- Dev pod could NOT start (host GPUs taken); volume file transfer goes
  via scp through the training pods meanwhile; dev pod may need
  delete+recreate (state survives — do when next needed for viz/sync).
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

- Rot axis SIGN flips ~13% (sign-aware metrics 2026-08-18). The fix is
  IMPLEMENTED + unit-tested (2026-08-23): midpoint screw-direction term
  in PredPredArticulationLoss (`dir_weight`; 1−cos between trajectory
  chords and the screw velocity field at chord midpoints — exactly 0 at
  consistency for any step size, 2 under a flip; L_pp's locus residuals
  are sign-blind, this is the oriented complement). GT convention
  VERIFIED 2026-08-23 (tests/test_gt_sign_convention.py, runs the real
  writer code via AST extraction): rot arcs sweep right-hand-positive
  about the stored axis and trans rays run along +axis BY CONSTRUCTION
  (writer e2 = n×e1, t∈[0,+90°]; v3 rebuild sign-preserving; reader
  order-preserving) — the term never fights GT supervision. Ready to
  wire to config/trainer and run.
- 2D-only articulation deadlock — candidates: track-curvature pseudo-type
  labels; analytic screw decode (survey option 3, routes 2D gradients
  into articulation params).
- Relational-grounding tail ("second drawer…" misses) — TALENT-style
  contrastive parked.
- Cost-map-without-taps on the current best base (g15's geometry gains).
- Finetune-from-2D vs scratch: DONE 2026-08-22 (label-efficiency study).
  Follow-ups if pursued: longer p90 pretrain (fixed 30 ep, kills the
  undertraining confound); ratio sweep (5%/25%); class-level holdout.
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
  stack — its create fallback REMOVED from train_pod.sh 2026-08-23 after
  a second bad auto-create: 29G shm truncates frames_512 → SIGBUS).
  Stock: poll creates every ~10 min via Monitor-wrapped script
  (~10-90 min to land); WK $1.89/hr, Server $2.09/hr. ALWAYS reconcile
  `pod list` after creates (orphans bill silently) AND verify the landed
  GPU (`nvidia-smi`) before launching; delete pods right after their
  test pass.
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
