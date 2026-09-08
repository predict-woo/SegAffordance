# SegAffordance — living project state

**The single source of truth for "where is this project right now".**
Update this document at every experiment wrap, decision, or infra change —
it is the first thing a fresh/compacted session should read. Keep entries
terse; details live in the linked specs/notes. Last update: 2026-09-07.

## COMPACTION SNAPSHOT (2026-08-28) — read this first after context loss

**Nothing is in flight.** No training pods, no downloads, no monitors
needed (monitors die with session restarts anyway — re-arm only for
things actually running). Dev pod `segaffordance-dev` IS RUNNING
($0.57/hr — user knows; stop-when-idle policy applies if they say so).
Volumes: main `bckt1t9uuf` 1TB (~584G used) + `segaffordance-hoi4d`
f2h0jczstn 500GB (163G of raw HOI4D archives, size-verified). Working
tree clean, everything pushed through `afdd56c`.

**The week's intellectual arc (full numbers in INDEX + notes.md):**
1. Trajectory labels are DERIVED from articulation (writer formula) yet
   co-training them is worth MA +7–8. Mechanism pinned by the analytic
   decode (zero-param writer-mirror): ~75–89% of the articulation gain
   is LOSS GEOMETRY, family-robust; mask gains are the DCT head's
   feature shaping (fdiff family inverts the mask ordering).
2. Continuous-limit theory (docs/slides/2026-08-28_continuous_trajectory_loss.html):
   the N→∞ loss = L² pullback metric; closes to Gram quadratics on
   radial/tangential residuals; trans rows = exactly the 1−cos axis
   loss. fdiff limit = Sobolev H¹ (angle/length terms have NO closed
   form — elliptic).
3. **The distillation holds at scale** (20260828_sf3d_closedform):
   closed-form quadratics alone (no trajectory head/curve/params) →
   MA 29.19 (0.7 shy of the all-time record) + ORIGIN RECORD 0.250;
   exact beats sampled +2.5 MA; concedes matched-axis 22.3°.
4. Other measured verdicts: L_pp family-dependent (off = +2.2 MA on
   DCT, −2.4 on fdiff); dir term harmful on DCT (drags trajectories)
   but sets flip-rate (11.24) + matched (14.62°) records on the plain
   head; joint > either alone on fdiff BOTH ways (art→traj +1.4, small
   but real); toy reductions of the mechanism: 3× null — needs the
   real system.

**Top follow-up candidates (parked, uncommissioned):** distilled gen-22
= closed-form quadratics + DCT head (+ maybe dir/detached); Gram
weight/Θ sweep; matched-axis-gap diagnosis (missing non-quadratic angle
term?); HOI4D 10-sequence hand prototype (unpack RGB/depth/2Dseg/objpose/
CAD for cameras ZY20210800001/2 via HOI4D-Instructions decode.py; hands
package at datasets/hoi4d_hands_package — WiLoR frames are 1-BASED,
per-subject depth bias +4–16cm; collaborator branch jiaqchen-epic-hand
NOT on our remote).

**Cross-session facts:** a fork session ("Review SF3D dataset
preprocessing", e85a4a52-…faab66) exists for the user's week-review —
standing by, owns nothing; the supervision-ablation session finished
and closed. New loss knobs since g21: analytic_trajectory_weight,
closed_form_trajectory_weight/velocity_weight (mutually exclusive with
analytic; all default 0, trainer blocks fire only with NO trajectory
head), fdiff-on-decode inside the analytic block. Suite at 312.

**Operational gotchas that keep biting:** background launch wrappers get
killed by the harness — ALWAYS verify pod state directly (staging sizes
+ pgrep '_better.py fi[t]') rather than trusting wrapper output; verify
GPU NAME and post-warmup clocks.sm on every fresh pod (power-capped
lemon hosts, 600W/670MHz signature); EarlyStopping exits print NO
"stopped" marker; CSV logger version_N bumps on relaunch; scratchpad
venv dies with session restarts (tests run on the dev pod instead);
silent mid-run deaths with truncated ~4.35G ckpts = volume quota.

## Current best checkpoints (all on the volume, `experiments/<id>/checkpoints/`)

| role | experiment | checkpoint |
|---|---|---|
| **best MA (3D), single seed** | 20260907_sf3d_g19_dct_ft_hoi4d_tf | best-epoch25-valloss0.9794 — MA **31.13**/signed 30.80 + PDet **23.27** + roughness 0.0079 (records); g19_dct recipe initialized from HOI4D v2 teacher_forcing; axis 28.0°/matched 19.7°, type 91.9 (worse than cf_h1only) |
| best articulation (3D), all-round | 20260828_sf3d_cf_h1only | best-epoch29-valloss1.1303 — MA 30.64/signed 30.11 + all-axis 24.5° + flips-all 9.8, NO trajectory head (H1 quadratic + axis anchor only) |
| **best origin** | 20260828_sf3d_closedform | best-epoch22-valloss1.1792 — origin 0.250 (record), MA 29.19 with NO trajectory head (closed-form pos+der quadratics) |
| prev best articulation | 20260821_sf3d_g19_fdiff | best-epoch29-valloss1.1780 — MA 29.9, traj_dir 96.1/0.819 (traj records stand) |
| **best axis precision (3D)** | 20260825_sf3d_fdiff_dir | best-epoch24-valloss1.2487 — matched 14.62°, rot flips 11.24 (both records) |
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

## Volume quota: RESOLVED 2026-08-24 (trim executed, user-approved)

The 2026-08-24 EDQUOT freeze is over. User approved keep-best-only for
superseded runs: **310.3G freed across 127 files** (41 dirs trimmed to
their single INDEX-reported best ckpt; 12 keep-all dirs untouched — the
STATE best table, OPD bests, label-eff v2 arms, p90_2d). Special cases:
twist_clip keeps last.ckpt (INDEX reference), opdreal_frozenclip keeps
best-epoch11; the supabl2 stale mid-run volume snapshots were deleted
entirely (real bests died with their pods). experiments/ is now 263G,
volume ~584G used, write probe 843 MB/s. Verified: every trimmed dir has
exactly 1 ckpt, keep-all dirs 4. Open volume items still parked:
venv-local.tar.tmp (4.9G, stale), 700GB scratch volume deletion
(needs user), possible resize. Quota lesson stays: MooseFS truncates
silently at quota — pause mutagen FIRST on any quota event; trash holds
nothing (deletes reclaim instantly).

## HOI4D articulation annotator LANDED (2026-09-01)

Web-based manual annotation of articulation parameters on the HOI4D
processed-2D dataset: `tools/hoi4d_annotate_articulation.py` (viser,
Apache-2.0; spec docs/superpowers/specs/2026-09-01-hoi4d-articulation-
annotator-design.md, plan in docs/superpowers/plans/). Per-sequence
WORLD-frame annotation (axis+origin gizmo, prismatic/revolute, motion
preview sweep, align-to-trajectory init) over a fused multi-frame
RGB-D cloud lifted with hands354 official_poses.npy; per-sequence JSONs
+ `export` mode projecting to SF3D-convention camera-frame fields
(sidecar pickle). 15 unit tests (geometry/store/scene/export) green in
a local venv; UI smoke ran locally against synthetic LMDBs (server up,
startup clean, export end-to-end). REAL-data smoke pending: needs the
processed LMDBs (ethz-workspace-65's build) + dev pod (start blocked
2026-09-01, host GPUs taken). Serve on the pod:
`python3 tools/hoi4d_annotate_articulation.py serve --data
/workspace/hoi4d_processed_2d --hands <hands354> --out
/workspace/hoi4d_processed_2d/annotations` + SSH port-forward 8080.
Known data quirk handled upstream: some action JSONs run a 10s clock
(13/354 furniture seqs + 8 missing the field) — found here 2026-09-01,
fix landed in hoi4d_process_2d.py load_windows by ethz-workspace-65.

## External screw-loss A/B (sibling session, ethz-workspace-17) — IN PROGRESS

Fine-tune A/B of `closed_form_screw_loss` on published articulation
models from released checkpoints. Spec:
docs/specs/2026-08-29_external_screw_loss_ab.md; results under
experiments/external/<method>/ (that session owns those paths + its
EU-FR-1 volumes).

**Result #1 — USDNet (Articulate3D): NULL / slightly negative
(2026-08-29).** Seed-matched pair: theirs AP50 M/axis/origin/both
0.552/0.439/0.446/0.094 vs ours-v2 (their losses kept + capped H1,
λ=0.05) 0.532/0.416/0.423/0.100 — deltas within their seed noise (~5 AP
between two "theirs" seeds; their config ships seed:null). Naive v1
(H1 REPLACING their line distance, uncapped) clearly harmful. Two
transfer lessons: (1) the |r*|² relative normalization EXPLODES on real
scenes (levers 0.1–1 m, early-training origins metres off → per-row
12–15) — usable only capped; (2) USDNet already supervises sign via
1−cos and its eval uses |cos|, so H1's sign sensitivity is invisible to
their metric — the anchor story can't even show up there. **Result #2 — SINGAPO (ICLR'25): NULL (2026-08-30).** Matched 20-epoch
fine-tunes from the released ckpt (seed/data-order/LR matched; ours =
x̂₀-space H1(1.0)+anchor(0.5) on the 6 joint channels, λ=0.05, min-SNR
γ=5, |r*|≥0.05 rows masked). Sign-aware axis eval: init 12.38°/6.7%
flips → theirs 12.94°/4.7% → ours 13.27°/8.1% — inside the init→theirs
drift band, and flips moved the WRONG way for the sign story. Their
IoU/CD metrics equally a wash. Diagnosis (theirs, sound): a converged
ε-prediction diffusion model is a fixed point of its own loss — "theirs"
barely moves either — and a λ=0.05 auxiliary is a light touch by
construction. The honest stronger test = training from the CAGE init
with the full 200-epoch recipe (~$50) — recorded as a candidate, NOT
commissioned (user decision). Notes: experiments/external/singapo/.

Follow-up read (per-input): the axis-error distribution is BIMODAL —
median per-input error ≈0.1° (axes snap to canonical directions), all
of the mean lives in a ~21% tail >20°; the apparent best-of-5
separation = 3–6 of 154 tail objects keeping one good sample, no tail
sharpening or mode shift. Confirms "regime, not loss."

**Result #3 — USDNet ADDITIVE follow-up (user-requested, 2026-08-30):
first non-null, and it INVERTS the SF3D ordering.** All their losses
kept, our term added at λ=0.05, seed 1, final epoch, vs theirs_s1
M/MA/MO/MAO 0.552/0.439/0.446/0.094: capped L2+H1 null; UNCAPPED L2+H1
MAO +0.037 but MA −0.089; **UNCAPPED L2-only MAO 0.144 (+0.050, best
+both of the campaign) with MA intact (−0.007)**. On USDNet the
position quadratic drives the combined axis+origin gate and H1 COSTS
axis AP — the inverse of our clean-data grid (where H1 is the engine
and L2 deadweight-with-anchor). Plausibly regime-consistent: USDNet
already has strong direct axis supervision (1−cos) and a |cos| eval, so
the marginal value lives in origin coupling — exactly where L2's Gram
weight sits. Caveats: one seed, final-epoch numbers on the reporting
split (user declined a best-epoch protocol), uncapped terms dominate
their line-dist loss (13–24× mean). Obvious next: second seed. Six arm
CSVs: experiments/external/usdnet/.

**Result #4 — SINGAPO + dominant L2 (user-requested, 2026-08-30):
NEGATIVE, and it completes the export rule.** Their loss + L2 position
only at λ=0.5 on the 6 joint channels: axis −0.9° (tail-count only,
4–5 objects), but their geometry metrics degrade across the board and
part inter-penetration (AOR) worsens 3.7× — a strong joint-channel term
perturbs the box channels through the shared denoiser (bbox+joints are
one tensor), and SINGAPO's boxes ARE its origin supervision. Combined
with #3, the transfer rule is clean: **export the constraint the host
model is missing, at a weight its own losses can absorb** — USDNet
lacked origin coupling (L2 helped), SINGAPO lacks nothing our terms
provide (everything hurts or is null). Ops note (folded into
train_pod.sh + README): 3 power-capped Workstation-edition PRO 6000
lemons observed across the two sessions; every Server Edition ran full
clocks.

**CAMPAIGN CLOSED (user, 2026-08-31).** All three external volumes
deleted (verified: only bckt1t9uuf + f2h0jczstn remain; only pod =
segaffordance-dev). Results/bundles preserved in
experiments/external/. The two follow-up candidates (USDNet second
seed; SINGAPO CAGE-init ~$50) stay uncommissioned and would need data
re-staging (~15min USDNet / ~1h SINGAPO) if ever picked up.

Summary of the closed program: replace-mode fine-tunes NULL; additive
arms split by host structure (USDNet L2-only positive at the origin
gate, SINGAPO negative via shared-tensor leakage);
DIPO dropped (incomplete release); Particulate blocked on the user's HF
dataset access. All external pods deleted; volumes kept.

## Closed-form composition grid COMPLETE (2026-08-29 overnight): the 2x2 + Theta

Three overnight single-knob arms (cf_h1_noaxis, cf_l2_noaxis,
cf_noaxis_2pi) completed the decomposition. MA grid:

|  | + anchor | no anchor |
|---|---|---|
| L2+H1 | 29.19 | 27.71 (2pi variant: 27.04) |
| H1 | **30.64 RECORD** | 25.92 |
| L2 | unrun (low value) | 23.80 |

Findings (details in the three notes.md):
1. **Additivity breaks (−3.2 MA interaction): the anchor and the
   position L2 are SUBSTITUTE stabilizers for H1.** H1 needs one
   complementary constraint; anchor is the better one (30.64 vs 27.71),
   and with it present L2 flips to deadweight (−1.45). Unanchored H1
   also blurs matched (19.3°) and degrades the trunk (masks 0.250/18.0,
   family-worst) and peaks early. Sweet spot = one scale-free direct
   constraint + one geometry-coupled derivative term.
2. **The Theta (sweep) knob is a sign-robustness ↔ direction-precision
   dial, and its mechanism is DECOUPLING, not penalty magnitude**
   (cf_noaxis_2pi: rot flips 15.9→13.0 — best no-anchor arm, beats
   closedform-with-anchor — while matched blurs 17.6→20.2 and origin
   worsens; ALL THREE registered predictions were directionally wrong).
   Third-time-confirmed corollary from cf_l2_noaxis (rot flips 20.1
   despite the family's largest flip penalty): what fixes revolute sign
   is gradient geometry (scale-free anchor or decoupled/derivative-side
   tangential gradients), never symmetric-point penalty size.
3. **L2 alone is the weak half decisively** (+3.4 over arm B vs +10.2
   for H1+anchor; earliest overfit; origin 0.277 worst — it helps
   origin only in combination).
Parked next: Gram weight/Theta sweep; H1(pi/2) + small pos(2pi)
anti-flip combo (reframed by the 2pi result); {L2+anchor} corner (low
value); distilled gen-22 = H1-only + DCT head. Ops: lemon-host check
moved UNDER LOAD in the runbook (056bcfb) after an idle-check pass
collapsed to 562 MHz / 4x slow mid-run (swap+resume worked cleanly).

## cf_h1only: NEW ALL-TIME MA RECORD (2026-08-29) — the distillation beats its teacher

The closed-form FDIFF (H1 derivative quadratic ONLY at weight 1.0,
position quadratic OFF, axis loss on, no trajectory head/L_pp):
**MA 30.64 / signed 30.11** (prev record g19_fdiff 29.91 with the full
trajectory apparatus), matched 16.6°, all-axis 24.5° + flips-all 9.8
(both best ever), pass_m 95.5, masks 0.266/21.8 (best of every
traj-supervised arm). Origin 0.254 — 4mm shy of closedform's record:
the position quadratic's only real job was absolute lever placement;
for MA it was deadweight. Composition picture: H1 supervises
shape-of-motion, 1-cos anchors sign (cf_noaxis showed what's lost
without it). Follow-ups parked: small-position sweep (0.1/0.2) to chase
both records in one run; H1-only + DCT head = distilled gen-22; seed
replicate. Notes: 20260828_sf3d_cf_h1only/notes.md.

## cf_noaxis ablation COMPLETE (2026-08-28)

Dropping the direct 1-cos axis loss from the closedform recipe
(vae_weight 0; identifiability argument says the cf quadratics + 3D
origin/point losses uniquely pin the articulation without it): **theory
confirmed** — MA 27.71 with zero direct axis supervision (+7.3 over arm
B) — but the anchor earns −1.5 MA, all through revolute sign (rot flips
13.7→15.9, the predicted antipodal-saddle cost), while matched axis
SHARPENS 22.3°→17.6° (the 1-cos was blurring precision to buy sign-
robustness). Verdict: axis loss on for MA runs, off for precision,
w≈0.1–0.25 the untested interpolation. Notes:
20260828_sf3d_cf_noaxis/notes.md. Operational note: with the dev pod
stopped, the mutagen mirror is dormant — new configs must be scp'd to
the volume before launching (bit this launch).

## Closed-form loss experiment COMPLETE (2026-08-28)

The continuous-limit distillation holds at scale: arm-B config + the two
closed-form Gram quadratics (position L2 + derivative H1; no trajectory
head, no sampled curve, zero params) reaches **MA 29.19** (0.7 shy of the
all-time record that needed head+fdiff+L_pp) and a **0.250 origin
record**; exact beats its own sampled approximation by +2.5 MA (the H1
quadratic outperforms the fdiff trio in the no-head setting); concedes
matched-axis sharpness (22.3°). Notes:
20260828_sf3d_closedform/notes.md. Follow-ups parked: Gram weight/Θ
sweep; closed form + DCT head = the distilled gen-22 candidate.

## DONE 2026-09-08 ~21:45 local: ARCTIC 2D dataset v1 BUILT (GT masks + GT articulation axes)

**`/workspace/datasets/arctic_processed_2d/`** — **2,559 records** (rebuilt after the stroke-splitter fix: strokes end at the
last moving frame, pauses > 5 frames break them — the first build (2,633)
absorbed rest phases into strokes; median stroke 22 frames, p90 38), SF3D reader format, one record per single articulation STROKE
(monotone angle run >= 10 deg, >= 10 frames) of 238 of the 239 "use"
sequences (s01_box_use_01's ego frames never downloaded — the MPI server
rate-limited the pod for the rest of the day; user: drop it for now).
Keys `<subject>_<object>/<seq>_s<k>_f<frame>`. Per object: espresso 332,
capsule machine 283, scissors 274, mixer 264, waffle iron 260, microwave
258, notebook 237, ketchup 211, phone 191, laptop 145, box 104. Drops:
hand outside the frame at the stroke start 130, no image 12, tiny mask 6,
far-from-mask 5, frame jump 3. What is DIFFERENT from HOI4D/EPIC:
mask = the moving part RENDERED from ARCTIC's dense per-part meshes with
the mocap GT pose (pixel-accurate; hands NOT cut out), depth = object-only
render (mm), `motion_info` = REAL revolute axis + origin in the camera
(type "rot") -> 3D articulation supervision is possible on this set;
description = template "open/close the <object> <part>". Conventions
verified the hard way: the moving part rotates by -angle about canonical
+z (checked against the collaborator's fingertip-to-part distances; the
+angle build looked fine on closed lids and was wrong on open ones — user
caught it), ego camera X_cam = R X_world + T, object translations in mm,
image index = frame + 1. Hand = the one with the smaller MEDIAN fingertip
distance to the moving part over the stroke. Outlier thresholds scaled by
fx (2414). Review: `viz/20260908_arctic_v1_lmdb_sample` + `..._random24b` (2 x 24
uniform random records, all masks on the moving part, single-arc tracks)
+ `viz/20260908_arctic_builder_test` (panels). Reader
smoke test (`config/arctic_v1_smoke.yaml`, fast_dev_run) passed.
Inputs on the volume: `arctic_gt_package/` (collaborator), `arctic/{meta,
raw_seqs}`, `arctic/images/<sid>/<seq>/0/` (60 GB, ego view only via
`tools/arctic_fetch_ego.py` byte-range reads of the per-sequence zips —
the MPI server 403-blocks an IP for 5+ min after bursts; credentials in
the dev pod's /root/.arctic_env only — ASK THE USER TO ROTATE THE
PASSWORD). Builder: `tools/arctic_process_2d.py` (~6 s per stroke on the
dense meshes; 25 min on 12 workers). Open: the s01_box_use_01 gap; hands
in masks; the model's depth input (object-only depth here, zeros on EPIC).

## DONE 2026-09-08 ~03:40 local: SF3D post-training from ALL FOUR HOI4D arms — table complete

| init (HOI4D v2 arm) | MA / signed | PDet / mIoU | rough | axis all / matched |
|---|---|---|---|---|
| scratch g19_dct | 25.98 / 25.83 | 21.72 / 0.2685 | 0.0090 | 25.3 / 18.2 |
| dct_baseline (DCT, detach) | 29.76 / 29.44 | 18.28 / 0.240 | 0.0081 | 31.0 / 22.9 |
| **teacher_forcing (DCT, GT anchor)** | **31.13 / 30.80** | **23.27** / 0.266 | 0.0079 | 28.0 / 19.7 |
| baseline (plain, detach) | 27.14 / 27.06 | 17.65 / 0.231 | 0.0083 | 27.8 / 16.4 |
| teacher_forcing_plain (plain, GT anchor) | 30.62 / 30.31 | 20.03 / 0.256 | 0.0079 | 28.0 / 20.0 |

Every HOI4D init raises MA; the teacher-forced arms transfer far better
than the detach arms on both heads (masks included), and the DCT head adds
on top. All single seeds. Pod E deleted automatically. **No pods running
except the dev pod.**

## DONE 2026-09-08 ~03:10 local: EPIC/VISOR 2D dataset v1 BUILT (overnight production run)

**`/workspace/datasets/epic_processed_2d/`** (main volume): `data.lmdb` +
`frames.lmdb` — **359 records** (FINAL for now, rebuilt ~04:55: trajectories = the full narrated action, onset..span_end + 4 frames, `--traj-frac 1.0`; stored length 5-86 stride-2 samples, median 20 — the READER resamples every trajectory to 20 points with linspace (scenefun3d.py ~l.732), so stored length only sets the time span. The collaborator's raw tracks ran 90 frames past the action (326-record uncut build had wandering tails); a half-span build (353, median 12) was tried and REVERTED by the user), SF3D reader format (same as HOI4D v2:
512x512 frame, thinned mask coords, 2D knuckle trajectory in pixels,
placeholder 3D, intrinsics, description = EPIC narration, motion stub);
keys `<video_id>/<narration_id>` (scene = kitchen video). **Depth = zeros**
(EPIC has none; `epic.hand_z_onset_m` keeps the WiLoR hand depth) — the
model's depth input must be made optional before training on it (user
parked this). `work/` (605 MB) = the 478 SAM2-propagated items (onset.jpg
full-res, mask.png, meta.json, panel.jpg QA) for every VISOR-covered
interaction with |d| <= 60; `build_stats.json`. Filters to 359: |d| <= 30
(-56), area-ratio 0.35-2.5 (-36), start > 300 px from mask (-14), frame-jump
(-9), < 5 points (-4). Per noun: drawer 170, fridge 73, cupboard 58, oven
24, dishwasher 9, others 25. Review sheet `viz/20260908_epic_v1_random20`
(this build: masks on the moving part 19/20, the miss = VISOR whole-cupboard
label); `..._random20_half` (reverted half-span) and `..._lmdb_sample`
(uncut) are superseded. Reader smoke test passed
(`config/epic_v1_smoke.yaml` fast_dev_run: 326 read, kitchen split, one
train+val step). Run facts: 115 videos (~700 GB streamed, deleted after
use; `epic_videos/` empty), 2 pods (dev + PRO 4000 pod B, deleted), 1 h 50
min wall-clock, 3.8 s per item; bugs fixed on the way (numpy float in
JSON, ffmpeg eating the shard's stdin, EPIC-55 test-split videos under
`videos/test/`, P12 videos are 720p while VISOR polygons stay 1080p).
Tools: `tools/epic_pipeline_shard.sh`, `epic_fetch_video.py`,
`epic_visor_propagate_batch.py`, `epic_process_2d.py`, `epic_lmdb_sample.py`.
Open: depth-optional model; a |d| <= 60 build is one command if more data
is wanted (478 items); the 800 non-VISOR interactions remain dropped;
cupboard whole-carcass masks (~49 records) unreviewed individually.

## EPIC prep — SAM2 propagation of VISOR masks VALIDATED (2026-09-08 ~00:30 local)

User decision: drop the ~800 non-VISOR interactions for now; masks = VISOR
polygon on the nearest sparse frame propagated to the contact onset with
SAM2 (video predictor, hiera-large; installed in the dev pod's /opt/venv
via `uv pip`, ckpt /workspace/models/sam2.1_hiera_large.pt). Test on 4
videos (P28_103, P04_05, P22_07, P01_09; full-HD MP4s at
/workspace/datasets/epic_videos/, 6-13 GB each): 18 sample propagations
onto the onset (offsets -15..+17) all clean; 24 sparse-to-sparse validation
round trips (gaps 15-40 frames) mean IoU 0.84, lows = VISOR's door-vs-
door+interior inconsistency (SAM2 keeps the moving part) plus ONE real
failure (dishwasher door swinging 90 deg over 40 frames, IoU 0.15 — flag
area-ratio outliers). Batch: viz/20260907_epic_visor_propagate (README has
the table). Tools committed: tools/epic_visor_{coverage,coverage_dense,
viz,propagate}.py, tools/epic_visor_fetch_{ann,interp}.sh,
tools/epic_fetch_video.py (chunked range download: 36 MB/s vs 2.7 single
stream; the HF mirror a1raman/epic_kitchens_100 is slower from EU-RO-1).
NEXT: the production pipeline on the user's separate pod/volume —
per-video: chunked fetch -> cut onset/window frames -> propagate the
VISOR mask for every covered interaction (~420 with |d|<=30) -> LMDB
records (RGB 512, mask, knuckle trajectory, intrinsics, narration); depth
still open (user: make the model's depth input optional later).

## EPIC-KITCHENS prep — VISOR mask audit DONE (2026-09-07 ~22:30 local)

User scope: VISOR-covered interactions only (the ~800 without VISOR are
DROPPED for now); depth: model change to make it optional, later. Data on
the main volume: `/workspace/datasets/visor/` = sparse annotation JSONs
(158 videos, 833 MB), dense interpolation zips for our 118 overlapping
videos (8.7 GB, `interpolations/`), frame_mapping.json, `coverage.json`
(sparse hits per interaction), `coverage_dense.json` + `dense_out/`
(exact-onset polygons). Viz: `viz/20260907_epic_visor_masks/`.
Numbers (1,305 ok interactions): 532 in VISOR videos; **500 have a
fixture mask on a sparse frame inside the window** (473 with the hand
annotated in contact); nearest sparse frame exactly at onset 9, ±15: 273,
±30: 422; before-or-at onset within 30 frames: 166. **Dense
interpolations give an exact-onset fixture mask for only 73** (VISOR
filters interpolations to entities present in both endpoints) — the dense
route is dead. Mask semantics: drawers + fridge/freezer/oven/dishwasher/
microwave/room doors = the moving part (tight polygons, hand cut out);
cupboard/cabinet inconsistent (door vs whole carcass). Open decision: how
to put the mask on the onset frame — (a) SAM2 propagation of the nearest
sparse mask to the onset (≤30 frames for 422), (b) re-anchor the record to
the sparse frame (collaborator has the EPIC-Fields poses), (c) accept
d≤0 frames only (166). Ops lessons: dev-pod cgroup is 31 GB (host shows
125); never parse zips from the FUSE mount (stage to /dev/shm or NVMe);
multiprocessing.Pool hung + OOM — per-video xargs workers with streaming
raw_decode worked (scratchpad visor_coverage_dense.py).

## DONE 2026-09-07 19:10 local: cross-eval — SF3D-trained checkpoints on the HOI4D held-out split

`experiments/20260907_xeval_sf3d_on_hoi4d` (dev pod, HOI4D teacher_forcing
config). mIoU / PDet: pure SF3D g19_dct 0.044 / 0.2; post-trained from
dct_baseline 0.131 / 2.4; from teacher_forcing 0.113 / 2.0; HOI4D-only
teacher_forcing 0.727 / 88.0. Pure SF3D has ZERO transfer to hand video;
30 SF3D epochs FORGET HOI4D almost completely. HOI4D pretraining is a
better init for SF3D, not a two-domain model (mixed-domain training would
be the route to one). Dev pod left RUNNING (user policy).

## IN FLIGHT 2026-09-07 ~18:15 local: SF3D g19_dct post-training from the PLAIN HOI4D arms

User-commissioned: the same g19_dct post-training as ft_hoi4d / ft_hoi4d_tf,
initialized from `20260907_hoi4d_2d_v2_baseline` (-> exp
`20260907_sf3d_g19_dct_ft_hoi4d_plain`) and from
`…_teacher_forcing_plain` (-> `…_ft_hoi4d_tf_plain`). Plain 20-point head
has no DCT counterpart: `load_finetune_weights` is name+shape filtered, so
only the trajectory head's output projection re-inits. Configs + notes
committed (8387a2a; the baseline ckpt name is resolved on the pod at launch
and copied back). PRO 6000 stock was EMPTY on both SKUs at 18:05 — a
persistent Monitor (`sf3d_plain_ft_driver.sh` in the scratchpad) retries
`train_pod.sh create sf3d-e` every 5 min, then ships configs + the pod
script `run_sf3d_plain_ft.sh` (dev-pod sync is DOWN, so files go by scp),
runs both arms back-to-back via sweep_queue + test pass, fetches
test.log/metrics.csv per arm, deletes the pod on SF3D_PLAIN_ALL_DONE.
Expected ~2.5 h per arm once a pod boots. If the monitor is gone after a
restart: `runpodctl pod list` (running pods only), ssh alias segaff-sf3d-e,
log /workspace/SegAffordance/sf3d_plain_ft.log.

## DONE 2026-09-07 17:30 local: HOI4D v2 2x2 grid complete (plain + teacher forcing)

**Nothing running; pod D deleted 17:27 local; dev pod + probe16 stopped.**
Last cell `20260907_hoi4d_2d_v2_teacher_forcing_plain` (plain head +
GT-anchored projection): val 0.3460 @ ep 81, held-out mIoU 0.708 / PDet
86.7 / shape 0.0413 — the WORST of the four. Full grid (mIoU / PDet /
shape, 110 objects, single seed): dct_baseline 0.720/86.5/0.0379,
baseline 0.716/88.2/0.0398, teacher_forcing 0.727/88.0/0.0377,
teacher_forcing_plain 0.708/86.7/0.0413. The GT anchor helps only with
the DCT basis; **DCT + teacher forcing is the HOI4D recipe** (and its
checkpoint is the init behind the SF3D MA record below).

**NEXT (user, 2026-09-07 evening): EPIC-KITCHENS hand data.** Package
`/workspace/datasets/epic_hands_package.zip` (11 MB, unpacked beside it;
collaborator branch jiaqchen-epic-hand, 2026-09-01): 1,305 open/close
fixture interactions in 340 videos (888 metric "grip" regime), 21-joint
WiLoR tracks in the onset-frame camera at 1920x1080 intrinsics, narration
+ verb/noun + side + span/window. Missing vs the HOI4D record: RGB (must
stream full-HD mp4s per video — pre-extracted rgb_frames are 456x256),
moving-part mask (VISOR object-level masks on sparse frames; coverage of
our onset frames UNMEASURED — fetch VISOR JSONs first; fallback = SAM
candidates at the onset knuckle + set-of-mark VLM like HOI4D), depth
(none in EPIC; model fuses a depth input — needs monocular metric depth
scale-aligned to the WiLoR hand z, user decision pending). EPIC-Fields
and EPIC-Sounds: not needed. HOI4D volume KEPT (user, $35/mo).

## DONE 2026-09-07 ~02:00 UTC: SF3D 3D-DCT post-training from HOI4D — NEW MA RECORD

**Nothing running; all training pods deleted; probe16 + dev pod stopped.**
Two g19_dct-recipe runs (full SF3D v3, 30 ep) initialized from HOI4D v2
depth-complete checkpoints, test protocol = 5,088 as always:

| run | init | val | MA / signed | PDet / mIoU | axis all / matched | flips | type | origin | rough |
|---|---|---|---|---|---|---|---|---|---|
| 20260821_sf3d_g19_dct (scratch) | — | 0.9652 | 25.98 / 25.83 | 21.72 / 0.2685 | 25.3 / 18.2 | 9.98 | 95.1 | 0.276 | 0.0090 |
| 20260907_sf3d_g19_dct_ft_hoi4d | dct_baseline | 1.0097 | 29.76 / 29.44 | 18.28 / 0.240 | 31.0 / 22.9 | 12.2 | 91.5 | 0.269 | 0.0081 |
| **20260907_sf3d_g19_dct_ft_hoi4d_tf** | teacher_forcing | 0.9794 | **31.13 / 30.80** | **23.27** / 0.2664 | 28.0 / 19.7 | 12.4 | 91.9 | 0.293 | **0.0079** |

**MA 31.13 = new all-time record** (prev cf_h1only 30.64 / signed 30.11) on
the plain g19_dct recipe, plus a PDet record (23.27 vs g21's 23.21) and
best DCT roughness — paid with worse axis precision (+2.7° all / +1.5°
matched), flips, type (−3.2) and origin. HOI4D init from dct_baseline
gets the MA gain but LOSES masks (PDet −3.4); the teacher_forcing init is
strictly better. Single seeds — the natural follow-ups: repeat with a
second seed; ft from teacher_forcing + the fdiff/dir 3D losses (the axis
-precision recipes); a shorter LR schedule (val already best at ep 25).
Ops: pods A/B/C deleted automatically after fetch; test logs + metrics in
each experiment dir (committed). Cost today ≈ $55 across 6 pod-instances.

## (superseded) OVERNIGHT 2026-09-06→07 (user asleep): two SF3D 3D-DCT post-training runs from HOI4D

HOI4D depth-complete arms DONE and scored (110 held-out objects): dct_baseline
0.720 mIoU / 86.5 PDet / shape 0.0379; baseline (plain head) 0.716 / 88.2 /
0.0398; teacher_forcing 0.727 / 88.0 / 0.0377 — a statistical tie; DCT
kept, teacher forcing = legitimate alternative to detach. Pod B deleted.
RUNNING: `20260907_sf3d_g19_dct_ft_hoi4d` (g19_dct recipe, full SF3D v3,
init dct_baseline best-epoch90) on pod C (jtto59pz4pjimb, from 21:38 UTC,
~5 h + test) and `20260907_sf3d_g19_dct_ft_hoi4d_tf` (init teacher_forcing
best-epoch88) on pod A (kku2vv165h75c3, from 21:52 UTC). Baseline to beat:
20260821_sf3d_g19_dct from scratch (val 0.9652 @ ep20, mIoU 0.2685, PDet
21.72, roughness 0.0090). Background waiters fetch test.log + metrics.csv
to the Mac and DELETE each pod when its run + test pass finish (markers
SF3D_FT_DONE in sweep_queue_c2.log, SF3D_FT_TF_DONE in sweep_queue_a3.log
on the volume). If this session dies: results live on the volume under
experiments/<id>/logs/test.log; pods must be deleted by hand
(`runpodctl get pod`). probe16 + dev pod stopped. Volume: main 1.5 TB.

## IN FLIGHT 2026-09-06 evening: teacher-forced anchor vs fullfix on depth-complete HOI4D v2

Findings: (1) HOI4D's `trajectory_3d_camera_coords` (WiLoR joints_3d_cam)
is SCALE-LESS — z ~25 m vs 0.7 m sensor, ~230 px reprojection error —
never use it as geometry; (2) the pred_depth projection anchor skips rows
without depth (`anchor_ok = z > 1e-3`), and v2 had depth only for C4/C6,
so the SWEEP'S TRAJECTORY TERM TRAINED ON FURNITURE ONLY (masks/points on
all). Done: depth extracted for all 2,973 seqs on probe16 (note: GNU tar
`-C` must precede the member pattern — the first attempt extracted into
cwd; HOI4D volume briefly hit quota; raw depth tar parts (143 GB) and the
per-frame LMDB deleted, volume now ~311/500 GB); `run_rebuild_v2d.sh`
built `/workspace/hoi4d_processed_2d_v2d` (3,084 records, depth for all
categories, window enumeration now FROZEN to the sweep verb set — the
09:16 one-per-window build had off-by-one windows after cut/binding
windows, ~12 C18 records affected) → now IS `datasets/hoi4d_processed_2d_v2`
on the main volume (old build kept as `hoi4d_processed_2d_v2_nodepth`). New
loss option `trajectory_proj_anchor: gt_point` (TEACHER FORCING = GT 2D
first point lifted with input depth; constant anchor; `depth_anchor_source`
likewise; test metric follows) — smoke-tested. Arms RUNNING on pod B since 19:50 UTC (queue log sweep_queue_c.log, then test passes, FINAL_DONE): `lr3e5_depth` (reference), `tf`, `fullfix` (plain head +
unnormalized proj MSE) — configs `config/hoi4d_v2_{lr3e5_depth,tf,fullfix}.yaml`,
dirs `experiments/20260907_hoi4d_2d_v2_*`. User: SF3D-init line NOT wanted.
**Main volume RESIZED to 1,500 GB (user, 2026-09-06).**

## VOLUME QUOTA HIT + TRIM (2026-09-06)

At 13:45 UTC the main volume (1 TB) hit "Disk quota exceeded" (sweep
checkpoints, 16.6 GB/arm) and silently killed both training pods' jobs for
2 h. Fix: `runpod/sweep_queue.sh` now keeps ONLY the best checkpoint per
arm; remaining arm configs use save_top_k 1. Then, user-approved, EVERY
finished experiment was trimmed to its single best-valloss checkpoint
(23 dirs, **284 GB freed**; experiments/ 523 → 239 GB; volume ~562/1000 GB).
Kept files match the "Current best checkpoints" table below. Dirs with only
a last.ckpt (opdreal_frozenclip, sf3d_twist_clip, hoi4d_2d_dct_v2 stub)
untouched. Lesson: 4.3 GB per ckpt × save_top_k 3 + last — never leave
that default on for sweeps.

## HOI4D v2 HYPER-PARAMETER SWEEP COMPLETE (2026-09-06 18:30 UTC)

**Winner (from scratch, user wants no SF3D-init line): `e100_lr3e5`** =
v1 recipe with lr 3e-5 (milestones 80/92), 100 epochs, batch 64 —
held-out (110 objects) mIoU 0.694 / PDet 86.7 / point 0.0148 / traj shape
0.0329, val 0.3518 @ ep 77; checkpoint
`experiments/20260906_hoi4d_2d_v2_e100_lr3e5/checkpoints/best-epoch77-valloss0.3518.ckpt`;
config `config/hoi4d_v2_sweep_e100_lr3e5.yaml`. Panels:
viz/20260906_hoi4d_v2_lr3e5_val_panels.

| arm | val | mIoU | PDet | point | shape |
|---|---|---|---|---|---|
| base (v1: 30 ep, lr 1e-5) | 0.391 | 0.551 | 68.0 | 0.0240 | 0.0369 |
| e100 | 0.386 | 0.626 | 78.7 | 0.0185 | 0.0346 |
| e100_lr3e6 | 0.438 | 0.521 | 61.3 | 0.0268 | 0.0379 |
| **e100_lr3e5** | **0.352** | **0.694** | **86.7** | **0.0148** | **0.0329** |
| e100_lr6e5 | 0.355 | 0.674 | 84.8 | 0.0161 | 0.0345 |
| e100_lr1e4 | 0.354 | 0.702 | 86.7 | 0.0156 | 0.0331 |
| e100_bs32 | 0.358 | 0.688 | 86.3 | 0.0153 | 0.0332 |
| e200 | 0.368 | 0.656 | 82.2 | 0.0167 | 0.0331 |
| e100_ft (SF3D init) | 0.364 | 0.690 | 85.2 | 0.0179 | 0.0330 |

Findings: LR is the lever — 3e-5…1e-4 all reach the same ~0.35 val floor
(1e-5 stalls at ~0.39, 3e-6 worse); reached by ep 20–40, later epochs are
noise/overfit (train 0.15 vs val 0.40 at lr 1e-5/100 ep); batch 32 ≈ more
updates ≈ same effect; 200 epochs confirms the horizon. traj_dir acc at
chance for every arm (as v1). SF3D init helps at low LR but less than LR;
the combination arm e100_ft_lr3e5 was CANCELLED at ep 1 (user).
Ops: 10 arms on two RTX PRO 6000 pods (~$35 incl. 2 h lost to the quota
incident). Pod A deleted; **pod B (segaff-hoi4d-b, $2.19/hr) still UP** for
optional extra visualizations — delete when done. Dev pod + probe16
stopped (two stopped dev-pod entries exist: 0dguj91q3tmy5c and
v0clt0iywmvoc5 — one is stale; check which `dev.sh` uses, delete the
other). The mirror is dormant: sweep configs/notes reached the volume by
scp; Mac-side commits are ahead of the volume's repo copy.

## DONE OVERNIGHT 2026-09-06: full-package VLM sweep v2 (terra) + HOI4D 2D LMDB v2

Results on the HOI4D volume:
- `/workspace/vlm_select_v2/selections.json`: 12,890 windows — 12,639
  answered by **gpt-5.6-terra** (24 workers, fast tier, 109 min, ONE
  attempt, 0 ERROR / 0 UNPARSED / 0 missing DESC): 9,813 single-part,
  2,719 multi-part, 107 NONE (77 of them single-candidate: 40 Pickup, 36
  binding, 31 putdown — worth a look), + 251 pre-filed NONE (no object
  mask). hand: 10,991 right / 905 left / 743 both (advisory; builder trusts
  WiLoR when they disagree). Luna's 1,574 partial answers kept in
  selections_luna_partial.json (terra won the head-to-head: hand 87% vs 62%
  WiLoR-agreement, safe-door miss fixed, whole-object multi-part answers).
- **SAMPLING CHANGED 2026-09-06 (user): ONE record per window** at the
  window's first frame with the FULL hand trajectory (SF3D semantics — the
  frame shows the object in its start state; v1's stride-2 "remaining
  trajectory" suffix sampling was NOT the SF3D convention and is now only
  behind `--per-frame`). Rebuilt into `/workspace/hoi4d_processed_2d_v2/`
  (09:30 UTC, 14 min) with 12,046 records = windows, then **Pickup/putdown
  REMOVED in place (user: "nothing interesting to train on")** → **FINAL:
  3,075 records** = open 1,181, close 1,039, dump 240, push 178, Press 171,
  pull 165, switch 101; categories TrashCan 770, Safe 581, StorageFurniture
  520, Lamp 378, ToyCar 331, Laptop 232, Bucket 99, Kettle 92, Mug 31,
  Bottle 27, Stapler 12, Scissors 1, Pliers 1 (Bowl/Knife/Chair gone);
  732 object instances, 1,116 seqs; 40/40 randomly sampled records
  reviewed correct (viz/20260906_hoi4d_v2_lmdb_sample). KEEP_VERBS in the
  builder matches. **Compacted copy (446 MB) transferred to the MAIN volume
  at `/workspace/datasets/hoi4d_processed_2d_v2/` (md5-verified); dev-pod
  fast_dev_run with the v2 config PASSED** (3,075 keys, scene split 622
  train / 110 val objects at 0.15, finite losses; batch 64 OOMs on the
  24 GB dev GPU — A100 for real runs). Dev pod RUNNING, mirror live again.
  Still before training: retune schedule for 3k samples (v1 was sized for
  15.6k), recipe/init decision, experiment dir + INDEX. The per-frame build was
  moved to `/workspace/hoi4d_processed_2d_v2_perframe/` (125,166 records
  after pruning: removed cut/paper-cut/binding — 1,873, VLM split tool vs
  material — and WiLoR trajectory outliers — 1,124; start >300 px from mask
  or per-frame jump >300 px, jump p99 = 128 px). Filters live in the
  builder (KEEP_VERBS, MAX_*_PX) so rebuilds reproduce them. 2,942 seqs ok / 29 no-samples / 1
  missing-2dseg / 1 empty-action seq; 1,460 object instances (split key);
  all 16 categories; verbs: putdown 59.3k, Pickup 25.1k, open 15.7k, close
  14.1k, dump 4.2k, push 2.9k, pull 2.8k, Press 691, switch 377 (putdown >>
  Pickup because Pickup windows average 13.9 frames vs 27). traj len
  5/16/108, ~2.4k unique descriptions. Review panels + findings in
  viz/20260906_hoi4d_v2_lmdb_sample/README.md: part-motion and whole-object
  picks all correct in ~44 inspected multi-candidate records; residual
  error rate not measured precisely (95% bound ≈ 7%, likely a few %). Old v1 set (`hoi4d_processed_2d`, 15.6k furniture
  records) untouched. Training NOT commissioned. NOTE the v2 config
  (`config/hoi4d_train_runpod_2d_dct_v2.yaml`) expects
  `/workspace/datasets/hoi4d_processed_2d_v2` on the MAIN volume, while the
  build lives on the HOI4D volume (`/workspace/hoi4d_processed_2d_v2`, 16G)
  — copy across (a pod can mount only one volume; go via scp/rsync between
  pods or a tar over the network) before any training run. Decisions today: verb set =
open/close/Press/push/pull/switch + dump/paper-cut/binding/cut +
Pickup/putdown (carry dropped); windows from collaborator CSV; anchor =
MANO joint 9 (verified visually); hand = VLM HAND field if WiLoR has >=5
dets on that side else most-detected side; motion_info = stub; multi-number
ANSWER unioned. Survey facts: 2Dseg index 2 = hand always, other indices
per-part and STABLE within a video but anonymous across videos; rigid
categories mostly single-class (ToyCar always); Bucket/Chair/Pliers/
Scissors/Laptop/Lamp/Safe/TrashCan/Stapler/Furniture are part-segmented;
second hand / unrelated objects can appear under other indices. All
2,973 seqs extracted (RGB + 2Dseg + action) under /workspace/ext; full
hands package at /workspace/hands2973. Code: tools/hoi4d_vlm_select_all.py
(v2 composite, 3-field prompt, shards), tools/hoi4d_process_2d.py (CSV
windows, knuckle, hand policy, colors union, shards/merge),
tools/codex_client.py (service_tier) — UNCOMMITTED as of this note.
Training NOT commissioned.

## COMPACTION SNAPSHOT 2 (2026-09-03) — VLM mask-selection saga + full-package plan

**IN FLIGHT: nothing running.** The VLM sweep is PAUSED at user request.
One pod up: `segaff-probe16` (id 6s8stprosx1j81, 16 vCPU/32GB cpu3c,
$0.48/hr, HOI4D volume) — **user's codex auth.json IS on it at
/root/.codex (wipe when the sweep story ends)**. Dev pod stopped
(mirror DORMANT — Mac commits do NOT reach the main volume; scp or
start dev pod). Volumes: main bckt1t9uuf + hoi4d f2h0jczstn only.

**The VLM mask-selection story (2026-09-02→03):** v1's masks were ~57%
HAND (motion-energy pick); wrist-exclusion heuristic rejected by user
after panel review; Set-of-Mark VLM selection adopted (gpt-5.6-luna via
codex CLI, effort high; pilot 8/8 parse, user-approved). State: 354-seq
furniture sweep is at **475/1,114 answered** (balanced 155/154/166,
zero errors, selections.json on the hoi4d volume at
/workspace/vlm_select_all/ + composites + jobs.json). Tools:
tools/hoi4d_vlm_select_all.py (prepare-only/consume-jobs split,
resume-safe, raw replies stored), hoi4d_vlm_part_pilot.py,
hoi4d_process_2d.py --selections. **These 475 will be DISCARDED** —
superseded by the planned unified full-package sweep.

**USER-COMMISSIONED NEXT (prep approved, sweep NOT yet — verb survey is
the go/no-go gate):** ONE sweep over the full 2,973-seq collaborator
package (/workspace/hoi4d_all_2973_hands.zip on the hoi4d volume,
verified complete: byte-identical superset of hands354, 16 categories,
+ hoi4d_action_segments.csv with FRAME-CONVERTED windows for 2,972).
Settled decisions: (1) descriptions PIGGYBACKED on the sweep call
(ANSWER:/DESC: two-field format, image-grounded imperative — re-pilot
the parse first); (2) trajectories/point switch wrist→MIDDLE KNUCKLE =
MANO joint index 9 (verify ordering empirically: ~40-80px from joint 0);
(3) windows from the CSV (retires 10s-clock + markResult landmines;
JSON fallback for the 1 missing seq); (4) generalized object-generic
prompt; (5) proposed defaults pending bless: per-window hand = most
detections (was right-only), furniture re-ask included. Prep steps:
unpack full hands; extract RGB+2Dseg for the other 14 categories
(~1-2h, +~60G); verify hand color (0,128,0) on non-furniture; verb
survey → user approves → sweep (~6-9k calls, 6-10h at 14 workers).

**Hard-won ops lessons this arc:** (a) RunPod CPU pods via runpodctl
are ALWAYS 2 vCPU/4GB with a 4GB cgroup cap — the "crashloops" were
container OOM from 12-24 codex app-servers; bigger pods ONLY via REST:
POST rest.runpod.io/v1/pods with vcpuCount (apikey in
~/.runpod/config.toml, single-quoted TOML); cpu3c = ~2GB/vCPU. (b)
codex app-server needs CLI >= ~0.15x — the Mac's mise 0.135 fails every
turn with status='failed'; standalone binaries from GitHub releases
work (musl for pods, aarch64-apple-darwin for Mac). (c) NEVER run many
codex workers on the user's Mac — 8 workers lagged the machine badly
(user displeased); pod-only from now on. (d) pkill/pgrep self-match:
any ssh command CONTAINING the pattern string kills/matches itself —
always bracket-trick every pattern incl. 'codex app-server'. (e) local
scratchpad (venv, binaries, data) is WIPED on session restart; only
volume copies survive. (f) prepare (decode) and consume (VLM) phases
must not share a small pod's CPUs.

## HOI4D 2D training line: FIRST RUN COMPLETE (2026-09-01 overnight)

The g17_2d_dct recipe from scratch on real HOI4D hand video
(20260901_hoi4d_2d_dct): **perception excels, motion geometry doesn't
emerge.** Data pipeline built end-to-end (tools/hoi4d_process_2d.py):
354 furniture seqs → 15,612 samples; format landmines survived
(shift_mask dirs on cams 2–4, markResult action JSONs, 10s-clock action
times ×13 — sibling-found, FFV1 16-bit depth, gather-grid mask coords).
Val: mIoU 0.477 / PDet 53.4 (~2× the SF3D arms), wrist point 0.0084,
traj shape 0.027 — but traj_dir 49% at chance
(jittery wrist tracks, GT roughness 0.72: no net-direction signal at
window scale); p_rev AUC 0.657 weak-but-real (CORRECTED 2026-09-02 —
the earlier chance reading was a float-input viz bug, harness metrics
unaffected); zero-shot to SF3D null (domain-locked). CAVEAT found via
panels: the moving-part mask selection often picked the HAND class (it
wins the motion-energy criterion), so mIoU is partly hand segmentation
— v2 fix: wrist-exclusion before the motion-energy pick, then rebuild. Verdict:
2D geometry emergence is a property of CLEAN track supervision. v2
candidates (parked): smoothed tracks, longer windows, mixed
SF3D+HOI4D, fine-tune from g17_2d_dct. LMDBs:
/workspace/datasets/hoi4d_processed_2d (main volume; backup on the
hoi4d volume — also what the annotator session needs). Panels rendered
on the volume, images pending next dev-pod sync. train_pod.sh now
routes *hoi4d* configs like sf3d.

## HOI4D official drawer package landed (2026-08-30)

`/workspace/datasets/hoi4d_official_drawer_package/` (1.2G unpacked;
zip kept beside it). Collaborator-produced on Euler (array 12077585,
2026-08-28, WiLoR fcb9113, fidelity-audited 0.7–1.0 px vs mirror).
Contents: (a) `data/` — 8 fully-kitted C4 drawer sequences, 4 subjects
× 2: WiLoR hands (MANO rotmats, joints 2D/3D, z-corrected
joints_3d_cam), official METRIC camera-to-world poses (300×4×4, 3Dseg
SLAM), official intrinsics, official action segments
(rest/Reachout/open/Stop… timestamps — labeled contact); (b)
`all_354_furniture_hands/` — the COMPLETE official furniture run: 354
seqs (187 StorageFurniture C4 + 167 Safe C6), 85,945 detections over
85,045 frames (~80% frame coverage), hands + camera per seq. Frame
numbers 1-BASED (max 300); ~99% right hand; priors: wrist 2D 14–21 px,
depth bias +4–16 cm/subject. This supersedes the 10-seq hands package
for furniture — full-category hand supervision is now in hand; RGB/
depth/2Dseg still come from the raw archives on segaffordance-hoi4d.

## HOI4D raw download COMPLETE (2026-08-28)

- New volume `segaffordance-hoi4d` (f2h0jczstn, 500GB EU-RO-1, $35/mo)
  + CPU pod `segaff-hoi4d-dl` (qo9j1a31r6cu49, $0.06/hr-class)
  downloading the OFFICIAL HOI4D release (~174GB: RGB
  HOI4D_release.zip 23G, depth tar.gz0-6 127G, annotations 22G, CAD
  1.5G, camera params + hand pose 0.5G) from the project's OneDrive
  shares via the anonymous badger-token API (no per-sequence access
  exists — monolithic archives). Script /workspace/hoi4d_download.py
  on that volume; log /workspace/hoi4d_download.log; per-file resume +
  size verification; monitor armed. DELETE THE POD when done; the
  volume holds the raw data for the 10-sequence hand-supervision
  prototype (hoi4d_hands_package on the main volume) and future
  furniture-category extraction. DONE: all 12 archives, 163G, every
  file size-verified against the share metadata; download pod DELETED.
  Next step when picked up: unpack selectively for the 10 prototype
  sequences (cameras ZY20210800001/2) — RGB+depth decode via
  HOI4D-Instructions utils/decode.py, plus 2Dseg/objpose/CAD.

## fdiff-family ablation transfer COMPLETE (2026-08-28)

Both arms done, wrapped, pods deleted. The user's two questions:
1. **Joint > either alone on fdiff? YES, on both sides.** Joint
   (fdnolpp) vs art-only (B): MA +7.1, matched −8.2°. Joint vs
   traj-only (C_f): traj_dir +1.4/+0.019 — small but real, where DCT
   measured ~nil. The coupling asymmetry survives but fdiff's weak
   direction is nonzero.
2. **The mechanism verdict is FAMILY-ROBUST and stronger on fdiff:**
   decode+fdiff (zero params) recovers ~89% of the B→joint gap (MA
   26.7 of 20.4→27.5; DCT: 75%) and BEATS the joint arm on flips,
   origin, and masks. fdiff geometry composes with the decode
   parameter-free (matched −1.1°, rot flips −2.4 vs plain decode).
Non-transfer: the DCT mask ordering (C>B>D) inverts on fdiff
(B>C_f>joint) — "trajectory supervision is trunk-friendly" was
DCT-specific (the smooth basis sends gentle gradients; plain+fdiff is
harsher). Full tables: 20260827_sf3d_{andec_fdiff,supabl3_traj_fdiff}
notes.md.

## Earlier (2026-08-27): fdiff-family ablation transfer

- User-commissioned: does the DCT-family ablation transfer to fdiff?
  Two new arms (spec 2026-08-27-fdiff-family-ablation-design.md; smokes
  passed; pollers hunting PRO 6000 stock): C_f
  `20260827_sf3d_supabl3_traj_fdiff` (trajectory-only + fdiff, plain
  head) and `20260827_sf3d_andec_fdiff` (analytic decode + fdiff ON THE
  DECODE — new trainer block, commit 13ffad8). Reused corners: arm B
  (art-only, shared by construction — fdiff dies with the head) and
  fdnolpp (joint). Readouts: joint-vs-either on fdiff; mechanism split
  (andec_fdiff−B)/(fdnolpp−B); mask ordering transfer.
- Dev pod RECREATED 2026-08-27 (host lost its GPUs again while stopped);
  mutagen session recreated cleanly after a git-archive sync — mirror
  healthy.

## Overnight program COMPLETE (2026-08-25) — mechanism study + fdiff grid

All four runs done, wrapped, pods deleted. The synthesis:

1. **WHY trajectory supervision helps articulation (user's question):
   ~75% is LOSS GEOMETRY.** The analytic screw decode (writer-mirror
   from predicted articulation params, ZERO new parameters) recovers MA
   26.5 of the arm-B→D 20.4→28.2 gap and most of the flip-rate gain,
   with NO shared-feature routing needed. The head's own contribution is
   the MASK gains (the decode arm's masks fall below arm B). Same
   information, better-conditioned parameterization = different
   optimization problem. (20260825_sf3d_analytic_decode/notes.md.)
   Toy probes CLOSED after three null regimes (easy, shared-trunk,
   underfitting): the transfer does not reduce to a generic
   low-dimensional mechanism — at-scale attribution + small-scale
   irreducibility is the final answer (viz/20260825_toy_traj_mechanism).
2. **The dir term verdict, revised:** g21's failure was substantially
   the dir×DCT INTERACTION. On the plain head (fddir) the term achieves
   its design goal: rot flips 15.2→**11.24 (record)**, matched axis
   **14.62° (record)**, traj_dir only −0.7. Detach-trajectory variant
   now doubly attractive (may keep flips and recover the −2.1 MA).
3. **The L_pp trade is FAMILY-DEPENDENT:** off = MA +2.2 on DCT
   (supabl2 D) but MA −2.4 on fdiff (fdnolpp) — while fdnolpp still
   takes precision columns (matched 15.09°, flips 10.4/13.5, radius
   0.113). No universal "drop L_pp".
4. Threshold-MA vs precision is the recurring trade: g19_fdiff keeps
   the MA crown (29.9); every intervention that sharpens precision
   (drop L_pp, add dir) pays ~2 MA at the pass threshold.
- **Gen-22 candidate (updated):** trajectory head + analytic decode +
  fdiff + dir‑term(plain head or detached); L_pp ±0.1 to be MEASURED.
  Not commissioned.
- Ops footnotes: all four ran on power-capped EU-RO-1 hosts (600W cap,
  0.6–1.2 it/s — verify clocks at launch); a Mac network outage + a
  Claude session restart cost three background watchers (re-armed) and
  delayed one pod delete + push (both recovered); mutagen mirror was
  stuck "connecting to beta" at last check — scp via dev pod works.

## Earlier in flight (2026-08-24)

- **Label-efficiency v2 DONE 2026-08-24** (all four arms wrapped, notes
  + INDEX in, MY pods deleted). Headline (all g21 recipe): B' ≫ A'
  (MA 11.4 vs 4.9, mIoU 0.188 vs 0.029), B' < C' (MA 26.6, mIoU 0.271);
  vs v1 the ARTICULATION transfer improved a lot (matched axis within
  2.2° of the 100% baseline) while mask transfer dipped. C' set mIoU
  0.2712 / PDet 23.21 RECORDS. **Dir-term verdict: FAILED as
  implemented at 0.1** — rot flips 13.3→15.4 and traj_dir 94.5→88.8 on
  3D (g21 vs g19_dct), traj_dir 84→64 on the 2D pretrain: the two-way
  gradient lets wrong axes drag trajectories. Twice-motivated fix,
  PARKED: detach the trajectory inside the term (axis-only gradients).
  Cross-read pending: the other session's supabl2 arms (their arm D =
  L_pp fully off on this recipe). B'2 was quota-cut at ep29 (val at
  plateau — negligible loss); its ep30 truncated ckpt was removed.
- **Supervision ablation v2 DONE 2026-08-24** (spec + full results table:
  docs/superpowers/specs/2026-08-24-supervision-ablation-v2-design.md;
  arms `20260824_sf3d_supabl2_{art,traj,nolpp}`, notes + INDEX rows in;
  ALL ITS PODS DELETED). Re-ran the 2026-08-15 joint-vs-either ablation on
  the g21 stack and added the deconfounding arm that spec deferred. Arm A
  was the REUSED g21_dct_dir run; A₀ = g19_dct (L_pp on, dir off) turned
  out to be the cleaner joint partner. Three answers:
  1. **Trajectory → articulation: YES, bigger than v1** (arm B vs D, no
     consistency coupling on either side): MA +7.8 (20.4→28.2), matched
     axis −6.2°. v1's effect was on TYPE; on the split-head stack type is
     flat and it all lands on the AXIS.
  2. **Articulation → trajectory: NO — v1's biggest effect does NOT
     replicate.** traj_dir 94.36 (C) vs 94.26 (D), flat, where v1 measured
     −5.5. g16's normalized traj loss + g19's DCT head now supply what
     articulation used to. **The coupling is ASYMMETRIC now.**
  3. **The v1 win was co-training, not L_pp.** A₀ vs D: turning L_pp OFF
     *improves* MA +2.2, matched axis −1.1°, radius −2.1cm; it buys type
     (+1.5) and rot sign stability (13.3 vs 14.8 flips) instead.
  Also: **consistency never emerges for free** — arm D's passive
  L_geo_pred_pred_art is FLAT 0.382→0.370 over 30 epochs vs A₀'s trained
  0.234→0.124 (only arm D could show this). And removing the trajectory
  nearly DOUBLES the rot flip rate (14.8→21.8) — the trajectory's time
  ordering is the only sign-aware signal in the loss set, so
  **trajectory-side supervision looks a better lever on the sign problem
  than another L_pp term** (relevant to the parked dir-term fix).
  Masks: fewer heads = better masks, C>B>D, replicating gen-9 exactly.
  CAVEAT: all three arms' weights were POD-LOCAL and died with their pods
  (quota freeze) — metrics in notes/INDEX/logs are the durable artifact;
  re-running an arm is ~4h/$8. No viz batch for the same reason.
- Dev pod RECREATED 2026-08-24 (`lltgv0y73agseu`, RTX PRO 4000,
  $0.57/hr, RUNNING) and the mutagen mirror was sync-reset to it —
  normal edit-locally/run-on-pod workflow restored; the mirror is
  reconciling the Mac tree (HEAD) onto the volume.

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
- ~~Scratch volume deletion~~ RESOLVED 2026-08-28: `s3qha8tz50` is
  already gone (verified: delete returns nonexistent, list shows only
  the main volume) — deleted during/around the 2026-08-24 quota
  resolution. Raw SceneFun3D is re-downloadable from public sources if
  ever needed; all training data lives on the main volume.
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
