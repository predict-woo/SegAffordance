# SegAffordance — living project state

**The single source of truth for "where is this project right now".**
Update this document at every experiment wrap, decision, or infra change —
it is the first thing a fresh/compacted session should read. Keep entries
terse; details live in the linked specs/notes. Last update: 2026-08-28.

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
| **best articulation (3D)** | 20260828_sf3d_cf_h1only | best-epoch29-valloss1.1303 — MA 30.64/signed 30.11 + all-axis 24.5° + flips-all 9.8 (ALL RECORDS), NO trajectory head (H1 quadratic + axis anchor only) |
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

**A/B program status: replace-mode fine-tunes NULL; additive arms split
by host structure** (USDNet L2-only positive, SINGAPO negative);
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
