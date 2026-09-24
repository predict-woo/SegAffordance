# Spec: external A/B of the closed-form screw loss on published articulation models

Status: DRAFT (no implementation yet). Written 2026-08-29.
Owner: Andy. Related: `knowledge/articulation-loss-survey.md` (workspace
root), per-paper notes `~/dev/knowledge/summary_*.md`, STATE.md
"COMPACTION SNAPSHOT (2026-08-28)".

## 1. Goal

Show that the closed-form screw loss
(`model/losses/geometric.py::closed_form_screw_loss`) improves axis/origin
accuracy when dropped into *other people's* supervised articulation models,
replacing their decoupled axis-direction + origin terms. Each experiment is a
matched A/B: same released checkpoint as init, same data, steps, seed, LR;
the only difference is the articulation loss. Deliverable per method: their
metric(s) + our signed metrics, for both arms, plus the exact patch applied.

This is a *fine-tune* A/B (cheap, ~$100–150 total on RunPod H100), not a
full retrain. If a method shows a clear win, a full retrain of that one
method is a separate follow-up.

## 2. What the loss is (the piece that crosses over)

For a revolute joint with predicted (axis dir n, origin q, a point p on the
moving part) and GT (n*, q*, p*): r = (p−q)⊥n, t = n×r, dr = r−r*,
dt = t−t*; position term (3π/4−2)|dr|² + (π/4)|dt|² − dr·dt over
(π−2)|r*|², derivative (H1) term (π/4)(|dr|²+|dt|²) − dr·dt over
(π/2)|r*|². Prismatic rows collapse to |n̂−n̂*|² = 2(1−cos). Rows routed by
GT type. It couples direction, origin and **sign** in one term and is
invariant to sliding q along n*. On SF3D the H1-only variant + a 1−cos axis
anchor set the MA record (30.64) and the closed-form pair set the origin
record (0.250) with no trajectory head — that recipe is what we transplant.

Port as a single dependency-free file `screw_loss.py` (function + the
sign-aware angle/origin metric) so all four experiments use bit-identical
loss code. `tests/test_closed_form_loss.py` (18 tests: gauge invariance,
exact flip values rot pos = π/(π−2), der = 2.0) is the executable spec —
the port must pass a copy of it. Signature:
`closed_form_screw_loss(motion_type_gt, axis_trans, axis_rot, origin_pred,
point_3d_pred, axis_gt, origin_gt, traj_start_gt) -> (pos, der)` per row,
GT-type routed (>0.5 = revolute). Needs a GT origin AND a GT point p on
the moving part. No other SegAffordance code is reused (`runpod/ensure_env.sh`
is OUR env — never reuse it for external repos).

## 3. Targets (from the survey; verdict High/Medium + code available)

| # | Method | Their arm (loss to replace) | Our arm (slot-in) | p (moving-part point) | Their metric | Init ckpt | Data |
|---|---|---|---|---|---|---|---|
| 1 | **SINGAPO** (ICLR'25) | ε-MSE on 30-D node tensor; axis = ch 12:15, origin = 15:18 | reconstruct x̂₀ from ε̂; add λ·screw(x̂₀[12:15], x̂₀[15:18], aabb-centre) masked by valid non-fixed nodes, SNR-weighted | AABB centre (predicted & GT) | RS/AS-dgIoU, dcDist, dCD, AOR, Acc — **no axis metric**, add ours | released ckpt | PartNet-Mobility (SINGAPO split, CAGE format) |
| 2 | ~~DIPO~~ (NeurIPS'25) — DROPPED 2026-08-29: shipped data has no features, debug training code | same ε-MSE (`systems/system_origin.py::compute_loss`) | same as SINGAPO | AABB centre | same family — add ours | released ckpt | PartNet-Mobility + PM-X |
| 3 | **Particulate-B** (CVPR'26) | `L_dra` = L1(dir) + `L_xra` = L1(per-point foot point) (+ `L_pd` L1 prismatic dir) | per-point screw: q = predicted foot point x̃_j, p = p_j, n = d̃_ra; r*_j = p_j − x_j from their dataloader; `L_pd` → 2(1−cos) | every part point (native) | articulated gIoU/PC/OC — **no axis metric**, add ours from `revolute_plucker` | HF ckpt | PartNet-Mobility cached `.npz` (skip GRScenes for the fine-tune) |
| 4 | **USDNet** (ICCV'25, Articulate3D) | `(1−cos) + ‖(o−o*)×a*‖/‖a*‖ + ‖(o*−o)×a‖/‖a‖` for rotation queries (`criterion.py::loss_articulations`, `regular_arti_loss=false` branch) | screw loss with p = GT-mask centroid for rotation; keep 1−cos for translation; keep seg/cls/aux terms | mask centroid | AP₅₀ +Origin/+Axis/+both (15°, 0.25 m gates) — coarse; also log raw mean errors | released final ckpt | ScanNet++ (license) + Articulate3D preprocessing |

Order: 1 → 2 → 3 → 4 (cheapest and least data-blocked first). USDNet is the
best scientific match (scene-level, closest to our setting, 69% of instances
fail their gate = large headroom) but is gated on ScanNet++ access.

Headroom check (why it's worth doing): these regressors sit at 8–24° axis /
0.06–0.17 origin on PartNet-Mobility, USDNet at 31 AP₅₀ on real scenes.
Saturated methods (ArtGS 0.02°, REArtGS++) are the self-supervised ones we
excluded anyway.

## 4. Infrastructure

Decisions already made with the user:
- **Isolation**: nothing touches the main volume `bckt1t9uuf` or the mutagen
  mirror. One RunPod **network volume per experiment**, each in **EU-FR-1**
  (H100 SXM provisioned first try there on 2026-08-29 at $3.29/hr on-demand;
  EU-RO-1 has no H100). Volumes are **kept** after the run; the user deletes
  them manually. Only small results come back.
- **Compute**: RunPod on-demand H100 SXM (non-preemptible → no
  supervisor/resume plumbing). Nebius was evaluated and dropped (billing
  setup friction). USDNet is sparse-conv/batch-1 bound and would be fine on
  a cheaper card if H100 stock is gone.
- **Nothing shared between experiments** except `screw_loss.py`; each uses
  the upstream repo at a pinned commit plus a minimal patch.

Layout in this repo (small, text, git-tracked):

```
SegAffordance/
  runpod/external/
    common/screw_loss.py          # the ported loss + signed metrics (+ unit test vs geometric.py)
    common/volume.sh              # create volume in EU-FR-1, create/delete pod, ssh helpers
    <method>/                     # singapo | dipo | particulate | usdnet
      setup.sh                    # on-pod: clone @ pinned commit, env, data download+preprocess, apply patch, 50-step smoke of both arms
      patch.diff                  # the loss swap + metric hook + checkpoint-interval change
      run.sh                      # arm A (theirs) and arm B (ours), identical steps/seed/init; then eval
      collect.sh                  # scp results bundle back, delete pod, `runpodctl pod list` must be empty of ours
  experiments/external/<method>/  # results bundle: metrics.json (both arms), curves csv, eval logs, patch.diff, config, notes.md
```

Per-experiment volume layout (`/workspace` on the pod): `repo/`, `data/`,
`ckpt_init/`, `runs/{theirs,ours}/`, `results/`, `DONE` markers so every
step is idempotent/re-runnable.

Data acquisition (verified 2026-08-29, all pod-side, nothing on the Mac):
- SINGAPO: `pm.zip` 13 GB + ckpts, open HTTP from aspis.cmpt.sfu.ca (200, no auth).
- DIPO: `wuruiqi0722/DIPO_data` + `HorizonRobotics/DIPO-Dataset` on HF, ungated.
- USDNet: preprocessed `processed.zip` **5.8 GB** on Google Drive
  (id `1QS_D5CBoF5AssleA3kdMMirwFEWPMaMW`), `gdown` resolves it and the
  server supports ranges (`gdown --continue`); Mask3D ScanNet200 init ckpt
  is an open URL. No ScanNet++ license needed for the A/B (form optional).
- Particulate: raw PartNet-Mobility from HF `sapien-sim/PartNetMobility`
  (SAPIEN team's own mirror, 2,350 per-object zips ≈ 3.3 GB, manual gate —
  access requested with ETH email); pulled with `HF_TOKEN` +
  `huggingface-cli download`. PartField ckpt is public.

Pod phasing: SINGAPO/DIPO/USDNet run on a single GPU pod (their data is
already preprocessed; the only CPU-heavy step, USDNet's MinkowskiEngine
build, must compile against the GPU image anyway). **Particulate is
two-phase**: a cheap CPU pod in EU-FR-1 on the same volume runs download →
`process_urdf.py` → `cache_points.py` (multi-core, ~1–2 h, CPU-only deps),
is deleted, then the GPU pod does env + PartField feature caching + smoke +
both arms. Saves H100 idle time and keeps the GPU session short.

## 5. Experimental protocol (identical across methods)

- Init: released checkpoint. Two arms only: **theirs** (unchanged loss,
  continued fine-tune) and **ours** (loss swapped). Continuing to fine-tune
  the baseline with its own loss is the control — it separates "more
  training" from "different loss".
- Matched: steps, batch, seed, LR (a low constant LR ~0.1–0.2× their
  final LR, no warm restart of schedule), data order. Checkpoint every
  ~15–30 min of wall-clock; keep last + best.
- Budget per arm: SINGAPO/DIPO ~20–30 epochs; Particulate ~3–5K steps at
  batch 16; USDNet ~50–100 epochs of the articulation stage at batch 1.
- Loss weight λ for our term: match the *magnitude* of the term it replaces
  at step 0 (measured in the smoke run), plus one ×0.5 / ×2 sensitivity
  arm only if the first result is ambiguous. Headline "ours" config = **H1
  derivative quadratic only (weight 1.0) + sign-sensitive 1−cos axis
  anchor** — the current all-time MA record (30.64, cf_h1only); pos+der
  0.5/0.5 is the conservative fallback (origin record 0.250). Triangulation
  from SF3D: H1 = sharpness, 1−cos = revolute sign (dropping it costs +2.2
  flips), position quadratic ≈ 4 mm of origin and deadweight for MA.
- **Keep their native origin/point supervision alongside ours.** The screw
  loss is gauge-invariant (origin sliding along the axis) and blind to
  absolute p/q placement — it reshapes, it does not anchor. "Replace" in
  §3 means: their direction + origin *coupling* terms are swapped for ours,
  but any absolute origin/point regression they have stays on.
- **Sign convention is load-bearing.** Our GT axes are canonical right-hand
  (t = n̂×r carries the sign). SINGAPO/DIPO canonicalize GT direction to
  majority-positive components (a data convention, not physics) — the
  signed terms are consistent with that only if GT and ranges use the same
  convention. Particulate/USDNet: verify per dataset whether axis sign is
  canonical or arbitrary; if arbitrary, either canonicalize their GT or
  expect 2(1−cos) to punish convention mismatch rather than error.
- **On-axis rows are noise** (r*→0): SF3D filters `min_revolute_radius`
  0.10 m. Apply the equivalent per method (in normalized PartNet frames,
  scale accordingly) and report how many rows are dropped.
- Eval: their official eval script unchanged (so numbers are comparable to
  the paper), **plus** our metrics on the same predictions: unsigned axis
  angle, signed axis angle (flip rate), origin point-to-GT-line distance,
  origin lever-arm error. Both at init and at end for both arms, so the
  delta-vs-init is visible even where their metric is indirect
  (SINGAPO/DIPO/Particulate).
- Pass criterion: ours beats theirs on axis angle *and* origin distance
  with the paper metric not regressing; report seeds=1 honestly as such.

## 6. Cost and time estimate (H100 SXM, $3.29/hr)

| Method | GPU-h (2 arms + eval + setup) | $ |
|---|---|---|
| SINGAPO | 5–8 | 15–25 |
| DIPO | 5–8 | 15–25 |
| Particulate | 4–8 | 15–25 |
| USDNet | 8–14 | 25–45 |
| overhead (env builds, debugging) | 8–12 | 25–40 |
| **total** | **30–50** | **≈ $100–160** |

Volumes: ~$0.07/GB/mo; 4 × 50–100 GB ≈ $15–25/mo while kept. Account
balance is ~$50 → needs a top-up before running more than one experiment.

## 7. Risks / open questions

- **Data access**: PartNet-Mobility requires a SAPIEN account/token (three
  experiments re-download it — accepted cost of isolation); ScanNet++ needs
  a signed license → USDNet is blocked on a human step. Particulate also
  needs PartField weights (public).
- **Environments**: USDNet = Mask3D + MinkowskiEngine (old CUDA/torch; may
  need a cu118 image, not the cu128 default); Particulate = PartField on
  the fly. Budget the overhead line for this.
- **Diffusion models (SINGAPO/DIPO)**: x̂₀-space auxiliary losses at high
  noise are ill-conditioned; need SNR weighting and per-node masking. Their
  sign canonicalization (majority-positive dir) must be respected in GT.
- **Metric mismatch**: none of the four reports a signed axis metric; the
  sign benefit only shows in our added metric. Say so in the writeup.
- **Fine-tune ≠ retrain**: a null result on a short fine-tune from a
  converged checkpoint is weak evidence; a positive result is decent
  evidence. Pre-register the step budget; don't extend it post hoc for one
  arm only.
- **Orphan pods** bill silently: every `collect.sh` ends with a reconciled
  pod list. Stopped pods do not appear in `runpodctl pod list` — check by
  id. Network-volume creates that return 500 CAN still create the volume —
  list before retrying. Volumes: DC-locked forever, resize up only, one
  volume per pod.
- **Lemon hosts**: on every fresh pod verify GPU name AND post-warmup
  `clocks.sm` under load (power-capped signature 670–1400 MHz = 2–3× slow;
  healthy ≈ max clocks at ~full power). Bake into `setup.sh`.
- **Pod defaults**: open-files ulimit 1024 (raise before many dataloader
  workers); check `/dev/shm` size (undersized → SIGBUS mid-epoch); volume
  storage is MooseFS and TRUNCATES writes silently at quota — size-check
  checkpoints against a known-good byte count.
- **Old-CUDA builds** (USDNet/MinkowskiEngine): no tested recipe in this
  project; pick a torch image matching their pinned CUDA and budget
  compile time.
- **Dev pod is STOPPED by user request — do not start it.** While it is
  stopped the mutagen mirror is down, so local edits do not reach the main
  volume; irrelevant to the isolated volumes, but nothing here should
  assume otherwise.
- **PartNet-Mobility**: no token on the volume; needs the user's own
  sapien.ucsd.edu account/download.

## 8. Coordination with the sibling session (ethz-workspace-65)

Consulted 2026-08-29: no collisions; it owns `STATE.md` and
`experiments/INDEX.md` (it already added a "PLANNED" pointer to this spec,
commit cbd1689) — send it result summaries to fold in rather than editing
those files here. Nothing of theirs is in flight; pod list empty of theirs.
Repo convention for specs is `docs/superpowers/specs/`; this one lives at
`docs/specs/` (noted, left as is).

Read before implementing: `knowledge/infra-lessons.md` (workspace root),
`docs/RUNNING_EXPERIMENTS.md`, `docs/slides/2026-08-28_continuous_trajectory_loss.html`
(derivation), `SegAffordance/knowledge/twist-screw-theory.md`,
`experiments/20260828_sf3d_cf_h1only/notes.md` (the H1-only result).

## 9. Not in scope

Full retrains; methods without code (GEOPARD, MonoArt, Artic-O, sim2art,
ArtiLatent, DICArt, DailyArt) — revisit when repos land; changing anything
in the SegAffordance training code itself.
