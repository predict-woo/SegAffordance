# Supervision ablation v2: joint vs trajectory-only vs articulation-only, deconfounded

**Date:** 2026-08-24
**Status:** commissioned by the user ("run this experiment on the newest
stack? I want to see if the results match").
**Question:** does jointly training on trajectory and articulation
supervision still beat either alone on the CURRENT recipe — and is the win
co-training, or is it the L_pp consistency loss?

## Why re-run

The original ablation (`2026-08-15-supervision-ablation-design.md`,
arms `20260815_sf3d_g9{_closeup010,abl_artonly,abl_trajonly}`) answered
"yes, both directions" on the **gen-9 stack**: frozen CLIP RN50, 256²
input, v2 data, no split axis heads, no normalized trajectory loss, no
DCT head. Since then mIoU went 0.146 → 0.265 and MA 21.9 → 29.9. Its two
headline effects were:

- removing articulation costs the trajectory −5.5 pts traj_dir (93.1 → 87.6);
- removing the trajectory costs articulation −3.0 pts type and +4.6°
  matched axis.

It also carried an **accepted, unresolved confound**, stated in its own
spec: arm A differed from B and C by *two* mechanisms — the extra
supervision AND the L_pp consistency coupling. The deconfounding arm was
named as "a possible follow-up, not part of this experiment" and never
ran. v2 runs it.

## The recipe: "g21", inherited from label-efficiency v2

All arms derive from `config/sf3d_train_runpod_g21_dct_dir.yaml` —
g17 split-axis + DINOv3 512 + normalized trajectory loss +
`trajectory_dct_coeffs: 6` + `pred_pred_art_dir_weight: 0.1`. Not the
g19-dct or g19-fdiff recipe: the joint arm is **reused, not retrained**
(see below), so the ablation is recipe-matched to a run that already
exists.

## Arms

Arm A is `20260823_sf3d_g21_dct_dir` — the label-efficiency v2 C' run
owned by the concurrent session (coordinated 2026-08-24). It is exactly
the joint arm of this ablation: 100% 3D supervision, g21 recipe, 30
epochs, standard test pass. **It is reused, not re-run.** Only the three
ablated arms are trained here.

| | A: joint (reused) | B: articulation-only | C: trajectory-only | D: joint − L_pp |
|---|---|---|---|---|
| Mask (DiceBCE) | ✓ | ✓ | ✓ | ✓ |
| Interaction point: heatmap + coord + z_p lift + 3D loss | ✓ | ✓ | ✓ | ✓ |
| Origin: heatmap + z_q lift + canonical-q* + map BCE | ✓ | ✓ | — | ✓ |
| Type CE | ✓ | ✓ | — | ✓ |
| Axis direction (split rot/trans heads, 1−cos) | ✓ | ✓ | — | ✓ |
| Trajectory head (DCT) + normalized per-point loss | ✓ | — | ✓ | ✓ |
| L_pp consistency + midpoint dir term | ✓ | — | — | **0 weight, still logged** |

Same decisions as v1, kept deliberately so the two generations compare:

1. **The interaction point is static grounding, not "trajectory data".**
   Every arm keeps the full point pipeline (heatmap, coord, z_p, 3D loss
   vs GT `trajectory[:, 0]`).
2. **Arm C removes articulation from the architecture**, not just its
   losses: no type head, no axis heads, no origin heatmap channel, no z_q
   head, `origin_uv` out of the condition vector.
3. **One standard test pass per arm** on the identical 5,088-sample test
   split; absent-head metrics stay absent (never a 0.0 placeholder).

### What each comparison answers

- **A vs B** — does trajectory supervision improve articulation? (v1's
  type/matched-axis effect: does it reproduce?)
- **A vs C** — does articulation supervision improve the trajectory?
  (v1's traj_dir effect.)
- **A vs D** — **new.** Isolates the L_pp consistency loss + dir term:
  identical supervision, coupling on vs off.
- **D vs B**, **D vs C** — **new.** The v1 comparisons with the
  consistency confound removed. If D still beats B and C, co-training
  itself is the mechanism; if D collapses to B/C, the v1 win was L_pp.
- **A/B/C/D on mIoU, 2D/3D point** — task competition on the shared
  trunk. v1 found the trajectory-only arm had the BEST masks; watch
  whether that survives at 512 with DINOv3.

## Config deltas (all vs `sf3d_train_runpod_g21_dct_dir.yaml`)

Config-only — every gate already exists in the model, and
`tests/test_supervision_ablation.py` already pins the trainer-side
skipping behaviour. No code changes, therefore nothing extra to sync to
the pods beyond the YAMLs.

**Arm B — `sf3d_train_runpod_supabl2_art.yaml`,
`experiments/20260824_sf3d_supabl2_art`:**

```yaml
model_params:
  use_trajectory_head: false        # TrajectoryMLP not constructed
  trajectory_dct_coeffs: 0          # inert with the head gone; explicit
loss_params:
  trajectory_weight: 0.0
  geometric_loss: "none"            # L_pp needs the predicted trajectory
  pred_pred_art_weight: 0.0
  pred_pred_art_dir_weight: 0.0
```

Note this is *forced*, not chosen: `PredPredArticulationLoss.forward`
requires `outputs.trajectory_pred`, so removing the head kills L_pp and
the dir term automatically (`model/losses/geometric.py:189`). Arm B
therefore cannot isolate co-training on its own — that is what D is for.

**Arm C — `sf3d_train_runpod_supabl2_traj.yaml`,
`experiments/20260824_sf3d_supabl2_traj`:**

```yaml
model_params:
  use_motion_head: false
  use_motion_type_head: false
  split_axis_heads: false           # REQUIRED: raises if the heads are off
  use_origin_heatmap: false         # projector back to 2 channels
  use_origin_local_feature: false   # REQUIRED: raises without the heatmap
loss_params:
  vae_weight: 0.0                   # the 1-cos axis loss
  motion_type_weight: 0.0
  origin_weight: 0.0
  origin_map_weight: 0.0
  geometric_loss: "none"
  pred_pred_art_weight: 0.0
  pred_pred_art_dir_weight: 0.0
```

The two `REQUIRED` lines are new vs v1's arm C: `split_axis_heads` and
`use_origin_local_feature` did not exist in gen-9 and both raise
`ValueError` in `CRIS.__init__` if left on without their dependencies.

**Arm D — `sf3d_train_runpod_supabl2_nolpp.yaml`,
`experiments/20260824_sf3d_supabl2_nolpp`:**

```yaml
loss_params:
  pred_pred_art_weight: 0.0
  pred_pred_art_dir_weight: 0.0
  # geometric_loss STAYS "pred_pred_art"
```

Deliberately keeping the module built at zero weight rather than
`geometric_loss: "none"`: `forward` always emits
`L_geo_pred_pred_art` into its log dict, so arm D still **measures**
head consistency every step while training on none of it — the
diagnostic that makes A vs D readable. Cost is one extra residual
computation per step. Known gap: the dir term's log is guarded by
`if self.dir_weight != 0.0`, so `L_geo_pp_dir` is absent in D (accepted;
fixing it would mean editing shared loss code the concurrent session is
running).

## Constants across arms

Everything else in `sf3d_train_runpod_g21_dct_dir.yaml` verbatim: DINOv3
ViT-L/16 frozen + dino.txt, input 512, sf3d_processed_v3, key cache
`sf3d_v2_keys_cutoff05_minrad010_maskfrac0010_edge05.pkl` (59,174
records), `min_revolute_radius 0.10`, `min_mask_area_frac 0.001`,
`edge_margin_frac 0.05`, `point_source "element"`, val_split_ratio 0.1,
batch 64, 30 epochs, lr 1e-5, milestones [24, 28], seed 42, precision 16,
channels_last + compile.

## Execution

- Own PRO 6000 pods, distinct names (`segaffordance-abl-*`), created in
  parallel with the concurrent session's two running pods. Verify
  `nvidia-smi` on every landed pod before launching (a "successful"
  create can land the wrong SKU) and reconcile `pod list` after each
  create — orphans bill silently.
- The mutagen mirror is DOWN (dev pod host lost). The volume's code tree
  is at commit `a991ae9`; these configs reach the pod by
  `git archive HEAD | ssh segaff-<pod> "tar -x -C /workspace/SegAffordance"`.
- No dev-pod smoke: no model code changed. Local validation instead —
  build `CRIS` under each arm's flags on a stub backbone and run
  forward + backward (the gate combinations `split_axis_heads: false` +
  DCT head + `use_origin_heatmap: false` have never been exercised
  together).
- ~4 h/arm (7:52/epoch × 30, measured on g19-dct), 3 arms ≈ $23–25.
- Delete every pod right after its test pass.

## Out of scope

- Anything owned by the concurrent session:
  `config/sf3d_train_runpod_{g21_dct_dir,p90_2d_dir,s10_3d_dir}.yaml`,
  `experiments/20260823_sf3d_*`, and its label-efficiency v2 wrap.
  `INDEX.md` / `STATE.md` edits are serialized with it: pull immediately
  before, commit promptly after, append-only.
- An "articulation + unsupervised trajectory head with L_pp on" arm
  (trajectory head present, `trajectory_weight: 0`, L_pp live). Offered
  and declined for this round; it would separate "GT trajectory labels
  matter" from "the consistency structure over a predicted curve
  matters".
- Cross-generation comparison to the gen-9 numbers on absolute metric
  values. Only the DIRECTION and rough SIZE of each effect transfer —
  different backbone, resolution, data version, and loss set.

## AMENDMENT 2026-08-24: a fifth arm exists for free, and it is the better partner

The concurrent session's label-efficiency v2 wrap landed while these arms
were training, and it changes which comparisons should be headlined.

**Finding that forces this:** the `pred_pred_art_dir_weight: 0.1` term in
the g21 recipe FAILED — vs `20260821_sf3d_g19_dct` (identical but for the
term) it made rot flips WORSE (13.3 → 15.4) and traj_dir worse (94.5 →
88.8). Arm A (= `20260823_sf3d_g21_dct_dir`) therefore carries a known
net-negative term, which would understate the joint arm in every A-vs-B
and A-vs-C comparison.

**The fix costs nothing:** `20260821_sf3d_g19_dct` IS the g21 recipe minus
the dir term — same data, same 59,174-key split, same 30 epochs, same
DCT head — and it is already run, tested, and in INDEX. Adopt it as a
fifth arm:

| arm | trajectory GT | articulation GT | L_pp | dir term |
|---|---|---|---|---|
| **A₀** `20260821_sf3d_g19_dct` | ✓ | ✓ | 0.1 | — |
| **A** `20260823_sf3d_g21_dct_dir` | ✓ | ✓ | 0.1 | 0.1 |
| **D** `20260824_sf3d_supabl2_nolpp` | ✓ | ✓ | — | — |
| **B** `20260824_sf3d_supabl2_art` | — | ✓ | — | — |
| **C** `20260824_sf3d_supabl2_traj` | ✓ | — | — | — |

Revised headline comparisons:

- **A₀ vs D is now the clean L_pp isolation** — identical recipe, identical
  supervision, consistency weight 0.1 vs 0, dir term absent from BOTH.
  This is strictly better than the A vs D comparison the original spec
  planned, which confounded L_pp with the failed dir term.
- **A₀ vs B and A₀ vs C** replace A vs B / A vs C as the restatement of
  the 2026-08-15 question ("does joint beat either alone?"), using the
  joint arm that does NOT carry the bad term.
- **D vs B and D vs C** answer the purer question the original ablation
  could not: does co-training alone help, with no consistency coupling
  anywhere in the comparison?
- A vs A₀ (the dir term) is the concurrent session's experiment, not
  this one's; cite it, don't re-derive it.

Nothing about the running arms changes — B, C and D are unaffected. This
is purely a decision about which columns the wrap table headlines.
