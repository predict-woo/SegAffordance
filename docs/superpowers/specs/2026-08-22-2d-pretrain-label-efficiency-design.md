# 2D-pretrain label efficiency (90/10) — design

**Date:** 2026-08-22
**Status:** approved in discussion (arms, scene-level split, g17 comparison,
data-fraction semantics all confirmed by the user)

## Question

2D supervision (point tracks) is cheap; 3D articulation annotation is
expensive. If we pretrain the model with the 2D-only recipe on 90% of the
training data and then train with full 3D supervision on only the remaining
10%, how close do we get to the model trained with 3D supervision on 100%?

## Arms

| arm | data | recipe | init |
|---|---|---|---|
| **A — scratch-10** (control) | 10% train scenes | g17 3D recipe | scratch |
| **B — pretrain+finetune** (hypothesis) | 90% (2D) then 10% (3D) | best 2D recipe → g17 3D recipe | B-pretrain ckpt via `model.finetune_from_path` |
| **C — full 3D** (upper bound) | 100% | g17 | *(exists: 20260818_sf3d_g17_splitax — no new run)* |

The claim is interesting iff B ≫ A and B ≈ C. Expected shape of the result:
B recovers mask/grounding/PDet near C (the 2D losses teach the trunk), while
axis/origin/MA lag C (articulation params only see the 10% 3D data — the 2D
line's known articulation deadlock).

## Data split (the part that can silently invalidate everything)

- The existing `split_dataset_by_scene` (seed 42, `val_split_ratio` 0.1)
  stays untouched — val and the standard 5,088-sample test split are
  IDENTICAL to every previous experiment.
- The **train** subset is further partitioned **at scene level** (keys are
  `visit_id/...`; a key-level split would leak near-duplicate frames of the
  same object between pretrain and finetune, inflating arm B):
  - Shuffle the train scene ids with a dedicated rng (`train_subset_seed`,
    default 4242 — deliberately not 42, so the partition is independent of
    the val split).
  - Greedily accumulate scenes into the **finetune** set until its sample
    count reaches `train_subset_ratio` (0.1) of train samples — greedy by
    sample count, not scene count, because scenes have very unequal sizes.
    Accept the overshoot of the last scene; print the realized fraction.
  - Remaining scenes = **pretrain** set. Disjoint by construction,
    deterministic on any machine from (seed 42, seed 4242, ratio 0.1).
- Accepted imperfection: a scene-level 10% has an unbalanced rot/trans and
  category mix. Report the realized rot fraction in notes.md.

### Datamodule interface (only code change)

`SF3DDataModule` gains three params, default-off (no behavior change for
existing configs):

```yaml
data:
  train_scene_subset: pretrain   # or: finetune; null = full train (default)
  train_subset_ratio: 0.1
  train_subset_seed: 4242
```

Implementation: `partition_subset_by_scene(subset, ratio, seed)` in
`datasets/scenefun3d.py` (operates on the train `Subset` returned by
`split_dataset_by_scene`, returns `(pretrain_subset, finetune_subset)`);
the datamodule picks one by name in `setup()` after the val split.

## Recipes and convergence

- **3D recipe** (arms A and B-finetune): exactly
  `config/sf3d_train_runpod_g17_splitax.yaml` — comparison target C already
  has full numbers on it.
- **2D recipe** (B-pretrain): decided at the wrap of the two in-flight 2D
  arms — whichever of {arm-A-detach, +DCT, +uv-fdiff} has the best proj2d
  shape without a mask regression vs arm-A-detach (mIoU 0.242). Fallback:
  arm-A-detach (`config/sf3d_train_runpod_g17_2d_detach.yaml`).
- **"Until convergence"** = `EarlyStopping(monitor="val/loss_total",
  mode="min", patience=5)` added to the callbacks of all three new configs.
  `max_epochs` caps: 40 (pretrain, 90% data), 60 (the 10% arms — epochs are
  ~10× shorter, and small-data runs need more of them; early stopping is
  the real terminator, and the 10% arms will hit it via overfitting well
  before 60).
- Val loss values are NOT comparable across arms (different loss terms /
  data volumes). Comparison happens only through the standard test pass.

## Experiments

- `20260822_sf3d_p90_2d` — B-pretrain (2D recipe, `pretrain` subset)
- `20260822_sf3d_ft10_3d` — B-finetune (g17 recipe, `finetune` subset,
  `finetune_from_path` = p90 best ckpt)
- `20260822_sf3d_s10_3d` — A (g17 recipe, `finetune` subset, scratch)

Configs: `sf3d_train_runpod_{p90_2d,ft10_3d,s10_3d}.yaml`. Standard wrap for
each (test pass on the training pod, delete pod, notes/INDEX/STATE.md).
Headline table: A vs B vs C on mIoU/PDet, type/MA, matched axis err, origin,
traj metrics. Sequencing: A and p90 can run in parallel on two pods;
ft10 needs p90's checkpoint first (staged from the volume — checkpoints
land there via the experiment dir).

## Tests

Unit tests for `partition_subset_by_scene` (synthetic dataset with
`item_keys`): determinism across calls; index disjointness + union = train
indices; no scene straddles the partition; realized finetune fraction within
[ratio, ratio + max-scene-size/N]; val subset unaffected; datamodule
`train_scene_subset` selection plumbing (`pretrain`/`finetune`/None) and
ValueError on unknown value.

## Cost

Pretrain ≈ a normal 30-epoch run (~$8–10); the two 10% arms are cheap
(~$1–2 each even at 60-epoch cap). Total ≈ $12–15 plus test passes.

## Out of scope (recorded, later)

- Class-level holdout (generalization to unseen affordance classes) — a
  different, harder question; do after the data-fraction result.
- Ratio sweeps (1%, 5%, 25%) — only if the 10% result is interesting.
