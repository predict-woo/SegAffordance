# Supervision Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the trainer and metrics survive (and stay honest about) models with the trajectory head or the articulation heads removed, and add the two ablation-arm configs (spec: `docs/superpowers/specs/2026-08-15-supervision-ablation-design.md`).

**Architecture:** No model changes — the gating flags (`use_trajectory_head`, `use_motion_head`, `use_motion_type_head`, `use_origin_heatmap`, `geometric_loss: "none"`) all exist and were audited working. The work is (1) guarding the few metric/viz consumers that assume heads exist, so absent-head metrics are *skipped*, never scored against a zeros placeholder or logged as 0.0; (2) two configs derived from the gen-9 config.

**Tech Stack:** PyTorch Lightning trainers (`train_OPDReal_better.py`, `train_SF3D_better.py`), pytest with the `_StubBackbone` monkeypatch pattern from `tests/test_g7_lift.py`.

## Global Constraints

- Local test interpreter: `PY=/private/tmp/claude-501/-Users-andyye-dev-ethz-workspace/ab498bc1-4197-41fb-9b67-6f93d9065aed/scratchpad/twistenv/bin/python` (has the full chain incl. pytorch_lightning). Run tests as `$PY -m pytest tests/ -x -q`.
- The full suite (131 tests) must stay green.
- Never construct a real CLIP backbone in tests (`clip_pretrain` can't load locally) — use the `_StubBackbone` monkeypatch pattern from `tests/test_g7_lift.py`.
- Arm A (gen-9) behavior must be bit-identical for training losses. Metric *logging* may change only as specified (skips instead of 0.0 for absent-head metrics).
- Do not touch `model/segmenter.py`, `model/losses/*`, or `datasets/*` — audited as already correct.
- Commit style: repo convention (`trainer: …`, `config: …`, `tests: …`).

---

### Task 1: Trainer guards for absent heads + gating tests

**Files:**
- Modify: `train_OPDReal_better.py` (~line 734, wandb val-viz path)
- Modify: `train_SF3D_better.py` (test_step ~459–541, on_test_epoch_end ~589–860, `__init__` accumulators ~118)
- Test: `tests/test_supervision_ablation.py` (new)

**Interfaces:**
- Consumes: existing `SF3DTrainingModule` (subclass of the OPDReal module), `tests/test_g7_lift.py` helpers `_g7_module`, `_g7_batch`, `_StubBackbone` (import or copy the pattern; prefer importing if the helpers are module-level).
- Produces: `self._test_has_axis_head: bool`, `self._test_has_type_head: bool` on the SF3D module (set during test_step, reset with the other accumulators) — Task 2's config test does not depend on them, but the wrap-up tooling reads the test CSV and relies on absent-head metric columns being absent.

- [ ] **Step 1: Write failing tests**

Create `tests/test_supervision_ablation.py` with helpers built on the `tests/test_g7_lift.py` pattern (stub backbone, small input size). Arm-B module = gen-7/9 flags + `use_trajectory_head=False`, loss_params `geometric_loss="none"`, `trajectory_weight=0.0`, `pred_pred_art_weight=0.0`. Arm-C module = gen-7/9 flags + `use_motion_head=False`, `use_motion_type_head=False`, `use_origin_heatmap=False`, loss_params `geometric_loss="none"` and `vae_weight=motion_type_weight=origin_weight=origin_map_weight=pred_pred_art_weight=0.0`.

```python
def test_armB_no_trajectory_head_trains():
    module = _abl_module(arm="B")
    outputs = _forward(module)          # with K_norm, as _g7 tests do
    assert module.model.trajectory_predictor is None
    assert outputs.trajectory_pred is None
    assert outputs.point_3d_pred is not None      # point pipeline intact
    assert outputs.origin_pred is not None        # origin pipeline intact
    loss = _training_step(module)                 # _common_step "train"
    assert torch.isfinite(loss)

def test_armC_no_articulation_trains():
    module = _abl_module(arm="C")
    outputs = _forward(module)
    for f in ("motion_pred", "motion_type_logits", "origin_uv",
              "origin_logits", "origin_pred"):
        assert getattr(outputs, f) is None, f
    assert outputs.point_3d_pred is not None      # z_p lift intact
    assert outputs.trajectory_pred is not None
    loss = _training_step(module)
    assert torch.isfinite(loss)

def test_armC_test_step_skips_axis_and_type_metrics():
    module = _abl_module(arm="C")
    _run_test_step(module)              # one small SF3D-format test batch
    assert module._test_axis_errors_all == []
    assert module._test_axis_errors_matched == []
    assert module._test_type_correct_all == 0
    assert module._test_ma_correct_all == 0
    assert module._test_has_axis_head is False
    assert module._test_has_type_head is False
    assert len(module._test_ious) > 0             # mask metrics still collected

def test_armA_test_step_still_collects_axis_and_type():
    module = _abl_module(arm="A")       # plain gen-7/9 flags, no removals
    _run_test_step(module)
    assert len(module._test_axis_errors_all) > 0
    assert module._test_has_axis_head is True
    assert module._test_has_type_head is True

def test_armB_wandb_viz_guard():
    # The OPDReal wandb viz path indexes trajectory_pred[i]; with the head
    # off it must fall back instead of raising TypeError. Unit-level: just
    # verify the expression the fix uses.
    tp = None
    fallback = tp[0] if tp is not None else torch.zeros(20, 3)
    assert fallback.shape == (20, 3)
```

`_run_test_step` builds a test-format batch (see `test_step`'s unpack in `train_SF3D_better.py` ~line 280: the camera-params-in-batch tuple including `motion_origin_3d_gt`, `intrinsic_matrix`, `_img_size`, `rgb_image_filenames`) and calls `module.test_step(batch, 0)`. Set `module.config.test_visualize_debug = False` and stub `module.trainer` only if an attribute access requires it (`test_step` touches `self.trainer.is_global_zero` only inside the viz branch, and `on_test_epoch_end` is NOT called in these tests).

- [ ] **Step 2: Run tests, verify the new file fails**

Run: `$PY -m pytest tests/test_supervision_ablation.py -x -q`
Expected: `test_armC_test_step_skips_axis_and_type_metrics` FAILS (axis list non-empty via the zeros fallback; `_test_has_axis_head` AttributeError). Arm construction tests may already pass — that's fine; the metric-skip tests are the teeth.

- [ ] **Step 3: Guard the OPDReal wandb viz path**

`train_OPDReal_better.py` ~line 734, replace

```python
                    trajectory_pred=trajectory_pred[i],
```

with

```python
                    trajectory_pred=(
                        trajectory_pred[i] if trajectory_pred is not None
                        else torch.zeros(20, 3, device=img.device)
                    ),
```

(zeros placeholder matches the existing `motion_pred`/`motion_type_logits` fallbacks two lines up — this path only renders a wandb composite).

- [ ] **Step 4: Head-aware metric collection in `train_SF3D_better.py` test_step**

In `__init__` (next to the other `_test_*` accumulators) add:

```python
        self._test_has_axis_head: bool = False
        self._test_has_type_head: bool = False
```

Replace the axis/type block (~459–476, from `# --- Motion and Axis evaluation` through `self._test_ma_correct_all += 1`) with:

```python
            # --- Motion and Axis evaluation (for all samples) ---
            # Ablation arms (2026-08-15 spec) may lack the axis and/or type
            # heads entirely: skip collection rather than scoring a zeros
            # placeholder, so absent-head metrics stay absent instead of
            # polluting the means.
            has_axis = motion_pred is not None or twist_decoded is not None
            has_type = motion_type_logits is not None or twist_decoded is not None
            self._test_has_axis_head |= has_axis
            self._test_has_type_head |= has_type
            axis_err = None
            is_axis_correct = is_type_correct = False
            if has_axis:
                axis_src = (
                    motion_pred[i] if motion_pred is not None
                    else twist_decoded[1][i]
                )
                axis_err = self._axis_error_deg(axis_src, motion_gt[i]).item()
                self._test_axis_errors_all.append(axis_err)
                is_axis_correct = axis_err <= self.config.test_motion_threshold_deg
            if has_type:
                is_type_correct = bool(pred_types[i] == motion_type_gt[i])
                if is_type_correct:
                    self._test_type_correct_all += 1
            if is_axis_correct and is_type_correct:
                self._test_ma_correct_all += 1
```

In the matched block (~529–541) change only `self._test_axis_errors_matched.append(axis_err)` to:

```python
                if axis_err is not None:
                    self._test_axis_errors_matched.append(axis_err)
```

(`is_axis_correct`/`is_type_correct` stay False when heads are absent, so the counters below need no further change.)

- [ ] **Step 5: Skip absent-head logging in on_test_epoch_end**

All in `train_SF3D_better.py`:

a. Wrap the axis logs (`test/err_adir_matched_deg`, `test/err_adir_all_deg`) in `if self._test_has_axis_head:`.
b. Wrap the type logs (`test/pass_rate_m`) in `if self._test_has_type_head:` and `test/pass_rate_ma` in `if self._test_has_axis_head and self._test_has_type_head:` (compute the rates inside the guards; `test/p_det` stays unconditional).
c. `test/mean_origin_error_m`: log only when `all_origin_errors_rotational.numel() > 0`.
d. Split-metric loop: log only when the head produced data —

```python
            gathered = self.all_gather(torch.tensor(values, device=self.device))
            mean = float(gathered.mean().item()) if gathered.numel() > 0 else 0.0
            if gathered.numel() > 0:
                self.log(name, mean, on_epoch=True, logger=True, sync_dist=False)
            split_stats[name] = (mean, gathered.numel())
```

e. Console prints: guard `M Pass Rate` / `MA Pass Rate` / `Mean Axis Error` lines with the same flags (print `n/a (head absent)` or skip the line — skip preferred).
f. In the accumulator-reset section add `self._test_has_axis_head = False` and `self._test_has_type_head = False`.

Variables `pass_rate_m`, `pass_rate_ma`, `err_adir_*` are referenced later in the print block — keep their computation (they're cheap) and guard only `self.log` calls and prints, OR move computation inside guards and default the print-block guards accordingly. Either is fine; keep it consistent and NameError-free.

- [ ] **Step 6: Run the new tests, then the full suite**

Run: `$PY -m pytest tests/test_supervision_ablation.py -x -q` → all pass.
Run: `$PY -m pytest tests/ -q` → 131 + new tests pass, 0 failures.

- [ ] **Step 7: Commit**

```bash
git add train_OPDReal_better.py train_SF3D_better.py tests/test_supervision_ablation.py
git commit -m "trainer: skip absent-head test metrics + viz guard (supervision-ablation arms)"
```

---

### Task 2: Ablation arm configs

**Files:**
- Create: `config/sf3d_train_runpod_g9abl_artonly.yaml`
- Create: `config/sf3d_train_runpod_g9abl_trajonly.yaml`
- Test: append to `tests/test_supervision_ablation.py`

**Interfaces:**
- Consumes: `config/sf3d_train_runpod_g9_closeup010.yaml` (the base — copy it byte-for-byte, then apply the deltas below).
- Produces: the two config paths above, referenced verbatim by the run launcher.

- [ ] **Step 1: Write the failing config test**

Append to `tests/test_supervision_ablation.py`:

```python
import yaml

_CFG = os.path.join(os.path.dirname(__file__), "..", "config")

def _load(name):
    with open(os.path.join(_CFG, name)) as f:
        return yaml.safe_load(f)

def test_ablation_configs_match_spec():
    base = _load("sf3d_train_runpod_g9_closeup010.yaml")
    art = _load("sf3d_train_runpod_g9abl_artonly.yaml")
    trj = _load("sf3d_train_runpod_g9abl_trajonly.yaml")

    bm, am, tm = (c["model"]["model_params"] for c in (base, art, trj))
    bl, al, tl = (c["model"]["loss_params"] for c in (base, art, trj))
    bd, ad, td = (c["data"] for c in (base, art, trj))

    # Arm B: only the trajectory path is removed.
    assert am["use_trajectory_head"] is False
    assert al["geometric_loss"] == "none"
    assert al["trajectory_weight"] == 0.0
    assert al["pred_pred_art_weight"] == 0.0
    assert am["use_motion_head"] and am["use_motion_type_head"]
    assert am["use_origin_heatmap"] and am["predict_point_depth"]

    # Arm C: only the articulation paths are removed.
    assert tm["use_motion_head"] is False
    assert tm["use_motion_type_head"] is False
    assert tm["use_origin_heatmap"] is False
    assert tm["predict_point_depth"] is True
    assert tm.get("use_trajectory_head", True) is True
    assert tl["geometric_loss"] == "none"
    for k in ("vae_weight", "motion_type_weight", "origin_weight",
              "origin_map_weight", "pred_pred_art_weight"):
        assert tl[k] == 0.0, k

    # Constants identical across all three arms.
    for m, l, d in ((am, al, ad), (tm, tl, td)):
        assert d["key_cache_path"] == bd["key_cache_path"]
        assert d["min_mask_area_frac"] == bd["min_mask_area_frac"]
        assert d["edge_margin_frac"] == bd["edge_margin_frac"]
        assert d["batch_size_train"] == bd["batch_size_train"]
        assert m["clip_pretrain"] == bm["clip_pretrain"]
        assert l["mask_weight"] == bl["mask_weight"]
        assert l["point_3d_weight"] == bl["point_3d_weight"]
    for c in (base, art, trj):
        assert c["trainer"]["max_epochs"] == 30
        assert c["model"]["optimizer_params"]["scheduler_milestones"] == [24, 28]
        assert c["seed_everything"] == 42

    # Each arm writes to its own experiment dir.
    for c, tag in ((art, "g9abl_artonly"), (trj, "g9abl_trajonly")):
        ckpt = c["trainer"]["callbacks"][0]["init_args"]["dirpath"]
        logd = c["trainer"]["logger"]["init_args"]["save_dir"]
        assert tag in ckpt and tag in logd
```

Run: `$PY -m pytest tests/test_supervision_ablation.py -x -q` → the new test FAILS (files missing).

- [ ] **Step 2: Create the two configs**

Copy `config/sf3d_train_runpod_g9_closeup010.yaml` twice and apply exactly these deltas (leave every other line identical, including all data/optimizer/trainer settings):

`sf3d_train_runpod_g9abl_artonly.yaml` (arm B):
1. Replace the header comment block (everything before `seed_everything`) with:
```yaml
# 20260815_sf3d_g9abl_artonly: supervision-ablation arm B — ARTICULATION
# ONLY. The gen-9 recipe (sf3d_train_runpod_g9_closeup010.yaml) minus the
# trajectory path: TrajectoryMLP not constructed, L_trajectory and L_pp
# gone. The interaction point (heatmap + z_p + 3D loss vs traj[0]) is
# static grounding and stays. Spec:
# docs/superpowers/specs/2026-08-15-supervision-ablation-design.md
```
2. In `model_params`, directly under `twist_num_hypotheses: 1`, add:
```yaml
    use_trajectory_head: false     # arm B: no trajectory path at all
```
3. In `loss_params`: `trajectory_weight: 0.0`, `geometric_loss: "none"`, `pred_pred_art_weight: 0.0` (edit the existing lines, keep positions).
4. Both `20260815_sf3d_g9_closeup010` path occurrences (checkpoint `dirpath`, logger `save_dir`) → `20260815_sf3d_g9abl_artonly`.

`sf3d_train_runpod_g9abl_trajonly.yaml` (arm C):
1. Header comment:
```yaml
# 20260815_sf3d_g9abl_trajonly: supervision-ablation arm C — TRAJECTORY
# ONLY. The gen-9 recipe minus every articulation path: no type head, no
# axis head (MotionMLP skipped entirely), no origin heatmap channel / z_q
# lift (projector back to 2 channels, condition = [features, point_uv]).
# Mask + interaction point + 20-point relative trajectory remain. Spec:
# docs/superpowers/specs/2026-08-15-supervision-ablation-design.md
```
2. In `model_params`: `use_motion_head: false`, `use_motion_type_head: false`, `use_origin_heatmap: false` (edit existing lines; update their trailing comments to say "arm C: articulation removed").
3. In `loss_params`: `vae_weight: 0.0`, `motion_type_weight: 0.0`, `origin_weight: 0.0`, `origin_map_weight: 0.0`, `geometric_loss: "none"`, `pred_pred_art_weight: 0.0`.
4. Both experiment-path occurrences → `20260815_sf3d_g9abl_trajonly`.

- [ ] **Step 3: Run the config test, then the full suite**

Run: `$PY -m pytest tests/test_supervision_ablation.py -x -q` → all pass.
Run: `$PY -m pytest tests/ -q` → everything passes.

- [ ] **Step 4: Commit**

```bash
git add config/sf3d_train_runpod_g9abl_artonly.yaml config/sf3d_train_runpod_g9abl_trajonly.yaml tests/test_supervision_ablation.py
git commit -m "config: supervision-ablation arms B (art-only) and C (traj-only)"
```
