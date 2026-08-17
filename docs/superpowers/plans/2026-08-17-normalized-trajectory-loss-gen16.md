# Gen-16 Normalized Trajectory Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the GT-energy-normalized trajectory loss behind `trajectory_loss_normalized` and the gen-16 config (spec: `docs/superpowers/specs/2026-08-17-normalized-trajectory-loss-gen16-design.md`). The empirical collapse-reversal probe is orchestrator-run afterward, not a plan task.

**Architecture:** One gated branch inside the existing relative-trajectory loss block in `train_OPDReal_better.py`; a LossParams flag; one config chained off gen-13. Flag off = byte-identical.

**Tech Stack:** PyTorch Lightning trainer, pytest (stub patterns from `tests/test_g7_lift.py` / `tests/test_split_heads.py`).

## Global Constraints

- Local test interpreter: `PY=/private/tmp/claude-501/-Users-andyye-dev-ethz-workspace/ab498bc1-4197-41fb-9b67-6f93d9065aed/scratchpad/twistenv/bin/python`; `cd /Users/andyye/dev/ethz-workspace/SegAffordance && $PY -m pytest tests/ -q`. Suite is currently 157 passed; must stay green.
- Flag default False → the trajectory loss value is BIT-identical (the existing `self.trajectory_loss_fn(...)` call untouched; no reordering).
- Exact values: eps floor `1e-4` (clamp on GT energy); weight in the g16 config `0.5`; flag names `trajectory_loss_normalized` (LossParams), config `sf3d_train_runpod_g16_trajnorm.yaml`, experiment `20260817_sf3d_g16_trajnorm`.
- `trajectory_loss_normalized` + `trajectory_absolute` must raise ValueError at module init.
- Commit style: repo convention, Co-Authored-By: Claude Fable 5 <noreply@anthropic.com> last line.

---

### Task 1: The normalized loss branch + tests

**Files:**
- Modify: `train_OPDReal_better.py` (`__init__` guard near the other loss setup ~line 78–95; the relative branch of the trajectory block ~line 434–442)
- Modify: `config/opd_train.py` (LossParams field next to `trajectory_weight`)
- Test: `tests/test_traj_norm_loss.py` (new)

**Interfaces:**
- Consumes: the existing trajectory block (read it fully first — the brief's snippet below replaces ONLY the else-branch body's final assignment region); `_g7_module`-style helpers for the one integration test.
- Produces: LossParams `trajectory_loss_normalized: bool = False` (Task 2's config test reads the name); log key `{step}/L_trajectory_m2` when the flag is on.

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import torch


def _rows(B=4, N=20, scale=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    gt = torch.randn(B, N, 3, generator=g) * scale
    gt = gt - gt[:, 0:1]
    pred = torch.randn(B, N, 3, generator=g) * scale
    pred = pred - pred[:, 0:1]
    return pred, gt


def _norm_loss(pred, gt):
    # The spec's formula, straight: used to pin the trainer's implementation.
    per_row = (pred - gt).pow(2).sum(-1).mean(-1)
    energy = gt.pow(2).sum(-1).mean(-1)
    return (per_row / energy.clamp(min=1e-4)).mean()


def test_collapsed_prediction_scores_one():
    _, gt = _rows()
    loss = _norm_loss(torch.zeros_like(gt), gt)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-6)


def test_scale_invariance():
    pred, gt = _rows(scale=0.3)
    l1 = _norm_loss(pred, gt)
    l2 = _norm_loss(5.0 * pred, 5.0 * gt)
    assert torch.isclose(l1, l2, rtol=1e-5)


def test_eps_floor_damps_degenerate_gt():
    # 1 cm GT stub: energy ~3e-5 < 1e-4 floor -> loss is measured against
    # the floor, not amplified by the tiny denominator.
    gt = torch.zeros(1, 20, 3)
    gt[:, :, 2] = torch.linspace(0, 0.01, 20)
    pred = torch.zeros_like(gt)
    loss = _norm_loss(pred, gt)
    energy = gt.pow(2).sum(-1).mean(-1)
    assert energy.item() < 1e-4
    assert loss.item() < 1.0            # damped below the collapsed score


def test_trainer_uses_normalized_when_flag_on():
    # Integration: module with the flag computes the SAME value as the
    # reference formula on a crafted batch, and logs L_trajectory_m2.
    # Build the module with the _g7_module pattern + loss flag override,
    # run _common_step on a batch whose GT trajectory is known, and compare
    # the logged train/L_trajectory against _norm_loss(pred, gt_rel)
    # computed from the module's own forward outputs. (Implementer: capture
    # logs via the module.log monkeypatch pattern already used in
    # tests/test_split_heads.py, or recompute the loss from outputs.)


def test_flag_off_bit_identical():
    # Module WITHOUT the flag: logged train/L_trajectory equals
    # F.mse_loss(trajectory_pred, gt_rel) exactly (the pre-change value).


def test_absolute_plus_normalized_raises():
    with pytest.raises(ValueError):
        _build_module(trajectory_absolute=True, trajectory_loss_normalized=True)
```

The last three tests are sketched: the implementer fleshes them out with the
existing stub-module machinery (read `tests/test_g7_lift.py` and
`tests/test_split_heads.py` first; whichever helper import pattern those
files use for `_common_step`-level tests is the pattern to follow). Every
assertion must be against values computed independently of the trainer's
own code path (the `_norm_loss` reference above, or `F.mse_loss` for the
off-path).

Run: `$PY -m pytest tests/test_traj_norm_loss.py -x -q` → the pure-formula
tests pass (they test the reference), the module tests FAIL (flag unknown).

- [ ] **Step 2: Implement**

`config/opd_train.py`, next to `trajectory_weight`:

```python
    # Gen-16: per-row GT-energy-normalized trajectory loss (relative-error;
    # a collapsed prediction scores exactly 1.0). Fixes the rot-sweep
    # collapse of the trajectory_weight=0.15 arms — see the 2026-08-17
    # spec. Relative-trajectory mode only.
    trajectory_loss_normalized: bool = False
```

`train_OPDReal_better.py` `__init__` (near the geometric-loss setup):

```python
        if getattr(loss_params, "trajectory_loss_normalized", False) and getattr(
            model_params, "trajectory_absolute", False
        ):
            raise ValueError(
                "trajectory_loss_normalized supports the RELATIVE trajectory "
                "head only (the normalization anchors on gt - gt[0])"
            )
```

The relative branch (else-arm of the trajectory block) becomes:

```python
                trajectory_gt_relative = (
                    trajectory_gt_device - trajectory_gt_device[:, 0:1, :]
                )
                if getattr(self.loss_params, "trajectory_loss_normalized", False):
                    # Gen-16: per-row RELATIVE error — each row's squared
                    # error over its own GT sweep energy, so rot and trans
                    # exert identical pressure regardless of sweep scale
                    # and a collapsed prediction scores exactly 1.0 (the
                    # rot-collapse fix; same philosophy as normalized
                    # L_pp). eps = (1 cm)^2 damps degenerate GT stubs.
                    _err = outputs.trajectory_pred - trajectory_gt_relative
                    _per_row = _err.pow(2).sum(-1).mean(-1)
                    _gt_energy = trajectory_gt_relative.pow(2).sum(-1).mean(-1)
                    L_trajectory = (_per_row / _gt_energy.clamp(min=1e-4)).mean()
                    self.log(
                        f"{step_type}/L_trajectory_m2",
                        _per_row.detach().mean(),
                        on_step=False, on_epoch=True, logger=True, sync_dist=True,
                    )
                else:
                    L_trajectory = self.trajectory_loss_fn(
                        outputs.trajectory_pred, trajectory_gt_relative
                    )
```

(Everything after — the weight multiply, grad_terms, existing logs — is
untouched. Match the neighboring `self.log` kwarg style exactly.)

- [ ] **Step 3: Run the new tests, then the full suite** — all green
  (157 + new).

- [ ] **Step 4: Commit**

```bash
git add train_OPDReal_better.py config/opd_train.py tests/test_traj_norm_loss.py
git commit -m "trainer: GT-energy-normalized trajectory loss (gen-16, flag-gated)"
```

---

### Task 2: Gen-16 config + config test

**Files:**
- Create: `config/sf3d_train_runpod_g16_trajnorm.yaml`
- Test: append to `tests/test_traj_norm_loss.py`

**Interfaces:**
- Consumes: `config/sf3d_train_runpod_g13_res512.yaml` (base; copy byte-for-byte, apply only the deltas), Task 1's flag name.
- Produces: the g16 config path.

- [ ] **Step 1: Failing config test** (append)

```python
def test_g16_config_matches_spec():
    base = _load_cfg("sf3d_train_runpod_g13_res512.yaml")
    g16 = _load_cfg("sf3d_train_runpod_g16_trajnorm.yaml")
    gl, bl = dict(g16["model"]["loss_params"]), dict(base["model"]["loss_params"])
    assert gl.pop("trajectory_loss_normalized") is True
    assert gl.pop("trajectory_weight") == 0.5 and bl.pop("trajectory_weight") == 0.15
    assert gl == bl
    assert g16["model"]["model_params"] == base["model"]["model_params"]
    assert g16["data"] == base["data"]
    assert "20260817_sf3d_g16_trajnorm" in g16["trainer"]["callbacks"][0]["init_args"]["dirpath"]
    assert "20260817_sf3d_g16_trajnorm" in g16["trainer"]["logger"]["init_args"]["save_dir"]
    assert g16["trainer"]["max_epochs"] == 30 and g16["seed_everything"] == 42
```

(`_load_cfg` as in `tests/test_dinov3_stack.py`.)

- [ ] **Step 2: Create the config** — copy g13's, apply: header block
  (replace everything before `seed_everything` with a gen-16 header citing
  the spec and the collapse diagnosis), `trajectory_weight: 0.15` →
  `0.5` (comment: "restored on the NORMALIZED term"), add
  `trajectory_loss_normalized: true` directly below it, both experiment
  paths → `20260817_sf3d_g16_trajnorm`.

- [ ] **Step 3: All tests green.**

- [ ] **Step 4: Commit**

```bash
git add config/sf3d_train_runpod_g16_trajnorm.yaml tests/test_traj_norm_loss.py
git commit -m "config: gen-16 normalized trajectory loss on the g13 base"
```
