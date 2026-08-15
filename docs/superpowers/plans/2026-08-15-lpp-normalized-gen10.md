# Gen-10 Normalized Consistency Loss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the normalized (dimensionless) branches to `PredPredArticulationLoss` behind a default-off flag, wire the two new loss params through, sync the probe tool, and create the gen-10 config (spec: `docs/superpowers/specs/2026-08-15-lpp-normalized-gen10-design.md`).

**Architecture:** One `if self.normalized:` block inside the existing loss forward converts both branch residuals to relative errors before the soft gate; `normalized=False` (default) leaves every existing config bit-identical. Config plumbing follows the existing `getattr(loss_params, ...)` pattern.

**Tech Stack:** PyTorch, pytest. Existing test helpers `_pp_outputs` and `_circle_traj` in `tests/test_split_heads.py` (module-level, importable).

## Global Constraints

- Local test interpreter: `PY=/private/tmp/claude-501/-Users-andyye-dev-ethz-workspace/ab498bc1-4197-41fb-9b67-6f93d9065aed/scratchpad/twistenv/bin/python`; run `cd /Users/andyye/dev/ethz-workspace/SegAffordance && $PY -m pytest tests/ -q`. Suite is currently 137 passed and must stay green.
- `normalized=False` must be BIT-identical to the current loss (same ops, same order) — gen-9/ablation checkpoints and configs must reproduce exactly.
- Exact spec values: `radius_floor` default **0.10**; gen-10 weight **0.1**; config flag names `pred_pred_art_normalized`, `pred_pred_art_radius_floor`.
- The gen-10 config differs from `config/sf3d_train_runpod_g9_closeup010.yaml` ONLY in the three loss lines, the header comment, and the two experiment-path strings.
- Commit style: repo convention, Co-Authored-By: Claude Fable 5 <noreply@anthropic.com> last line.

---

### Task 1: Normalized branches in the loss + params + tests

**Files:**
- Modify: `model/losses/geometric.py` (PredPredArticulationLoss `__init__` ~159–168, forward ~209–216; `build_geometric_loss` "pred_pred_art" branch ~899–903)
- Modify: `config/opd_train.py` (LossParams dataclass — add two fields next to `pred_pred_art_weight`)
- Test: `tests/test_lpp_normalized.py` (new)

**Interfaces:**
- Consumes: `_pp_outputs(traj, dhat, p_rev_logit, origin, anchor)` and `_circle_traj(center_rel, dhat, start_rel, thetas)` from `tests/test_split_heads.py`; `StepTargets` from `model/targets.py`.
- Produces: `PredPredArticulationLoss(weight, degenerate_threshold=1e-6, trajectory_is_absolute=False, normalized=False, radius_floor=0.10)`; LossParams fields `pred_pred_art_normalized: bool = False`, `pred_pred_art_radius_floor: float = 0.10` (Task 2's config test reads these names).

- [ ] **Step 1: Write the failing tests**

```python
import torch

from model.losses.geometric import PredPredArticulationLoss, build_geometric_loss
from model.targets import StepTargets
from tests.test_split_heads import _circle_traj, _pp_outputs

torch.manual_seed(0)


def _rand_case(B=4, N=20):
    traj = torch.randn(B, N, 3) * 0.2
    traj = traj - traj[:, :1]                       # relative frame
    dhat = torch.nn.functional.normalize(torch.randn(B, 3), dim=1)
    return _pp_outputs(traj, dhat, p_rev_logit=1.3,
                       origin=torch.randn(B, 3), anchor=torch.randn(B, 3))


def test_normalized_false_bit_identical():
    out = _rand_case()
    old = PredPredArticulationLoss(weight=0.5)
    new = PredPredArticulationLoss(weight=0.5, normalized=False, radius_floor=0.10)
    t_old, _ = old(out, StepTargets())
    t_new, _ = new(out, StepTargets())
    assert torch.equal(t_old, t_new)


def test_normalized_line_branch_at_most_one():
    # All-prismatic gate: normalized line = off-axis energy fraction <= 1.
    out = _rand_case()
    out.motion_type_logits[:, 1] = -20.0            # P(rev) ~ 0
    loss = PredPredArticulationLoss(weight=1.0, normalized=True)
    _, terms = loss(out, StepTargets())
    assert terms["L_geo_pred_pred_art"].item() <= 1.0 + 1e-6


def test_normalized_circle_scale_invariant():
    # Uniformly scaling the scene must not change the relative orbit error
    # (r_hat stays above the floor at both scales).
    dhat = torch.nn.functional.normalize(torch.tensor([[0.0, 1.0, 0.0]]), dim=1)
    anchor = torch.tensor([[0.5, 0.2, 1.0]])
    origin = torch.tensor([[0.2, 0.2, 1.0]])        # r_hat = 0.3 m
    center_rel = (origin - anchor)[0]
    traj = _circle_traj(center_rel, dhat[0], torch.zeros(3),
                        torch.linspace(0, 1.2, 20))[None]
    traj = traj + 0.03 * torch.randn_like(traj)     # imperfect orbit
    traj = traj - traj[:, :1]
    loss = PredPredArticulationLoss(weight=1.0, normalized=True)
    _, t1 = loss(_pp_outputs(traj, dhat, 20.0, origin, anchor), StepTargets())
    s = 3.0
    _, t2 = loss(_pp_outputs(s * traj, dhat, 20.0, s * origin, s * anchor),
                 StepTargets())
    assert abs(t1["L_geo_pred_pred_art"].item()
               - t2["L_geo_pred_pred_art"].item()) < 1e-5


def test_normalized_radius_floor_engages():
    # r_hat ~ 0 (origin == anchor): denominator is the floor squared, so
    # halving the floor quadruples the loss.
    out = _rand_case(B=1)
    out.origin_pred = out.point_3d_pred.clone()     # r_hat -> 0
    out.motion_type_logits[:, 1] = 20.0             # P(rev) ~ 1
    lo = PredPredArticulationLoss(weight=1.0, normalized=True, radius_floor=0.10)
    hi = PredPredArticulationLoss(weight=1.0, normalized=True, radius_floor=0.05)
    _, t_lo = lo(out, StepTargets())
    _, t_hi = hi(out, StepTargets())
    ratio = t_hi["L_geo_pred_pred_art"].item() / max(
        t_lo["L_geo_pred_pred_art"].item(), 1e-12)
    assert abs(ratio - 4.0) < 1e-3


def test_build_geometric_loss_forwards_new_params():
    class LP:
        geometric_loss = "pred_pred_art"
        pred_pred_art_weight = 0.1
        pred_pred_art_normalized = True
        pred_pred_art_radius_floor = 0.10
    loss = build_geometric_loss(LP())
    assert loss.normalized is True
    assert loss.radius_floor == 0.10
    assert loss.weight == 0.1


def test_lossparams_defaults():
    from config.opd_train import LossParams
    lp = LossParams()
    assert lp.pred_pred_art_normalized is False
    assert lp.pred_pred_art_radius_floor == 0.10
```

- [ ] **Step 2: Run them, verify failure**

Run: `$PY -m pytest tests/test_lpp_normalized.py -x -q`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'normalized'` on the first normalized test (the bit-identity test may pass by accident once the params exist; the teeth are the other five).

- [ ] **Step 3: Implement the loss change**

`model/losses/geometric.py` — `__init__` (~159): add the two params and store them:

```python
    def __init__(
        self,
        weight: float,
        degenerate_threshold: float = 1e-6,
        trajectory_is_absolute: bool = False,
        normalized: bool = False,
        radius_floor: float = 0.10,
    ):
        super().__init__()
        self.weight = weight
        self.degenerate_threshold = degenerate_threshold
        self.trajectory_is_absolute = trajectory_is_absolute
        self.normalized = normalized
        self.radius_floor = radius_floor
```

In forward, between the `l_circle` computation and the `p_rev` line (current ~213–215), insert:

```python
        if self.normalized:
            # Gen-10 (2026-08-15 spec): dimensionless relative errors, so
            # the soft gate blends consistent units and big-radius doors
            # don't dominate small ones. Line: fraction of the displacement
            # energy perpendicular to the axis (intrinsically <= 1).
            # Circle: orbit residual relative to the predicted radius,
            # floored at radius_floor (= the dataset's min_revolute_radius)
            # so an implausibly small r_hat is measured against that scale
            # instead of exploding the ratio.
            mean_sq_disp = d.pow(2).sum(-1).mean(-1)              # (B,)
            l_line = l_line / mean_sq_disp.clamp(min=1e-8)
            l_circle = l_circle / (
                r_hat.squeeze(1).clamp(min=self.radius_floor).pow(2)
            )
```

Also update the class docstring with two sentences describing the
normalized mode. Update `build_geometric_loss`'s "pred_pred_art" branch:

```python
    if name == "pred_pred_art":
        return PredPredArticulationLoss(
            weight=getattr(loss_params, "pred_pred_art_weight", 0.5),
            trajectory_is_absolute=trajectory_is_absolute,
            normalized=getattr(loss_params, "pred_pred_art_normalized", False),
            radius_floor=getattr(loss_params, "pred_pred_art_radius_floor", 0.10),
        )
```

`config/opd_train.py` — in LossParams, directly after `pred_pred_art_weight`:

```python
    # Gen-10: dimensionless L_pp branches (relative orbit error / off-axis
    # energy fraction) and the r_hat floor, = min_revolute_radius.
    pred_pred_art_normalized: bool = False
    pred_pred_art_radius_floor: float = 0.10
```

- [ ] **Step 4: Run the new tests, then the full suite**

Run: `$PY -m pytest tests/test_lpp_normalized.py -x -q` → 6 pass.
Run: `$PY -m pytest tests/ -q` → 137 + 6 pass, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add model/losses/geometric.py config/opd_train.py tests/test_lpp_normalized.py
git commit -m "loss: normalized (relative-error) L_pp branches behind default-off flag"
```

---

### Task 2: Gen-10 config + probe floor arg + config test

**Files:**
- Create: `config/sf3d_train_runpod_g10_closeup010.yaml`
- Modify: `tools/diag_lpp_samples.py` (floor 0.05 → `--radius-floor` arg, default 0.10)
- Test: append to `tests/test_lpp_normalized.py`

**Interfaces:**
- Consumes: `config/sf3d_train_runpod_g9_closeup010.yaml` (base, copy byte-for-byte then apply deltas); Task 1's LossParams field names.
- Produces: the gen-10 config path, launched verbatim by the run pipeline.

- [ ] **Step 1: Write the failing config test**

Append to `tests/test_lpp_normalized.py`:

```python
import os

import yaml

_CFG = os.path.join(os.path.dirname(__file__), "..", "config")


def test_g10_config_matches_spec():
    with open(os.path.join(_CFG, "sf3d_train_runpod_g9_closeup010.yaml")) as f:
        base = yaml.safe_load(f)
    with open(os.path.join(_CFG, "sf3d_train_runpod_g10_closeup010.yaml")) as f:
        g10 = yaml.safe_load(f)

    gl = g10["model"]["loss_params"]
    assert gl["pred_pred_art_weight"] == 0.1
    assert gl["pred_pred_art_normalized"] is True
    assert gl["pred_pred_art_radius_floor"] == 0.10
    assert gl["geometric_loss"] == "pred_pred_art"

    # Everything else identical to gen-9.
    bl = dict(base["model"]["loss_params"])
    gl2 = dict(gl)
    for k in ("pred_pred_art_weight", "pred_pred_art_normalized",
              "pred_pred_art_radius_floor"):
        bl.pop(k, None), gl2.pop(k, None)
    assert gl2 == bl
    assert g10["model"]["model_params"] == base["model"]["model_params"]
    assert g10["data"] == base["data"]
    assert g10["model"]["optimizer_params"] == base["model"]["optimizer_params"]
    assert g10["seed_everything"] == 42
    assert g10["trainer"]["max_epochs"] == 30
    ckpt = g10["trainer"]["callbacks"][0]["init_args"]["dirpath"]
    logd = g10["trainer"]["logger"]["init_args"]["save_dir"]
    assert "20260815_sf3d_g10_closeup010" in ckpt
    assert "20260815_sf3d_g10_closeup010" in logd
```

Run: `$PY -m pytest tests/test_lpp_normalized.py::test_g10_config_matches_spec -x -q` → FAIL (file missing).

- [ ] **Step 2: Create the gen-10 config**

Copy `config/sf3d_train_runpod_g9_closeup010.yaml` and apply exactly:

1. Header comment block (everything before `seed_everything`) →

```yaml
# 20260815_sf3d_g10_closeup010: gen-9 recipe + NORMALIZED, full-strength
# consistency loss. ONLY delta vs sf3d_train_runpod_g9_closeup010.yaml:
# L_pp branches become dimensionless relative errors (line: off-axis
# energy fraction; circle: orbit residual / max(r_hat, 0.10)^2) at weight
# 0.1 — calibrated on the g9 best ckpt so the weighted contribution
# (~0.016) sits at the top of the trajectory/axis/origin band instead of
# 0.2% of the total. Same split/schedule/seed as the whole gen-9 family.
# Spec: docs/superpowers/specs/2026-08-15-lpp-normalized-gen10-design.md
```

2. In `loss_params`: `pred_pred_art_weight: 0.1` (edit the existing line;
   comment: `# calibrated for normalized branches (spec)`), and directly
   below it add:

```yaml
    pred_pred_art_normalized: true   # dimensionless relative-error branches
    pred_pred_art_radius_floor: 0.10 # = min_revolute_radius
```

3. Both `20260815_sf3d_g9_closeup010` path occurrences (checkpoint
   `dirpath`, logger `save_dir`) → `20260815_sf3d_g10_closeup010`.

- [ ] **Step 3: Probe floor argument**

`tools/diag_lpp_samples.py`: add `ap.add_argument("--radius-floor", type=float, default=0.10, help="r_hat floor for the normalized circle branch (keep = the loss's radius_floor)")`; give `lpp_breakdown` a `radius_floor=0.10` keyword and replace the hard-coded `clamp(min=0.05 ** 2)` with `clamp(min=radius_floor ** 2)`... careful, the current code clamps `r_hat...pow(2)` with `min=0.05 ** 2` — change to clamping r_hat BEFORE squaring (`r_hat.squeeze(1).clamp(min=radius_floor).pow(2)`) so probe and loss are formula-identical; pass `args.radius_floor` at both call sites. Update the comment from "floored at 0.05 m" wording to reference the arg.

- [ ] **Step 4: Run all tests**

Run: `$PY -m pytest tests/test_lpp_normalized.py -x -q` → 7 pass.
Run: `$PY -m pytest tests/ -q` → full suite green (144 expected).

- [ ] **Step 5: Commit**

```bash
git add config/sf3d_train_runpod_g10_closeup010.yaml tools/diag_lpp_samples.py tests/test_lpp_normalized.py
git commit -m "config/tools: gen-10 normalized-L_pp config + probe radius-floor arg"
```
