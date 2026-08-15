# Gen-11 Origin Local Sample Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the origin depth head ẑ_q the same grid-sampled local feature the point depth head has, behind a default-off flag, and create the gen-11 config (v3 dataset) — spec: `docs/superpowers/specs/2026-08-16-origin-local-sample-gen11-design.md`.

**Architecture:** One flag (`use_origin_local_feature`), one conditional input-dim change in `__init__`, one gated `grid_sample` in `forward` mirroring the point path's. Dataset change is already done (`sf3d_processed_v3`); the config only points at it.

**Tech Stack:** PyTorch; pytest with the `_StubBackbone`/`_g7_module` patterns from `tests/test_g7_lift.py` and `tests/test_split_heads.py`.

## Global Constraints

- Local test interpreter: `PY=/private/tmp/claude-501/-Users-andyye-dev-ethz-workspace/ab498bc1-4197-41fb-9b67-6f93d9065aed/scratchpad/twistenv/bin/python`; run `cd /Users/andyye/dev/ethz-workspace/SegAffordance && $PY -m pytest tests/ -q`. Suite is currently 144 passed and must stay green.
- Flag default False → model construction and forward BIT-identical to gen-10 (no reordering of existing ops; the new code is purely additive and gated).
- Exact names/values: `use_origin_local_feature` (ModelParams, default False); requires `use_origin_heatmap` (ValueError otherwise); ẑ_q input dim becomes `vae_condition_dim + model_params.fpn_out[1]`; config `config/sf3d_train_runpod_g11_closeup010.yaml`, experiment `20260816_sf3d_g11_closeup010`, data root `/workspace/datasets/sf3d_processed_v3`.
- Commit style: repo convention, Co-Authored-By: Claude Fable 5 <noreply@anthropic.com> last line.

---

### Task 1: Model flag + gated origin local sample + tests

**Files:**
- Modify: `model/segmenter.py` (flag read near the gen-7 flag block ~127–132; `origin_depth_head_g7` construction ~307–315; forward z_q block ~483–484)
- Modify: `config/opd_train.py` (ModelParams — add the flag next to `use_origin_heatmap`/`predict_point_depth`)
- Test: `tests/test_origin_local_sample.py` (new)

**Interfaces:**
- Consumes: `_g7_module`-style construction helpers from `tests/test_g7_lift.py` (import module-level helpers; `_StubBackbone` pattern from `tests/test_split_heads.py`).
- Produces: `use_origin_local_feature: bool = False` on ModelParams (Task 2's config test reads this name); `self.use_origin_local_feature` on the segmenter.

- [ ] **Step 1: Write the failing tests**

```python
import torch

def test_flag_requires_origin_heatmap():
    # use_origin_local_feature without use_origin_heatmap must raise.
    # Build params exactly like the g7 tests but flip the flags.
    with pytest.raises(ValueError):
        _build_module(use_origin_heatmap=False, use_origin_local_feature=True)

def test_origin_head_input_dim_grows():
    m_off = _build_module(use_origin_local_feature=False)
    m_on = _build_module(use_origin_local_feature=True)
    d_off = _first_linear_in_features(m_off.model.origin_depth_head_g7)
    d_on = _first_linear_in_features(m_on.model.origin_depth_head_g7)
    assert d_on == d_off + m_on.model_params.fpn_out[1]

def test_forward_with_flag_on_lifts_origin():
    module = _build_module(use_origin_local_feature=True)
    outputs = _forward_with_K(module)
    assert outputs.origin_pred is not None
    assert torch.isfinite(outputs.origin_pred).all()
    loss = _training_step(module)
    assert torch.isfinite(loss)

def test_flag_off_path_unchanged():
    # Default-off: the ẑ_q input dim must equal the POINT depth head's
    # input dim minus fpn_out[1] — z_p is [condition, local], gen-10's z_q
    # is [condition] — an invariant of the pre-change architecture that
    # doesn't reference this change's own code. Forward + step still run.
    module = _build_module(use_origin_local_feature=False)
    d_q = _first_linear_in_features(module.model.origin_depth_head_g7)
    d_p = _first_linear_in_features(module.model.point_depth_head)
    assert d_q == d_p - module.model_params.fpn_out[1]
    outputs = _forward_with_K(module)
    assert outputs.origin_pred is not None
    assert torch.isfinite(_training_step(module))
```

Helpers: `_build_module(**flag_overrides)` mirrors `_g7_module` (stub
backbone, small input size, use_cvae False, twist off, use_origin_heatmap
True unless overridden, predict_point_depth True); `_first_linear_in_features`
walks `head.modules()` for the first `nn.Linear`; `_forward_with_K` and
`_training_step` reuse the g7 test-file patterns (import if module-level,
else replicate).

Run: `$PY -m pytest tests/test_origin_local_sample.py -x -q`
Expected: FAIL — ValueError test gets no error, dim test sees equal dims
(flag unknown → getattr default False).

- [ ] **Step 2: Implement**

`config/opd_train.py`, in ModelParams next to `predict_point_depth`:

```python
    # Gen-11: ẑ_q consumes a grid-sampled local feature at origin_uv —
    # the mirror of ẑ_p's sample at point_uv (the hinge-seam pixel IS
    # depth evidence; reverses the gen-7 condition-only choice). Requires
    # use_origin_heatmap.
    use_origin_local_feature: bool = False
```

`model/segmenter.py`, next to the other gen-7 flag reads:

```python
        self.use_origin_local_feature = getattr(
            model_params, "use_origin_local_feature", False
        )
        if self.use_origin_local_feature and not self.use_origin_heatmap:
            raise ValueError(
                "use_origin_local_feature needs use_origin_heatmap: the "
                "sample location IS the origin heatmap's soft-argmax"
            )
```

`origin_depth_head_g7` construction:

```python
        self.origin_depth_head_g7 = (
            OriginDepthHead(
                input_dim=vae_condition_dim
                + (model_params.fpn_out[1] if self.use_origin_local_feature else 0),
                hidden_dim=model_params.vae_hidden_dim,
            )
            if self.use_origin_heatmap
            else None
        )
```

Forward (replace the two-line z_q block):

```python
        if self.origin_depth_head_g7 is not None:
            zq_in = vae_condition
            if self.use_origin_local_feature:
                # Gen-11: mirror of the point path's local sample — the
                # origin heatmap's argmax pixel is typically the visible
                # hinge seam, whose appearance is depth evidence.
                ogrid = origin_uv.view(-1, 1, 1, 2) * 2.0 - 1.0
                olocal = F.grid_sample(
                    fq, ogrid, align_corners=False
                ).flatten(1)                                  # (B, fpn_out[1])
                zq_in = torch.cat([vae_condition, olocal], dim=1)
            z_q = self.origin_depth_head_g7(zq_in)
```

Also update the `__init__` comment above the gen-7 depth heads ("z_q
(origin): condition only …") to note the flag-gated local sample.

- [ ] **Step 3: Run the new tests, then the full suite**

Run: `$PY -m pytest tests/test_origin_local_sample.py -x -q` → all pass.
Run: `$PY -m pytest tests/ -q` → 144 + new pass, 0 failures.

- [ ] **Step 4: Commit**

```bash
git add model/segmenter.py config/opd_train.py tests/test_origin_local_sample.py
git commit -m "model: flag-gated origin local sample for z_q (gen-11 path symmetry)"
```

---

### Task 2: Gen-11 config + config test

**Files:**
- Create: `config/sf3d_train_runpod_g11_closeup010.yaml`
- Test: append to `tests/test_origin_local_sample.py`

**Interfaces:**
- Consumes: `config/sf3d_train_runpod_g10_closeup010.yaml` (base — copy byte-for-byte, apply only the deltas below); Task 1's flag name.
- Produces: the gen-11 config path, launched verbatim by the run pipeline.

- [ ] **Step 1: Write the failing config test**

```python
import os
import yaml

_CFG = os.path.join(os.path.dirname(__file__), "..", "config")

def _load_cfg(name):
    with open(os.path.join(_CFG, name)) as f:
        return yaml.safe_load(f)

def test_g11_config_matches_spec():
    base = _load_cfg("sf3d_train_runpod_g10_closeup010.yaml")
    g11 = _load_cfg("sf3d_train_runpod_g11_closeup010.yaml")

    gm = dict(g11["model"]["model_params"])
    bm = dict(base["model"]["model_params"])
    assert gm.pop("use_origin_local_feature") is True
    assert gm == bm  # nothing else in model_params changed

    gd = dict(g11["data"])
    bd = dict(base["data"])
    assert gd.pop("train_data_dir") == "/workspace/datasets/sf3d_processed_v3"
    bd.pop("train_data_dir")
    assert gd == bd  # incl. same key_cache_path (validated against v3)

    assert g11["model"]["loss_params"] == base["model"]["loss_params"]
    assert g11["model"]["optimizer_params"] == base["model"]["optimizer_params"]
    assert g11["seed_everything"] == 42
    assert g11["trainer"]["max_epochs"] == 30
    ckpt = g11["trainer"]["callbacks"][0]["init_args"]["dirpath"]
    logd = g11["trainer"]["logger"]["init_args"]["save_dir"]
    assert "20260816_sf3d_g11_closeup010" in ckpt
    assert "20260816_sf3d_g11_closeup010" in logd
```

Run: `$PY -m pytest tests/test_origin_local_sample.py::test_g11_config_matches_spec -x -q` → FAIL (missing file).

- [ ] **Step 2: Create the config**

Copy `config/sf3d_train_runpod_g10_closeup010.yaml`; apply exactly:

1. Header comment block (everything before `seed_everything`) →

```yaml
# 20260816_sf3d_g11_closeup010: gen-10 recipe + ORIGIN LOCAL SAMPLE, on
# sf3d_processed_v3. Two deltas vs sf3d_train_runpod_g10_closeup010.yaml:
#   * use_origin_local_feature: z_q consumes grid_sample(fq, origin_uv) —
#     the mirror of z_p's sample (hinge-seam pixel = depth evidence;
#     reverses the gen-7 condition-only choice).
#   * train_data_dir -> sf3d_processed_v3: prismatic sweeps are 0.7 m (the
#     revolute median) instead of 0.1 m. Same records/filters/caches; traj
#     magnitude metrics on trans rows are a FRESH baseline vs gens 8-10.
# Spec: docs/superpowers/specs/2026-08-16-origin-local-sample-gen11-design.md
```

2. In `model_params`, directly under `use_origin_heatmap: true`, add:
```yaml
    use_origin_local_feature: true # gen-11: z_q gets the hinge-pixel sample
```
3. `data.train_data_dir` → `"/workspace/datasets/sf3d_processed_v3"`
   (update its comment to mention v3's 0.7 m trans sweeps; `key_cache_path`
   and `frame_cache_path` stay — v3 symlinks v2's frames and the cache
   validates against v3's identical entry count).
4. Both `20260815_sf3d_g10_closeup010` experiment-path occurrences →
   `20260816_sf3d_g11_closeup010`.

- [ ] **Step 3: Run all tests**

Run: `$PY -m pytest tests/test_origin_local_sample.py -x -q` → all pass.
Run: `$PY -m pytest tests/ -q` → full suite green.

- [ ] **Step 4: Commit**

```bash
git add config/sf3d_train_runpod_g11_closeup010.yaml tests/test_origin_local_sample.py
git commit -m "config: gen-11 origin local sample on sf3d_processed_v3"
```
