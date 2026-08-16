# DINOv3 Standard Stack Implementation Plan (stages R/T/C, no runs)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and test all three stages of `docs/superpowers/specs/2026-08-16-dinov3-standard-stack-design.md` — multi-layer taps (T), patch-text cost map + gate subspace fix (C), and the three staged configs (R/T/C) — WITHOUT launching the gen-13/14/15 experiments. A separate on-pod smoke (run by the orchestrator, not a plan task) validates real-weight model construction afterward.

**Architecture:** Two flag-gated model changes (`dinov3_multilayer_taps` in the backbone, `text_cost_map` in the segmenter+backbone interface), both default-off and bit-identical when off; three chained configs, each = predecessor + one stage's knob(s).

**Tech Stack:** PyTorch; pytest with stub patterns from `tests/test_split_heads.py` / `tests/test_g7_lift.py`. The real dino.txt weights are pod-only — local tests use stubs; shape/finiteness against real weights is the orchestrator's pod smoke.

## Global Constraints

- Local test interpreter: `PY=/private/tmp/claude-501/-Users-andyye-dev-ethz-workspace/ab498bc1-4197-41fb-9b67-6f93d9065aed/scratchpad/twistenv/bin/python`; `cd /Users/andyye/dev/ethz-workspace/SegAffordance && $PY -m pytest tests/ -q`. Suite is currently 149 passed and must stay green.
- Both new flags default False; flag-off construction/forward BIT-identical (purely additive, gated; no reordering). Gen-12 and every CLIP config must reproduce exactly.
- Exact names: ModelParams `dinov3_multilayer_taps: bool = False`, `text_cost_map: bool = False`; taps = ViT-L blocks **[4, 11, 17]** captured by forward hook + final tap from the existing pass; configs `sf3d_train_runpod_g13_res512.yaml` / `g14_taps.yaml` / `g15_costmap.yaml`; experiments `20260817_sf3d_g13_res512` / `20260817_sf3d_g14_taps` / `20260817_sf3d_g15_costmap`; frames cache path `/workspace/datasets/sf3d_frames_512.lmdb`.
- Do not touch `tools/sf3d_build_frame_cache.py` (already parameterized) or any training/loss code.
- Commit style: repo convention, Co-Authored-By: Claude Fable 5 <noreply@anthropic.com> last line.

---

### Task 1: MultiTapPyramid + `dinov3_multilayer_taps`

**Files:**
- Modify: `model/backbones/pyramid.py` (new class beside SimpleFeaturePyramid)
- Modify: `model/backbones/dinov3.py` (flag param, hook capture, adapter selection)
- Modify: `model/backbones/__init__.py` (pass the flag), `config/opd_train.py` (ModelParams field)
- Test: `tests/test_dinov3_stack.py` (new)

**Interfaces:**
- Consumes: existing `SimpleFeaturePyramid`, `tokens_to_map`.
- Produces: `MultiTapPyramid(in_dim, out_channels)` with `forward(taps: List[Tensor 4×(B,C,h,w)], x_deep=None) -> (v8, v16, v32)`; `DINOv3Backbone(..., multilayer_taps=False)`; ModelParams `dinov3_multilayer_taps: bool = False` (Task 3's configs set it).

- [ ] **Step 1: Write the failing tests**

```python
import torch

from model.backbones.pyramid import MultiTapPyramid, SimpleFeaturePyramid


def test_multitap_pyramid_shapes():
    torch.manual_seed(0)
    fpn_in = [512, 1024, 1024]
    pyr = MultiTapPyramid(in_dim=64, out_channels=fpn_in)
    taps = [torch.randn(2, 64, 16, 16) for _ in range(4)]
    v8, v16, v32 = pyr(taps)
    assert v8.shape == (2, 512, 32, 32)
    assert v16.shape == (2, 1024, 16, 16)
    assert v32.shape == (2, 1024, 8, 8)


def test_multitap_pyramid_x_deep_replaces_deep_source():
    torch.manual_seed(0)
    pyr = MultiTapPyramid(in_dim=8, out_channels=[16, 16, 16])
    taps = [torch.randn(1, 8, 4, 4) for _ in range(4)]
    x_deep = torch.randn(1, 8, 4, 4)
    _, _, a = pyr(taps)
    _, _, b = pyr(taps, x_deep=x_deep)
    assert not torch.allclose(a, b)          # deep level follows x_deep
    v8a, v16a, _ = pyr(taps)
    v8b, v16b, _ = pyr(taps, x_deep=x_deep)
    assert torch.equal(v8a, v8b) and torch.equal(v16a, v16b)  # others don't


def test_modelparams_has_taps_flag_default_false():
    import dataclasses
    from config.opd_train import ModelParams
    fields = {f.name: f for f in dataclasses.fields(ModelParams)}
    assert fields["dinov3_multilayer_taps"].default is False
```

Run: `$PY -m pytest tests/test_dinov3_stack.py -x -q` → FAIL (ImportError: MultiTapPyramid).

- [ ] **Step 2: Implement `MultiTapPyramid`** (`model/backbones/pyramid.py`)

```python
class MultiTapPyramid(nn.Module):
    """4 intermediate-layer taps -> strides 8/16/32 (frozen-DINO standard).

    DPT-style reassembly (DINOv2 "lin-4" / Depth Anything / SegDINO): the
    fine level comes from an EARLY layer, which still holds the high-
    frequency spatial detail the final layer has abstracted away —
    SimpleFeaturePyramid's deconv-from-final-layer cannot recover it (its
    ViTDet evidence base is fine-tuned MAE, a different regime; see
    knowledge/dinov3-dense-adapter-survey.md).

    forward(taps, x_deep=None): taps are 4 maps (B, in_dim, h, w) ordered
    shallow->deep (ViT-L blocks 4/11/17/23). x_deep, if given, replaces
    the deep tap as the /32 source (the dino.txt-aligned map — the same
    contract as SimpleFeaturePyramid).
    """

    def __init__(self, in_dim: int, out_channels: List[int]):
        super().__init__()
        if len(out_channels) != 3:
            raise ValueError(f"expected 3 output channel counts, got {out_channels}")
        self.up = nn.Sequential(                       # tap[0] -> /8
            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_dim // 2),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, out_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[0]),
        )
        self.mid = nn.Sequential(                      # cat(tap[1], tap[2]) -> /16
            nn.Conv2d(2 * in_dim, out_channels[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels[1]),
        )
        self.down = nn.Sequential(                     # tap[3] (or x_deep) -> /32
            nn.Conv2d(in_dim, out_channels[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels[2]),
        )

    def forward(self, taps, x_deep=None):
        if len(taps) != 4:
            raise ValueError(f"expected 4 taps, got {len(taps)}")
        deep = taps[3] if x_deep is None else x_deep
        return (
            self.up(taps[0]),
            self.mid(torch.cat([taps[1], taps[2]], dim=1)),
            self.down(deep),
        )
```

(`List` is already imported in the file.)

- [ ] **Step 3: Wire the flag through the backbone**

`model/backbones/__init__.py`, dinov3 branch: add
`multilayer_taps=getattr(model_params, "dinov3_multilayer_taps", False),`.

`model/backbones/dinov3.py`:
- `__init__` gains `multilayer_taps: bool = False`, stored as
  `self.multilayer_taps` BEFORE the `_init_*` calls.
- In `_init_dinotxt` AND `_init_hf_vision`: build
  `self.adapter = MultiTapPyramid(backbone_dim, self.fpn_in)` when the
  flag is on, `SimpleFeaturePyramid(...)` otherwise (import MultiTapPyramid
  alongside the existing pyramid imports).
- **Tap capture (dinotxt path)** — forward hooks on the frozen ViT so the
  existing single pass yields the intermediates (no second forward):

```python
    def _install_tap_hooks(self, blocks):
        # Blocks 4/11/17 of ViT-L (the "lin-4" spread minus the final
        # layer, which the existing pass already returns). Hook outputs are
        # the post-block hidden states incl. prefix tokens.
        self._tap_cache = {}
        for i in (4, 11, 17):
            def _mk(idx):
                def hook(_m, _inp, out):
                    self._tap_cache[idx] = out
                return hook
            blocks[i].register_forward_hook(_mk(i))
```

  Call it at the end of `_init_dinotxt` / `_init_hf_vision` when the flag
  is on (`vis.backbone.blocks` / `self.vision_model` — for the HF path use
  `self.vision_model.encoder.layer` if present, else raise ValueError
  saying taps are dinotxt-only for now; keep the committed scope dinotxt).
- `encode_image` (dinotxt branch), when the flag is on: after the existing
  `encode_image_with_patch_tokens` call, assemble taps —

```python
            norm = self.dinotxt.visual_model.backbone.norm
            taps = []
            for i in (4, 11, 17):
                t = self._tap_cache.pop(i)
                if isinstance(t, tuple):
                    t = t[0]
                t = norm(t)[:, -(grid_h * grid_w):, :]      # final LN + strip prefix
                taps.append(tokens_to_map(t, grid_h, grid_w))
            taps.append(raw_map)                             # block-23 tokens, already normed
            return self.adapter(taps, x_deep=aligned_map)
```

  (`raw_map`/`aligned_map` are the existing variables; the aligned /32
  source is PRESERVED — the spec keeps the gate-subspace question in
  Stage C.) Applying the backbone's final LayerNorm to each tap mirrors
  `get_intermediate_layers(norm=True)` — the DINOv2/DPT convention.
- Flag off: no hooks installed, no behavior change anywhere.

`config/opd_train.py` ModelParams, next to the other dinov3 fields:

```python
    # Gen-14 (dinov3 standard stack, stage T): build the pyramid from
    # 4 intermediate-layer taps (ViT-L blocks 4/11/17 by hook + the final
    # tokens) instead of the final layer only. dinotxt path only.
    dinov3_multilayer_taps: bool = False
```

- [ ] **Step 4: Run tests + full suite**

`$PY -m pytest tests/test_dinov3_stack.py -x -q` → pass;
`$PY -m pytest tests/ -q` → 149 + new pass.

- [ ] **Step 5: Commit**

```bash
git add model/backbones/ config/opd_train.py tests/test_dinov3_stack.py
git commit -m "backbone: MultiTapPyramid + dinov3_multilayer_taps (stage T, flag-gated)"
```

---

### Task 2: Patch-text cost map + gate subspace fix (`text_cost_map`)

**Files:**
- Modify: `model/backbones/base.py` (extended encode), `model/backbones/dinov3.py` (aligned-map extras), `model/segmenter.py` (flag, FPN dims, forward injection), `config/opd_train.py` (field)
- Test: append to `tests/test_dinov3_stack.py`

**Interfaces:**
- Consumes: Task 1's file state; `FPN` (`model/layers.py:678` — constructor takes `in_channels`, `text_dim`; NOT modified).
- Produces: `BackboneBase.encode_image_full(img) -> (maps, extras: dict)` (default `{}`); DINOv3Backbone dinotxt extras `{"aligned_map": (B, half, H/16, W/16)}`; ModelParams `text_cost_map: bool = False`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_encode_image_full_default_empty_extras():
    # Any backbone without an override returns (maps, {}).
    class _B(BackboneBase):
        def tokenize(self, texts, l): ...
        def encode_text(self, tokens): ...
        def encode_image(self, img):
            return "MAPS"
    maps, extras = _B().encode_image_full("img")
    assert maps == "MAPS" and extras == {}


def test_text_cost_map_requires_dinotxt():
    with pytest.raises(ValueError):
        _build_module_flags(text_cost_map=True)   # default backbone=clip_rn50


def test_text_cost_map_forward_shapes_and_gate_half():
    # Stub backbone advertising the dinotxt contract: state_dim 64,
    # aligned_map channels 32 (= half), extras wired through.
    module = _build_costmap_module()              # helper, see notes below
    outputs = _forward_with_K(module)
    assert outputs.mask_logits is not None        # end-to-end runs
    # FPN was built with +2 in_channels and half text_dim:
    neck = module.model.neck
    assert neck.txt_proj[0].in_features == 32     # half of the stub's 64


def test_flag_off_fpn_dims_unchanged():
    module = _build_module_flags()                # all flags off, stub clip-style
    neck = module.model.neck
    assert neck.txt_proj[0].in_features == module.model.backbone.state_dim
```

Helper notes for the implementer: `_build_costmap_module` monkeypatches
`build_backbone` with a stub whose `encode_image_full` returns 3 maps AND
`{"aligned_map": torch.randn(B, 32, H//16, W//16)}`, `state_dim = 64`,
`word_dim` matching the stub pattern in `tests/test_split_heads.py`; the
ModelParams for it set `backbone="dinov3"`, `text_source="dinotxt"`,
`text_cost_map=True`. `txt_proj` is `linear_layer(text_dim, ...)` — a
Sequential whose first element is `nn.Linear`; adjust the assertion to the
actual structure after reading `model/layers.py::linear_layer`.

Run → FAIL (no `encode_image_full`, no flag).

- [ ] **Step 2: Implement**

`model/backbones/base.py`:

```python
    def encode_image_full(self, img):
        """(maps, extras). Extras are backbone-specific side outputs —
        dinotxt returns its text-aligned /16 patch map for the cost-map
        path (gen-15). Default: no extras."""
        return self.encode_image(img), {}
```

`model/backbones/dinov3.py` (dinotxt branch): restructure so the existing
`encode_image` body becomes `_encode(img)` returning
`(maps, {"aligned_map": tokens_to_map(aligned_tokens, grid_h, grid_w)})`
— the RAW ψ tokens (before `deep_proj`), whose channel width is the
text-embedding HALF (dino.txt: text = [CLS-half; patch-half]). Then
`encode_image` returns `self._encode(img)[0]` and `encode_image_full`
returns `self._encode(img)`. The `text_source="clip"` branch keeps the
default (empty extras).

`config/opd_train.py` ModelParams:

```python
    # Gen-15 (stage C): explicit patch-text similarity maps (local + global
    # halves of the dino.txt embedding vs the aligned /16 tokens) injected
    # as 2 extra FPN input channels per level, and the FPN text gate reads
    # only the patch-aligned LOCAL half of the state. dinotxt only.
    text_cost_map: bool = False
```

`model/segmenter.py`:
- `__init__`, near the other flag reads:

```python
        self.text_cost_map = getattr(model_params, "text_cost_map", False)
        if self.text_cost_map and not (
            getattr(model_params, "backbone", "clip_rn50") == "dinov3"
            and getattr(model_params, "text_source", "dinotxt") == "dinotxt"
        ):
            raise ValueError(
                "text_cost_map needs the dinov3 backbone with dino.txt text "
                "(the cost map is cosine vs the dino.txt-aligned patch tokens)"
            )
```

- FPN construction (~line 90): when the flag is on, pass
  `in_channels=[c + 2 for c in fpn_in_channels]` and
  `text_dim=state_dim // 2` (state stays full-width everywhere else —
  decoder word features and the projector's dynamic kernel are unchanged
  by spec).
- Forward (~line 368): replace the single `encode_image` call with

```python
        if self.text_cost_map:
            vis, _bb_extras = self.backbone.encode_image_full(img)
        else:
            vis = self.backbone.encode_image(img)
```

  and after depth fusion, before the neck:

```python
        if self.text_cost_map:
            aligned = F.normalize(_bb_extras["aligned_map"].float(), dim=1)
            half = state.shape[1] // 2
            t_glob = F.normalize(state[:, :half].float(), dim=1)
            t_loc = F.normalize(state[:, half:].float(), dim=1)
            cost_g = torch.einsum("bchw,bc->bhw", aligned, t_glob)[:, None]
            cost_l = torch.einsum("bchw,bc->bhw", aligned, t_loc)[:, None]
            costs = torch.cat([cost_g, cost_l], dim=1).to(vis_fused[0].dtype)
            vis_fused = tuple(
                torch.cat(
                    [v, F.interpolate(costs, size=v.shape[-2:], mode="bilinear",
                                      align_corners=False)],
                    dim=1,
                )
                for v in vis_fused
            )
            state_for_gate = state[:, state.shape[1] // 2:]
        else:
            state_for_gate = state
        fq = self.neck(vis_fused, state_for_gate)
```

  (The existing `fq = self.neck(vis_fused, state)` line is REPLACED by the
  gated version; flag off ⇒ `state_for_gate = state`, identical call. If
  the aligned-map channel width does not equal `half`, raise a RuntimeError
  with both numbers — a mis-paired backbone/text checkpoint should fail
  loudly, not silently miscompute.)

- [ ] **Step 3: Run tests + full suite** (as Task 1 Step 4).

- [ ] **Step 4: Commit**

```bash
git add model/backbones/ model/segmenter.py config/opd_train.py tests/test_dinov3_stack.py
git commit -m "model: patch-text cost map + FPN gate subspace fix (stage C, flag-gated)"
```

---

### Task 3: Staged configs g13/g14/g15 + config tests

**Files:**
- Create: `config/sf3d_train_runpod_g13_res512.yaml`, `config/sf3d_train_runpod_g14_taps.yaml`, `config/sf3d_train_runpod_g15_costmap.yaml`
- Test: append to `tests/test_dinov3_stack.py`

**Interfaces:**
- Consumes: `config/sf3d_train_runpod_g12_dinov3.yaml` (chain base); Task 1/2 flag names.
- Produces: the three config paths; each = predecessor + its stage knobs + experiment paths.

- [ ] **Step 1: Write the failing chain test** (append)

```python
def _load_yaml(name):
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "..", "config", name)) as f:
        return yaml.safe_load(f)


def test_g13_g14_g15_config_chain():
    g12 = _load_yaml("sf3d_train_runpod_g12_dinov3.yaml")
    g13 = _load_yaml("sf3d_train_runpod_g13_res512.yaml")
    g14 = _load_yaml("sf3d_train_runpod_g14_taps.yaml")
    g15 = _load_yaml("sf3d_train_runpod_g15_costmap.yaml")

    # g13 = g12 + resolution block
    m13, m12 = dict(g13["model"]["model_params"]), dict(g12["model"]["model_params"])
    assert m13.pop("backbone_image_size") == 512 and m12.pop("backbone_image_size") == 256
    assert m13 == m12
    assert g13["model"]["config"]["input_size"] == [512, 512]
    assert g13["model"]["loss_params"]["point_sigma"] == 16.0
    assert g13["data"]["frame_cache_path"] == "/workspace/datasets/sf3d_frames_512.lmdb"
    assert g13["data"]["batch_size_train"] == 64 and g13["data"]["batch_size_val"] == 64
    d13, d12 = dict(g13["data"]), dict(g12["data"])
    for k in ("frame_cache_path", "batch_size_train", "batch_size_val"):
        d13.pop(k), d12.pop(k)
    assert d13 == d12

    # g14 = g13 + taps flag
    m14, m13b = dict(g14["model"]["model_params"]), dict(g13["model"]["model_params"])
    assert m14.pop("dinov3_multilayer_taps") is True
    assert m14 == m13b
    assert g14["data"] == g13["data"]
    assert g14["model"]["loss_params"] == g13["model"]["loss_params"]

    # g15 = g14 + cost-map flag
    m15, m14b = dict(g15["model"]["model_params"]), dict(g14["model"]["model_params"])
    assert m15.pop("text_cost_map") is True
    assert m15 == m14b
    assert g15["data"] == g14["data"]

    for cfg, tag in ((g13, "20260817_sf3d_g13_res512"),
                     (g14, "20260817_sf3d_g14_taps"),
                     (g15, "20260817_sf3d_g15_costmap")):
        assert tag in cfg["trainer"]["callbacks"][0]["init_args"]["dirpath"]
        assert tag in cfg["trainer"]["logger"]["init_args"]["save_dir"]
        assert cfg["trainer"]["max_epochs"] == 30
        assert cfg["seed_everything"] == 42
```

Run → FAIL (files missing).

- [ ] **Step 2: Create the three configs**

Each derived byte-for-byte from its predecessor with only these deltas
(plus a stage-appropriate header comment block replacing everything before
`seed_everything`, each pointing at the spec):

`g13_res512` (from g12): `backbone_image_size: 512`;
`config.input_size: [512, 512]`; `loss_params.point_sigma: 16.0` (comment:
"8 px at 256 = 16 px at 512 — same physical extent");
`data.frame_cache_path: "/workspace/datasets/sf3d_frames_512.lmdb"`
(comment: built by tools/sf3d_build_frame_cache.py --depth-size 512;
staged into /dev/shm by the launch — 6000-class pods only);
`batch_size_train: 64`, `batch_size_val: 64` (comment: 4× activations at
512; revisit toward 128 after the first epoch's memory high-water);
paths → `20260817_sf3d_g13_res512`.

`g14_taps` (from g13): `dinov3_multilayer_taps: true` (one added line,
comment "stage T: DPT-style taps from ViT-L blocks 4/11/17 + final");
paths → `20260817_sf3d_g14_taps`.

`g15_costmap` (from g14): `text_cost_map: true` (one added line, comment
"stage C: dual half-similarity channels + local-half FPN gate");
paths → `20260817_sf3d_g15_costmap`.

- [ ] **Step 3: Run all tests** — chain test passes, full suite green.

- [ ] **Step 4: Commit**

```bash
git add config/sf3d_train_runpod_g1[345]_*.yaml tests/test_dinov3_stack.py
git commit -m "config: staged g13/g14/g15 configs (512 res / taps / cost map) — not launched"
```
