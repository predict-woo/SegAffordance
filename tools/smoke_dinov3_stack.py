"""Pod smoke for the dinov3 standard stack (gens 12-15): real weights,
no data — construct each stage's model, forward random tensors, backward.

Validates what the local stub tests cannot: the dinotxt hub load, the
tap hooks against the real ViT-L blocks, the aligned-map channel width vs
the text-embedding half (gen-15's RuntimeError guard), and 512-input
shapes. Run on a pod with the gated weights (~2 min):

  /opt/venv/bin/python -u tools/smoke_dinov3_stack.py
"""
import dataclasses
import sys

import torch
import yaml

sys.path.insert(0, "/workspace/SegAffordance")

from config.opd_train import ModelParams  # noqa: E402
from model.segmenter import CRIS  # noqa: E402

CONFIGS = [
    ("g12", "config/sf3d_train_runpod_g12_dinov3.yaml"),
    ("g13", "config/sf3d_train_runpod_g13_res512.yaml"),
    ("g14", "config/sf3d_train_runpod_g14_taps.yaml"),
    ("g15", "config/sf3d_train_runpod_g15_costmap.yaml"),
]

device = torch.device("cuda")
field_names = {f.name for f in dataclasses.fields(ModelParams)}

for tag, path in CONFIGS:
    raw = yaml.safe_load(open(path))["model"]["model_params"]
    raw = {k: v for k, v in raw.items() if k in field_names}
    raw["compile_model"] = False          # smoke tests eager correctness
    raw["channels_last"] = True           # the real runs' layout
    mp = ModelParams(**raw)
    size = int(getattr(mp, "backbone_image_size", 256))

    model = CRIS(mp).to(device)
    model.train()
    img = torch.randn(2, 3, size, size, device=device).to(
        memory_format=torch.channels_last
    )
    depth = torch.randn(2, 1, size, size, device=device)
    word = model.tokenize(["open the top oven door", "pull the middle drawer"],
                          mp.word_len).to(device)
    K = torch.tensor([[[1.1, 0.0, 0.5], [0.0, 1.5, 0.5], [0.0, 0.0, 1.0]]],
                     device=device).repeat(2, 1, 1)
    # Train-mode pooling teacher-forces the GT mask — forward needs one.
    mask = (torch.rand(2, 1, size, size, device=device) > 0.7).float()

    out = model(img, depth, word, mask, None, None, None, K)

    checks = {
        "mask_logits": out.mask_logits,
        "point_uv": out.point_uv,
        "origin_uv": out.origin_uv,
        "point_3d_pred": out.point_3d_pred,
        "origin_pred": out.origin_pred,
        "trajectory_pred": out.trajectory_pred,
        "motion_pred": out.motion_pred,
        "motion_type_logits": out.motion_type_logits,
    }
    for name, t in checks.items():
        assert t is not None, f"{tag}: {name} is None"
        assert torch.isfinite(t.float()).all(), f"{tag}: {name} not finite"

    loss = sum(t.float().mean() for t in checks.values())
    loss.backward()
    neck_grads = [p.grad for p in model.neck.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in neck_grads), \
        f"{tag}: no gradient reached the neck"

    hm = out.mask_logits.shape[-1]
    print(f"{tag}: OK  input {size}  mask_logits {tuple(out.mask_logits.shape)} "
          f"traj {tuple(out.trajectory_pred.shape)}", flush=True)
    del model, out, loss
    torch.cuda.empty_cache()

print("SMOKE OK — all stages construct, forward, and backprop with real weights")
