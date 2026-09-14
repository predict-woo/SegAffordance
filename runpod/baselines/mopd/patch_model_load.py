"""Fit-to-data patch for MOPD's model constructor: non-strict checkpoint load.

MOPD (maskformer_model.py, MaskFormer.__init__) loads its full released
checkpoint with a strict ``self.load_state_dict(torch.load(MODEL.WEIGHTS))``.
That checkpoint is Baidu-only, so we compose the init from our OPDFormer-P RGB
weights + EfficientSAM ViT-S + geffnet ImageNet weights
(tools/baselines_sf3d/mopd_compose_ckpt.py). MOPD's OWN added layers
(``seg_cross_attention_layers``, the "normal" cross-attention and
``input_proj2`` in the transformer decoder) exist in no public checkpoint and
must start from their random init; this patch makes the load non-strict and
prints what was missing / unexpected. Idempotent.

    python patch_model_load.py <MOPD>/opdformer/mask2former/maskformer_model.py
"""
import sys
from pathlib import Path

OLD = "        state_dict = torch.load(checkpoint_path)\n        self.load_state_dict(state_dict)\n"
NEW = ("        state_dict = torch.load(checkpoint_path)\n"
       "        _res = self.load_state_dict(state_dict, strict=False)  # SF3D baselines: MOPD-only layers start from init\n"
       "        print(f'[MOPD init] missing {len(_res.missing_keys)} keys (random init), unexpected {len(_res.unexpected_keys)}')\n"
       "        for _k in sorted({k.rsplit('.', 2)[0] for k in _res.missing_keys})[:40]: print('   missing:', _k)\n"
       "        for _k in _res.unexpected_keys[:20]: print('   unexpected:', _k)\n")

for path in sys.argv[1:]:
    p = Path(path)
    src = p.read_text()
    if NEW in src:
        print(f"already patched: {p}")
        continue
    assert src.count(OLD) == 1, f"expected exactly one occurrence in {p}, found {src.count(OLD)}"
    p.write_text(src.replace(OLD, NEW))
    print(f"patched: {p}")
