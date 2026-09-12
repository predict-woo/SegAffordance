"""Minimal patches to 3DOI/monoarti for SF3D. Idempotent; every replacement is asserted unique.

1. depth validity: their loader marks holes as -1 and the loss checks only pixel (0, 0) to decide
   whether a frame has depth at all (fine for hole-free Taskonomy renders). SF3D sensor depth has
   holes, so the check becomes "any valid pixel" (the per-pixel mask ``tgt_depths > 1e-8`` already
   excludes holes from the loss).
2. train.py / stats.py import visdom and submitit at module level although neither is used with
   the basic launcher; guard the imports so the env needs neither.
3. optional iteration cap for smokes: ``SF3D_LIMIT_ITERS`` (env) breaks the epoch loop early.

  python patch_train.py <path to 3DOI/monoarti>
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(path, edits, marker):
    p = root / path
    s = p.read_text()
    if marker in s:
        print("already patched", p)
        return
    for old, new in edits:
        assert s.count(old) == 1, (path, old)
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", p)


for f in ("monoarti/sam_transformer.py", "monoarti/transformer.py"):
    patch(f, [("valid_depth = depth[:, 0, 0] > 0", "valid_depth = (depth > 0).flatten(1).any(1)  # SF3D_PATCHED")], "SF3D_PATCHED")

patch("train.py", [
    ("from visdom import Visdom\n", "try:  # SF3D_PATCHED\n    from visdom import Visdom\nexcept ImportError:\n    Visdom = None\n"),
    # Their validation walks the whole val split, and `epoch % interval == 0` fires at epoch 0, so a
    # 3,495-frame pass costs ~1 h per validation. Nothing selects a checkpoint from it (train.py just
    # overwrites checkpoint.pth every interval and we export from the last one), so cap the number of
    # val batches. Training is untouched.
    ("    for iteration, batch in enumerate(val_dataloader):\n        loss = 0.0\n",
     "    for iteration, batch in enumerate(val_dataloader):\n"
     "        if os.environ.get('SF3D_LIMIT_VAL_ITERS') and iteration >= int(os.environ['SF3D_LIMIT_VAL_ITERS']):\n"
     "            break\n"
     "        loss = 0.0\n"),
    ("        for iteration, batch in enumerate(train_dataloader):\n            optimizer.zero_grad()\n",
     "        for iteration, batch in enumerate(train_dataloader):\n"
     "            if os.environ.get('SF3D_LIMIT_ITERS') and iteration >= int(os.environ['SF3D_LIMIT_ITERS']):\n"
     "                break\n"
     "            optimizer.zero_grad()\n"),
], "SF3D_PATCHED")

patch("monoarti/stats.py", [
    ("from visdom import Visdom\n", "try:\n    from visdom import Visdom  # SF3D_PATCHED: optional\nexcept ImportError:\n    Visdom = None\n"),
], "SF3D_PATCHED")

