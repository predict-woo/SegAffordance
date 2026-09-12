"""Fit-to-env patch for MOPD's matcher: skip the optional `uotod` Sinkhorn block.

MOPD's HungarianMatcher (opdformer/mask2former/modeling/matcher.py) computes an
UnbalancedSinkhorn matching with the `uotod` package inside the per-image loop,
then RETURNS THE HUNGARIAN INDICES IN BOTH BRANCHES (the Sinkhorn result only
feeds an unused `indices2` list). `uotod` 0.3 fails to import on our pods
(missing packaged sample asset), so the block is guarded: if the import fails
the loop skips the dead computation and the function returns the same
Hungarian indices it always returned. Idempotent.

    python patch_matcher.py <MOPD>/opdformer/mask2former/modeling/matcher.py
"""
import sys
from pathlib import Path

OLD = "            from uotod.match import BalancedSinkhorn,Hungarian,UnbalancedSinkhorn\n"
NEW = ("            try:\n"
       "                from uotod.match import BalancedSinkhorn,Hungarian,UnbalancedSinkhorn\n"
       "            except Exception:  # SF3D baselines: uotod unusable; its matching is discarded below anyway\n"
       "                matching = torch.zeros(1)\n"
       "                continue\n")

for path in sys.argv[1:]:
    p = Path(path)
    src = p.read_text()
    if NEW in src:
        print(f"already patched: {p}")
        continue
    assert src.count(OLD) == 1, f"expected exactly one occurrence in {p}, found {src.count(OLD)}"
    p.write_text(src.replace(OLD, NEW))
    print(f"patched: {p}")
