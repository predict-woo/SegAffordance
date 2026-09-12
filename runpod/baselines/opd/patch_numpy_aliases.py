"""Fit-to-env patch for OPDMulti/MOPD: numpy >= 1.24 removed np.float/np.int/np.bool.

Their evaluator (mask2former/evaluation/motion_coco_eval.py, `accumulate`) uses
`dtype=np.float`, which crashes every validation pass under a modern numpy.
Replace the removed aliases (word-bounded, so np.float32 etc. are untouched)
in every .py file under the given directories. Idempotent.

    python patch_numpy_aliases.py <repo>/opdformer/mask2former
"""
import re
import sys
from pathlib import Path

PAT = re.compile(r"\bnp\.(float|int|bool|object|str)\b(?!_)")
for root in sys.argv[1:]:
    n = 0
    for p in Path(root).rglob("*.py"):
        src = p.read_text()
        new = PAT.sub(lambda m: m.group(1), src)
        if new != src:
            p.write_text(new)
            n += 1
            print(f"patched: {p}")
    print(f"{root}: {n} files patched")
