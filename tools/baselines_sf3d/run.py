#!/usr/bin/env python
"""Run a tools/baselines_sf3d script inside a foreign environment.

The upstream baseline envs (USDNet/OPDMulti pods) ship a site-packages
package literally named ``tools`` which shadows this repo's namespace
package, so ``from tools.baselines_sf3d import common`` fails there. This
runner pins ``tools`` to the repo's directory first, then executes the
script as ``__main__``:

    python tools/baselines_sf3d/run.py tools/baselines_sf3d/sf3d_to_opd.py --out ...
"""
import runpy
import sys
import types
from pathlib import Path

root = Path(__file__).resolve().parents[2]
pkg = types.ModuleType("tools")
pkg.__path__ = [str(root / "tools")]
sys.modules["tools"] = pkg
sys.path.insert(0, str(root))
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
