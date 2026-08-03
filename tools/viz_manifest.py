"""Auto-provenance for viz/ batches.

Every tool that renders images into a `viz/YYYYMMDD_<subject>_<what>/`
batch calls :func:`write_manifest` on its output directory, so the batch
records how to reproduce itself even before a human writes the README.
The manifest is tracked in git (see .gitignore); the images are not.
"""

import datetime
import os
import subprocess
import sys

import yaml


def write_manifest(out_dir, **extra):
    """Write ``<out_dir>/manifest.yaml``: argv, git commit, timestamp, extras."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ).stdout.strip() or None
    except OSError:
        commit = None
    manifest = {
        "command": " ".join(sys.argv),
        "tool_commit": commit,
        "generated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    with open(os.path.join(out_dir, "manifest.yaml"), "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
