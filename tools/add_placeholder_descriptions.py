"""
Add PLACEHOLDER natural-language descriptions to OPD annotation JSONs.

The SegAffordance training pipeline is language-grounded: every annotation
needs a `description` field, which the published OPD datasets do not have.
The original (Euler-era) descriptions and their generation pipeline were
lost — until they are regenerated, this script fills the field with simple
template sentences derived from the part category and motion type, so the
training/eval code paths can run end to end.

Style is modeled on the surviving real examples in
`.old-dont-run/MotionNet_test.json` (e.g. "Open the bottom drawer.",
"Pull the top left drawer straight out."), minus the spatial qualifiers.

Every processed file gets `info.description_source = "placeholder-template-v1"`
so placeholder files are identifiable. Re-running is idempotent: real
descriptions (any file without that marker) are never overwritten unless
--force is given.

Usage:
  python tools/add_placeholder_descriptions.py <annotation.json> [more.json ...]
"""

import argparse
import json
import random

MARKER = "placeholder-template-v1"

TRANS_TEMPLATES = [
    "Pull out the {cat}.",
    "Push in the {cat}.",
    "Slide the {cat} open.",
    "Pull the {cat} straight out.",
]
ROT_TEMPLATES = [
    "Open the {cat}.",
    "Close the {cat}.",
    "Swing the {cat} open.",
    "Pull the {cat} open.",
]
FALLBACK_TEMPLATES = [
    "Open the {cat}.",
    "Move the {cat}.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_files", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing descriptions even if the file is not marked as placeholder.",
    )
    args = parser.parse_args()

    for path in args.json_files:
        with open(path) as f:
            data = json.load(f)

        info = data.get("info") or {}
        already_marked = info.get("description_source") == MARKER
        has_descriptions = any("description" in a for a in data["annotations"][:50])
        if has_descriptions and not already_marked and not args.force:
            print(f"SKIP {path}: has descriptions not made by this script (use --force to overwrite)")
            continue

        cat_by_id = {c["id"]: c["name"] for c in data.get("categories", [])}
        rng = random.Random(args.seed)

        n = 0
        for anno in data["annotations"]:
            cat = cat_by_id.get(anno.get("category_id"), "part").replace("_", " ")
            motion = anno.get("motion") or {}
            mtype = (motion.get("type") or motion.get("motion_type") or "").lower()
            if mtype.startswith("trans"):
                tpl = rng.choice(TRANS_TEMPLATES)
            elif mtype.startswith("rot"):
                tpl = rng.choice(ROT_TEMPLATES)
            else:
                tpl = rng.choice(FALLBACK_TEMPLATES)
            anno["description"] = tpl.format(cat=cat)
            n += 1

        info["description_source"] = MARKER
        data["info"] = info
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"OK {path}: {n} placeholder descriptions written")


if __name__ == "__main__":
    main()
