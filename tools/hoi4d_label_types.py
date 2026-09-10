"""Rule-based rot/trans/none labels for the HOI4D 2D LMDB (v2, full package).

The v2 records carry a placeholder motion_type ("trans" for all 3,084). The
labels follow the user's rules (2026-09-11):
  rot  : trash can (C14) open/close, safe (C6), lamp (C17) open/close, laptop (C3),
         storage furniture (C4) whose description names a door/cabinet/closet
  trans: storage furniture (C4) whose description names a drawer (or the one
         "front panel"), toy car (C1) push/pull, every Press / switch event
  none : dump (bucket C8, kettle C12, mug C2, bottle C5), stapler C18, pliers C11,
         scissors C9 — no articulated part; kept for masks/points only.
"none" records are excluded from the type CE and the trajectory losses by the
loader (motion_type "none").

  python tools/hoi4d_label_types.py --lmdb /workspace/datasets/hoi4d_processed_2d_v2 [--dry-run]
Writes motion_type + motion_type_source ("rules-2026-09-11-v1") into every record after
backing up data.lmdb -> data.lmdb.bak_pre_rule_types.
"""
import argparse
import pickle
import re
import shutil
from collections import Counter
from pathlib import Path

NONE_CATS = {"C8", "C12", "C2", "C5", "C18", "C11", "C9"}
ROT_CATS = {"C6", "C3"}


def label(rec) -> str:
    h = rec["hoi4d"]; cat, ev = h["category"], (h.get("event") or "").lower()
    desc = (rec.get("description") or "").lower()
    if "press" in ev or "switch" in ev:
        return "trans"
    if cat in NONE_CATS:
        return "none"
    if cat in ROT_CATS:
        return "rot"
    if cat == "C1":
        return "trans"
    if cat in ("C14", "C17"):
        return "rot"                      # open / close of a lid or a folding arm
    if cat == "C4":
        if re.search(r"drawer|front panel", desc):
            return "trans"
        if re.search(r"door|cabinet|cupboard|closet|wardrobe", desc):
            return "rot"
        raise ValueError(f"C4 description not resolvable: {desc!r}")
    raise ValueError(f"unhandled category {cat} event {ev!r}")


def main():
    import lmdb
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb", default="/workspace/datasets/hoi4d_processed_2d_v2")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src = Path(a.lmdb) / "data.lmdb"
    env = lmdb.open(str(src), readonly=True, lock=False)
    plan = {}; c = Counter(); by = Counter()
    with env.begin() as t:
        for k, v in t.cursor():
            rec = pickle.loads(v); lab = label(rec)
            plan[k] = lab; c[lab] += 1; by[(rec["hoi4d"]["category"], rec["hoi4d"]["event"], lab)] += 1
    env.close()
    print("totals:", dict(c))
    for (cat, ev, lab), n in sorted(by.items(), key=lambda x: -x[1]):
        print(f"  {cat:4s} {ev:22s} -> {lab:5s} {n}")
    if a.dry_run:
        return
    bak = Path(a.lmdb) / "data.lmdb.bak_pre_rule_types"
    if not bak.exists():
        shutil.copytree(src, bak); print(f"backup -> {bak}")
    env = lmdb.open(str(src), map_size=1 << 36)
    with env.begin(write=True) as t:
        for k, v in list(t.cursor()):
            rec = pickle.loads(v)
            rec["motion_info"]["original_motion_data"]["motion_type"] = plan[k]
            rec["motion_info"]["original_motion_data"]["motion_type_source"] = "rules-2026-09-11-v1"
            t.put(k, pickle.dumps(rec, protocol=4))
    print(f"applied {len(plan)} labels")


if __name__ == "__main__":
    main()
