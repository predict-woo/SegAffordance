"""A3VLM eval outputs -> shared prediction JSONL, plus the chained REC -> REG-Joint questions.

Two protocols, both from the same trained model:

* ``chain``  (the language-query protocol): the REC answer to our description
  ("Please provide the 3D bounding box of the region this sentence describes: ...")
  is pasted into their REG-Joint question, whose answer gives type + axis.  The
  mask is the filled 2D hull of the predicted box (so PDet is a box IoU).
* ``gtbox``  (their own REG-Joint protocol): the GT box is given in the question;
  only type + axis are scored (mask = GT box hull -> PDet not meaningful).

Answers are parsed with the converter's regexes and un-normalised with the
per-image sidecar (meta_test.json: pad, K, d_min, d_max).  ``origin_cam`` is the
first axis endpoint for revolute joints and the segment midpoint for prismatic.

  # 1) build the chained joint questions from the REC results
  python a3vlm_preds_to_jsonl.py make-joint --rec-results vqa_logs/<flag>/rec_test.json \
      --rec-questions rec_test.json --out joint_pred_test.json
  # 2) export
  python a3vlm_preds_to_jsonl.py export --joint-results vqa_logs/<flag>/joint_pred_test.json \
      --meta meta_test.json --out preds_chain.jsonl [--rec-results ...]   # rec results give the mask box
  python a3vlm_preds_to_jsonl.py export --joint-results vqa_logs/<flag>/joint_test.json \
      --meta meta_test.json --gt-box-questions joint_test.json --out preds_gtbox.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.baselines_sf3d import common as C  # noqa: E402
from tools.baselines_sf3d.sf3d_to_a3vlm import (  # noqa: E402
    JOINT_INSTRUCT,
    fmt_box,
    parse_axis,
    parse_box,
    unproject_uvd,
    vqa,
)

TYPE_ID = {"revolute": 1, "prismatic": 0}


def question_text(prompt):
    """The eval script stores the full conversation prompt; recover the human turn."""
    if "### Human: " in prompt:
        prompt = prompt.split("### Human: ", 1)[1]
    return prompt.split("\n###")[0].strip()


def results_by_key(results, questions):
    """Eval outputs carry image + question only; join them with the question JSON (which has our key)."""
    lookup = {}
    for q in questions:
        lookup[(q["image"], q["conversations"][0]["value"].strip())] = q["key"]
    out = {}
    n_miss = 0
    for r in results:
        k = r.get("key")  # written by the patched eval script (runpod/baselines/a3vlm/patch_eval.py)
        if k is None:  # unpatched outputs: join on (image, question); ambiguous when two elements share a question
            k = lookup.get((r["image"], question_text(r["question"])))
        if k is None:
            n_miss += 1
            continue
        out[k] = r
    if n_miss:
        print(f"WARNING: {n_miss} results could not be joined to a key", file=sys.stderr)
    return out


def make_joint_questions(rec_results, rec_questions):
    """REC answers -> REG-Joint questions with the predicted box (GT box when the answer is unparsable,
    flagged ``rec_failed`` so the export can mark them unmatched)."""
    by_key = results_by_key(rec_results, rec_questions)
    out = []
    for q in rec_questions:
        key = q["key"]
        r = by_key.get(key)
        box = parse_box(r["answer"]) if r else None
        failed = box is None
        box_str = fmt_box(box) if box is not None else q["conversations"][1]["value"]
        item = vqa(q["image"], JOINT_INSTRUCT.format(REF=box_str), None, key=key)
        item["rec_failed"] = failed
        item["box_pred"] = box_str
        out.append(item)
    return out


def box_mask(box_uvd, meta):
    """Filled convex hull of the 8 projected vertices, in the native frame (H, W)."""
    w, h = meta["wh"]
    x0, y0, s = meta["pad"]
    pts = np.asarray(box_uvd)[:, :2] * s - [x0, y0]
    m = np.zeros((h, w), np.uint8)
    hull = cv2.convexHull(np.round(pts).astype(np.int32).reshape(-1, 1, 2))
    cv2.fillConvexPoly(m, hull, 1)
    return m


def to_prediction(key, joint_answer, box_uvd, meta, matched=True):
    rec = {"key": key, "matched": False, "score": 1.0, "mask_rle": None, "type": None, "axis_cam": None, "origin_cam": None}
    if box_uvd is not None:
        rec["mask_rle"] = C.rle_encode(box_mask(box_uvd, meta))
    parsed = parse_axis(joint_answer) if matched else None
    if parsed is None or parsed[0] not in TYPE_ID:
        return rec
    jt, uvd = parsed
    pts = unproject_uvd(uvd, meta["K"], meta["pad"], meta["d_min"], meta["d_max"])
    d = pts[1] - pts[0]
    n = np.linalg.norm(d)
    if n < 1e-6:
        return rec
    rec["matched"] = True
    rec["type"] = TYPE_ID[jt]
    rec["axis_cam"] = (d / n).tolist()
    rec["origin_cam"] = (pts[0] if jt == "revolute" else pts.mean(0)).tolist()
    return rec


def export(joint_results, joint_questions, metas, rec_results=None, gt_box=False):
    by_key = results_by_key(joint_results, joint_questions)
    rec_by_key = results_by_key(rec_results, joint_questions) if rec_results else {}
    lines = []
    n_ok = 0
    for q in joint_questions:
        key = q["key"]
        meta = metas[Path(q["image"]).name]
        r = by_key.get(key)
        if gt_box:
            box = parse_box(q["conversations"][0]["value"])
        elif q.get("box_pred"):
            box = None if q.get("rec_failed") else parse_box(q["box_pred"])
        else:
            rr = rec_by_key.get(key)
            box = parse_box(rr["answer"]) if rr else None
        matched = r is not None and not q.get("rec_failed", False)
        p = to_prediction(key, r["answer"] if r else "", box, meta, matched=matched)
        n_ok += p["matched"]
        lines.append(p)
    print(f"{len(lines)} predictions, {n_ok} parsed", file=sys.stderr)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-joint")
    m.add_argument("--rec-results", required=True)
    m.add_argument("--rec-questions", required=True)
    m.add_argument("--out", required=True)
    e = sub.add_parser("export")
    e.add_argument("--joint-results", required=True)
    e.add_argument("--joint-questions", required=True, help="the question JSON the results were generated from")
    e.add_argument("--meta", required=True)
    e.add_argument("--rec-results", default=None)
    e.add_argument("--gt-box", action="store_true")
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make-joint":
        out = make_joint_questions(json.load(open(a.rec_results)), json.load(open(a.rec_questions)))
        json.dump(out, open(a.out, "w"))
        print(f"{len(out)} joint questions, {sum(o['rec_failed'] for o in out)} with unparsable REC answers")
    else:
        lines = export(json.load(open(a.joint_results)), json.load(open(a.joint_questions)), json.load(open(a.meta)),
                       rec_results=json.load(open(a.rec_results)) if a.rec_results else None, gt_box=a.gt_box)
        with open(a.out, "w") as f:
            for p in lines:
                f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    main()
