"""Regenerate OPD text descriptions with Codex (app-server) at scale.

Covers OPDReal + OPDMulti, train/valid/test (~63k annotations). For each
annotation it renders the (original, annotated) image pair via
label_render.render_pair, asks gpt-5.6-luna (medium effort) for a
part-identifying, mechanics-free instruction (label_render.PROMPT_TEMPLATE),
validates the sentence, and appends the result to a JSONL checkpoint.

Safe by construction:
  - every finished annotation is appended to results.jsonl immediately;
    re-running the same command skips everything already done (resume);
  - on a rate-limit/usage-limit error the pool stops taking new work,
    drains in-flight calls, checkpoints, and exits 0 with a summary —
    re-run after the limit window resets to continue;
  - Ctrl+C drains the same way.

Usage (on the pod, from the repo root):
  python tools/gen_descriptions.py run --workers 100          # full job
  python tools/gen_descriptions.py run --workers 5 --limit 50 # trial
  python tools/gen_descriptions.py status                     # progress
  python tools/gen_descriptions.py merge                      # write JSONs
"""

import argparse
import collections
import datetime
import json
import os
import queue
import re
import sys
import threading
import time

import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codex_client import CodexClient, CodexError  # noqa: E402
from label_render import (DATASETS, FrameSource, render_pair)  # noqa: E402
from datasets.opd_intrinsics import intrinsic_matrix_from_camera  # noqa: E402

SPLITS = ["train", "valid", "test"]
RESULTS_DEFAULT = "/workspace/desc_gen/results.jsonl"
RENDER_DIR = "/tmp/descgen"
MODEL_DEFAULT = "gpt-5.6-luna"
EFFORT_DEFAULT = "medium"
DESCRIPTION_SOURCE = "codex-gpt-5.6-luna-medium-v1"
CLIENT_RECYCLE_AFTER = 25  # describe() calls per app-server process
# Empty local dir as the codex thread cwd — keeps the app-server from
# indexing/watching the repo on the network volume (heavy background CPU).
CODEX_CWD = "/root/codex-empty-cwd"

RATE_LIMIT_RE = re.compile(
    r"rate.?limit|usage.?limit|quota|429|too many requests|limit reached|"
    r"usage cap|plan limit", re.I)

# Vocabulary that would leak the annotations or the motion mechanics into
# the label. Direction words that can IDENTIFY a part (left/right/top/...)
# are allowed; only unambiguous leaks are banned.
BANNED_RE = re.compile(
    r"highlight|\bgreen\b|\borange\b|\barrows?\b|\bmarkers?\b|outlin|"
    r"annotat|overlay|\bmasks?\b|"
    r"image 1|image 2|first image|second image|red dot|"
    r"rotat|hinge|clockwise|\binwards?\b|\boutwards?\b|toward you|"
    r"\berror\b|unavailable|as an ai|i cannot|i can't|unable to", re.I)


def validate(desc: str) -> str | None:
    """Return a rejection reason, or None if the description is acceptable."""
    if not desc:
        return "empty"
    if "\n" in desc.strip():
        return "multiline"
    words = desc.split()
    if len(words) < 3:
        return "too short"
    if len(desc) > 140:
        return "too long"
    if not desc[0].isalpha():
        return "not a sentence"
    if BANNED_RE.search(desc):
        return f"banned vocabulary: {BANNED_RE.search(desc).group(0)!r}"
    return None


class Source:
    """FrameSource + lock + prompt helper for one (dataset, split)."""

    def __init__(self, dataset: str, split: str):
        self.dataset, self.split = dataset, split
        self.fs = FrameSource(dataset, split)
        self.lock = threading.Lock()

    def jobs(self):
        for image_id, annos in self.fs.by_image.items():
            for anno in annos:
                yield {"dataset": self.dataset, "split": self.split,
                       "image_id": image_id, "anno_id": anno["id"]}

    def render(self, image_id, anno_id, out_dir):
        with self.lock:
            annos = self.fs.by_image[image_id]
            target = next(a for a in annos if a["id"] == anno_id)
            im = self.fs.img_by_id[image_id]
            K = intrinsic_matrix_from_camera(im, self.fs.is_multi)
            bgr = self.fs.load_bgr(image_id)
            prompt = self.fs.prompt_for(target)
        orig, ann = render_pair(bgr, annos, target, K)
        stem = f"{self.dataset}_{self.split}_{anno_id}"
        p_orig = os.path.join(out_dir, stem + "_orig.jpg")
        p_ann = os.path.join(out_dir, stem + "_ann.jpg")
        cv2.imwrite(p_orig, orig)
        cv2.imwrite(p_ann, ann)
        return prompt, p_orig, p_ann


def result_key(r):
    return (r["dataset"], r["split"], r["anno_id"])


def load_results(path):
    """Last occurrence wins; only entries with a description count as done."""
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[result_key(r)] = r
    return done


class Driver:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()
        self.stop_reason = None
        self.results_path = args.results
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        os.makedirs(RENDER_DIR, exist_ok=True)
        os.makedirs(CODEX_CWD, exist_ok=True)
        self.write_lock = threading.Lock()
        self.stats = collections.Counter()
        self.tokens = collections.Counter()
        self.t0 = time.time()

    def record(self, rec):
        with self.write_lock:
            with open(self.results_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()

    def signal_stop(self, reason):
        if not self.stop.is_set():
            self.stop_reason = reason
            self.stop.set()
            print(f"\n*** stopping gracefully: {reason}\n"
                  "    in-flight calls will finish; progress is checkpointed; "
                  "re-run the same command to resume.", flush=True)

    def worker(self, wid, jobs_q, sources):
        client = None
        calls = 0
        while not self.stop.is_set():
            try:
                job = jobs_q.get_nowait()
            except queue.Empty:
                break
            try:
                src = sources[(job["dataset"], job["split"])]
                prompt, p_orig, p_ann = src.render(
                    job["image_id"], job["anno_id"], RENDER_DIR)
            except Exception as e:  # bad annotation/image — record and move on
                self.record({**job, "description": None, "valid": False,
                             "error": f"render: {e}"})
                self.stats["render_error"] += 1
                continue

            desc, reject, error, rejected_text = None, None, None, None
            for attempt in range(3):
                if self.stop.is_set() and attempt > 0:
                    break
                try:
                    if client is None:
                        client = CodexClient(model=self.args.model,
                                             effort=self.args.effort,
                                             cwd=CODEX_CWD)
                        calls = 0
                    elif calls > 0:
                        client.new_thread()
                    calls += 1
                    out = client.describe(prompt, images=[p_orig, p_ann])
                    usage = client.last_usage or {}
                    self.tokens["input"] += usage.get("inputTokens", 0)
                    self.tokens["cached"] += usage.get("cachedInputTokens", 0)
                    self.tokens["output"] += usage.get("outputTokens", 0)
                    candidate = " ".join(out.strip().split())
                    reject = validate(candidate)
                    if reject is None:
                        desc = candidate
                        break
                    rejected_text = candidate
                    self.stats["validation_retry"] += 1
                except Exception as e:  # CodexError, pipe/OS errors — never
                    error = str(e)      # let a worker thread die

                    try:
                        client.close()
                    except Exception:
                        pass
                    client = None
                    if RATE_LIMIT_RE.search(error):
                        self.signal_stop(f"rate/usage limit: {error[:200]}")
                        break
                    time.sleep(2 * (attempt + 1))
            # Recycle the app-server process periodically — its RSS/CPU grows
            # with thread count. (Must live outside the retry loop: the
            # success path breaks out of it.)
            if client is not None and calls >= CLIENT_RECYCLE_AFTER:
                client.close()
                client = None

            for p in (p_orig, p_ann):
                try:
                    os.remove(p)
                except OSError:
                    pass

            if desc is not None:
                self.record({**job, "description": desc, "valid": True})
                self.stats["ok"] += 1
            elif error and RATE_LIMIT_RE.search(error):
                # not recorded: the job stays pending for the resume run
                self.stats["deferred"] += 1
            else:
                self.record({**job, "description": None, "valid": False,
                             "error": reject or error,
                             "rejected_text": rejected_text})
                self.stats["failed"] += 1

            n = self.stats["ok"] + self.stats["failed"]
            if n and n % 25 == 0:
                rate = n / (time.time() - self.t0)
                print(f"[{n}] ok={self.stats['ok']} failed={self.stats['failed']} "
                      f"{rate:.2f}/s in={self.tokens['input']//1000}k "
                      f"(cached {self.tokens['cached']//1000}k) "
                      f"out={self.tokens['output']//1000}k", flush=True)
        if client is not None:
            client.close()

    def run(self):
        args = self.args
        datasets = args.datasets.split(",")
        splits = args.splits.split(",")
        done = load_results(self.results_path)
        if args.retry_failed:
            n_before = len(done)
            done = {k: r for k, r in done.items() if r.get("valid")}
            print(f"--retry-failed: re-queuing {n_before - len(done)} failed results")
        print(f"checkpoint: {len(done)} results already recorded")

        sources = {}
        pending = []
        for ds in datasets:
            for sp in splits:
                src = Source(ds, sp)
                sources[(ds, sp)] = src
                for job in src.jobs():
                    if result_key(job) not in done:
                        pending.append(job)
        if args.limit:
            pending = pending[: args.limit]
        print(f"pending: {len(pending)} annotations "
              f"({args.workers} workers, model={args.model}, effort={args.effort})")
        if not pending:
            return

        jobs_q = queue.Queue()
        for j in pending:
            jobs_q.put(j)
        threads = [threading.Thread(target=self.worker,
                                    args=(i, jobs_q, sources), daemon=True)
                   for i in range(args.workers)]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(1)
        except KeyboardInterrupt:
            self.signal_stop("interrupted by user")
        for t in threads:
            t.join()

        dt = time.time() - self.t0
        print(f"\ndone in {dt/60:.1f} min: {dict(self.stats)}")
        print(f"tokens: in={self.tokens['input']:,} "
              f"(cached {self.tokens['cached']:,}) out={self.tokens['output']:,}")
        if self.stop_reason:
            print(f"stopped early ({self.stop_reason}) — re-run to resume.")


def cmd_status(args):
    done = load_results(args.results)
    counts = collections.Counter()
    valid = collections.Counter()
    for r in done.values():
        k = f"{r['dataset']}/{r['split']}"
        counts[k] += 1
        if r.get("valid"):
            valid[k] += 1
    total_expected = 0
    for ds in args.datasets.split(","):
        for sp in args.splits.split(","):
            path = os.path.join(DATASETS[ds]["root"], "annotations_bwdf",
                                f"MotionNet_{sp}.json")
            with open(path) as f:
                n = len(json.load(f)["annotations"])
            total_expected += n
            k = f"{ds}/{sp}"
            print(f"{k:16s} {counts[k]:7d} done ({valid[k]} valid) / {n}")
    print(f"{'TOTAL':16s} {len(done):7d} / {total_expected}")


COLOR_HUES = {"green": (35, 85), "orange": (8, 25)}  # OpenCV H in [0,180)


def cmd_salvage(args):
    """Accept color-banned rejections when the target part really is that color.

    The validator bans "green"/"orange" to prevent overlay leakage, but
    OPDMulti homes contain genuinely green/orange furniture. For each failed
    result whose rejected_text is clean apart from a color word, check the
    actual hue of the target mask's pixels in the ORIGINAL image; if the part
    is that color, append the description as valid (last-wins on merge)."""
    import numpy as np
    from label_render import decode_mask

    done = load_results(args.results)
    sources = {}
    salvaged, rejected = 0, 0
    with open(args.results, "a") as out_f:
        for r in done.values():
            text = r.get("rejected_text")
            if r.get("valid") or not text:
                continue
            colors = [c for c in COLOR_HUES if re.search(rf"\b{c}\b", text, re.I)]
            if not colors:
                continue
            # must be valid once the color words are neutralized
            neutral = re.sub(r"\b(green|orange)\b", "gray", text, flags=re.I)
            if validate(neutral) is not None:
                continue
            skey = (r["dataset"], r["split"])
            if skey not in sources:
                sources[skey] = Source(*skey)
            src = sources[skey]
            try:
                annos = src.fs.by_image[r["image_id"]]
                target = next(a for a in annos if a["id"] == r["anno_id"])
                im = src.fs.img_by_id[r["image_id"]]
                bgr = src.fs.load_bgr(r["image_id"])
                mask = decode_mask(target["segmentation"], im["height"], im["width"])
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                sel = mask > 0
                if not sel.any():
                    rejected += 1
                    continue
                h = hsv[..., 0][sel].astype(int)
                s = hsv[..., 1][sel].astype(int)
                v = hsv[..., 2][sel].astype(int)
                ok = True
                for c in colors:
                    lo, hi = COLOR_HUES[c]
                    frac = float(np.mean((h >= lo) & (h <= hi) & (s > 60) & (v > 40)))
                    if frac < args.min_color_frac:
                        ok = False
                if ok:
                    rec = {k: r[k] for k in ("dataset", "split", "image_id", "anno_id")}
                    rec.update(description=" ".join(text.split()), valid=True,
                               salvage=f"color-verified:{','.join(colors)}")
                    out_f.write(json.dumps(rec) + "\n")
                    salvaged += 1
                else:
                    rejected += 1
            except Exception:
                rejected += 1
    print(f"salvaged {salvaged} color-banned descriptions; "
          f"{rejected} did not pass the pixel-hue check")


def cmd_merge(args):
    done = load_results(args.results)
    by_file = collections.defaultdict(dict)
    invalid = collections.Counter()
    for r in done.values():
        if r.get("valid") and r.get("description"):
            by_file[(r["dataset"], r["split"])][r["anno_id"]] = r["description"]
        else:
            invalid[(r["dataset"], r["split"])] += 1

    for ds in args.datasets.split(","):
        for sp in args.splits.split(","):
            path = os.path.join(DATASETS[ds]["root"], "annotations_bwdf",
                                f"MotionNet_{sp}.json")
            with open(path) as f:
                data = json.load(f)
            descs = by_file.get((ds, sp), {})
            missing = [a["id"] for a in data["annotations"] if a["id"] not in descs]
            cover = 1 - len(missing) / max(1, len(data["annotations"]))
            print(f"{ds}/{sp}: {len(descs)} descriptions, "
                  f"{len(missing)} missing ({cover:.1%} coverage), "
                  f"{invalid[(ds, sp)]} invalid")
            if missing and not args.partial:
                print("  -> skipped (use --partial to merge incomplete files)")
                continue
            for a in data["annotations"]:
                if a["id"] in descs:
                    a["description"] = descs[a["id"]]
            data.setdefault("info", {})
            data["info"]["description_source"] = DESCRIPTION_SOURCE
            data["info"]["description_date"] = datetime.date.today().isoformat()
            with open(path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"  -> wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--results", default=RESULTS_DEFAULT)
    common.add_argument("--datasets", default="opdreal,opdmulti")
    common.add_argument("--splits", default=",".join(SPLITS))

    runp = sub.add_parser("run", parents=[common])
    runp.add_argument("--workers", type=int, default=25)
    runp.add_argument("--limit", type=int, default=None)
    runp.add_argument("--model", default=MODEL_DEFAULT)
    runp.add_argument("--effort", default=EFFORT_DEFAULT)
    runp.add_argument("--retry-failed", action="store_true",
                      help="re-attempt annotations recorded as valid=false")

    sub.add_parser("status", parents=[common])
    mergep = sub.add_parser("merge", parents=[common])
    mergep.add_argument("--partial", action="store_true")
    salv = sub.add_parser("salvage", parents=[common])
    salv.add_argument("--min-color-frac", type=float, default=0.25,
                      help="min fraction of mask pixels matching the color")

    args = ap.parse_args()
    if args.cmd == "run":
        Driver(args).run()
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "merge":
        cmd_merge(args)
    elif args.cmd == "salvage":
        cmd_salvage(args)


if __name__ == "__main__":
    main()
