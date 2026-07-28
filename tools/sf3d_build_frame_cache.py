"""Build the SF3D frame cache: one LMDB of training-sized frame bytes.

Why this exists (profiled 2026-07-28 on a training pod): every __getitem__
read ~826 KB through the MooseFS FUSE mount (356 KB full-res JPEG + 470 KB
full-res 16-bit depth PNG), and all dataloader workers funnel through the
single FUSE daemon — steady throughput capped at ~73 samples/s whether the
page cache was warm or not, leaving the GPU idle ~75% of the time. This cache
re-encodes each unique frame at the size training actually consumes:

  * RGB: draft-decoded to ~480x360 (the 1/4-DCT scale for 256-target),
    re-encoded JPEG quality 92  (~38 KB)
  * depth: decoded, NEAREST-resized to the training input size on uint16,
    re-encoded lossless PNG      (~43 KB)
  * original (width, height) stored alongside — mask coordinates and point
    normalisation are in original pixels and the small JPEG no longer knows
    them.

~13.4 GB for all 159,845 frames: a single sequentially-warmable, mmap-served
file instead of 320k small files. Depth values are bit-identical to what the
old path produced (same uint16 NEAREST resize before the mm->m cast); RGB
differs only by one JPEG re-encode at quality 92.

Records are keyed by the record's `rgb_image_path` string, so the reader
needs no path parsing. A `__metadata__` entry pins the build parameters; the
reader refuses a cache whose depth size does not match its input size.

Usage (on a pod, ~30-45 min with 64 workers, FUSE-read-bound):
    python tools/sf3d_build_frame_cache.py \
        --data-root /workspace/datasets/sf3d_processed_v2 \
        --out /workspace/datasets/sf3d_processed_v2/frames.lmdb
"""

import argparse
import io
import pickle
import time
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import cv2
import lmdb
import numpy as np
from PIL import Image

_ROOT = None
_DEPTH_SIZE = None
_JPEG_QUALITY = None


def _init(root, depth_size, jpeg_quality):
    global _ROOT, _DEPTH_SIZE, _JPEG_QUALITY
    _ROOT, _DEPTH_SIZE, _JPEG_QUALITY = Path(root), depth_size, jpeg_quality


def _encode_frame(stem):
    try:
        jpg_bytes = (_ROOT / "images" / f"{stem}.jpg").read_bytes()
        png_bytes = (_ROOT / "depth" / f"{stem}.png").read_bytes()

        im = Image.open(io.BytesIO(jpg_bytes))
        orig_size = im.size  # header, before draft
        im.draft("RGB", (_DEPTH_SIZE, _DEPTH_SIZE))
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=_JPEG_QUALITY)

        depth = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.dtype != np.uint16:
            return stem, None, "bad depth decode"
        depth = cv2.resize(
            depth, (_DEPTH_SIZE, _DEPTH_SIZE), interpolation=cv2.INTER_NEAREST
        )
        ok, depth_enc = cv2.imencode(".png", depth)
        if not ok:
            return stem, None, "png encode failed"

        value = pickle.dumps(
            {
                "jpeg": buf.getvalue(),
                "depth_png": depth_enc.tobytes(),
                "orig_size": orig_size,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return stem, value, None
    except Exception as exc:  # noqa: BLE001 — collect, report, keep building
        return stem, None, repr(exc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--depth-size", type=int, default=256)
    ap.add_argument("--jpeg-quality", type=int, default=92)
    args = ap.parse_args()

    root = Path(args.data_root)
    env = lmdb.open(str(root / "data.lmdb"), readonly=True, lock=False)
    stems = set()
    with env.begin() as txn:
        cur = txn.cursor()
        cur.first()
        for k in cur.iternext(keys=True, values=False):
            if k == b"__metadata__":
                continue
            visit, video, ts, _ = k.decode().split("/")
            stems.add(f"{visit}_{video}_{ts}")
    stems = sorted(stems)
    print(f"{len(stems)} unique frames")

    out = lmdb.open(args.out, map_size=40 * 1024**3)
    failures = []
    t0 = time.time()
    with Pool(
        args.workers, initializer=_init,
        initargs=(str(root), args.depth_size, args.jpeg_quality),
    ) as pool:
        txn = out.begin(write=True)
        for i, (stem, value, err) in enumerate(
            pool.imap_unordered(_encode_frame, stems, chunksize=64)
        ):
            if err is not None:
                failures.append((stem, err))
                continue
            txn.put(f"{stem}.jpg".encode(), value)
            if (i + 1) % 2000 == 0:
                txn.commit()
                txn = out.begin(write=True)
                rate = (i + 1) / (time.time() - t0)
                print(
                    f"{i + 1}/{len(stems)}  {rate:.0f} frames/s  "
                    f"eta {(len(stems) - i - 1) / rate / 60:.1f} min",
                    flush=True,
                )
        txn.put(
            b"__metadata__",
            pickle.dumps(
                {
                    "built_from": str(root),
                    "date": date.today().isoformat(),
                    "depth_size": args.depth_size,
                    "jpeg_quality": args.jpeg_quality,
                    "num_frames": len(stems) - len(failures),
                }
            ),
        )
        txn.commit()
    out.sync()
    print(f"done: {len(stems) - len(failures)} frames in {(time.time() - t0) / 60:.1f} min")
    if failures:
        print(f"{len(failures)} FAILURES (first 10): {failures[:10]}")


if __name__ == "__main__":
    main()
