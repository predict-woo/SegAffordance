"""Chunked, resumable download of one EPIC-KITCHENS video from data.bris.

data.bris serves ~2.7 MB/s per connection but honours byte ranges, and 16
parallel ranges measured 36 MB/s from EU-RO-1 (2026-09-07). Downloads the
remainder of the file after any existing prefix as N parallel range parts,
then appends them in order. Usage:
    python tools/epic_fetch_video.py P01_09 /workspace/datasets/epic_videos [--chunks 16]
EPIC-55 ids (2-digit) come from the original-sequences DOI, extension ids
(3-digit) from the EPIC-100 extension DOI.
"""
import argparse, os, subprocess, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

D55 = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/train"
DEXT = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"

D55_TEST = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/test"

def candidates(vid):
    p, n = vid.split("_")
    if len(n) == 3: return [f"{DEXT}/{p}/videos/{vid}.MP4"]
    return [f"{D55}/{p}/{vid}.MP4", f"{D55_TEST}/{p}/{vid}.MP4"]  # EPIC-55 test-split videos live under videos/test/

def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r: return int(r.headers["Content-Length"])
    except urllib.error.HTTPError as e:
        if e.code == 404: return None
        raise

def url_for(vid):
    for u in candidates(vid):
        if head_size(u) is not None: return u
    raise FileNotFoundError(f"{vid}: not found at any of {candidates(vid)}")

def total_size(url): return head_size(url)

def fetch_range(url, a, b, path, tries=5):
    for _ in range(tries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= b - a + 1: return True
        rc = subprocess.run(["curl", "-s", "-m", "3600", "-r", f"{a + have}-{b}", "-o", path + ".part", url]).returncode
        if rc == 0 and os.path.exists(path + ".part"):
            with open(path, "ab") as f, open(path + ".part", "rb") as g: f.write(g.read())
            os.remove(path + ".part")
    return os.path.getsize(path) == b - a + 1

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("vid"); ap.add_argument("out_dir"); ap.add_argument("--chunks", type=int, default=16); a = ap.parse_args()
    url = url_for(a.vid); out = os.path.join(a.out_dir, f"{a.vid}.MP4"); os.makedirs(a.out_dir, exist_ok=True)
    total = total_size(url); have = os.path.getsize(out) if os.path.exists(out) else 0
    if have >= total: print(f"{a.vid}: already complete ({total} bytes)"); return
    print(f"{a.vid}: {total/2**30:.2f} GB total, have {have/2**30:.2f} GB, fetching the rest in {a.chunks} ranges", flush=True)
    rem = total - have; step = (rem + a.chunks - 1) // a.chunks
    parts = [(have + i * step, min(have + (i + 1) * step, total) - 1, f"{out}.r{i:02d}") for i in range(a.chunks) if have + i * step < total]
    with ThreadPoolExecutor(a.chunks) as ex: ok = list(ex.map(lambda p: fetch_range(url, *p), parts))
    if not all(ok): print("FAILED parts:", [p[2] for p, o in zip(parts, ok) if not o]); sys.exit(1)
    with open(out, "ab") as f:
        for _, _, path in parts:
            with open(path, "rb") as g:
                while True:
                    buf = g.read(1 << 24)
                    if not buf: break
                    f.write(buf)
            os.remove(path)
    final = os.path.getsize(out); print(f"{a.vid}: done, {final} bytes ({'OK' if final == total else 'SIZE MISMATCH'})")
    sys.exit(0 if final == total else 1)

if __name__ == "__main__": main()
