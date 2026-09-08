"""Fetch ONLY the egocentric view (view 0) of ARCTIC image zips via HTTP ranges.

ARCTIC ships one zip per sequence with all 10 views (~2.6 GB, 8k entries;
the ego view is ~1/9 of it). The MPI download server honours byte ranges
after a cookie login, so: read the zip's central directory remotely, pick the
entries under `<seq>/0/`, fetch them in a few large contiguous ranges, and
write the JPEGs to <out>/<subject>/<seq>/0/<frame>.jpg. Resumable (skips
sequences whose frame count already matches).
Credentials: ARCTIC_USERNAME / ARCTIC_PASSWORD in the environment (never in
the repo). Usage:
    python tools/arctic_fetch_ego.py --seqs s01_box_use_01,... --out /workspace/datasets/arctic/images [--view 0]
    python tools/arctic_fetch_ego.py --list /workspace/datasets/arctic_gt_package/arctic_sample_list.csv --out ... --workers 6
"""
import argparse, csv, io, os, struct, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
import requests, urllib3
urllib3.disable_warnings()
BASE = ("https://download.is.tue.mpg.de/download.php?domain=arctic&resume=1&sfile=arctic_release/"
        "c7216c3b205186106a1f8326ed7b948f838e4907e69b21c8b3c87bb69d87206e/v1_0/data/images_zips")

def session():
    s = requests.Session(); s.verify = False
    u, p = os.environ["ARCTIC_USERNAME"], os.environ["ARCTIC_PASSWORD"]
    r = s.post(f"{BASE}/../meta.zip", data={"username": u, "password": p}, allow_redirects=True, stream=True, timeout=60)
    r.close()
    if not s.cookies: raise RuntimeError("login gave no session cookie — credentials?")
    return s

def get_range(s, url, a, b, tries=5):
    for t in range(tries):
        try:
            r = s.get(url, headers={"Range": f"bytes={a}-{b}"}, timeout=600, stream=True)
            if r.status_code == 206:
                data = r.content
                if len(data) == b - a + 1: return data
            elif r.status_code == 401: raise RuntimeError("401 — login lost")
        except (requests.RequestException,) as e:
            err = e
        time.sleep(2 * (t + 1))
    raise RuntimeError(f"range {a}-{b} failed after {tries} tries")

def central_directory(s, url):
    total = None
    for t in range(6):  # the server occasionally answers the first ranged GET with a 302/200 — retry
        head = s.get(url, headers={"Range": "bytes=0-0"}, timeout=60, stream=True); head.close()
        if head.status_code == 206 and "Content-Range" in head.headers: total = int(head.headers["Content-Range"].split("/")[1]); break
        if head.status_code == 200 and "Content-Length" in head.headers and int(head.headers["Content-Length"]) > 1 << 20:
            total = int(head.headers["Content-Length"]); break
        time.sleep(2 * (t + 1))
    if total is None: raise RuntimeError(f"size probe failed: http {head.status_code} {dict(head.headers)}")
    tail = get_range(s, url, max(0, total - 65536), total - 1)
    i = tail.rfind(b"PK\x05\x06"); assert i >= 0, "no EOCD"
    n, cd_size, cd_off = struct.unpack("<H", tail[i + 10:i + 12])[0], *struct.unpack("<II", tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF or n == 0xFFFF or cd_size == 0xFFFFFFFF:  # zip64
        j = tail.rfind(b"PK\x06\x06"); assert j >= 0, "zip64 EOCD missing"
        n = struct.unpack("<Q", tail[j + 32:j + 40])[0]; cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = get_range(s, url, cd_off, cd_off + cd_size - 1)
    entries = []; p = 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        (method, csize, usize, nlen, xlen, clen, off) = (struct.unpack("<H", cd[p + 10:p + 12])[0], *struct.unpack("<II", cd[p + 20:p + 28]),
                                                        *struct.unpack("<HHH", cd[p + 28:p + 34]), struct.unpack("<I", cd[p + 42:p + 46])[0])
        name = cd[p + 46:p + 46 + nlen].decode(); extra = cd[p + 46 + nlen:p + 46 + nlen + xlen]
        if 0xFFFFFFFF in (csize, usize, off):  # zip64 extra field
            q = 0
            while q + 4 <= len(extra):
                hid, hl = struct.unpack("<HH", extra[q:q + 4]); body = extra[q + 4:q + 4 + hl]
                if hid == 1:
                    vals = list(struct.unpack("<" + "Q" * (len(body) // 8), body[: (len(body) // 8) * 8]))
                    if usize == 0xFFFFFFFF: usize = vals.pop(0)
                    if csize == 0xFFFFFFFF: csize = vals.pop(0)
                    if off == 0xFFFFFFFF: off = vals.pop(0)
                q += 4 + hl
        entries.append((name, method, csize, usize, off)); p += 46 + nlen + xlen + clen
    return entries, total

def fetch_seq(s, seq, out, view, gap=2 << 20):
    sid, name = seq.split("_", 1); url = f"{BASE}/{sid}/{name}.zip"
    dst = f"{out}/{sid}/{name}/{view}"; os.makedirs(dst, exist_ok=True)
    entries, total = central_directory(s, url)
    # entries are named "<view>/<frame>.jpg" (no sequence prefix), deflate-compressed, NOT sorted by view
    want = sorted([e for e in entries if e[0].startswith(f"{view}/") and e[0].lower().endswith(".jpg")], key=lambda e: e[4])
    if not want: return f"{seq}: NO view-{view} entries (of {len(entries)})"
    have = len([f for f in os.listdir(dst) if f.endswith(".jpg")])
    if have >= len(want): return f"{seq}: already complete ({have} frames)"
    # group into contiguous byte runs (local header .. data end), fetch each run once
    runs = []; cur = None
    for e in want:
        a = e[4]; b = e[4] + 30 + len(e[0].encode()) + 65536 + e[2]  # upper bound; trimmed by the next entry's offset
        if cur and a - cur[1] < gap: cur[1] = max(cur[1], b); cur[2].append(e)
        else:
            if cur: runs.append(cur)
            cur = [a, b, [e]]
    runs.append(cur)
    written = 0; t0 = time.time(); nbytes = 0
    for a, b, es in runs:
        b = min(b, total - 1); blob = get_range(s, url, a, b); nbytes += len(blob)
        for name_, method, csize, usize, off in es:
            p = off - a; assert blob[p:p + 4] == b"PK\x03\x04", (seq, name_)
            nlen, xlen = struct.unpack("<HH", blob[p + 26:p + 30]); d = p + 30 + nlen + xlen
            raw = blob[d:d + csize]
            data = raw if method == 0 else zlib.decompress(raw, -15)
            assert len(data) == usize, (seq, name_, len(data), usize)
            open(f"{dst}/{os.path.basename(name_)}", "wb").write(data); written += 1
    return f"{seq}: {written} frames, {nbytes / 2**20:.0f} MB in {time.time() - t0:.0f}s ({len(runs)} ranges)"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seqs", default=None); ap.add_argument("--list", default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--view", default="0"); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    seqs = a.seqs.split(",") if a.seqs else [r["sequence_id"] for r in csv.DictReader(open(a.list))]
    s = session()
    def job(seq):
        try: return fetch_seq(s, seq, a.out, a.view)
        except Exception as e: return f"{seq}: FAILED {type(e).__name__}: {e}"
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(job, seqs): print(msg, flush=True)
    print("ARCTIC_FETCH_DONE")

if __name__ == "__main__": main()
