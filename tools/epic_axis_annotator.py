"""Manual ground-truth 3D articulation axes for the EPIC-KITCHENS test records (stdlib + numpy only).

Serves the point clouds written by tools/epic_cloud_precompute.py to a single-page three.js explorer
(tools/epic_axis_annotator.html) in which two clicked points define the axis, and exports the saved
annotations in the schema of experiments/hoi4d_test_gt_articulation.json.

  # on the pod (detached), then `ssh -N -L 8080:localhost:8080 segaff-dev` and open http://localhost:8080
  python tools/epic_axis_annotator.py serve --clouds /workspace/datasets/epic_gt_annot/clouds \\
      --out /workspace/datasets/epic_gt_annot/annotations --port 8080
  python tools/epic_axis_annotator.py export --clouds ... --out ... --json experiments/epic_test_gt_articulation.json

API: GET /api/records (list + annotation status), GET /api/cloud/<i> (binary: uint32 meta length, meta
JSON, float32 xyz, uint8 rgb, uint8 in_mask), POST /api/annot/<i> ({p1, p2, type, note} -> <out>/<key>.json).
Annotations and the export are in the camera frame of the cloud (OpenCV: x right, y down, z forward, m):
axis_cam = normalised p2 - p1, origin_cam = p1. Records without an annotation get valid=false.
"""
import argparse
import glob
import json
import os
import struct
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epic_axis_annotator.html")


def safe_name(key):
    return key.replace("/", "__")


def list_records(clouds_dir):
    """Test-order record list from index.json (written by precompute) or, failing that, from the npz files."""
    idx = os.path.join(clouds_dir, "index.json")
    if os.path.exists(idx):
        return json.load(open(idx))["records"]
    recs = []
    for i, fn in enumerate(sorted(glob.glob(os.path.join(clouds_dir, "*.npz")))):
        z = np.load(fn)
        recs.append(dict(i=i, key=str(z["key"]), file=os.path.basename(fn), desc=str(z["desc"]),
                         type_label=int(z["type_label"]), n_points=int(z["xyz"].shape[0]), size=z["size"].tolist()))
    return recs


def load_annotations(out_dir, records):
    ann = {}
    for r in records:
        fn = os.path.join(out_dir, safe_name(r["key"]) + ".json")
        if os.path.exists(fn):
            ann[r["key"]] = json.load(open(fn))
    return ann


def build_export(records, annots):
    """{key: {valid, type, axis_cam, origin_cam, ...}} — same shape as hoi4d_test_gt_articulation.json."""
    out = {}
    for r in records:
        a = annots.get(r["key"])
        base = dict(seq=r["key"].split("/")[0], desc=r.get("desc", ""), type_label=r.get("type_label"),
                    source="manual:tools/epic_axis_annotator.py")
        if a is None:
            out[r["key"]] = dict(valid=False, type="rot" if r.get("type_label") == 1 else "trans", **base)
            continue
        p1, p2 = np.asarray(a["p1"], np.float64), np.asarray(a["p2"], np.float64)
        d = p2 - p1
        n = float(np.linalg.norm(d))
        out[r["key"]] = dict(valid=n > 1e-6, type=a["type"], axis_cam=(d / max(n, 1e-12)).tolist(), origin_cam=p1.tolist(),
                             p1_cam=p1.tolist(), p2_cam=p2.tolist(), axis_len_m=n, note=a.get("note", ""),
                             saved_at=a.get("saved_at"), **base)
    return out


def encode_cloud(npz_path, rec):
    z = np.load(npz_path)
    xyz = np.ascontiguousarray(z["xyz"], np.float32)
    rgb = np.ascontiguousarray(z["rgb"], np.uint8)
    msk = np.ascontiguousarray(z["in_mask"], np.uint8)
    meta = dict(i=rec["i"], key=str(z["key"]), desc=str(z["desc"]), type_label=int(z["type_label"]), n=int(xyz.shape[0]),
                point_uv=z["point_uv"].tolist(), K_render=z["K_render"].tolist(), size=z["size"].tolist())
    mb = json.dumps(meta).encode()
    return struct.pack("<I", len(mb)) + mb + xyz.tobytes() + rgb.tobytes() + msk.tobytes()


def make_handler(clouds_dir, out_dir):
    records = list_records(clouds_dir)
    cache = {}

    class H(BaseHTTPRequestHandler):
        def _send(self, body, ctype="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/index.html"):
                return self._send(open(HTML, "rb").read(), "text/html; charset=utf-8")
            if p == "/api/records":
                ann = load_annotations(out_dir, records)
                rows = [dict(r, annot=ann.get(r["key"]), annotated=r["key"] in ann) for r in records]
                return self._send(json.dumps(rows).encode())
            if p.startswith("/api/cloud/"):
                i = int(p.rsplit("/", 1)[1])
                if not 0 <= i < len(records):
                    return self.send_error(404)
                if i not in cache:
                    cache[i] = encode_cloud(os.path.join(clouds_dir, records[i]["file"]), records[i])
                return self._send(cache[i], "application/octet-stream")
            self.send_error(404)

        def do_POST(self):
            p = self.path.split("?")[0]
            if not p.startswith("/api/annot/"):
                return self.send_error(404)
            i = int(p.rsplit("/", 1)[1])
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            a = dict(key=records[i]["key"], p1=[float(v) for v in body["p1"]], p2=[float(v) for v in body["p2"]],
                     type=body.get("type", "rot"), note=body.get("note", ""), saved_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, safe_name(a["key"]) + ".json"), "w") as f:
                json.dump(a, f, indent=1)
            self._send(json.dumps(dict(ok=True, annot=a)).encode())

        def log_message(self, fmt, *args):  # quiet: one line per request is noise over the ssh tunnel
            pass

    return H, records


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--clouds", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--port", type=int, default=8080)
    e = sub.add_parser("export")
    e.add_argument("--clouds", default="/workspace/datasets/epic_gt_annot/clouds")
    e.add_argument("--out", required=True)
    e.add_argument("--json", default="experiments/epic_test_gt_articulation.json")
    a = ap.parse_args()
    if a.cmd == "serve":
        handler, records = make_handler(a.clouds, a.out)
        print(f"{len(records)} records from {a.clouds}; annotations -> {a.out}; http://localhost:{a.port}", flush=True)
        ThreadingHTTPServer(("0.0.0.0", a.port), handler).serve_forever()
    else:
        records = list_records(a.clouds)
        gt = build_export(records, load_annotations(a.out, records))
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(gt, open(a.json, "w"), indent=1)
        nv = sum(v["valid"] for v in gt.values())
        print(f"wrote {a.json}: {len(gt)} records, {nv} with a valid axis")


if __name__ == "__main__":
    main()
