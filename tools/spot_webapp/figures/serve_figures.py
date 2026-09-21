#!/usr/bin/env python3
"""serve_figures: local server for figure_maker.html (nice still images of a session's first far cloud + EgoArt prediction).

    python3 serve_figures.py /Users/andyye/dev/egoart-recordings/final      -> http://127.0.0.1:8771

Scans <root>/*/ *_session/run/*_farpred.data.json (the FIRST far prediction of each session), serves the page, the data,
and saves what the page renders next to the session:  <root>/egoart_NN/egoart_NN_far.png  +  egoart_NN_camera.json
(camera + style, reloaded automatically next time so a figure can be re-rendered identically).
"""
import base64
import glob
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
PORT = int(os.environ.get("PORT", 8771))


def index():
    items = []
    for d in sorted(glob.glob(os.path.join(ROOT, "*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        far = sorted(glob.glob(os.path.join(d, "*_session", "run", "*_farpred.data.json")))
        if not far:
            continue
        sess = os.path.join(os.path.dirname(os.path.dirname(far[0])), "session.json")
        prompt = ""
        try:
            s = json.load(open(sess)); prompt = (s.get("far") or {}).get("prompt", "")
        except Exception:
            pass
        cam = os.path.join(d, f"{name}_camera.json")
        items.append({"name": name, "data": "/files/" + os.path.relpath(far[0], ROOT), "prompt": prompt,
                      "camera": json.load(open(cam)) if os.path.exists(cam) else None,
                      "png": f"{name}_far.png" if os.path.exists(os.path.join(d, f"{name}_far.png")) else None})
    return items


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, open(os.path.join(HERE, "figure_maker.html"), "rb").read(), "text/html; charset=utf-8")
        if u.path == "/index.json":
            return self._send(200, index())
        if u.path.startswith("/files/"):
            f = os.path.abspath(os.path.join(ROOT, unquote(u.path[len("/files/"):])))
            if not f.startswith(ROOT) or not os.path.isfile(f):
                return self._send(404, {"error": "not found"})
            ctype = "application/json" if f.endswith(".json") else "application/octet-stream"
            return self._send(200, open(f, "rb").read(), ctype)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path != "/save":
            return self._send(404, {"error": "unknown"})
        name = os.path.basename(body["name"])
        d = os.path.join(ROOT, name)
        if not os.path.isdir(d):
            return self._send(400, {"error": f"no folder {name}"})
        out = {}
        if body.get("camera") is not None:
            json.dump(body["camera"], open(os.path.join(d, f"{name}_camera.json"), "w"), indent=1); out["camera"] = f"{name}_camera.json"
        if body.get("png_b64"):
            suffix = body.get("suffix") or "far"
            p = os.path.join(d, f"{name}_{suffix}.png")
            open(p, "wb").write(base64.b64decode(body["png_b64"].split(",", 1)[-1])); out["png"] = os.path.relpath(p, ROOT)
        print("saved", out, flush=True)
        return self._send(200, {"ok": True, **out})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"figure maker: {len(index())} sessions under {ROOT}  ->  http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
