#!/usr/bin/env python3
"""Tiny persistent SAM 3 text-prompt segmentation server for the dev pod (model stays loaded, ~1 s per call).

  /workspace/venvs/sam3/bin/python sam3_serve.py [--port 12190] [--ckpt /workspace/models/sam3.pt]

  POST /segment  {"image": "/path.jpg", "prompt": "handle", "out": "/path/stem", "thr": 0.3}
    -> {"n": N, "instances": [{"score", "box", "centroid_uv", "area"}...], "npz": "<out>_masks.npz"}
       and writes <out>_masks.npz (masks (N,H,W) bool, boxes, scores, prompt) + <out>_mask<i>.png
  GET  /health   -> {"ok": true, "device": ...}

Mirrors hermann's sam3_server.py (set_image + set_text_prompt) but with file paths instead of array transport,
so the webapp can call it with one `ssh pod curl`. Binds to localhost only.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/tools/sam3")
from sam3.model_builder import build_sam3_image_model  # noqa: E402
from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402

LOCK = threading.Lock()
MODEL = PROC = None


def _np(t):
    if hasattr(t, "detach"):
        t = t.detach()
        if t.dtype in (torch.bfloat16, torch.float16):   # numpy has no bf16
            t = t.float()
        return t.cpu().numpy()
    return np.asarray(t)


def segment(image_path, prompt, out, thr):
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    with LOCK, torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):   # as in the SAM 3 examples
        state = PROC.set_image(img)
        o = PROC.set_text_prompt(state=state, prompt=prompt)
    masks = _np(o["masks"]).astype(bool).reshape(-1, H, W) if o.get("masks") is not None and len(o["masks"]) else np.zeros((0, H, W), bool)
    boxes = _np(o["boxes"]).reshape(-1, 4) if o.get("boxes") is not None and len(o["boxes"]) else np.zeros((0, 4))
    scores = _np(o["scores"]).ravel() if o.get("scores") is not None and len(o["scores"]) else np.zeros((0,))
    keep = scores >= thr
    masks, boxes, scores = masks[keep], boxes[keep], scores[keep]
    order = np.argsort(-scores); masks, boxes, scores = masks[order], boxes[order], scores[order]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out + "_masks.npz", masks=masks, boxes=boxes, scores=scores, prompt=prompt)
    inst = []
    for i, (m, b, s) in enumerate(zip(masks, boxes, scores)):
        ys, xs = np.nonzero(m)
        Image.fromarray(m.astype(np.uint8) * 255).save(f"{out}_mask{i}.png")
        inst.append({"score": float(s), "box": [float(v) for v in b], "area": int(m.sum()),
                     "centroid_uv": [float((xs.mean() + 0.5) / W), float((ys.mean() + 0.5) / H)] if xs.size else None})
    return {"n": len(inst), "instances": inst, "npz": out + "_masks.npz", "size": [W, H]}


class Hd(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"ok": MODEL is not None, "device": str(next(MODEL.parameters()).device) if MODEL else None})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            t0 = time.time()
            r = segment(req["image"], req["prompt"], req["out"], float(req.get("thr", 0.3)))
            r["seconds"] = round(time.time() - t0, 2)
            self._send(200, r)
        except Exception as ex:
            self._send(500, {"error": repr(ex)})

    def log_message(self, *a):
        pass


def main():
    global MODEL, PROC
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=12190)
    ap.add_argument("--ckpt", default="/workspace/models/sam3.pt")
    ap.add_argument("--bpe", default="/workspace/tools/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    a = ap.parse_args()
    t0 = time.time()
    MODEL = build_sam3_image_model(bpe_path=a.bpe, checkpoint_path=a.ckpt, load_from_HF=False)
    MODEL = MODEL.cuda().eval()
    PROC = Sam3Processor(MODEL)
    print(f"sam3 loaded in {time.time() - t0:.0f}s on {next(MODEL.parameters()).device}; serving http://127.0.0.1:{a.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Hd).serve_forever()


if __name__ == "__main__":
    main()
