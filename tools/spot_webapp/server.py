#!/usr/bin/env python3
"""EgoArt x Spot door pipeline, as a local web app (runs on the Mac, orchestrates spot22 + the dev pod over ssh).

  python3 server.py [--port 8770] [--moves]      then open http://127.0.0.1:8770

Stages (each button on the page = one stage; the page polls /state):
  1 snap_far       hand camera RGB-D from far away          -> far point cloud
  2 predict_far    user prompt -> EgoArt on the dev pod     -> mask / hinge / axis / arc on the far cloud
  3 accept_far     plan standoff -> WALK -> aim the camera at the predicted handle from AIM_DIST -> snap close
  4 predict_close  SAM 3 text prompt "handle" (pod, sam3_serve.py) -> instance nearest the expected handle -> centroid
                   + door plane -> far hinge carried over -> close cloud with the recalibrated arc
  5 accept_close   plan the real arc -> arm HOME -> smooth timed trajectory (open -> ease in -> grasp -> arc -> release
                   -> retreat) -> arm back HOME
  stop             spotctl stop at any time;  home = arm to half-down;  reset = clear the run

Robot MOTION (walk, aim, home, arc) only happens when moves are enabled (page toggle or --moves); otherwise those
steps run as dry runs and the close-up frame is taken from wherever the camera is. Snapshots are always live.
Everything is stdlib: the Mac has no numpy; math runs on spot22 (plan_*.py) and the pod (predict + pc_export).

Hosts: `spot` (spot22, spotd/spotctl in ~/andrew_ws) and `segaff-dev` (dev pod: EgoArt in /opt/venv, SAM 3 server
on localhost:12190 from /workspace/tools/sam3_serve.py, exporter in /workspace/datasets/itw/tools). See spot22/ here
for copies of the spot22 + pod scripts.
"""
import argparse
import json
import os
import re
import subprocess
import threading
import time
import traceback
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.join(HERE, "runs")
SPOT, POD = "spot", "segaff-dev"
SPOT_WS = "~/andrew_ws"
POD_REPO = "/workspace/SegAffordance"
POD_TOOLS = "/workspace/datasets/itw/tools"
POD_RUNS = "/workspace/datasets/itw/webapp_runs"
POD_ENV = "export PATH=/opt/venv/bin:$PATH; "     # the pod's project venv is not on PATH for non-interactive ssh
K_PX = "552.03 552.03 320 240"
CKPT = "experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt"
CFG = "config/sf3d_test_decoder_rgb_scalefree_dense.yaml"
CLOSE_PROMPT = "handle"          # SAM 3 text prompt for the close-up (EgoArt is only used on the far frame)
SAM3_PORT = 12190                # sam3_serve.py on the pod (tmux session sam3)

STATE = {
    "stage": "idle", "busy": None, "moves_enabled": False, "log": [], "run_id": None,
    "params": {"turn_deg": 30, "slide_m": 0.15, "speed": 0.01, "standoff": 1.10, "aim_dist": 0.50, "approach": 0.12, "grasp_bias": 0.02},
    "handle": "auto",           # auto = from the SAM 3 mask shape; or vertical / horizontal override. Sets the gripper roll.
    "far": {}, "close": {}, "plans": {}, "error": None,
}
LOCK = threading.Lock()


def log(msg):
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    with LOCK:
        STATE["log"].append(line); STATE["log"] = STATE["log"][-200:]
    print(line, flush=True)


def sh(cmd, timeout=600):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r.stdout


def ssh(host, cmd, timeout=600):
    if host == POD:
        cmd = POD_ENV + cmd
    return sh(f"ssh -o BatchMode=yes {host} {json.dumps(cmd)}", timeout)


def spotctl(args, timeout=300):
    return ssh(SPOT, f"{SPOT_WS}/spotctl {args}", timeout).strip()


def run_dir():
    d = os.path.join(RUN_ROOT, STATE["run_id"]); os.makedirs(d, exist_ok=True); return d


def pod_run():
    return f"{POD_RUNS}/{STATE['run_id']}"


# ---- stages -----------------------------------------------------------------------------------
def snap(tag):
    """Live RGB-D snapshot -> local run dir + pod run dir. Returns the stem (hand_YYYYmmdd_HHMMSS)."""
    out = spotctl("snap --depth")
    m = re.search(r"(/\S+/snaps/(hand_\d+_\d+))\.jpg", out)
    if not m:
        raise RuntimeError(f"snap failed: {out}")
    remote_stem, stem = m.group(1), m.group(2)
    d = run_dir()
    sh(f"scp -q {SPOT}:{remote_stem}.jpg {SPOT}:{remote_stem}_depth.png {SPOT}:{remote_stem}.json {d}/")
    ssh(POD, f"mkdir -p {pod_run()}")
    sh(f"scp -q {d}/{stem}.jpg {d}/{stem}_depth.png {d}/{stem}.json {POD}:{pod_run()}/")
    meta = json.load(open(f"{d}/{stem}.json"))
    hand = meta.get("hand_in_body")
    log(f"[{tag}] snapshot {stem}: {meta['width']}x{meta['height']}, hand at {['%.2f' % v for v in hand['xyz']] if hand else 'UNKNOWN (no TF: driver state publisher down?)'}")
    if "T_body_cam" not in meta:
        raise RuntimeError("snapshot has no body->camera transform (TF missing); fix the driver before continuing")
    return stem


def export_cloud(stem, tag, preds=None, recalib=None, models="dense", masks_npz=None, expect_body=None):
    """pc_export on the pod -> fetch .data.json (+ .pred.json). Returns local paths."""
    p = STATE["params"]
    out = f"{pod_run()}/{stem}_{tag}"
    extra = f" --preds {preds}" if preds else ""
    extra += f" --masks-npz {masks_npz} --grasp-bias-m {p['grasp_bias']}" if masks_npz else ""
    extra += f" --expect-body {expect_body[0]:.4f} {expect_body[1]:.4f} {expect_body[2]:.4f}" if expect_body else ""
    extra += f" --recalib {recalib}" if recalib else ""
    txt = ssh(POD, f"cd {POD_REPO} && python {POD_TOOLS}/pc_export.py {pod_run()}/{stem}.jpg{extra} --models {models} --turn-deg {p['turn_deg']} --slide-m {p['slide_m']} -o {out}.html 2>&1 | grep -v Warning")
    for line in txt.strip().splitlines():
        log(f"[{tag}] {line.strip()}")
    d = run_dir()
    sh(f"scp -q {POD}:{out}.data.json {d}/{stem}_{tag}.data.json")
    files = {"data": f"/files/{STATE['run_id']}/{stem}_{tag}.data.json"}
    if preds or masks_npz:
        sh(f"scp -q {POD}:{out}.pred.json {d}/{stem}_{tag}.pred.json")
        files["pred_local"] = f"{d}/{stem}_{tag}.pred.json"; files["pred_pod"] = f"{out}.pred.json"
    return files


def predict(stem, prompt, tag):
    """EgoArt (dense) on the pod -> dump jsonl path on the pod."""
    dump = f"{pod_run()}/{stem}_{tag}_preds.jsonl"
    ssh(POD, f"cd {POD_REPO} && python tools/predict_image.py --model dense {CFG} {CKPT} --case {pod_run()}/{stem}.jpg {json.dumps(prompt)} "
             f"--out {pod_run()}/out_{tag} --K {K_PX} --dump {dump} 2>&1 | grep -c '^wrote'", timeout=900)
    log(f"[{tag}] EgoArt done for prompt {prompt!r}")
    return dump


def sam3(stem, prompt, tag, thr=0.3):
    """Text-prompted segmentation on the pod's SAM 3 server -> masks npz path on the pod."""
    out = f"{pod_run()}/{stem}_{tag}_sam3"
    req = json.dumps({"image": f"{pod_run()}/{stem}.jpg", "prompt": prompt, "out": out, "thr": thr})
    txt = ssh(POD, f"curl -s -m 120 -X POST localhost:{SAM3_PORT}/segment -d {json.dumps(req)}", timeout=180)
    r = json.loads(txt or "{}")
    if "error" in r:
        raise RuntimeError(f"sam3: {r['error']}")
    inst = ", ".join(f"#{i} {x['score']:.2f} at ({x['centroid_uv'][0]:.2f},{x['centroid_uv'][1]:.2f})" for i, x in enumerate(r["instances"]))
    log(f"[{tag}] SAM 3 {prompt!r}: {r['n']} instance(s) in {r.get('seconds', 0):.1f}s: {inst or 'none'}")
    if r["n"] == 0:
        raise RuntimeError(f"sam3 found no {prompt!r} in the close-up frame")
    return r["npz"]


def job_snap_far():
    STATE["run_id"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    STATE["far"], STATE["close"], STATE["plans"] = {}, {}, {}
    stem = snap("far")
    files = export_cloud(stem, "far")
    STATE["far"] = {"stem": stem, "cloud": files["data"], "pred": None}
    STATE["stage"] = "far_ready"


def job_predict_far(prompt):
    far = STATE["far"]
    dump = predict(far["stem"], prompt, "far")
    files = export_cloud(far["stem"], "farpred", preds=dump)
    try:
        far_type = json.load(open(files["pred_local"]))["preds"][0]["type"]
    except Exception:
        far_type = "unknown"
    far.update({"prompt": prompt, "cloud": files["data"], "pred": files["pred_local"], "pred_pod": files["pred_pod"], "type": far_type})
    log(f"[far] EgoArt type: {far_type}")
    STATE["stage"] = "far_predicted"


def job_accept_far():
    p = STATE["params"]; far = STATE["far"]; moves = STATE["moves_enabled"]
    # 1. standoff plan on spot22
    sh(f"scp -q {far['pred']} {SPOT}:{SPOT_WS}/snaps/far.pred.json")
    txt = ssh(SPOT, f"cd {SPOT_WS} && python3 plan_standoff.py snaps/far.pred.json --standoff {p['standoff']} -o standoff.json")
    for line in txt.strip().splitlines():
        log(f"[standoff] {line}")
    so = json.loads(ssh(SPOT, f"cat {SPOT_WS}/standoff.json"))
    STATE["plans"]["standoff"] = so
    gx, gy, gyaw = so["goal_body_xy_yaw_deg"]
    # 2. walk
    log(f"[walk] {'MOVING' if moves else 'dry run'}: walkto {gx:.3f} {gy:.3f} {gyaw:.1f}")
    log("[walk] " + spotctl(f"walkto {gx:.3f} {gy:.3f} {gyaw:.1f} --t 25{'' if moves else ' --dry'}"))
    # 3. aim the camera at the handle from aim_dist (handle expressed in the post-walk body frame; door normal there is -x)
    T_hand_cam = spotctl("tf spot/hand spot/hand_color_image_sensor")
    h = so["handle_in_goal_frame"] if moves else so["handle_body"]
    n = [-1.0, 0.0, 0.0] if moves else so["door_normal_body"]
    txt = ssh(SPOT, f"cd {SPOT_WS} && python3 plan_aim.py --target {h[0]} {h[1]} {h[2]} --normal {n[0]} {n[1]} {n[2]} --dist {p['aim_dist']} --T-hand-cam '{T_hand_cam}' -o aim.json")
    for line in txt.strip().splitlines():
        log(f"[aim] {line}")
    aim = json.loads(ssh(SPOT, f"cat {SPOT_WS}/aim.json")); STATE["plans"]["aim"] = aim
    if moves:
        q = aim["hand_quat_xyzw"]; x = aim["hand_xyz"]
        log("[aim] MOVING hand: " + spotctl(f"poseq {x[0]:.4f} {x[1]:.4f} {x[2]:.4f} {q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}"))
        time.sleep(1.0)
    else:
        log("[aim] dry run: hand not moved, close-up frame will be taken from the current camera pose")
    # 4. close snapshot + segmentation
    stem = snap("close")
    STATE["close"] = {"stem": stem}
    STATE["stage"] = "close_snapped"
    job_predict_close()


def job_predict_close():
    far, close = STATE["far"], STATE["close"]
    npz = sam3(close["stem"], CLOSE_PROMPT, "close")
    so = STATE["plans"].get("standoff") or {}
    expect = so.get("handle_in_goal_frame") if STATE["moves_enabled"] else so.get("handle_body")
    files = export_cloud(close["stem"], "closerecal", masks_npz=npz, expect_body=expect, recalib=far["pred_pod"])
    try:
        pr = json.load(open(files["pred_local"]))["preds"][0]
        detected = pr.get("handle_orient"); ang = pr.get("handle_angle_deg")
    except Exception:
        detected, ang = None, None
    close.update({"prompt": CLOSE_PROMPT, "cloud": files["data"], "pred": files["pred_local"], "handle_detected": detected})
    log(f"[close] handle bar from the mask: {detected or 'undetermined'}" + (f" ({ang:.0f} deg in the image)" if ang is not None else ""))
    STATE["stage"] = "close_predicted"


def job_accept_close():
    p = STATE["params"]; close = STATE["close"]; moves = STATE["moves_enabled"]
    sh(f"scp -q {close['pred']} {SPOT}:{SPOT_WS}/snaps/close_recal.pred.json")
    handle = STATE["handle"]
    if handle == "auto":
        handle = close.get("handle_detected") or "vertical"
        log(f"[plan] handle bar: {handle} ({'from the SAM 3 mask' if close.get('handle_detected') else 'mask undetermined, defaulting to vertical'})")
    else:
        log(f"[plan] handle bar: {handle} (manual override)")
    txt = ssh(SPOT, f"cd {SPOT_WS} && python3 plan_traj.py snaps/close_recal.pred.json --real --turn {p['turn_deg']} --slide {p['slide_m']} --steps 30 --approach {p['approach']} --handle {handle} -o traj_web.json")
    for line in txt.strip().splitlines():
        log(f"[plan] {line}")
    if "OUT OF REACH" in txt or "WARNING" in txt:
        log("[plan] reach warning present: check the numbers above before enabling moves")
    if moves:
        log("[home] MOVING: go half-down -> " + spotctl("go half-down"))
        time.sleep(0.5)
        log("[arc] MOVING: " + spotctl(f"traj traj_web.json --smooth --speed {p['speed']}", timeout=600))
        time.sleep(0.5)
        # back to the home pose after the release + retreat, so the camera is in the far-view position for the next run
        log("[home] MOVING: return to half-down -> " + spotctl("go half-down", timeout=60))
    else:
        log("[arc] dry run: " + spotctl(f"traj traj_web.json --smooth --speed {p['speed']} --dry"))
        log("[home] dry run: would return the arm to half-down afterwards")
    STATE["stage"] = "done"


JOBS = {"snap_far": job_snap_far, "predict_far": job_predict_far, "accept_far": job_accept_far,
        "predict_close": job_predict_close, "accept_close": job_accept_close}


def start_job(name, **kw):
    with LOCK:
        if STATE["busy"]:
            raise RuntimeError(f"busy with {STATE['busy']}")
        STATE["busy"] = name; STATE["error"] = None

    def run():
        try:
            JOBS[name](**kw)
        except Exception as ex:
            STATE["error"] = f"{name}: {ex}"
            log(f"ERROR {name}: {ex}")
            traceback.print_exc()
        finally:
            STATE["busy"] = None
    threading.Thread(target=run, daemon=True).start()


# ---- http ---------------------------------------------------------------------------------------
class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, open(os.path.join(HERE, "index.html"), "rb").read(), "text/html; charset=utf-8")
        if u.path == "/state":
            return self._send(200, STATE)
        if u.path.startswith("/files/"):
            rel = os.path.normpath(u.path[len("/files/"):])
            f = os.path.join(RUN_ROOT, rel)
            if not f.startswith(RUN_ROOT) or not os.path.exists(f):
                return self._send(404, {"error": "not found"})
            ctype = "application/json" if f.endswith(".json") else "image/jpeg" if f.endswith(".jpg") else "application/octet-stream"
            return self._send(200, open(f, "rb").read(), ctype)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        try:
            if u.path == "/api/stop":
                out = spotctl("stop", timeout=20); log(f"STOP -> {out}")
                return self._send(200, {"ok": True, "msg": out})
            if u.path == "/api/reset":
                if STATE["busy"]:
                    raise RuntimeError(f"busy with {STATE['busy']}; press STOP first or wait")
                STATE.update({"stage": "idle", "run_id": None, "far": {}, "close": {}, "plans": {}, "error": None})
                log("reset: state cleared (robot untouched, moves switch unchanged)")
                return self._send(200, {"ok": True})
            if u.path == "/api/home":
                if not STATE["moves_enabled"]:
                    raise RuntimeError("enable robot moves first")
                out = spotctl("go half-down", timeout=60); log(f"[home] arm to half-down -> {out}")
                return self._send(200, {"ok": True, "msg": out})
            if u.path == "/api/moves":
                STATE["moves_enabled"] = bool(body.get("enabled")); log(f"moves {'ENABLED' if STATE['moves_enabled'] else 'disabled'}")
                return self._send(200, {"ok": True})
            if u.path == "/api/params":
                if body.get("handle") in ("auto", "vertical", "horizontal"):
                    STATE["handle"] = body["handle"]
                STATE["params"].update({k: float(v) for k, v in body.items() if k in STATE["params"]})
                return self._send(200, {"ok": True, "params": STATE["params"]})
            if u.path == "/api/snap_far":
                start_job("snap_far")
            elif u.path == "/api/predict_far":
                if not STATE["far"].get("stem"):
                    raise RuntimeError("take the far snapshot first")
                start_job("predict_far", prompt=body.get("prompt", "").strip() or "open the cabinet door")
            elif u.path == "/api/accept_far":
                if not STATE["far"].get("pred"):
                    raise RuntimeError("no far prediction to accept")
                start_job("accept_far")
            elif u.path == "/api/accept_close":
                if not STATE["close"].get("pred"):
                    raise RuntimeError("no close prediction to accept")
                start_job("accept_close")
            else:
                return self._send(404, {"error": "unknown action"})
            return self._send(200, {"ok": True})
        except Exception as ex:
            return self._send(400, {"ok": False, "error": str(ex)})

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--moves", action="store_true", help="start with robot moves ENABLED (default: dry runs)")
    a = ap.parse_args()
    STATE["moves_enabled"] = a.moves
    os.makedirs(RUN_ROOT, exist_ok=True)
    log(f"server on http://127.0.0.1:{a.port}  moves {'ENABLED' if a.moves else 'disabled (dry runs)'}")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
