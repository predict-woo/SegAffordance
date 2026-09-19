#!/usr/bin/env python3
"""Browser teleop for Spot, served FROM spot22 (no ssh in the loop): http://192.168.1.213:8780

  bash ~/andrew_ws/teleop_web.sh          (tmux session `teleop`; sources the ROS env, binds 0.0.0.0:8780)

What it does
  * Hold-to-drive: the page posts the current velocity vector at 10 Hz while any drive key is held and posts zeros
    the instant the last key is released, so a short tap turns a few degrees instead of a fixed increment. The
    server streams Twist on /spot/cmd_vel only while a command is fresh (deadman 0.35 s) and publishes ONE zero
    when it expires; when idle it publishes nothing, so it never pre-empts spotd's walkto / trajectory commands.
  * Precise nudges (turn 2/5/15 deg, step 5/20 cm) go through spotd's `walkto` (the driver's trajectory action).
  * Live camera as MJPEG (/stream.mjpg?cam=hand|front|back|left|right|frontleft|frontright); one ROS subscription
    at a time, switched on demand, JPEG-encoded at the source rate (hand ~5 Hz, body cams ~11 Hz).
  * Robot commands (stop, sit, stand, stow, unstow, home, open, close, estop) are forwarded to spotd's socket.
Stdlib HTTP only (no websockets): POST /api/vel {vx,vy,wz}; POST /api/cmd {cmd}; POST /api/cam {cam};
GET /api/state; GET /stream.mjpg?cam=...
"""
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image as ImageMsg

HERE = os.path.dirname(os.path.abspath(__file__))
SPOTD_SOCK = "/tmp/spotd.sock"
CAMS = {"hand": "/spot/camera/hand/image", "front": "/spot/camera/frontmiddle_virtual/image",
        "frontleft": "/spot/camera/frontleft/image", "frontright": "/spot/camera/frontright/image",
        "left": "/spot/camera/left/image", "right": "/spot/camera/right/image", "back": "/spot/camera/back/image"}
VMAX, WMAX = 1.0, 1.5            # hard caps (m/s, rad/s) regardless of what the page sends
DEADMAN_S = 0.35
ALLOWED_CMDS = {"stop", "sit", "stand", "stow", "unstow", "open", "close", "estop", "claim", "take", "poweron", "poweroff",
                "selfright", "estop-release"}


def spotctl(*args, timeout=30.0):
    """Talk to spotd over its Unix socket (same protocol as spotctl)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(SPOTD_SOCK)
    except (FileNotFoundError, ConnectionRefusedError):
        return False, "spotd not running"
    s.sendall(("\t".join(args) + "\n").encode())
    buf = b""
    while True:
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            break
        if not chunk:
            break
        buf += chunk
    s.close()
    txt = buf.decode().rstrip("\n")
    return txt.startswith("OK"), txt[3:] if txt[:3] in ("OK ", "ERR") else txt[4:] if txt.startswith("ERR ") else txt


class TeleopNode(Node):
    def __init__(self):
        super().__init__("teleop_web")
        self.pub = self.create_publisher(Twist, "/spot/cmd_vel", 10)
        self.lock = threading.Lock()
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_until = 0.0
        self.was_active = False
        self.create_timer(0.1, self._tick)
        self.cam_name, self.cam_sub = None, None
        self.jpeg, self.jpeg_t, self.frame_n, self.enc = None, 0.0, 0, ""
        self.status, self.status_t = "", 0.0
        self.last_vel_client_t = 0.0

    def set_vel(self, vx, vy, wz):
        vx = float(np.clip(vx, -VMAX, VMAX)); vy = float(np.clip(vy, -VMAX, VMAX)); wz = float(np.clip(wz, -WMAX, WMAX))
        with self.lock:
            self.cmd = (vx, vy, wz)
            self.cmd_until = time.time() + DEADMAN_S if any(abs(v) > 1e-6 for v in (vx, vy, wz)) else 0.0
            self.last_vel_client_t = time.time()

    def _tick(self):
        with self.lock:
            active = time.time() < self.cmd_until
            vx, vy, wz = self.cmd
        if active:
            m = Twist(); m.linear.x, m.linear.y, m.angular.z = vx, vy, wz
            self.pub.publish(m); self.was_active = True
        elif self.was_active:
            self.pub.publish(Twist()); self.was_active = False       # one zero, then silence

    def stop_now(self):
        with self.lock:
            self.cmd, self.cmd_until = (0.0, 0.0, 0.0), 0.0
        self.pub.publish(Twist()); self.was_active = False

    # ---- camera --------------------------------------------------------------------------------
    def select_cam(self, name):
        if name not in CAMS:
            return False
        with self.lock:
            if name == self.cam_name:
                return True
            if self.cam_sub is not None:
                self.destroy_subscription(self.cam_sub)
            self.cam_name, self.jpeg, self.frame_n = name, None, 0
            self.cam_sub = self.create_subscription(ImageMsg, CAMS[name], self._img_cb, 1)
        return True

    def _img_cb(self, m):
        enc = m.encoding.lower()
        try:
            if enc in ("rgb8", "bgr8"):
                a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
                bgr = a[:, :, ::-1] if enc == "rgb8" else a
            elif enc in ("mono8", "8uc1"):
                bgr = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
            elif enc in ("16uc1", "mono16"):
                d = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32)
                bgr = cv2.applyColorMap(np.clip(d / 4000.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            else:
                return
            ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(bgr), [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with self.lock:
                    self.jpeg, self.jpeg_t, self.frame_n, self.enc = buf.tobytes(), time.time(), self.frame_n + 1, m.encoding
        except Exception as ex:
            self.get_logger().warning(f"image convert failed: {ex}")

    def get_status(self):
        if time.time() - self.status_t > 2.0:
            ok, txt = spotctl("status", timeout=5.0)
            self.status, self.status_t = (txt if ok else f"spotd: {txt}"), time.time()
        return self.status


NODE: TeleopNode = None


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self._send(200, open(os.path.join(HERE, "teleop.html"), "rb").read(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            with NODE.lock:
                vx, vy, wz = NODE.cmd; active = time.time() < NODE.cmd_until
                cam, n, age = NODE.cam_name, NODE.frame_n, (time.time() - NODE.jpeg_t) if NODE.jpeg else None
            return self._send(200, {"vel": [vx, vy, wz], "active": active, "cam": cam, "frames": n, "frame_age": age,
                                    "encoding": NODE.enc, "status": NODE.get_status(), "cams": list(CAMS)})
        if u.path == "/stream.mjpg":
            cam = (parse_qs(u.query).get("cam") or ["hand"])[0]
            NODE.select_cam(cam)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            last = -1
            try:
                while True:
                    with NODE.lock:
                        j, n = NODE.jpeg, NODE.frame_n
                    if j is not None and n != last:
                        last = n
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return
        self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        if u.path == "/api/vel":
            NODE.set_vel(body.get("vx", 0.0), body.get("vy", 0.0), body.get("wz", 0.0))
            return self._send(200, {"ok": True})
        if u.path == "/api/cam":
            return self._send(200, {"ok": NODE.select_cam(body.get("cam", "hand"))})
        if u.path == "/api/cmd":
            cmd = str(body.get("cmd", ""))
            NODE.stop_now()                                   # any discrete command first silences teleop streaming
            if cmd == "stop":
                ok, txt = spotctl("stop")
            elif cmd in ALLOWED_CMDS:
                ok, txt = spotctl(cmd, "--yes") if cmd == "estop-hard" else spotctl(cmd)
            elif cmd == "home":
                ok, txt = spotctl("go", "half-down")
            elif cmd == "turn":
                deg = float(np.clip(float(body.get("deg", 0.0)), -45, 45)); ok, txt = spotctl("walkto", "0", "0", f"{deg:.1f}", "--t", "10")
            elif cmd == "step":
                dx = float(np.clip(float(body.get("dx", 0.0)), -0.5, 0.5)); dy = float(np.clip(float(body.get("dy", 0.0)), -0.5, 0.5))
                ok, txt = spotctl("walkto", f"{dx:.3f}", f"{dy:.3f}", "0", "--t", "10")
            else:
                return self._send(400, {"ok": False, "msg": f"unknown cmd {cmd!r}"})
            NODE.status_t = 0.0
            return self._send(200, {"ok": ok, "msg": txt})
        self._send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


def main():
    global NODE
    rclpy.init()
    NODE = TeleopNode()
    ex = MultiThreadedExecutor(num_threads=3); ex.add_node(NODE)
    threading.Thread(target=ex.spin, daemon=True).start()
    NODE.select_cam("hand")
    port = int(os.environ.get("TELEOP_PORT", 8780))
    print(f"teleop_web serving on http://0.0.0.0:{port}  (open http://192.168.1.213:{port} on the LAN)", flush=True)
    srv = ThreadingHTTPServer(("0.0.0.0", port), H); srv.daemon_threads = True
    try:
        srv.serve_forever()
    finally:
        NODE.stop_now(); rclpy.shutdown()


if __name__ == "__main__":
    main()
