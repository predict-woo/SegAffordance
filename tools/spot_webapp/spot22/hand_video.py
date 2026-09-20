#!/usr/bin/env python3
"""hand_video: full-rate hand camera video for the Rerun world view, as an H.264 stream, in its own process.

Started by world.sh next to spot_world.py. Polls the robot's image service for hand_color_image as JPEG (the camera runs
at ~30 Hz; one fetch takes ~8 ms on the wired link), decodes it, re-encodes it with x264 and logs the Annex-B NAL units
to Rerun's VideoStream archetype on the server that spot_world.py hosts, in the same recording. The browser viewer
decodes the H.264 itself.

Why not the driver's topic: its image publisher fetches all cameras in one loop and gets ~4 Hz for this camera.
Why a separate process: polling from a thread inside spot_world.py lost frames to the GIL (18 fps).
Why H.264 and not JPEG frames: JPEG q75 at 30 fps is ~15 Mbit/s and fills the 16 MiB viewer history in ~8 s; the
H.264 stream is capped at HAND_KBPS (2 Mbit/s default) and costs ~2 ms of one core per frame on spot22.
Gotcha: frames decoded from MJPEG carry pict_type=I, which makes x264 emit only keyframes (10 Mbit/s); reset it.

env: RR_GRPC_PORT (9876), HAND_CAMERA (hand_color_image), HAND_KBPS (2000), HAND_CODEC (h264 | jpeg), HAND_JPEG_QUALITY (75)
"""
import os
import socket
import time
from fractions import Fraction

import av
import numpy as np
import rerun as rr
import yaml
import bosdyn.client
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

SPOT_CFG = "/home/spot/dev/ros2_ws/install/locopt_ros/share/locopt_ros/config/spot_config.yaml"   # hostname/username/password
GRPC_PORT = int(os.environ.get("RR_GRPC_PORT", 9876))
CAMERA = os.environ.get("HAND_CAMERA", "hand_color_image")
KBPS = int(os.environ.get("HAND_KBPS", "2000"))
CODEC = os.environ.get("HAND_CODEC", "h264")
QUALITY = int(os.environ.get("HAND_JPEG_QUALITY", "75"))
FPS = 30
RECORDING_ID = "spot-world-live"                   # must match spot_world.py
ENTITY = "cams/hand_rgb"


def log(msg):
    print(f"[hand_video] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def wait_for_server():
    while True:
        try:
            with socket.create_connection(("127.0.0.1", GRPC_PORT), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)


def make_encoder(width, height):
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = width, height, "yuv420p"
    enc.time_base, enc.framerate = Fraction(1, FPS), Fraction(FPS, 1)
    enc.bit_rate = KBPS * 1000
    enc.options = {"preset": "veryfast", "tune": "zerolatency",
                   # 1 keyframe/s so a joining viewer starts within a second; no B-frames (latency); SPS/PPS repeated in-band
                   "x264-params": f"keyint={FPS}:min-keyint={FPS}:scenecut=0:bframes=0:repeat-headers=1:annexb=1"
                                  f":vbv-maxrate={KBPS}:vbv-bufsize={KBPS // 2}"}
    enc.open()
    return enc


def main():
    wait_for_server()
    rr.init("spot_world", recording_id=RECORDING_ID)
    rr.connect_grpc(f"rerun+http://127.0.0.1:{GRPC_PORT}/proxy")
    if CODEC == "h264":
        rr.log(ENTITY, rr.VideoStream.from_fields(codec=rr.VideoCodec.H264), static=True)
    req = [build_image_request(CAMERA, quality_percent=QUALITY, image_format=image_pb2.Image.FORMAT_JPEG)]
    while True:
        try:
            cfg = yaml.safe_load(open(SPOT_CFG))["/**"]["ros__parameters"]
            robot = bosdyn.client.create_standard_sdk("spot_world_hand").create_robot(cfg["hostname"])
            robot.authenticate(cfg["username"], cfg["password"])
            ic = robot.ensure_client(ImageClient.default_service_name)
            dec = av.CodecContext.create("mjpeg", "r")
            enc = None
            last, n, nbytes, t0, pts = None, 0, 0, time.time(), 0
            log(f"streaming {CAMERA} as {CODEC}" + (f" ({KBPS} kbit/s cap)" if CODEC == "h264" else f" (JPEG q{QUALITY})"))
            while True:
                r = ic.get_image(req)[0]
                ts = r.shot.acquisition_time
                key = (ts.seconds, ts.nanos)
                if key == last:                      # camera has not produced a new frame yet
                    time.sleep(0.003)
                    continue
                last = key
                data = bytes(r.shot.image.data)
                rr.set_time("ros", timestamp=time.time())
                if CODEC == "h264":
                    frame = dec.decode(av.Packet(data))[0].reformat(format="yuv420p")
                    if enc is None:
                        enc = make_encoder(frame.width, frame.height)
                    frame.pts, pts = pts, pts + 1
                    frame.pict_type = av.video.frame.PictureType.NONE   # let x264 choose I/P (MJPEG frames arrive as I)
                    for pkt in enc.encode(frame):
                        sample = np.frombuffer(bytes(pkt), np.uint8)
                        rr.log(ENTITY, rr.VideoStream.from_fields(sample=sample, is_keyframe=bool(pkt.is_keyframe)))
                        nbytes += len(sample)
                else:
                    rr.log(ENTITY, rr.EncodedImage(contents=np.frombuffer(data, np.uint8), media_type="image/jpeg"))
                    nbytes += len(data)
                n += 1
                if time.time() - t0 >= 30.0:
                    dt = time.time() - t0
                    log(f"{n / dt:.1f} fps, {8 * nbytes / dt / 1e6:.2f} Mbit/s")
                    n, nbytes, t0 = 0, 0, time.time()
        except KeyboardInterrupt:
            return
        except Exception as e:                       # robot rebooting, auth hiccup, network: retry
            log(f"{type(e).__name__}: {e}; retrying in 3 s")
            time.sleep(3.0)


if __name__ == "__main__":
    main()
