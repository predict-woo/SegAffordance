#!/usr/bin/env python3
"""video_streams: full-rate camera video for the Rerun world view as H.264 streams, in its own process.

Started by world.sh next to spot_world.py. One thread per camera polls the robot's image service for the JPEG image
(hand colour camera 30 Hz, body cameras 15 Hz; one fetch is 5-8 ms on the wired link), decodes it, re-encodes it with
x264 and logs the Annex-B NAL units to Rerun's VideoStream archetype on the server that spot_world.py hosts, in the
same recording. Each stream is logged onto the entity that carries that camera's Pinhole (cams/<cam>), so the 3D view
shows the live video in the camera frustum and a 2D view of the entity shows the plain video. The browser decodes it.

Why not the driver's topics: its image publisher fetches all cameras in one loop and gets ~4 Hz for the hand camera.
Why a separate process: polling from a thread inside spot_world.py lost frames to the GIL (18 fps).
Why H.264 and not JPEG frames: JPEG q75 at 30 fps is ~15 Mbit/s and fills the 16 MiB viewer history in ~8 s; each
H.264 stream is capped (2 Mbit/s hand, 1 Mbit/s per grey body camera) and costs ~2 ms of one core per frame.
Gotcha: frames decoded from MJPEG carry pict_type=I, which makes x264 emit only keyframes (10 Mbit/s); reset it.
Note: the front body cameras are mounted sideways, so their video is rotated ~90 deg in a 2D view (it is correct in 3D).

env: RR_GRPC_PORT (9876), VIDEO_CAMS (comma list of robot image sources; default hand + both front cameras),
     VIDEO_KBPS_HAND (2000), VIDEO_KBPS_BODY (1000), VIDEO_CODEC (h264 | jpeg), VIDEO_JPEG_QUALITY (75)
"""
import os
import socket
import threading
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
# robot image source -> (Rerun entity = the entity spot_world.py logs that camera's Pinhole on, camera fps, kbit/s cap)
SOURCES = {
    "hand_color_image": ("cams/hand", 30, int(os.environ.get("VIDEO_KBPS_HAND", "2000"))),
    "frontleft_fisheye_image": ("cams/frontleft", 15, int(os.environ.get("VIDEO_KBPS_BODY", "1000"))),
    "frontright_fisheye_image": ("cams/frontright", 15, int(os.environ.get("VIDEO_KBPS_BODY", "1000"))),
    "left_fisheye_image": ("cams/left", 15, int(os.environ.get("VIDEO_KBPS_BODY", "1000"))),
    "right_fisheye_image": ("cams/right", 15, int(os.environ.get("VIDEO_KBPS_BODY", "1000"))),
    "back_fisheye_image": ("cams/back", 15, int(os.environ.get("VIDEO_KBPS_BODY", "1000"))),
}
CAMS = [c.strip() for c in os.environ.get("VIDEO_CAMS", "hand_color_image,frontleft_fisheye_image,frontright_fisheye_image").split(",") if c.strip()]
CODEC = os.environ.get("VIDEO_CODEC", "h264")
QUALITY = int(os.environ.get("VIDEO_JPEG_QUALITY", "75"))
RECORDING_ID = "spot-world-live"                   # must match spot_world.py


def log(msg):
    print(f"[video] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def wait_for_server():
    while True:
        try:
            with socket.create_connection(("127.0.0.1", GRPC_PORT), timeout=1.0):
                return
        except OSError:
            time.sleep(1.0)


def make_encoder(width, height, fps, kbps):
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = width, height, "yuv420p"
    enc.time_base, enc.framerate = Fraction(1, fps), Fraction(fps, 1)
    enc.bit_rate = kbps * 1000
    enc.options = {"preset": "veryfast", "tune": "zerolatency",
                   # 1 keyframe/s so a joining viewer starts within a second; no B-frames (latency); SPS/PPS repeated in-band
                   "x264-params": f"keyint={fps}:min-keyint={fps}:scenecut=0:bframes=0:repeat-headers=1:annexb=1"
                                  f":vbv-maxrate={kbps}:vbv-bufsize={kbps // 2}"}
    enc.open()
    return enc


def stream(robot, source):
    entity, fps, kbps = SOURCES[source]
    if CODEC == "h264":
        rr.log(entity, rr.VideoStream(codec=rr.VideoCodec.H264), static=True)
    req = [build_image_request(source, quality_percent=QUALITY, image_format=image_pb2.Image.FORMAT_JPEG)]
    while True:
        try:
            ic = robot.ensure_client(ImageClient.default_service_name)
            dec = av.CodecContext.create("mjpeg", "r")
            enc = None
            last, n, nbytes, t0, pts = None, 0, 0, time.time(), 0
            log(f"{source} -> {entity} as {CODEC}" + (f" ({kbps} kbit/s cap, {fps} fps)" if CODEC == "h264" else f" (JPEG q{QUALITY})"))
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
                    frame = dec.decode(av.Packet(data))[0].reformat(format="yuv420p")   # grey body cams: gray -> yuv420p
                    if enc is None:
                        enc = make_encoder(frame.width, frame.height, fps, kbps)
                    frame.pts, pts = pts, pts + 1
                    frame.pict_type = av.video.frame.PictureType.NONE   # let x264 choose I/P (MJPEG frames arrive as I)
                    for pkt in enc.encode(frame):
                        sample = np.frombuffer(bytes(pkt), np.uint8)
                        rr.log(entity, rr.VideoStream.from_fields(sample=sample, is_keyframe=bool(pkt.is_keyframe)))
                        nbytes += len(sample)
                else:
                    rr.log(entity, rr.EncodedImage(contents=np.frombuffer(data, np.uint8), media_type="image/jpeg"))
                    nbytes += len(data)
                n += 1
                if time.time() - t0 >= 30.0:
                    dt = time.time() - t0
                    log(f"{source}: {n / dt:.1f} fps, {8 * nbytes / dt / 1e6:.2f} Mbit/s")
                    n, nbytes, t0 = 0, 0, time.time()
        except Exception as e:                       # robot rebooting, auth hiccup, network: retry
            log(f"{source}: {type(e).__name__}: {e}; retrying in 3 s")
            time.sleep(3.0)


def main():
    wait_for_server()
    rr.init("spot_world", recording_id=RECORDING_ID)
    rr.connect_grpc(f"rerun+http://127.0.0.1:{GRPC_PORT}/proxy")
    while True:
        try:
            cfg = yaml.safe_load(open(SPOT_CFG))["/**"]["ros__parameters"]
            robot = bosdyn.client.create_standard_sdk("spot_world_video").create_robot(cfg["hostname"])
            robot.authenticate(cfg["username"], cfg["password"])
            break
        except Exception as e:
            log(f"auth: {type(e).__name__}: {e}; retrying in 3 s"); time.sleep(3.0)
    threads = [threading.Thread(target=stream, args=(robot, c), daemon=True, name=c) for c in CAMS if c in SOURCES]
    for t in threads:
        t.start()
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
