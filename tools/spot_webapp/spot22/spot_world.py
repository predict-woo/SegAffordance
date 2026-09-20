#!/usr/bin/env python3
"""spot_world: real-time world-frame visualizer + recorder for Spot, logged to Rerun.

  bash ~/andrew_ws/world.sh            (tmux session `world`)  ->  web viewer at http://192.168.1.213:9090
                                        recordings: ~/andrew_ws/world/recordings/spot_<timestamp>.rrd

What is logged (all in the fixed frame `spot/vision`, Spot's visual-odometry world frame):
  * the official Spot URDF (Boston Dynamics meshes from spot_description) animated from /spot/joint_states and TF
  * live point clouds from the 5 body depth cameras + the hand depth camera, backprojected on spot22, voxel-
    downsampled, transformed with TF at the image timestamp
  * a persistent voxel map that accumulates as Spot walks (occupied voxels are kept until the cap is hit)
  * depth camera frustums (pinholes) with live H.264 video in them for the hand and both front cameras, at the cameras'
    full rate (30 / 15 fps), logged by the separate video_streams.py (own process: the driver republishes the hand image at
    only ~4 Hz, and in-process polling lost frames to the GIL); HAND_SDK=0 falls back to the driver's hand topic (2 Hz)
  * the trail of the body through the room
Everything is time-stamped with ROS time, so the .rrd recording scrubs like a video.
"""
import os
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
from scipy.spatial import cKDTree
import rerun as rr
import rerun.blueprint as rrb
import rerun.urdf as rr_urdf
import tf2_ros
from nav_msgs.msg import Odometry
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image as ImageMsg, JointState

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "spot.urdf")
WORLD = "spot/vision"
BODY = "spot/body"
DEPTH_CAMS = {  # name -> (registered depth image, its camera_info, the matching intensity/RGB image, stride, hz cap)
    "frontleft": ("/spot/depth_registered/frontleft/image", "/spot/depth_registered/frontleft/camera_info", "/spot/camera/frontleft/image", 4, 4.0),
    "frontright": ("/spot/depth_registered/frontright/image", "/spot/depth_registered/frontright/camera_info", "/spot/camera/frontright/image", 4, 4.0),
    "left": ("/spot/depth_registered/left/image", "/spot/depth_registered/left/camera_info", "/spot/camera/left/image", 4, 4.0),
    "right": ("/spot/depth_registered/right/image", "/spot/depth_registered/right/camera_info", "/spot/camera/right/image", 4, 4.0),
    "back": ("/spot/depth_registered/back/image", "/spot/depth_registered/back/camera_info", "/spot/camera/back/image", 4, 4.0),
    "hand": ("/spot/depth_registered/hand/image", "/spot/depth_registered/hand/camera_info", "/spot/camera/hand/image", 4, 4.0),
}
CAM_COLORS = {"frontleft": (255, 140, 0), "frontright": (255, 200, 0), "left": (0, 200, 255), "right": (120, 120, 255),
              "back": (200, 80, 255), "hand": (60, 255, 120)}
Z_MIN, Z_MAX = 0.25, 4.0
VOXEL_LIVE, VOXEL_MAP = 0.04, 0.05
MAP_MAX_VOXELS = 800_000
MAP_LOG_PERIOD, POSE_HZ = 2.0, 15.0
TILE = 2.0                     # map is logged as 2 m tiles; only tiles that gained voxels are re-sent
MAP_MIN_HITS = 3               # a voxel enters the map after being seen in this many frames (kills depth-noise inflation)
MAP_ON = os.environ.get("WORLD_MAP", "0") == "1"   # cumulative map is OFF by default: live per-frame clouds only
HAND_SDK = os.environ.get("HAND_SDK", "1") == "1"  # camera video comes from video_streams.py (own process, full rate) instead of the ~4 Hz driver topic
RECORDING_ID = "spot-world-live"                   # shared with video_streams.py so both land in the same recording
FRUSTUM_M = 0.15                                   # image-plane distance of the drawn frustums (fisheye f=257 px makes them wide)
COLOR_MATCH_M = float(os.environ.get("COLOR_MATCH_M", "0.02"))   # grey body-cam points within this distance of a hand (RGB) point take its colour
RGB_CLOUD_MAX_AGE = 1.5                            # s; older hand clouds are not used for colouring


def turbo(t):
    """t in [0,1] -> RGB uint8 (a compact Turbo-like ramp)."""
    t = np.clip(t, 0, 1)
    r = np.clip(1.6 * t - 0.2, 0, 1); g = np.clip(1 - np.abs(2 * t - 1) * 1.2, 0, 1); b = np.clip(1.4 - 1.8 * t, 0, 1)
    return (np.stack([r, g, b], 1) * 255).astype(np.uint8)


class SpotWorld(Node):
    def __init__(self):
        super().__init__("spot_world")
        self.tf_buf = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=15.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self, spin_thread=False)
        self.lock = threading.Lock()
        self.K = {}
        self.grids = {}
        self.last_cam_t = {c: 0.0 for c in DEPTH_CAMS}
        self.map = {}                    # tile key (ix,iy,iz) -> {voxel key -> [x, y, z]} (world)
        self.dirty_tiles = set()
        self.n_voxels = 0
        self.pending = {}                # voxel key -> hit count, for voxels not yet committed
        self.trail = deque(maxlen=20000)
        self.tree = rr_urdf.UrdfTree.from_file_path(URDF, entity_path_prefix="world/spot")
        self.tree.log_urdf_to_recording()
        self.joints = {j.name: j for j in self.tree.joints()}
        self.live_frame_set = set()
        self.color = {}                  # cam -> latest intensity/RGB image as (H,W,3) uint8
        self.rgb_cloud = None            # (KD-tree of the latest hand cloud in the world frame, its colours, wall time)
        for cam, (dt, ct, it, _, _) in DEPTH_CAMS.items():
            self.create_subscription(CameraInfo, ct, lambda m, c=cam: self._caminfo(c, m), 5)
            self.create_subscription(ImageMsg, dt, lambda m, c=cam: self._depth(c, m), 2)
            self.create_subscription(ImageMsg, it, lambda m, c=cam: self._color(c, m), 1)
        if not HAND_SDK:                 # otherwise video_streams.py (started by world.sh) logs the video onto world/cams/<cam>/video at full rate
            self.create_subscription(ImageMsg, "/spot/camera/hand/image", self._hand_rgb, 1)
        self.create_subscription(JointState, "/spot/joint_states", self._joints, 10)
        self.create_subscription(Odometry, "/spot/odometry", self._odom, 10)
        if MAP_ON:
            self.create_timer(MAP_LOG_PERIOD, self._log_map)
        self.last_joint_t = self.last_rgb_t = 0.0
        self.n_depth = 0
        self.n_cam_msgs = self.n_state_msgs = 0   # per publisher group: FastDDS discovery on spot22 fails silently, sometimes only partially
        self._discovery_timer = self.create_timer(15.0, self._discovery_check)
        self.get_logger().info(f"spot_world up: URDF {len(self.joints)} joints, {len(DEPTH_CAMS)} depth cams, cumulative map {'ON' if MAP_ON else 'off'}, camera video via {'video_streams.py' if HAND_SDK else 'ROS'}")

    # ---- helpers -------------------------------------------------------------------------------
    def _discovery_check(self):
        """Exit with code 3 if no topic delivered anything in the first 15 s: this node instance never discovered the
        driver (FastDDS discovery on spot22 fails at random with this many participants). world.sh restarts us."""
        # the cameras come from the driver's depth_image_proc container, joints/odometry/TF from its state publisher: a
        # node instance can discover one and not the other (seen 2026-09-20: cameras only -> empty world, frustums at origin)
        missing = [name for name, n in (("camera infos", self.n_cam_msgs), ("joint states / odometry", self.n_state_msgs)) if n == 0]
        if self._T_latest(WORLD, BODY) is None:
            missing.append("TF vision->body")
        if missing:
            self.get_logger().error(f"nothing received in 15 s from: {', '.join(missing)} (DDS discovery failed) -> exiting for a restart")
            os._exit(3)
        self.destroy_timer(self._discovery_timer) if hasattr(self, "_discovery_timer") else None

    @staticmethod
    def _stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _T(self, target, source, stamp):
        """4x4 target_T_source at stamp (falls back to latest)."""
        try:
            tf = self.tf_buf.lookup_transform(target, source, Time(seconds=stamp.sec, nanoseconds=stamp.nanosec), timeout=rclpy.duration.Duration(seconds=0.05))
        except Exception:
            try:
                tf = self.tf_buf.lookup_transform(target, source, Time())
            except Exception:
                return None
        return self._tf_to_matrix(tf)

    def _T_latest(self, target, source):
        """4x4 target_T_source from the latest TF, no waiting."""
        try:
            return self._tf_to_matrix(self.tf_buf.lookup_transform(target, source, Time()))
        except Exception:
            return None

    @staticmethod
    def _tf_to_matrix(tf):
        q, t = tf.transform.rotation, tf.transform.translation
        x, y, z, w = q.x, q.y, q.z, q.w
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = [t.x, t.y, t.z]
        return M

    @staticmethod
    def _voxel_keys(pts, size):
        ijk = np.floor(pts / size).astype(np.int64) + (1 << 20)
        return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]

    # ---- cameras -------------------------------------------------------------------------------
    def _caminfo(self, cam, m):
        self.n_cam_msgs += 1
        if cam in self.K:
            return
        K = np.array(m.k).reshape(3, 3)
        stride = DEPTH_CAMS[cam][3]
        us, vs = np.meshgrid(np.arange(0, m.width, stride), np.arange(0, m.height, stride))
        self.grids[cam] = ((us + 0.5 - K[0, 2]) / K[0, 0], (vs + 0.5 - K[1, 2]) / K[1, 1], stride)
        self.K[cam] = (K, m.width, m.height, m.header.frame_id)
        # frustum: a plain entity under world/ (frame spot/vision); its pose is re-logged in _joints as an ordinary
        # entity transform computed from TF on spot22, like the clouds. Hanging the pinhole on the camera's TF frame via
        # parent/child frame edges drew the hand frustum under the belly although the logged transform was correct.
        rr.log(f"world/cams/{cam}", rr.Pinhole(image_from_camera=K, resolution=[m.width, m.height], camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=FRUSTUM_M), static=True)
        self.get_logger().info(f"{cam}: {m.width}x{m.height} f={K[0, 0]:.0f} frame {m.header.frame_id}")

    def _color(self, cam, m):
        enc = m.encoding.lower()
        if enc in ("rgb8", "bgr8"):
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            self.color[cam] = a if enc == "rgb8" else a[:, :, ::-1]
        elif enc in ("mono8", "8uc1"):
            g = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
            self.color[cam] = np.repeat(g[:, :, None], 3, axis=2)

    def _depth(self, cam, m):
        now = time.time()
        if cam not in self.K or cam not in self.color or now - self.last_cam_t[cam] < 1.0 / DEPTH_CAMS[cam][4]:
            return
        self.last_cam_t[cam] = now
        K, W, H, frame = self.K[cam]
        if m.encoding.lower() not in ("16uc1", "mono16"):
            return
        d = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width)
        xn, yn, stride = self.grids[cam]
        z = d[::stride, ::stride].astype(np.float32) / 1000.0
        ok = (z > Z_MIN) & (z < Z_MAX)
        if ok.sum() < 20:
            return
        pc = np.stack([xn[ok] * z[ok], yn[ok] * z[ok], z[ok]], 1)
        img = self.color[cam]
        if img.shape[0] != m.height or img.shape[1] != m.width:
            return
        rgb = img[::stride, ::stride][ok]                       # (N,3) uint8, same grid as the depth samples
        # camera frame -> world at the image stamp (the depth frame id is the optical frame the driver publishes)
        Mwc = self._T(WORLD, m.header.frame_id, m.header.stamp)
        if Mwc is None:
            return
        pw = pc @ Mwc[:3, :3].T + Mwc[:3, 3]
        # live cloud: voxel downsample
        keys = self._voxel_keys(pw, VOXEL_LIVE)
        _, first = np.unique(keys, return_index=True)
        live, live_rgb = pw[first], rgb[first]
        # colour transfer: the body cameras are greyscale; points close to the (RGB) hand cloud borrow its colours
        if cam == "hand":
            self.rgb_cloud = (cKDTree(live), live_rgb, now)
        elif self.rgb_cloud is not None and now - self.rgb_cloud[2] < RGB_CLOUD_MAX_AGE and COLOR_MATCH_M > 0:
            tree, rgb_colors, _ = self.rgb_cloud
            # cheap prefilter: only points inside the hand cloud's bounding box can be within COLOR_MATCH_M of it
            inbox = np.all((live >= tree.mins - COLOR_MATCH_M) & (live <= tree.maxes + COLOR_MATCH_M), axis=1)
            if inbox.any():
                dist, idx = tree.query(live[inbox], k=1, distance_upper_bound=COLOR_MATCH_M)
                near = dist < COLOR_MATCH_M
                if near.any():
                    sel = np.flatnonzero(inbox)[near]
                    live_rgb = live_rgb.copy(); live_rgb[sel] = rgb_colors[idx[near]]
        rr.set_time("ros", timestamp=self._stamp(m))
        if cam not in self.live_frame_set:
            rr.log(f"world/live/{cam}", rr.CoordinateFrame(WORLD), static=True); self.live_frame_set.add(cam)
        rr.log(f"world/live/{cam}", rr.Points3D(live.astype(np.float32), colors=live_rgb, radii=0.012))
        self.n_depth += 1
        if not MAP_ON:
            return
        # map: add voxels (world frame, coarser)
        mk = self._voxel_keys(pw, VOXEL_MAP)
        _, first = np.unique(mk, return_index=True)          # one candidate per voxel
        tiles = np.floor(pw[first] / TILE).astype(np.int64)
        with self.lock:
            pend = self.pending
            for k, p, c, tk in zip(mk[first].tolist(), pw[first], rgb[first], map(tuple, tiles.tolist())):
                tile = self.map.get(tk)
                if tile is not None and k in tile:
                    continue
                n = pend.get(k, 0) + 1
                if n >= MAP_MIN_HITS:
                    pend.pop(k, None)
                    self.map.setdefault(tk, {})[k] = np.concatenate([p, c.astype(np.float32)]); self.n_voxels += 1; self.dirty_tiles.add(tk)
                else:
                    pend[k] = n
            if len(pend) > 3_000_000:
                pend.clear()

    def _hand_rgb(self, m):
        """ROS fallback (HAND_SDK=0): the driver's hand image, ~4 Hz at best, throttled to 2 Hz."""
        if time.time() - self.last_rgb_t < 0.5:
            return
        self.last_rgb_t = time.time()
        enc = m.encoding.lower()
        if enc not in ("rgb8", "bgr8"):
            return
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        rgb = a[:, :, ::-1] if enc == "bgr8" else a
        rr.set_time("ros", timestamp=self._stamp(m))
        rr.log("world/cams/hand/video", rr.Image(np.ascontiguousarray(rgb)).compress(jpeg_quality=70))

    # ---- robot ---------------------------------------------------------------------------------
    def _log_tf(self, path, parent, child, M, static=False):
        R = M[:3, :3]
        # rotation matrix -> quaternion (x, y, z, w)
        t = np.trace(R)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2; w = 0.25 * s; x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
        else:
            i = int(np.argmax(np.diag(R)))
            if i == 0:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
            elif i == 1:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
        if parent is None:
            rr.log(path, rr.Transform3D(translation=M[:3, 3], quaternion=[x, y, z, w]), static=static)
        else:
            rr.log(path, rr.Transform3D(translation=M[:3, 3], quaternion=[x, y, z, w], parent_frame=parent, child_frame=child), static=static)

    def _odom(self, m):
        self.n_state_msgs += 1
        now = time.time()
        if now - getattr(self, "_last_odom_t", 0.0) < 1.0 / POSE_HZ:
            return
        self._last_odom_t = now
        M = self._T(WORLD, BODY, m.header.stamp)
        if M is None:
            return
        rr.set_time("ros", timestamp=self._stamp(m))
        self._log_tf("tf/body", WORLD, BODY, M)
        p = M[:3, 3]
        if not self.trail or np.linalg.norm(p - self.trail[-1]) > 0.02:
            self.trail.append(p.copy())
            if len(self.trail) % 5 == 0:
                rr.log("world/trail", rr.LineStrips3D([np.array(self.trail, np.float32)], colors=[[255, 255, 255]], radii=0.01))

    def _joints(self, m):
        self.n_state_msgs += 1
        now = time.time()
        if now - self.last_joint_t < 1.0 / POSE_HZ:
            return
        self.last_joint_t = now
        rr.set_time("ros", timestamp=self._stamp(m))
        for name, pos in zip(m.name, m.position):
            j = self.joints.get(name)
            if j is not None:
                rr.log(f"tf/joints/{name}", j.compute_transform(float(pos), clamp=False))
        # camera frustum poses in the world frame (entity transforms). Latest TF, not the joint stamp: a stamped lookup
        # with a timeout polls can_transform for up to 50 ms while the TF for that stamp is still in flight, which cost
        # most of a core at 6 cameras x 15 Hz.
        for cam, (K, W, H, frame) in list(self.K.items()):
            M = self._T_latest(WORLD, frame)
            if M is not None:
                self._log_tf(f"world/cams/{cam}", None, None, M)

    # ---- map -----------------------------------------------------------------------------------
    def _log_map(self):
        with self.lock:
            if not self.dirty_tiles:
                return
            if self.n_voxels > MAP_MAX_VOXELS:               # drop whole tiles farthest from the robot's last position
                here = self.trail[-1] if self.trail else np.zeros(3)
                far = sorted(self.map, key=lambda tk: -np.linalg.norm(np.array(tk) * TILE + TILE / 2 - here))
                while self.n_voxels > MAP_MAX_VOXELS * 0.8 and far:
                    tk = far.pop(0); self.n_voxels -= len(self.map.pop(tk)); self.dirty_tiles.discard(tk)
                    rr.log(f"world/map/{tk[0]}_{tk[1]}_{tk[2]}", rr.Clear(recursive=False))
            todo = [(tk, np.array(list(self.map[tk].values()), np.float32)) for tk in self.dirty_tiles if tk in self.map]
            self.dirty_tiles.clear()
            total = self.n_voxels
        rr.set_time("ros", timestamp=time.time())
        for tk, arr in todo:
            rr.log(f"world/map/{tk[0]}_{tk[1]}_{tk[2]}", rr.Points3D(arr[:, :3], colors=arr[:, 3:6].astype(np.uint8), radii=0.02))
        if not getattr(self, "_map_frame_set", False):
            rr.log("world/map", rr.CoordinateFrame(WORLD), static=True)
            rr.log("world/trail", rr.CoordinateFrame(WORLD), static=True)
            self._map_frame_set = True
        self.get_logger().info(f"map {total} voxels in {len(self.map)} tiles ({len(todo)} re-sent), {self.n_depth} depth frames", throttle_duration_sec=30.0)

def main():
    rr.init("spot_world", recording_id=RECORDING_ID)
    grpc_port, web_port = int(os.environ.get("RR_GRPC_PORT", 9876)), int(os.environ.get("RR_WEB_PORT", 9090))
    record = os.environ.get("RR_RECORD", "0") == "1"           # streaming only by default; RR_RECORD=1 also writes an .rrd
    CORS = [f"http://192.168.1.213:{web_port}", f"http://localhost:{web_port}", f"http://127.0.0.1:{web_port}", "*"]
    # History replayed to a (re)connecting viewer. Static data (URDF meshes, frames) is always kept; this bounds only the
    # dynamic part, so a fresh browser tab gets the model plus the last seconds instead of minutes of clouds. Do not go
    # much lower: at 16 MiB with the H.264 video also in the buffer the server evicted ALL of this node's transforms and
    # clouds and viewers showed an empty world (only the video survived); 64 MiB is a few seconds of clouds + video.
    HISTORY = os.environ.get("RR_HISTORY", "64MiB")
    if record:
        rec_dir = os.path.join(HERE, "recordings"); os.makedirs(rec_dir, exist_ok=True)
        rec = os.path.join(rec_dir, f"spot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.rrd")
        rr.set_sinks(rr.GrpcServerSink(port=grpc_port, server_memory_limit=HISTORY, newest_first=True, cors_allow_origin=CORS), rr.FileSink(rec))
    else:
        rec = "off (RR_RECORD=1 to record)"
        rr.serve_grpc(grpc_port=grpc_port, server_memory_limit=HISTORY, newest_first=True, cors_allow_origin=CORS)
    rr.serve_web_viewer(web_port=web_port, open_browser=False, connect_to=f"rerun+http://192.168.1.213:{grpc_port}/proxy")
    viewer_url = f"http://192.168.1.213:{web_port}/?url=rerun%2Bhttp%3A%2F%2F192.168.1.213%3A{grpc_port}%2Fproxy"
    print(f"OPEN THIS: {viewer_url}\n(the bare :{web_port} page is an empty viewer; the ?url= part tells it where the data is)  | recording {rec}", flush=True)
    # world frame + view coordinates (z up)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world", rr.CoordinateFrame(WORLD), static=True)
    # link the odometry world frame to the viewer root frame (identity), so views rooted at "/" can place everything
    rr.log("tf/world", rr.Transform3D(translation=[0.0, 0.0, 0.0], quaternion=[0.0, 0.0, 0.0, 1.0], parent_frame="tf#/", child_frame=WORLD), static=True)
    # explicit layout: the 3D world and the hand video. Sent as active+default so a viewer's stale saved layout (e.g. a
    # panel for an entity that no longer exists) is replaced instead of persisting across restarts. The 2D view is rooted
    # at the video entity, not at the Pinhole entity: that one's frame is the 3D camera frame, which has no pinhole root.
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(rrb.Spatial3DView(origin="/", name="world"), rrb.Spatial2DView(origin="world/cams/hand/video", name="hand camera"), column_shares=[3, 1]),
        collapse_panels=False), make_active=True, make_default=True)
    rclpy.init()
    node = SpotWorld()
    ex = MultiThreadedExecutor(num_threads=4); ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == "__main__":
    main()
