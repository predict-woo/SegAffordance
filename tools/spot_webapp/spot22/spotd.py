#!/usr/bin/env python3
"""spotd: resident Spot command daemon (spot22).

One long-lived ROS 2 node that stays connected to the spot_ros2 driver and answers one-line commands
on a Unix socket, so a command costs ~50 ms instead of a fresh `ros2` CLI start-up (2-5 s) per call.
Holds warm: Trigger service clients, a persistent arm-pose publisher (no dropped one-shot messages),
a cmd_vel publisher with an expiring target, a TF listener for the hand pose, and cached status topics.

Run through spotd.sh (sets RMW/domain and sources ROS). Talk to it with `spotctl <cmd> [args]`.
Protocol: client sends one line, args separated by TAB; server replies "OK <text>" or "ERR <text>"
(possibly several lines) and closes.  Every command is appended to spotd.log.

Safety: `stop` is handled on its own connection thread, so it goes through even while a pose/vel is
in flight; arm poses are clamped to WORKSPACE unless --force; nudges are bounded per call; base
velocity is capped and time-limited with auto-stop; estop-hard needs --yes.
"""
from __future__ import annotations

import json
import math
import os
import socket
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from scipy.spatial.transform import Rotation

import tf2_ros
import yaml
import bosdyn.client
from bosdyn.client.robot_state import RobotStateClient
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import CameraInfo, Image as ImageMsg
from spot_msgs.msg import BatteryStateArray, EStopStateArray, LeaseArray, PowerState
from spot_msgs.srv import SetGripperAngle
from spot_msgs.action import Trajectory as TrajectoryAction
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration as DurationMsg
from spot_msgs.srv import RobotCommand as RobotCommandSrv
from bosdyn_api_msgs.msg import RobotCommand as RobotCommandMsg
from bosdyn_msgs.conversions import convert as bd_convert
from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2, trajectory_pb2
from bosdyn.util import seconds_to_duration
from std_srvs.srv import Trigger

HERE = os.path.dirname(os.path.abspath(__file__))
SOCK = os.environ.get("SPOTD_SOCK", "/tmp/spotd.sock")
LOG = os.path.join(HERE, "spotd.log")
POSES = os.path.join(HERE, "poses.json")

NS = "/spot"
SPOT_CFG = "/home/spot/dev/ros2_ws/install/locopt_ros/share/locopt_ros/config/spot_config.yaml"   # username/password/hostname
BASE_FRAME, HAND_FRAME, CMD_FRAME = "spot/body", "spot/hand", "body"

# body-frame box the hand may be commanded into (m). Unstow is x=1.0 z=0.18; half-stow x=0.7 z=0.3.
WORKSPACE = dict(x=(0.30, 1.10), y=(-0.55, 0.55), z=(-0.25, 0.75))
NUDGE_MAX_M, NUDGE_MAX_DEG = 0.15, 30.0
VEL_MAX, TURN_MAX, VEL_T_MAX = 0.5, 1.0, 3.0          # m/s, rad/s, s
POSE_TOL_M, POSE_TOL_DEG, POSE_WAIT = 0.02, 3.0, 8.0

TRIGGERS = {  # command -> service (relative to NS); MOVES marks commands that move the robot
    "claim": "claim", "release": "release", "take": "take_lease",
    "poweron": "power_on", "poweroff": "power_off",
    "stand": "stand", "sit": "sit", "stop": "stop", "selfright": "self_right",
    "stow": "arm_stow", "unstow": "arm_unstow", "carry": "arm_carry",
    "open": "open_gripper", "close": "close_gripper",
    "estop": "estop/gentle", "estop-hard": "estop/hard", "estop-release": "estop/release",
    "clear-fault": "clear_behavior_fault",
}
MOTOR = {0: "unknown", 1: "off", 2: "on", 3: "powering-on", 4: "powering-off", 5: "ERROR"}
SHORE = {0: "unknown", 1: "ON-SHORE-POWER", 2: "off"}
ESTOP = {0: "unknown", 1: "ESTOPPED", 2: "clear"}
BATT = {0: "unknown", 1: "missing", 2: "charging", 3: "discharging", 4: "booting"}

HELP = """spotctl commands (one line each; MOVES = the robot moves)
  status | hand | help                    health line / hand pose in body frame / this text
  claim | take | release                  lease (take = force-take a stale lease)
  poweron | poweroff                      motors
  stand | sit | selfright                 MOVES
  stop                                    halt base + arm, stay standing (handled with priority)
  stow | unstow | carry                   MOVES arm presets
  open | close | grip <0-90>              gripper (grip = opening angle in degrees)
  tf PARENT CHILD                         4x4 transform as json (e.g. tf spot/hand spot/hand_color_image_sensor)
  poseq X Y Z QX QY QZ QW [--nowait] [--force]   MOVES hand, quaternion form
  pose X Y Z [-r ROLL] [-p PITCH] [-y YAW] [--nowait] [--force]   MOVES hand to body-frame pose (m, deg)
  nudge [-x DX] [-y DY] [-z DZ] [-r DR] [-p DP] [-w DYAW]         MOVES hand relative to where it is
  go <name> | save <name> | poses         named poses from poses.json (save = store current hand pose)
  vel VX VY YAW [-t SECS]                 MOVES base for SECS (default 1.0, max 3.0), then auto-stops
  estop | estop-release | estop-hard --yes    gentle E-stop (settles + motors off) / clear it / cut power
  recover [--no-stand]                    estop-release -> poweroff -> poweron -> stand
  clear-fault                             clear behaviour fault
  walkto X Y YAW_DEG [--t S] [--loose] [--dry]   MOVES BASE to a pose in the current body frame (m, deg); `stop` cancels
  traj FILE.json --smooth [--speed 0.01] [--approach-speed 0.015] [--reverse] [--no-grasp] [--dry]
                                          MOVES arm as ONE timed trajectory (continuous + slow): open -> glide to pre-grasp -> glide into
                                          grasp -> close -> arc -> open -> glide back; `stop` cancels
  traj FILE.json [--pause S] [--approach-step M] [--approach-pause S] [--reverse] [--no-grasp] [--dry]   stepped fallback
                                          MOVES arm: open -> ease to pre-grasp -> ease into grasp -> close -> waypoints -> open -> ease back; `stop` aborts
  snap [PATH] [--depth]                   save one hand-camera RGB frame (jpg) + K in PATH.json; --depth adds 16-bit depth png"""


def log(line: str) -> None:
    with open(LOG, "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {line}\n")


def parse_kv(args, flags):
    """Pull `-k value` pairs listed in `flags` out of args; return (positional, {k: float}, set(bare flags))."""
    pos, kv, bare = [], {}, set()
    i = 0
    while i < len(args):
        a = args[i]
        if a in flags and i + 1 < len(args):
            kv[a] = float(args[i + 1]); i += 2
        elif a.startswith("--"):
            bare.add(a); i += 1
        else:
            pos.append(a); i += 1
    return pos, kv, bare


class SpotD(Node):
    def __init__(self):
        super().__init__("spotd")
        self.lock = threading.Lock()
        self.cli = {k: self.create_client(Trigger, f"{NS}/{v}") for k, v in TRIGGERS.items()}
        self.grip_cli = self.create_client(SetGripperAngle, f"{NS}/set_gripper_angle")
        self.traj_action = ActionClient(self, TrajectoryAction, f"{NS}/trajectory")
        self.walk_goal = None
        self.rc_cli = self.create_client(RobotCommandSrv, f"{NS}/robot_command")   # service, not action: the rclpy action client crashed the daemon from a worker thread
        self.rc_goal = None
        self.pose_pub = self.create_publisher(PoseStamped, f"{NS}/arm_pose_commands", 10)
        self.vel_pub = self.create_publisher(Twist, f"{NS}/cmd_vel", 10)
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)
        self.power = self.estop = self.leases = self.batt = None
        self.create_subscription(PowerState, f"{NS}/status/power_states", self._cb("power"), 10)
        self.create_subscription(EStopStateArray, f"{NS}/status/estop", self._cb("estop"), 10)
        self.create_subscription(LeaseArray, f"{NS}/status/leases", self._cb("leases"), 10)
        self.create_subscription(BatteryStateArray, f"{NS}/status/battery_states", self._cb("batt"), 10)
        # base velocity: target twist with expiry, streamed at 10 Hz so the driver's cmd_duration never gaps
        self.vel_target, self.vel_until = None, 0.0
        self.create_timer(0.1, self._vel_tick)
        self.abort = threading.Event()   # set by `stop`; traj execution checks it between waypoints
        # Direct SDK robot-state client (read-only, no lease). The driver's C++ state_publisher publishes garbage in
        # /spot/status/power_states after a startup race (2026-09-19), so power / shore / faults come from here.
        self.sdk_robot, self.sdk_state, self.sdk_err = None, None, None
        self.sdk_lock = threading.Lock()
        self.traj_busy = False

    def _cb(self, name):
        def f(msg):
            setattr(self, name, msg)
        return f

    # ---- low level -------------------------------------------------------------------------
    def trigger(self, key: str, timeout=20.0) -> tuple[bool, str]:
        c = self.cli[key]
        if not c.wait_for_service(timeout_sec=2.0):
            return False, f"service {TRIGGERS[key]} not available (driver down or wrong domain)"
        fut = c.call_async(Trigger.Request())
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                return False, f"{TRIGGERS[key]} timed out after {timeout:.0f}s"
            time.sleep(0.02)
        r = fut.result()
        return bool(r.success), r.message or ("Success" if r.success else "failed")

    def hand_pose(self):
        """(xyz[3], rpy_deg[3], quat_xyzw[4]) of spot/hand in spot/body, or None."""
        try:
            t = self.tf_buf.lookup_transform(BASE_FRAME, HAND_FRAME, rclpy.time.Time())
        except Exception:
            return None
        tr, q = t.transform.translation, t.transform.rotation
        quat = np.array([q.x, q.y, q.z, q.w])
        rpy = Rotation.from_quat(quat).as_euler("xyz", degrees=True)
        return np.array([tr.x, tr.y, tr.z]), rpy, quat

    def send_pose(self, xyz, rpy_deg, wait=True, force=False) -> tuple[bool, str]:
        for k, v in zip("xyz", xyz):
            lo, hi = WORKSPACE[k]
            if not force and not (lo <= v <= hi):
                return False, f"{k}={v:.3f} outside workspace {lo}..{hi} m (use --force to override)"
        q = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_quat()
        msg = PoseStamped()
        msg.header.frame_id = CMD_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, xyz)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, q)
        self.pose_pub.publish(msg)
        if not wait:
            return True, "sent"
        t0 = time.time()
        while time.time() - t0 < POSE_WAIT:
            time.sleep(0.1)
            hp = self.hand_pose()
            if hp is None:
                continue
            dpos = float(np.linalg.norm(hp[0] - np.asarray(xyz)))
            dang = float(np.degrees((Rotation.from_quat(hp[2]).inv() * Rotation.from_quat(q)).magnitude()))
            if dpos < POSE_TOL_M and dang < POSE_TOL_DEG:
                return True, f"converged in {time.time() - t0:.1f}s (pos err {dpos * 100:.1f} cm, ang err {dang:.1f} deg)"
        return False, f"not converged after {POSE_WAIT:.0f}s: pos err {dpos * 100:.1f} cm, ang err {dang:.1f} deg"

    def send_pose_q(self, xyz, quat_xyzw, wait_s=0.0, tol_m=0.02, tol_deg=4.0):
        msg = PoseStamped()
        msg.header.frame_id = CMD_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = map(float, xyz)
        msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = map(float, quat_xyzw)
        self.pose_pub.publish(msg)
        t0 = time.time()
        while wait_s > 0 and time.time() - t0 < wait_s and not self.abort.is_set():
            time.sleep(0.05)
            hp = self.hand_pose()
            if hp is None:
                continue
            dpos = float(np.linalg.norm(hp[0] - np.asarray(xyz)))
            dang = float(np.degrees((Rotation.from_quat(hp[2]).inv() * Rotation.from_quat(quat_xyzw)).magnitude()))
            if dpos < tol_m and dang < tol_deg:
                break

    def _ease(self, label, xyz0, q0, xyz1, q1, step_m, pause):
        """Move from pose 0 to pose 1 in straight-line steps of step_m (slerp for orientation), pausing at each."""
        xyz0, xyz1 = np.asarray(xyz0, float), np.asarray(xyz1, float)
        n = max(1, int(np.ceil(np.linalg.norm(xyz1 - xyz0) / step_m)))
        r0, r1 = Rotation.from_quat(q0), Rotation.from_quat(q1)
        for k in range(1, n + 1):
            if self.abort.is_set():
                raise InterruptedError(f"{label} step {k}/{n}")
            s = k / n
            q = (r0 * ((r0.inv() * r1) ** s)).as_quat()
            self.send_pose_q(xyz0 + (xyz1 - xyz0) * s, q, wait_s=pause)
            time.sleep(pause)
        return n

    def run_traj(self, spec, pause, grasp, dry, reverse=False, approach_step=0.01, approach_pause=0.4) -> tuple[bool, str]:
        """open -> ease to pre-grasp -> ease into the grasp pose -> close -> waypoints (slow) -> open -> ease back.
        `stop` aborts between steps. --reverse runs the waypoints backwards (close the door again)."""
        if self.traj_busy:
            return False, "a trajectory is already running (spotctl stop aborts it)"
        wps = list(spec["waypoints"])
        if reverse:
            wps = wps[::-1]
        approach_m = float(spec.get("approach_m", 0.12))
        # pre-grasp = first waypoint backed off along the hand's own -x (into-the-door direction)
        r_first = Rotation.from_quat(wps[0]["quat_xyzw"])
        hand_x = r_first.apply([1.0, 0.0, 0.0])
        pre_xyz = np.asarray(wps[0]["xyz"]) - hand_x * approach_m
        pre_q = wps[0]["quat_xyzw"]
        if dry:
            return True, (f"dry run: {'REVERSED ' if reverse else ''}{len(wps)} waypoints, pause {pause}s each, approach in "
                          f"{approach_step * 100:.0f} cm steps at {approach_pause}s, {'with' if grasp else 'without'} gripper close; nothing sent")
        self.traj_busy = True
        self.abort.clear()
        try:
            self.trigger("open"); time.sleep(0.5)
            hp = self.hand_pose()
            if hp is None:
                return False, "no hand TF"
            # 1. current pose -> pre-grasp, 3 cm steps (free space)
            self._ease("to-pre-grasp", hp[0], hp[2], pre_xyz, pre_q, 0.03, approach_pause)
            time.sleep(0.5)
            # 2. pre-grasp -> grasp pose, fine steps: this is the part that used to snap
            self._ease("approach", pre_xyz, pre_q, wps[0]["xyz"], wps[0]["quat_xyzw"], approach_step, approach_pause)
            self.send_pose_q(wps[0]["xyz"], wps[0]["quat_xyzw"], wait_s=3.0); time.sleep(0.5)
            if grasp:
                self.trigger("close"); time.sleep(1.0)
            # 3. the arc
            t0 = time.time()
            for k, w in enumerate(wps[1:], 1):
                if self.abort.is_set():
                    raise InterruptedError(f"wp{k}")
                self.send_pose_q(w["xyz"], w["quat_xyzw"], wait_s=pause)
                time.sleep(pause)
            dt = time.time() - t0
            # 4. release and ease back out along the final hand -x
            self.trigger("open"); time.sleep(0.8)
            r_last = Rotation.from_quat(wps[-1]["quat_xyzw"])
            back = np.asarray(wps[-1]["xyz"]) - r_last.apply([1.0, 0.0, 0.0]) * approach_m
            self._ease("retreat", wps[-1]["xyz"], wps[-1]["quat_xyzw"], back, wps[-1]["quat_xyzw"], approach_step, approach_pause)
            return True, (f"done{' (reversed)' if reverse else ''}: {len(wps)} waypoints in {dt:.1f}s, released, eased back {approach_m * 100:.0f} cm; "
                          f"hand now {self.hand_str()}")
        except InterruptedError as ex:
            self.trigger("stop")
            return False, f"ABORTED at {ex} by stop; arm holds its current pose"
        finally:
            self.traj_busy = False

    # ---- smooth (timed) arm trajectories via /spot/robot_command --------------------------------
    @staticmethod
    def _arm_cart_cmd(points, times, ref_frame=CMD_FRAME, max_lin=0.06, max_ang=0.5):
        """One ArmCartesianCommand whose SE3Trajectory holds every waypoint with a timestamp: the robot
        interpolates smoothly through them at the speed the timestamps imply. Velocity caps are a safety net."""
        traj = trajectory_pb2.SE3Trajectory()
        for (xyz, q), tt in zip(points, times):
            pt = traj.points.add()
            pt.pose.CopyFrom(geometry_pb2.SE3Pose(position=geometry_pb2.Vec3(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                                                  rotation=geometry_pb2.Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))))
            pt.time_since_reference.CopyFrom(seconds_to_duration(float(tt)))
        cart = arm_command_pb2.ArmCartesianCommand.Request(root_frame_name=ref_frame, pose_trajectory_in_task=traj,
                                                           force_remain_near_current_joint_configuration=True)
        cart.max_linear_velocity.value = max_lin
        cart.max_angular_velocity.value = max_ang
        arm = arm_command_pb2.ArmCommand.Request(arm_cartesian_command=cart)
        sync = synchronized_command_pb2.SynchronizedCommand.Request(arm_command=arm)
        return robot_command_pb2.RobotCommand(synchronized_command=sync)

    def _send_robot_command(self, proto, timeout_s, label, final_pose=None) -> tuple[bool, str]:
        """Send a RobotCommand proto through the driver's robot_command SERVICE (it holds the lease), then wait out
        the trajectory duration while watching for `stop`. The service returns as soon as the robot accepts the
        command; the robot executes the timed trajectory on its own. final_pose=(xyz, quat) enables early exit."""
        msg = RobotCommandMsg()
        bd_convert(proto, msg)
        if not self.rc_cli.wait_for_service(timeout_sec=2.0):
            return False, f"{label}: /spot/robot_command service not available"
        req = RobotCommandSrv.Request(command=msg)
        req.duration = DurationMsg(sec=int(timeout_s) + 2, nanosec=0)          # command end time on the robot
        fut = self.rc_cli.call_async(req)
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > 5:
                return False, f"{label}: service call timed out"
            time.sleep(0.02)
        res = fut.result()
        if not res.success:
            return False, f"{label}: rejected: {res.message}"
        while time.time() - t0 < timeout_s:
            if self.abort.is_set():
                self.trigger("stop")
                return False, f"{label}: ABORTED by stop after {time.time() - t0:.1f}s"
            time.sleep(0.1)
            if final_pose is not None and time.time() - t0 > 1.0:
                hp = self.hand_pose()
                if hp is not None:
                    dpos = float(np.linalg.norm(hp[0] - np.asarray(final_pose[0])))
                    dang = float(np.degrees((Rotation.from_quat(hp[2]).inv() * Rotation.from_quat(final_pose[1])).magnitude()))
                    if dpos < 0.01 and dang < 2.0:
                        break
        hp = self.hand_pose()
        tail = ""
        if final_pose is not None and hp is not None:
            dpos = float(np.linalg.norm(hp[0] - np.asarray(final_pose[0])))
            tail = f", end error {dpos * 100:.1f} cm"
        return True, f"{label}: done in {time.time() - t0:.1f}s{tail}"

    def _smooth_move(self, xyz, q, speed, label, min_t=2.0):
        """Single-target timed move: duration = distance / speed (>= min_t), so the arm glides instead of snapping."""
        hp = self.hand_pose()
        dist = float(np.linalg.norm(np.asarray(xyz) - hp[0])) if hp is not None else 0.3
        T = max(min_t, dist / speed)
        ok, msg = self._send_robot_command(self._arm_cart_cmd([(xyz, q)], [T]), T + 3.0, f"{label} ({dist * 100:.0f} cm in {T:.0f}s)", final_pose=(xyz, q))
        if not ok:
            raise RuntimeError(msg)
        return msg

    def run_traj_smooth(self, spec, speed, grasp, dry, reverse=False, approach_speed=0.015) -> tuple[bool, str]:
        """open -> glide to pre-grasp -> glide into grasp -> close -> ONE timed arc -> open -> glide back."""
        if self.traj_busy:
            return False, "a trajectory is already running (spotctl stop aborts it)"
        wps = list(spec["waypoints"])[::-1] if reverse else list(spec["waypoints"])
        approach_m = float(spec.get("approach_m", 0.12))
        r_first = Rotation.from_quat(wps[0]["quat_xyzw"])
        pre_xyz = (np.asarray(wps[0]["xyz"]) - r_first.apply([1.0, 0.0, 0.0]) * approach_m).tolist()
        P = np.array([w["xyz"] for w in wps])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        times = np.concatenate([[0.5], 0.5 + np.cumsum(seg) / speed])
        arc_pts = [(w["xyz"], w["quat_xyzw"]) for w in wps]
        arc_cmd = self._arm_cart_cmd(arc_pts, times)
        r_last = Rotation.from_quat(wps[-1]["quat_xyzw"])
        back_xyz = (np.asarray(wps[-1]["xyz"]) - r_last.apply([1.0, 0.0, 0.0]) * approach_m).tolist()
        plan = (f"{'REVERSED ' if reverse else ''}{len(wps)} waypoints, arc {seg.sum() * 100:.0f} cm over {times[-1]:.0f}s at {speed * 100:.1f} cm/s; "
                f"approach {approach_m * 100:.0f} cm at {approach_speed * 100:.1f} cm/s (~{approach_m / approach_speed:.0f}s); "
                f"{'with' if grasp else 'without'} gripper close")
        if dry:
            msg = RobotCommandMsg(); bd_convert(arc_cmd, msg)   # prove the proto -> ROS conversion works
            n = len(arc_cmd.synchronized_command.arm_command.arm_cartesian_command.pose_trajectory_in_task.points)
            return True, f"dry run: {plan}; built + converted a {n}-point timed arc; nothing sent"
        self.traj_busy = True
        self.abort.clear()
        log_lines = [plan]
        try:
            self.trigger("open"); time.sleep(0.5)
            log_lines.append(self._smooth_move(pre_xyz, wps[0]["quat_xyzw"], 0.05, "to pre-grasp"))
            time.sleep(0.5)
            log_lines.append(self._smooth_move(wps[0]["xyz"], wps[0]["quat_xyzw"], approach_speed, "ease into grasp", min_t=4.0))
            time.sleep(0.5)
            if grasp:
                self.trigger("close"); time.sleep(1.0)
            if self.abort.is_set():
                raise RuntimeError("ABORTED by stop before the arc")
            ok, msg = self._send_robot_command(arc_cmd, float(times[-1]) + 4.0, "arc", final_pose=(wps[-1]["xyz"], wps[-1]["quat_xyzw"]))
            log_lines.append(msg)
            if not ok:
                raise RuntimeError(msg)
            self.trigger("open"); time.sleep(0.8)
            log_lines.append(self._smooth_move(back_xyz, wps[-1]["quat_xyzw"], approach_speed, "retreat", min_t=4.0))
            return True, "done: " + " | ".join(log_lines) + f" | hand now {self.hand_str()}"
        except RuntimeError as ex:
            self.trigger("stop")
            return False, f"{ex}  [steps so far: {' | '.join(log_lines)}]  arm holds its current pose"
        finally:
            self.traj_busy = False

    # ---- base ------------------------------------------------------------------------------
    def walkto(self, x, y, yaw_deg, duration_s=20.0, precise=True, dry=False) -> tuple[bool, str]:
        """Base goal in the CURRENT body frame via the driver's /spot/trajectory action (it requires frame 'body')."""
        dist = math.hypot(x, y)
        if dist > 3.0 or abs(yaw_deg) > 180:
            return False, f"refusing: {dist:.2f} m / {yaw_deg:.0f} deg is more than the 3 m single-step limit"
        if dry:
            return True, f"dry run: would walk to x={x:+.3f} y={y:+.3f} yaw={yaw_deg:+.1f} deg (body frame), {dist:.2f} m, timeout {duration_s:.0f}s; nothing sent"
        if not self.traj_action.wait_for_server(timeout_sec=2.0):
            return False, "/spot/trajectory action server not available"
        self.abort.clear()
        g = TrajectoryAction.Goal()
        g.target_pose.header.frame_id = "body"
        g.target_pose.header.stamp = self.get_clock().now().to_msg()
        g.target_pose.pose.position.x, g.target_pose.pose.position.y = float(x), float(y)
        q = Rotation.from_euler("z", math.radians(yaw_deg)).as_quat()
        g.target_pose.pose.orientation.x, g.target_pose.pose.orientation.y, g.target_pose.pose.orientation.z, g.target_pose.pose.orientation.w = map(float, q)
        g.duration = DurationMsg(sec=int(duration_s), nanosec=0)
        g.precise_positioning = bool(precise)
        fut = self.traj_action.send_goal_async(g)
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > 5:
                return False, "goal not accepted within 5 s"
            time.sleep(0.05)
        gh = fut.result()
        if not gh.accepted:
            return False, "trajectory goal rejected by the driver"
        self.walk_goal = gh
        rf = gh.get_result_async()
        while not rf.done():
            if time.time() - t0 > duration_s + 5:
                gh.cancel_goal_async(); self.walk_goal = None
                return False, f"walk timed out after {duration_s:.0f}s; goal cancelled"
            time.sleep(0.1)
        self.walk_goal = None
        res = rf.result().result
        if self.abort.is_set():
            return False, "walk ABORTED by stop"
        return bool(res.success), f"{res.message or ('arrived' if res.success else 'failed')} after {time.time() - t0:.1f}s"

    def _vel_tick(self):
        if self.vel_target is None:
            return
        if time.time() >= self.vel_until:
            self.vel_pub.publish(Twist())          # zero = stop
            self.vel_target = None
            return
        self.vel_pub.publish(self.vel_target)

    def stop_all(self):
        self.abort.set()
        for gh in (self.walk_goal,):
            if gh is not None:
                try:
                    gh.cancel_goal_async()
                except Exception:
                    pass
        self.vel_target = None
        self.vel_pub.publish(Twist())


    # ---- camera ----------------------------------------------------------------------------
    def _grab_once(self, msg_type, topic, timeout=3.0):
        """Subscribe, wait for one message, unsubscribe. Local DDS on spot22, so no WiFi cost."""
        ev, box = threading.Event(), {}

        def cb(m):
            if not box:
                box["m"] = m
                ev.set()
        sub = self.create_subscription(msg_type, topic, cb, 1)
        try:
            ev.wait(timeout)
        finally:
            self.destroy_subscription(sub)
        return box.get("m")

    @staticmethod
    def _img_to_np(m):
        enc = m.encoding.lower()
        if enc in ("rgb8", "bgr8"):
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            return a[:, :, ::-1].copy() if enc == "rgb8" else a.copy(), enc
        if enc in ("mono8", "8uc1"):
            return np.frombuffer(m.data, np.uint8).reshape(m.height, m.width).copy(), enc
        if enc in ("16uc1", "mono16"):
            return np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).copy(), enc
        if enc == "rgba8" or enc == "bgra8":
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 4)[:, :, :3]
            return a[:, :, ::-1].copy() if enc == "rgba8" else a.copy(), enc
        raise ValueError(f"unsupported encoding {m.encoding}")

    def snap(self, path=None, depth=False) -> tuple[bool, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path or os.path.join(HERE, "snaps", f"hand_{ts}.jpg")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        m = self._grab_once(ImageMsg, f"{NS}/camera/hand/image")
        if m is None:
            return False, "no image on /spot/camera/hand/image within 3 s (driver up? camera streaming?)"
        img, enc = self._img_to_np(m)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        meta = {"topic": f"{NS}/camera/hand/image", "encoding": m.encoding, "width": m.width, "height": m.height,
                "frame_id": m.header.frame_id, "stamp": m.header.stamp.sec + m.header.stamp.nanosec * 1e-9}
        ci = self._grab_once(CameraInfo, f"{NS}/camera/hand/camera_info", timeout=2.0)
        if ci is not None:
            meta["K"] = [list(map(float, ci.k[i * 3:(i + 1) * 3])) for i in range(3)]
            meta["distortion_model"], meta["D"] = ci.distortion_model, list(map(float, ci.d))
        hp = self.hand_pose()
        if hp is not None:
            meta["hand_in_body"] = {"xyz": [float(v) for v in hp[0]], "rpy_deg": [float(v) for v in hp[1]], "quat_xyzw": [float(v) for v in hp[2]]}
        try:  # camera optical frame in body: lets a viewer place the cloud in the robot frame
            tf = self.tf_buf.lookup_transform(BASE_FRAME, m.header.frame_id, rclpy.time.Time())
            tr, q = tf.transform.translation, tf.transform.rotation
            T = np.eye(4); T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(); T[:3, 3] = [tr.x, tr.y, tr.z]
            meta["T_body_cam"] = T.tolist()
        except Exception as ex:
            meta["T_body_cam_error"] = str(ex)
        out = [f"{path}  {m.width}x{m.height} {m.encoding}"]
        if depth:
            dm = self._grab_once(ImageMsg, f"{NS}/depth_registered/hand/image")
            if dm is None:
                out.append("depth: none within 3 s")
            else:
                d, denc = self._img_to_np(dm)
                dpath = os.path.splitext(path)[0] + "_depth.png"
                cv2.imwrite(dpath, d)
                valid = d[d > 0]
                meta["depth"] = {"path": dpath, "encoding": dm.encoding, "units": "mm" if d.dtype == np.uint16 else "?",
                                 "valid_frac": float(valid.size / d.size), "min_mm": float(valid.min()) if valid.size else None,
                                 "max_mm": float(valid.max()) if valid.size else None}
                out.append(f"{dpath}  {dm.width}x{dm.height} {dm.encoding}  valid {100 * valid.size / d.size:.0f}%")
        json.dump(meta, open(os.path.splitext(path)[0] + ".json", "w"), indent=1)
        return True, "\n".join(out)

    # ---- robot state via the SDK -----------------------------------------------------------------
    def robot_state(self, max_age=1.0):
        """Fresh RobotState proto from the robot (cached for max_age seconds), or None with self.sdk_err set."""
        with self.sdk_lock:
            if self.sdk_state is not None and time.time() - self.sdk_state[0] < max_age:
                return self.sdk_state[1]
            try:
                if self.sdk_robot is None:
                    cfg = yaml.safe_load(open(SPOT_CFG))["/**"]["ros__parameters"]
                    robot = bosdyn.client.create_standard_sdk("spotd").create_robot(cfg["hostname"])
                    robot.authenticate(cfg["username"], cfg["password"])
                    self.sdk_robot = (robot, robot.ensure_client(RobotStateClient.default_service_name))
                st = self.sdk_robot[1].get_robot_state(timeout=2.0)
                self.sdk_state, self.sdk_err = (time.time(), st), None
                return st
            except Exception as ex:
                self.sdk_err = f"{type(ex).__name__}: {str(ex)[:120]}"
                self.sdk_robot = None          # re-authenticate next time
                return None

    def power_info(self):
        """(motor_state_name, shore_state_name, charge_pct, faults[list], source) from the SDK, else the topic."""
        st = self.robot_state()
        if st is not None:
            ps = st.power_state
            faults = [f.error_message for f in st.system_fault_state.faults]
            return MOTOR.get(ps.motor_power_state, str(ps.motor_power_state)), SHORE.get(ps.shore_power_state, str(ps.shore_power_state)), \
                float(ps.locomotion_charge_percentage.value), faults, "sdk"
        p = self.power
        if p is not None and p.motor_power_state in MOTOR and p.shore_power_state in SHORE:
            return MOTOR[p.motor_power_state], SHORE[p.shore_power_state], float(p.locomotion_charge_percentage), [], "topic"
        return "unknown", "unknown", float("nan"), [], f"none ({self.sdk_err})"

    # ---- status ----------------------------------------------------------------------------
    def status(self) -> str:
        p, e, l, b = self.power, self.estop, self.leases, self.batt
        driver = any(n == "spot_ros2" for n, _ in self.get_node_names_and_namespaces())
        out = [f"driver: {'up' if driver else 'DOWN'}"]
        motor, shore, charge, faults, src = self.power_info()
        out.append(f"motors: {motor}  shore: {shore}  charge: {charge:.0f}%  (via {src})")
        if faults:
            out.append("robot faults: " + "; ".join(faults))
        if e:
            out.append("estop: " + "  ".join(f"{s.name.replace('_estop', '')}={ESTOP.get(s.state, s.state)}" for s in e.estop_states))
        if l:
            owners = {r.lease_owner.client_name for r in l.resources if r.lease_owner.client_name}
            out.append("lease: " + (", ".join(sorted(owners)) if owners else "nobody"))
        if b and b.battery_states:
            bs = b.battery_states[0]
            out.append(f"battery: {bs.charge_percentage:.0f}% {BATT.get(bs.status, bs.status)} {bs.voltage:.1f} V")
        out.append("hand: " + self.hand_str())
        return "\n".join(out)

    def hand_str(self) -> str:
        hp = self.hand_pose()
        if hp is None:
            return "no TF (driver down?)"
        (x, y, z), (r, pch, yw), _ = hp
        return f"x={x:.3f} y={y:.3f} z={z:.3f} m  roll={r:.1f} pitch={pch:.1f} yaw={yw:.1f} deg  (body frame)"

    # ---- command dispatch -----------------------------------------------------------------
    def dispatch(self, args: list[str]) -> tuple[bool, str]:
        if not args:
            return False, HELP
        cmd, rest = args[0].lower(), args[1:]
        if cmd in ("help", "-h", "--help"):
            return True, HELP
        if cmd == "status":
            return True, self.status()
        if cmd == "hand":
            return True, self.hand_str()
        if cmd == "stop":
            self.stop_all()
            return self.trigger("stop")
        if cmd == "estop-hard":
            if "--yes" not in rest:
                return False, "estop-hard cuts motor power and the robot COLLAPSES. Re-run with --yes."
            return self.trigger("estop-hard")
        if cmd in TRIGGERS:
            return self.trigger(cmd, timeout=45.0 if cmd in ("poweron", "stand", "selfright") else 20.0)
        if cmd == "walkto":
            pos, kv, bare = parse_kv(rest, {"--t"})
            if len(pos) != 3:
                return False, "usage: walkto X Y YAW_DEG [--t SECS] [--loose] [--dry]   (current body frame)"
            x, y, yaw = (float(v) for v in pos)
            return self.walkto(x, y, yaw, duration_s=kv.get("--t", 20.0), precise="--loose" not in bare, dry="--dry" in bare)
        if cmd == "traj":
            pos, kv, bare = parse_kv(rest, {"--pause", "--approach-step", "--approach-pause", "--speed", "--approach-speed"})
            if not pos:
                return False, ("usage: traj FILE.json --smooth [--speed M/S] [--approach-speed M/S] [--reverse] [--no-grasp] [--dry]\n"
                               "       traj FILE.json [--pause SECS] [--approach-step M] [--approach-pause SECS] [--reverse] [--no-grasp] [--dry]  (stepped)")
            path = pos[0] if os.path.isabs(pos[0]) else os.path.join(HERE, pos[0])
            if not os.path.exists(path):
                return False, f"no such file {path}"
            spec = json.load(open(path))
            if "--smooth" in bare:
                return self.run_traj_smooth(spec, speed=kv.get("--speed", 0.01), grasp="--no-grasp" not in bare, dry="--dry" in bare,
                                            reverse="--reverse" in bare, approach_speed=kv.get("--approach-speed", 0.015))
            return self.run_traj(spec, pause=kv.get("--pause", 0.5), grasp="--no-grasp" not in bare, dry="--dry" in bare, reverse="--reverse" in bare,
                                 approach_step=kv.get("--approach-step", 0.01), approach_pause=kv.get("--approach-pause", 0.4))
        if cmd == "snap":
            pos = [a for a in rest if not a.startswith("--")]
            return self.snap(pos[0] if pos else None, depth="--depth" in rest)
        if cmd == "grip":
            if not rest:
                return False, "usage: grip <angle 0-90>"
            a = float(rest[0])
            if not 0 <= a <= 90:
                return False, "angle must be 0..90 degrees"
            if not self.grip_cli.wait_for_service(timeout_sec=2.0):
                return False, "set_gripper_angle service not available"
            fut = self.grip_cli.call_async(SetGripperAngle.Request(gripper_angle=a))
            t0 = time.time()
            while not fut.done() and time.time() - t0 < 15:
                time.sleep(0.02)
            if not fut.done():
                return False, "set_gripper_angle timed out"
            return bool(fut.result().success), fut.result().message
        if cmd == "tf":
            if len(rest) != 2:
                return False, "usage: tf PARENT CHILD   (4x4 matrix, json)"
            try:
                tr = self.tf_buf.lookup_transform(rest[0], rest[1], rclpy.time.Time())
            except Exception as ex:
                return False, f"tf lookup failed: {ex}"
            q, tt = tr.transform.rotation, tr.transform.translation
            M = np.eye(4); M[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(); M[:3, 3] = [tt.x, tt.y, tt.z]
            return True, json.dumps(M.tolist())
        if cmd == "poseq":
            pos, kv, bare = parse_kv(rest, set())
            if len(pos) != 7:
                return False, "usage: poseq X Y Z QX QY QZ QW [--nowait] [--force]   (body frame)"
            xyz = [float(v) for v in pos[:3]]; q = [float(v) for v in pos[3:]]
            rpy = Rotation.from_quat(q).as_euler("xyz", degrees=True)
            return self.send_pose(xyz, rpy, wait="--nowait" not in bare, force="--force" in bare)
        if cmd == "pose":
            pos, kv, bare = parse_kv(rest, {"-r", "-p", "-y"})
            if len(pos) != 3:
                return False, "usage: pose X Y Z [-r ROLL] [-p PITCH] [-y YAW] [--nowait] [--force]"
            xyz = [float(v) for v in pos]
            rpy = [kv.get("-r", 0.0), kv.get("-p", 0.0), kv.get("-y", 0.0)]
            return self.send_pose(xyz, rpy, wait="--nowait" not in bare, force="--force" in bare)
        if cmd == "nudge":
            _, kv, bare = parse_kv(rest, {"-x", "-y", "-z", "-r", "-p", "-w"})
            hp = self.hand_pose()
            if hp is None:
                return False, "no hand TF"
            d = np.array([kv.get("-x", 0.0), kv.get("-y", 0.0), kv.get("-z", 0.0)])
            dr = np.array([kv.get("-r", 0.0), kv.get("-p", 0.0), kv.get("-w", 0.0)])
            if np.abs(d).max() > NUDGE_MAX_M or np.abs(dr).max() > NUDGE_MAX_DEG:
                return False, f"nudge limited to {NUDGE_MAX_M} m and {NUDGE_MAX_DEG} deg per call"
            xyz = hp[0] + d
            rot = Rotation.from_euler("xyz", dr, degrees=True) * Rotation.from_quat(hp[2])
            return self.send_pose(xyz, rot.as_euler("xyz", degrees=True), wait="--nowait" not in bare, force="--force" in bare)
        if cmd in ("go", "save", "poses"):
            poses = json.load(open(POSES)) if os.path.exists(POSES) else {}
            if cmd == "poses":
                return True, "\n".join(f"{k:12s} {v}" for k, v in poses.items()) or "no saved poses"
            if not rest:
                return False, f"usage: {cmd} <name>"
            name = rest[0]
            if cmd == "save":
                hp = self.hand_pose()
                if hp is None:
                    return False, "no hand TF"
                poses[name] = {"xyz": [round(float(v), 3) for v in hp[0]], "rpy": [round(float(v), 1) for v in hp[1]]}
                json.dump(poses, open(POSES, "w"), indent=1)
                return True, f"saved {name}: {poses[name]}"
            if name not in poses:
                return False, f"unknown pose {name!r}; have: {', '.join(poses) or 'none'}"
            p = poses[name]
            return self.send_pose(p["xyz"], p["rpy"], wait="--nowait" not in rest, force="--force" in rest)
        if cmd == "vel":
            pos, kv, _ = parse_kv(rest, {"-t"})
            if len(pos) != 3:
                return False, "usage: vel VX VY YAW [-t SECS]"
            vx, vy, wz = (float(v) for v in pos)
            dur = min(kv.get("-t", 1.0), VEL_T_MAX)
            if abs(vx) > VEL_MAX or abs(vy) > VEL_MAX or abs(wz) > TURN_MAX:
                return False, f"limits: |vx|,|vy| <= {VEL_MAX} m/s, |yaw| <= {TURN_MAX} rad/s"
            tw = Twist()
            tw.linear.x, tw.linear.y, tw.angular.z = vx, vy, wz
            self.vel_target, self.vel_until = tw, time.time() + dur
            return True, f"driving vx={vx} vy={vy} yaw={wz} for {dur:.1f}s, then auto-stop"
        if cmd == "recover":
            steps = [("estop-release", "estop-release"), ("poweroff", "poweroff"), ("poweron", "poweron")]
            if "--no-stand" not in rest:
                steps.append(("stand", "stand"))
            out = []
            for label, key in steps:
                ok, msg = self.trigger(key, timeout=45.0)
                out.append(f"{label}: {'ok' if ok else 'FAILED'} - {msg}")
                if not ok and key != "poweroff":
                    if self.power_info()[1] == SHORE[1]:
                        out.append("robot is on shore power: unplug it, then run recover again")
                    return False, "\n".join(out)
                time.sleep(1.5)
            return True, "\n".join(out)
        return False, f"unknown command {cmd!r}\n{HELP}"


def serve(node: SpotD):
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    os.chmod(SOCK, 0o666)
    srv.listen(8)
    node.get_logger().info(f"spotd listening on {SOCK}")

    def client(conn):
        try:
            conn.settimeout(120)
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            args = [a for a in data.decode().rstrip("\n").split("\t") if a != ""]
            t0 = time.time()
            try:
                ok, msg = node.dispatch(args)
            except Exception as ex:  # never let one bad command kill the daemon
                ok, msg = False, f"internal error: {ex!r}"
            log(f"{' '.join(args)!r} -> {'OK' if ok else 'ERR'} ({time.time() - t0:.2f}s) {msg.splitlines()[0] if msg else ''}")
            conn.sendall(f"{'OK' if ok else 'ERR'} {msg}\n".encode())
        finally:
            conn.close()

    while rclpy.ok():
        conn, _ = srv.accept()
        threading.Thread(target=client, args=(conn,), daemon=True).start()   # one thread per command: `stop` never queues


def main():
    rclpy.init()
    node = SpotD()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    log("spotd started")
    try:
        serve(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_all()
        log("spotd stopped")
        if os.path.exists(SOCK):
            os.unlink(SOCK)
        rclpy.shutdown()


if __name__ == "__main__":
    main()
