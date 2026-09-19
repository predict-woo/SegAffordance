# EgoArt × Spot: door-opening pipeline, teleop and live 3D view

This folder holds everything we built on 2026-09-18/19 to run EgoArt on the lab's Boston Dynamics Spot:
a web app that goes from a far-away RGB-D frame to the robot opening a cabinet door, a browser teleop with live
cameras, a real-time 3D world view, and the robot-side daemon they all talk through. This document explains what
runs where, how each stage works, how to operate it, and what to do when it breaks.

Nothing here modifies Julia's stack (`~/julia_ws` on hermann, the driver scripts on spot22); it sits next to it and
uses the same ROS 2 driver.

---

## 1. Machines and what runs on each

```
 Mac (your laptop)                    spot22 (192.168.1.213, "spot")              dev pod (segaff-dev, RunPod)
 ─────────────────                    ──────────────────────────────              ─────────────────────────────
 server.py  :8770  ── ssh ──►         spotd.py      (robot daemon, Unix socket)   EgoArt   tools/predict_image.py
   (pipeline web app)                 spotctl       (thin client)                 SAM 3    /workspace/tools/sam3_serve.py :12190
                                      teleop_web.py :8780 (browser teleop)        exporter /workspace/datasets/itw/tools/pc_export.py
 browser ──── LAN ───────────────►    spot_world.py :9090/:9876 (Rerun view)
                                      spot_ros2 driver  ◄── wired 192.168.50.x ──►  Spot (192.168.50.3)
```

| Host | Role | Access |
|---|---|---|
| **spot22** | Ubuntu 22.04 NUC wired to the robot. Runs the ROS 2 Humble driver (`spot_ros2`, RAI Institute's open-source driver, lab fork) and all of our robot-side code under `~/andrew_ws`. | `ssh spot` (alias in the Mac's `~/.ssh/config`, key auth) |
| **dev pod** | RunPod GPU box with the SegAffordance repo. Runs EgoArt inference, a SAM 3 server, and the point-cloud exporter. Nothing robot-specific lives here except those tools. | `ssh segaff-dev` |
| **Mac** | Runs the pipeline web server (`server.py`) and the browser. Orchestrates the other two over ssh. No ROS, no numpy needed. | local |
| **hermann** | Julia's workstation. **Not used** by this pipeline. Its GPU was dead on 2026-09-19; it recovered after a reboot. | `ssh hermann` |

ROS on spot22 is FastDDS on `ROS_DOMAIN_ID=22`. Every shell that talks to the driver needs that (the scripts set it).

### 1.1 What is on spot22 (`/home/spot/andrew_ws`)

| File | Purpose |
|---|---|
| `spotd.py`, `spotd.sh`, `spotd_loop.sh` | The robot daemon: one resident ROS node that keeps service clients, a persistent arm-pose publisher, a TF listener and a read-only Spot-SDK state client warm, and answers one-line commands on `/tmp/spotd.sock` in ~50 ms. `spotd_loop.sh` respawns it if it dies. Runs in tmux session `spotd`. |
| `spotctl` | Client for spotd (`spotctl status`, `spotctl stand`, ... see §4). Symlinked into `~/.local/bin`. |
| `plan_standoff.py` | Where the base must stand to reach the predicted handle (§3.3). |
| `plan_aim.py` | Hand pose that points the hand camera at a target from a given distance (§3.3). |
| `plan_traj.py` | Body-frame gripper trajectory for a door arc or a drawer pull (§3.5). |
| `pc_export.py`, `pc_viewer.html` | Point-cloud + prediction exporter (also on the pod, which is where the app runs it). |
| `state_pub_loop.sh` | Relaunches the driver's C++ state publisher until it survives (§6). |
| `teleop_web.py`, `teleop.html`, `teleop_web.sh` | Browser teleop server, tmux session `teleop`, http://192.168.1.213:8780. |
| `world/spot_world.py`, `world/world.sh`, `world/spot.urdf`, `world/venv/` | Rerun world view, tmux session `world`, viewer URL in `WORLD.md`. |
| `poses.json` | Named arm poses (`half`, `half-down` = the camera "home" pose, ...). |
| `snaps/` | Every camera snapshot taken by `spotctl snap` (jpg + 16-bit depth png + json with K, hand pose, body→camera transform). |
| `spotd.log` | Every command spotd executed, with timing and result. |

Copies of all these scripts are in `spot22/` in this folder (source of truth is the repo; deploy with `scp`).

Changes on spot22 **outside** `~/andrew_ws`: one PATH line added at the top of `~/.bashrc` (backup `~/.bashrc.bak-20260919`),
the Mac's key in `~/.ssh/authorized_keys`, the `~/.local/bin/spotctl` symlink, and a now-unused user-site `rerun-sdk`.

### 1.2 What is on the pod

| Path | Purpose |
|---|---|
| `/workspace/SegAffordance` (`/opt/venv` python) | The repo; EgoArt is run with `tools/predict_image.py --model dense <config> <ckpt> --case IMG "prompt" --K fx fy cx cy --dump preds.jsonl`. Checkpoint: `experiments/20260913_joint4_decoder_l2anchor_dense/checkpoints/best-epoch13-sf3dval0.9789.ckpt`. |
| `/workspace/tools/sam3_serve.py`, `/workspace/venvs/sam3`, `/workspace/models/sam3.pt` | SAM 3 text-prompt segmentation server (tmux `sam3`, `POST localhost:12190/segment {image, prompt, out, thr}`). Weights copied from hermann's HuggingFace cache (3.45 GB). |
| `/workspace/datasets/itw/tools/pc_export.py` | The exporter the web app calls (same file as `spot22/pc_export.py`). |
| `/workspace/datasets/itw/webapp_runs/<run_id>/` | Per-run inputs and outputs (frames, dumps, masks, clouds). |

---

## 2. The robot daemon (`spotd`) and why it exists

The stock way to command the driver is `ros2 service call ...` / `ros2 topic pub ...`. Each call starts a fresh ROS
process (2–5 s of discovery), and one-shot topic publishes are dropped when the driver is busy, so an arm pose command
took 20–30 s with retries. `spotd` is a single long-lived `rclpy` node that holds everything open and answers over a Unix
socket. `spotctl` connects, sends one tab-separated line, prints the reply. Every other component (web app, teleop)
goes through it, so there is exactly one place that talks to the driver for us.

Design points worth knowing:

- **Kill switch**: `spotctl stop` (or the STOP buttons) runs on its own thread inside spotd, so it goes through while a
  trajectory is executing. It sets an abort flag, cancels in-flight walk goals and calls the driver's `/spot/stop`.
- **Robot state comes from the Spot SDK**, not from ROS topics. The driver's C++ state publisher emits garbage in
  `/spot/status/power_states` after its startup race (§6); spotd authenticates a read-only `RobotStateClient` with the
  credentials from the driver's yaml and reads power, shore power, battery and faults directly. No lease is taken.
- **Arm motion has two modes**. *Stepped*: one pose command per waypoint (the driver turns each into a one-point
  trajectory with no duration, so the arm dashes and stops: jerky). *Smooth* (`traj FILE --smooth`): the whole arc is one
  timed Cartesian trajectory sent through the driver's `/spot/robot_command` **service** (the ROS action client crashed
  the daemon from a worker thread), so the robot interpolates at exactly the commanded speed. The approach and retreat are
  timed moves too. Velocity caps (6 cm/s, 0.5 rad/s) are set on every command as a safety net.
- **Poses are in the body frame** (`spot/body`, x forward, y left, z up) and sent with `frame_id: body`, which is what
  the driver requires. A workspace box clamps `pose` commands unless `--force`.
- **Self-right clears faults first and retries**: every rollover leaves a "fall" behaviour fault that blocks all motion,
  and it can appear a second after the robot lands on its back, so a single clear can race it.

---

## 3. The pipeline (`server.py` + `index.html`)

Open http://127.0.0.1:8770 after `python3 tools/spot_webapp/server.py` on the Mac. Five stages, each a button; the page
polls `/state` once a second and shows a throbber while a stage runs. **Robot moves are off by default**: with the
switch off, the walk, the aim, the home move and the arc are dry runs (planned, logged, not sent), and the close-up frame
is taken from wherever the camera happens to be. Tick "Robot moves enabled" (with a confirm) for the real thing.

All frames of the robot in this text: **body frame** = `spot/body`; **camera frame** = optical frame of the hand colour
camera (`spot/hand_color_image_sensor`, x right, y down, z forward). Every snapshot json records `T_body_cam` at capture
time, so anything computed in the camera frame can be moved into the body frame even after the arm has moved.

### 3.1 Stage 1: far snapshot

`spotctl snap --depth` grabs one hand-camera RGB frame (640×480), the registered 16-bit depth, the intrinsics
(fx = fy = 552 px, principal point at the centre, no distortion) and TF. The app copies the three files to the pod and
runs `pc_export.py` without predictions to get a point cloud (base64 float32 xyz + uint8 rgb) for the far view.
The camera should be at the "home" pose (`spotctl go half-down`: hand at x 0.70, z 0.30, pitched 15° down) for a
view that contains the whole door; the **Head to home** button does that.

Depth from the hand camera is sparse (≈10 % of pixels: a low-resolution ToF sensor registered into the colour image) and
good to about 4 m.

### 3.2 Stage 2: EgoArt on the far frame

Your prompt goes to `predict_image.py` with the camera's real intrinsics (`--K`). EgoArt returns the motion type,
the part mask, the contact point, the axis direction and (for revolute) the hinge point, in the camera frame, and a
decoded trajectory we do not use. The exporter then:

- takes the **mask centroid** as the interaction point (the model's mask head is more precise than its point head), using
  the connected component nearest the point head when the mask covers several handles;
- **rescales** the prediction to the cloud: EgoArt's depth is scale-free, so anchor and hinge are multiplied by
  measured depth at the contact pixel ÷ predicted depth (typically ×1.2–1.3 at 1.3 m);
- re-decodes the trajectory as a pure **60° rotation** of the contact point about the axis through the hinge (the model's
  arc length is a 2D-training quantity) and draws hinge, axis, arc and contact on the cloud.

Prompt wording matters. SceneFun3D, the training set, phrases things as "open the left/right cabinet door under the
sink"; "open the cabinet door with the vertical handle" gave the best hinge on our door, "open the door on the right" a
much worse one, and the verb "pull down" flips ovens to prismatic. Adjectives like "narrow", "wide", "horizontal" are
not understood.

### 3.3 Stage 3: accept → standoff → walk → aim → close-up

`plan_standoff.py` turns the far prediction into a base goal in the current body frame:

- handle **h**, hinge **o**, axis **a** in the body frame; lever **l = h − o** along the door face;
- door normal **n** = a × l projected to the horizontal, signed toward the robot (for a drawer, **n** = the slide axis);
- goal = h + n·standoff + l̂·lateral (standoff 1.10 m default, lateral 0.15 m away from the hinge so the door swings past
  the body; no lateral for drawers); yaw faces the door;
- a reach check: with the base at the goal, the whole arc must stay within 0.9 m of the shoulder (0.29 m ahead of the body
  origin). Standoff ≤ 1.15 m keeps a 60° arc reachable.

`spotctl walkto X Y YAW` uses the driver's `/spot/trajectory` action (body-frame goal, the driver converts to odometry).
**Julia's `spot_teleop.py` must not be running**: it streams zero velocities at 10 Hz even when idle and pre-empts the
walk (the walk then times out or stops short). Our teleop only publishes while a key is held.

`plan_aim.py` then computes the hand pose that puts the camera 0.50 m in front of the (post-walk) handle position looking
straight at it, using the live hand→camera transform from TF; `spotctl poseq` moves there; `snap --depth` again.

### 3.4 Stage 4: close-up → SAM 3 → recalibration

From 0.5 m the camera sees only the handles, not the door edges, and EgoArt's own hinge on that frame is useless. So the
close-up is used **only to locate the handle precisely** and the geometry is carried over from the far frame:

- **SAM 3** (text prompt "handle") returns instance masks with scores. The instance nearest the projection of the
  expected handle position (from the standoff plan) is chosen, so two handles in view are not a coin toss.
- The contact is that mask's **centroid**, backprojected at the measured depth (dense at 0.5 m), then moved
  **grasp-bias** (2 cm default, editable) closer to the camera along the pixel ray so the jaws close around the bar rather
  than on its surface.
- **Handle orientation** from the mask's principal axis (within 45° of horizontal = horizontal bar) sets the gripper roll:
  vertical bar → roll 90° (jaws close horizontally), horizontal bar → roll 0°. The UI can override it.
- **Recalibration**: the yaw between the two body frames is measured as the angle between the far frame's door normal and
  a plane fitted to the close-up cloud (tens of thousands of points, so it is precise). Hinge-to-handle vector and axis
  are rotated by that yaw and attached to the new handle. For a drawer the pull axis is snapped to the fitted front normal.
- Result: hinge, axis and arc drawn on the close cloud, in the current body frame.

### 3.5 Stage 5: accept → home → execute → home

`plan_traj.py --real` builds the gripper path: pre-grasp 12 cm short of the handle along the door normal, grasp pose with
hand x into the door and the roll from the handle orientation, then either a rotation of the contact about the hinge axis
(`turn`, 30° default; the hand rotates with the door as a hand on a handle would) or a straight pull along the axis
toward the robot (`pull`, 15 cm default). 30 waypoints. Then, with moves enabled: arm to home, `traj --smooth` at 1 cm/s
(open gripper → glide to pre-grasp at 5 cm/s → ease 12 cm in at 1.5 cm/s → close → arc → open → ease 12 cm back), arm
to home again. A 30° door arc is ~30 s; the whole stage ~1 min.

**The door must be closed at the start of every run.** There is deliberately no perceive-before-grasp or grasp check
(the user declined them); the plan is always an opening from the closed state. There is no `--reverse`.

Measured on the real door: the arm tracks the planned path to 1–2 cm free-swinging; with the door in hand it lags the
plan by ~2 cm median and the arc runs ~4 s over, which is the door loading a stiff pose controller. The end pose repeats
to millimetres between runs.

### 3.6 Files per run

`tools/spot_webapp/runs/<run_id>/` on the Mac (gitignored): frames, depth, json, `*.data.json` clouds, `*.pred.json`
predictions. The same under `/workspace/datasets/itw/webapp_runs/<run_id>/` on the pod, plus EgoArt dumps and SAM 3 masks.
On spot22, `~/andrew_ws/traj_web.json` is the last executed trajectory and `standoff.json` / `aim.json` the last plans.

---

## 4. `spotctl` command reference

```
status | hand | tf PARENT CHILD             health line (power/faults via SDK, estop, lease, battery, hand pose) / transforms
claim | take | release                      lease (take = force-take a stale lease after a driver restart)
poweron | poweroff | stand | sit | selfright | stop | estop | estop-release | clear-fault | rollover
stow | unstow | carry | open | close | grip A
pose X Y Z [-r -p -y] | poseq X Y Z QX QY QZ QW | nudge [-x -y -z -r -p -w] | go NAME | save NAME | poses
walkto X Y YAW_DEG [--t S] [--dry]          base goal in the current body frame (driver trajectory action)
traj FILE.json --smooth [--speed 0.01] [--no-grasp] [--dry]      one timed trajectory (recommended)
traj FILE.json [--pause S] [--dry]                                 stepped fallback
snap [PATH] [--depth]                        hand RGB (+ depth) + json with K, hand pose, T_body_cam
vel VX VY YAW [-t S]                         base velocity for S seconds (auto-stop)
```

Every command is logged to `~/andrew_ws/spotd.log`.

---

## 5. Teleop and world view

**Teleop** (http://192.168.1.213:8780, served by spot22, no ssh): hold W/S/A/D/Q/E or the arrows to drive; release =
immediate stop; Shift = half speed; sliders for speed. Velocity is posted at 10 Hz while a key is held and the server
stops the robot 0.35 s after the last message (deadman); when idle it publishes nothing. Exact nudges (turn 2/5/15°, step
5/20 cm) go through `walkto`. Buttons for stand/sit/self-right/claim/power, stow/unstow/home/gripper, flip-over
(= stand, sit, rollover: the driver only allows a rollover after it has watched a sit complete), gentle E-stop and
clear-faults. Live MJPEG from any driver camera and a battery chip (SDK) at the top.

**World view** (Rerun): see `spot22/WORLD.md`. Boston Dynamics' official Spot meshes (from `spot_description`, MIT)
animated from joint states, live coloured clouds from the 5 body depth cameras + hand (registered depth, so real colours:
RGB from the hand, grey from the body cams), frustums, hand image and body trail, all in `spot/vision` (the visual-odometry
world frame). Streaming only by default; `RR_RECORD=1` writes an `.rrd`; `WORLD_MAP=1` enables a cumulative voxel map.

---

## 6. Operations

### 6.1 Bring-up after the robot or spot22 was off

```
# spot22
~/start_spot.sh -d --no-attach              # driver in tmux `spot_ros2`; relaunch until `pgrep -f spot_image_publisher_node`
                                            #   is alive and /spot/camera/hand/image has a rate (~6 Hz). Took 3 tries once.
tmux new-window -d -t spot_ros2 -n state_pub "bash ~/andrew_ws/state_pub_loop.sh"   # TF + status (retries until it sticks)
tmux new -d -s spotd  "bash ~/andrew_ws/spotd_loop.sh"
tmux new -d -s teleop "bash ~/andrew_ws/teleop_web.sh"
tmux new -d -s world  "bash ~/andrew_ws/world/world.sh"
spotctl take; spotctl poweron; spotctl stand
# Mac
python3 tools/spot_webapp/server.py         # http://127.0.0.1:8770
# pod (if the SAM 3 server is not up): ssh segaff-dev 'tmux new -d -s sam3 "/workspace/venvs/sam3/bin/python /workspace/tools/sam3_serve.py --port 12190"'
```

Why the retries: the driver's C++ side nodes (state publisher = TF and status, image publisher = cameras, plus two we do
not need) die at random on launch with `std::system_error: Invalid argument` thrown from `bosdyn::client::Robot::GetUserToken`,
an authentication race inside Boston Dynamics' C++ SDK. The Python SDK authenticates fine with the same credentials.
A surviving state publisher may still publish garbage power states, which is why spotd reads them via the SDK.

### 6.2 Shutdown

`spotctl stow; spotctl sit; spotctl poweroff; spotctl release`, then the power button on the robot. Or Julia's
`~/stop_spot.sh`, which does the same and kills the driver. Never kill the driver while the robot is standing: it is the
E-stop endpoint and the robot drops.

### 6.3 Battery

Motors on while sitting idle drains ~30 %/h; at 0 % the robot cuts motor power and collapses. Spot supports a hot swap
with the charger cable plugged in (the driver connection survives it). Sequence: `rollover` (battery-change pose; it
needs a completed sit first, so stand → sit → rollover), swap, then a fall fault blocks everything until cleared
(`spotctl clear-fault`; `selfright` does it itself), then `selfright` → `sit`.

### 6.4 Known issues and fixes

| Symptom | Cause / fix |
|---|---|
| Walk times out, robot moves in bursts | Julia's `spot_teleop.py` is running: stop it (Ctrl-C in its pane). |
| `spotctl status` says driver DOWN right after launch | Driver still booting, or spotd cannot see it: wait, or restart spotd. |
| No TF / "no hand pose" | Driver's state publisher died: `state_pub_loop.sh`. |
| No camera images | Driver's image publisher died: relaunch the whole driver. |
| `poweron` fails with an internal error | Shore power plugged in (unplug), or the motor controller is in the error state after an E-stop: `poweroff` then `poweron`. |
| `poweron` fails "no lease for resource body" | Stale lease from a dead driver: `spotctl take`. |
| Any motion refused with BehaviorFaultError | Fall fault after a rollover/collapse: `spotctl clear-fault`. |
| Robot reports payload fault "Leica BLK ARC collides with the shoulder" | Payload registration geometry vs the arm; informational so far, arm still works. Fix on the tablet/admin page if it starts blocking stow/unstow. |
| Rerun viewer shows the welcome screen | Use the URL with `?url=rerun%2Bhttp%3A%2F%2F192.168.1.213%3A9876%2Fproxy`. |
| Rerun "Data source failed: Failed to fetch" | CORS: the data server must list the viewer origin (it does now). |
| Web app snapshot fails with no `hand_in_body` | No TF (see above). |

---

## 7. Where things came from

- Driver: `spot_ros2` by the RAI Institute (formerly Boston Dynamics AI Institute), lab fork `cvg/spot_ros2_deck`, built
  in `~/dev/ros2_ws` on spot22 with the lab's `locopt_ros` launch (`spot_bringup.launch.xml`) and config
  (auto claim/power/stand off, point clouds on, stitched front image).
- Robot model: `spot_description` (URDF + Boston Dynamics meshes), MIT licence, part of the same workspace.
- Our code: this folder. `spot22/` = deployed copies; `server.py`/`index.html` = the Mac side. Commit history has the
  details of each fix.
