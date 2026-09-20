# spot_world: real-time world-frame visualizer + recorder (Rerun)

On spot22: `~/andrew_ws/world/` = `spot_world.py`, `world.sh`, `spot.urdf` (generated), `venv/` (numpy 2 + rerun-sdk),
`recordings/*.rrd`.  Start: `tmux new -d -s world "bash ~/andrew_ws/world/world.sh"` (streaming only; `RR_RECORD=1 bash ...` also writes an .rrd).  View: http://192.168.1.213:9090/?url=rerun%2Bhttp%3A%2F%2F192.168.1.213%3A9876%2Fproxy
(the bare :9090 page is an empty viewer, the ?url= tells it where the data is; native viewer: `rerun rerun+http://192.168.1.213:9876/proxy`).

Setup once (spot22):
  python3 -m venv --system-site-packages ~/andrew_ws/world/venv
  ~/andrew_ws/world/venv/bin/pip install "numpy>=2,<2.3" pyarrow rerun-sdk      # system numpy 1.25 mismatches the wheel
  source /opt/ros/humble/setup.bash; source ~/dev/ros2_ws/install/setup.bash
  xacro $(ros2 pkg prefix spot_description)/share/spot_description/urdf/spot.urdf.xacro arm:=true tf_prefix:=spot/ > ~/andrew_ws/world/spot.urdf

Logged, all in `spot/vision` (visual-odometry world frame): Boston Dynamics URDF meshes animated from /spot/joint_states +
TF body pose; live per-frame clouds from the 5 body depth cams + hand depth (4 Hz, stride 4, real colours; grey body-cam points within `COLOR_MATCH_M` = 2 cm of a point of the latest hand RGB cloud take that point's colour); optional cumulative voxel map (OFF by default; `WORLD_MAP=1` enables: 2 m
tiles of 5 cm voxels, 3-hit filter, dirty tiles re-sent every 2 s); depth-camera frustums; live H.264 video of the hand camera in its frustum and in a 2D panel (30 fps, 2 Mbit/s; the five body cameras can be added with `VIDEO_CAMS`, 15 fps, 1 Mbit/s each) via `video_streams.py` (JPEG from the robot's image service, x264 on spot22, Rerun `VideoStream`; `VIDEO_CAMS` picks the cameras, `HAND_SDK=0` falls back to the driver's hand topic, ~4 Hz); body trail.
Recording (opt-in) grows ~0.5-1 MB/s while on; streaming keeps a 1 GiB in-memory buffer on spot22 and writes nothing. Known cosmetic warning: ViewCoordinatesBatch numpy ABI (rerun 0.38 + py3.10).
