# spot_world: real-time world-frame visualizer + recorder (Rerun)

On spot22: `~/andrew_ws/world/` = `spot_world.py`, `world.sh`, `spot.urdf` (generated), `venv/` (numpy 2 + rerun-sdk),
`recordings/*.rrd`.  Start: `tmux new -d -s world "bash ~/andrew_ws/world/world.sh"` (streaming only; `RR_RECORD=1 bash ...` also writes an .rrd).  View: http://192.168.1.213:9090
(web viewer served by spot22; gRPC on 9876 for the native viewer: `rerun rerun+http://192.168.1.213:9876/proxy`).

Setup once (spot22):
  python3 -m venv --system-site-packages ~/andrew_ws/world/venv
  ~/andrew_ws/world/venv/bin/pip install "numpy>=2,<2.3" pyarrow rerun-sdk      # system numpy 1.25 mismatches the wheel
  source /opt/ros/humble/setup.bash; source ~/dev/ros2_ws/install/setup.bash
  xacro $(ros2 pkg prefix spot_description)/share/spot_description/urdf/spot.urdf.xacro arm:=true tf_prefix:=spot/ > ~/andrew_ws/world/spot.urdf

Logged, all in `spot/vision` (visual-odometry world frame): Boston Dynamics URDF meshes animated from /spot/joint_states +
TF body pose; live voxelised clouds from the 5 body depth cams + hand depth (2 Hz, stride 4); persistent voxel map in 2 m
tiles (5 cm voxels, a voxel needs 3 hits; only dirty tiles re-sent every 2 s); depth-camera frustums; hand RGB (2 Hz); body trail.
Recording (opt-in) grows ~0.5-1 MB/s while on; streaming keeps a 1 GiB in-memory buffer on spot22 and writes nothing. Known cosmetic warning: ViewCoordinatesBatch numpy ABI (rerun 0.38 + py3.10).
