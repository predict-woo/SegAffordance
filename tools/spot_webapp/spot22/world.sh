#!/usr/bin/env bash
# Launch the Rerun world visualizer/recorder on spot22:  tmux new -d -s world "bash ~/andrew_ws/world/world.sh"
# Web viewer: http://192.168.1.213:9090   Recordings: ~/andrew_ws/world/recordings/*.rrd
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=22
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source "$HOME/dev/ros2_ws/install/setup.bash"     # AMENT_PREFIX_PATH lets Rerun resolve package://spot_description meshes
cd "$(dirname "$0")"
exec "$(dirname "$0")/venv/bin/python" -u spot_world.py     # venv: numpy 2 + rerun-sdk (system numpy 1.25 mismatches the wheel)
