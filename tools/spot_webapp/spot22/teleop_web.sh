#!/usr/bin/env bash
# Launch the browser teleop server on spot22:  tmux new -d -s teleop "bash ~/andrew_ws/teleop_web.sh"
# Then open http://192.168.1.213:8780 from any machine on the lab LAN. Needs spotd for the discrete commands.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=22
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source "$HOME/dev/ros2_ws/install/setup.bash"
cd "$(dirname "$0")"
exec python3 -u teleop_web.py
