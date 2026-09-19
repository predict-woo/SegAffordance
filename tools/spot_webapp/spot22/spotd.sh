#!/usr/bin/env bash
# Launch spotd with the spot22 ROS environment (FastDDS, domain 22). Run in a tmux pane:
#   tmux new -d -s spotd ~/andrew_ws/spotd.sh      # background
#   ~/andrew_ws/spotd.sh                            # foreground (Ctrl-C stops it)
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=22
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source "$HOME/dev/ros2_ws/install/setup.bash"
cd "$(dirname "$0")"
exec python3 -u spotd.py
