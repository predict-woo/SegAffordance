#!/usr/bin/env bash
# spot_ros2 C++ state_publisher_node (TF + /spot/status/*) dies at startup on an auth race inside the C++ SDK
# (std::system_error EINVAL in Robot::GetUserToken). It survives on some attempts, so relaunch until it stays up.
# usage: state_pub_loop.sh [max_tries]   (run inside the spot_ros2 tmux session; needs ~/spot_env.sh)
source "$HOME/spot_env.sh" >/dev/null 2>&1
CFG=/home/spot/dev/ros2_ws/install/locopt_ros/share/locopt_ros/config/spot_config.yaml
BIN=/home/spot/dev/ros2_ws/install/spot_driver/lib/spot_driver/state_publisher_node
MAX=${1:-40}
for i in $(seq 1 "$MAX"); do
  echo "$(date +%T) state_publisher attempt $i"
  $BIN --ros-args -r __ns:=/spot --params-file "$CFG" -p spot_name:=spot &
  PID=$!
  sleep 12
  if kill -0 $PID 2>/dev/null; then echo "$(date +%T) state_publisher up (pid $PID) after $i attempt(s)"; wait $PID; echo "$(date +%T) state_publisher exited, relaunching"; else echo "  died at startup"; fi
  sleep 2
done
echo "gave up after $MAX attempts"
