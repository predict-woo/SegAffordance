#!/usr/bin/env bash
# Launch the Rerun world visualizer/recorder on spot22:  tmux new -d -s world "bash ~/andrew_ws/world/world.sh"
# Web viewer: http://192.168.1.213:9090   Recordings: ~/andrew_ws/world/recordings/*.rrd
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=22
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source "$HOME/dev/ros2_ws/install/setup.bash"     # AMENT_PREFIX_PATH lets Rerun resolve package://spot_description meshes
cd "$(dirname "$0")"
PY="$(dirname "$0")/venv/bin/python"                    # venv: numpy 2 + rerun-sdk + pandas>=2.2 (system numpy 1.25 mismatches the wheel; without a numpy-2 pandas in the
                                                       #  venv, pyarrow imports the system pandas and every rerun batch fails with "numpy.dtype size changed")
# spot_world.py hosts the Rerun server; video_streams.py (full-rate camera video, own process) is a client of it and
# must be restarted with it: a client that outlives the server keeps encoding into a dead connection
# ("Write messages call failed: transport error") and its stream never reappears in the new server.
# Restart both while the node reports failed DDS discovery (exit 3: it received nothing from the driver in 15 s).
VIDEO_PID=""
stop_video() { [ -n "$VIDEO_PID" ] && kill "$VIDEO_PID" 2>/dev/null; VIDEO_PID=""; }
trap 'stop_video' EXIT INT TERM
while true; do
  "$PY" -u spot_world.py &
  WORLD_PID=$!
  if [ "${HAND_SDK:-1}" = "1" ]; then
    "$PY" -u video_streams.py &                            # waits for the server port, then connects
    VIDEO_PID=$!
  fi
  wait $WORLD_PID; rc=$?
  stop_video
  [ "$rc" -eq 3 ] || break
  echo "world.sh: spot_world exited with 3 (no DDS discovery), restarting both in 2 s"; sleep 2
done
