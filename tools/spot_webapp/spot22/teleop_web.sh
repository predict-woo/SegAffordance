#!/usr/bin/env bash
# Launch the browser teleop server on spot22:  tmux new -d -s teleop "bash ~/andrew_ws/teleop_web.sh"
# Then open http://192.168.1.213:8780 from any machine on the lab LAN. Needs spotd for the discrete commands.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=22
unset CYCLONEDDS_URI
source /opt/ros/humble/setup.bash
source "$HOME/dev/ros2_ws/install/setup.bash"
cd "$(dirname "$0")"
# Watchdog: the server deadlocked once (all threads waiting on a lock, HTTP hung with a full listen backlog). Run it in the
# background and restart it when it stops answering on localhost for ~30 s. `kill -USR1 <pid>` first if you want the stacks.
while true; do
  python3 -u teleop_web.py &
  PID=$!
  fails=0
  while kill -0 $PID 2>/dev/null; do
    sleep 10
    if curl -s -m 5 -o /dev/null http://127.0.0.1:8780/; then fails=0; else fails=$((fails + 1)); fi
    if [ "$fails" -ge 3 ]; then
      echo "$(date +%T) teleop_web.sh: server unresponsive for 30 s, dumping stacks and restarting"
      kill -USR1 $PID 2>/dev/null; sleep 1; kill -TERM $PID 2>/dev/null; sleep 2; kill -9 $PID 2>/dev/null
      break
    fi
  done
  wait $PID 2>/dev/null
  echo "$(date +%T) teleop_web.sh: server exited, restarting in 2 s"; sleep 2
done
