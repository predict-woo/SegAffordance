#!/usr/bin/env bash
# Stop every ROS/DDS process on spot22 (ours + the driver) and clear stale FastDDS shared-memory files.
# Run as `bash ~/andrew_ws/stop_all_ros.sh` so no process pattern appears on a live command line (pkill self-kill).
. ~/.spot_ros2_panes 2>/dev/null
echo "== our services"
for s in world teleop spotd; do tmux kill-session -t $s 2>/dev/null; done
for p in video_streams.py spot_world.py teleop_web.py spotd.py teleop_web.sh world.sh spotd_loop.sh spotd.sh; do pkill -TERM -f "$p" 2>/dev/null; done
echo "== retry loops and their nodes"
tmux kill-window -t spot_ros2:state_pub 2>/dev/null; tmux kill-window -t spot_ros2:image_pub 2>/dev/null
for p in state_pub_loop.sh image_pub_loop.sh state_publisher_node spot_image_publisher_node; do pkill -TERM -f "$p" 2>/dev/null; done
echo "== driver launch"
[ -n "$P_DRIVER" ] && tmux send-keys -t "$P_DRIVER" C-c
for i in $(seq 1 15); do sleep 1; pgrep -f "ros2 launch locopt_ros" >/dev/null || break; done
pkill -TERM -f "ros2 launch locopt_ros" 2>/dev/null; sleep 3
for p in "spot_driver/spot_ros2 " "ros2 launch locopt_ros" component_container_mt robot_state_publisher image_stitcher_node spot_alerts wifi_scanner.py bluetooth_scanner.py state_publisher_node spot_image_publisher_node spot_inverse_kinematics object_synchronizer; do pkill -KILL -f "$p" 2>/dev/null; done
sleep 2
echo "remaining: $(pgrep -fa "ros2|spot_|publisher|spotd|teleop|world|video_streams|component_container|stitcher|scanner" | grep -v stop_all_ros | wc -l)"
pgrep -fa "ros2|spot_|publisher|spotd|teleop|world|video_streams|component_container|stitcher|scanner" | grep -v stop_all_ros | cut -c1-90
echo "== stale shm files: $(ls /dev/shm | grep -c fastrtps)"
rm -f /dev/shm/fastrtps_*
echo "after cleanup: $(ls /dev/shm | grep -c fastrtps)"
