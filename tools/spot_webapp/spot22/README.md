# andrew_ws on spot22: fast Spot control

`spotd` is a resident ROS 2 node that stays connected to the spot_ros2 driver; `spotctl` is a thin
client that sends it one line over a Unix socket. A command answers in about 50 ms instead of the
2 to 5 s a fresh `ros2 service call` needs, and arm poses go out on a persistent publisher so they
are not dropped the way one-shot `ros2 topic pub` messages were.

```
start:   tmux new -d -s spotd ~/andrew_ws/spotd.sh     # once per boot; the driver must be up
use:     spotctl status | hand | stand | sit | stow | unstow | open | close | grip 45
         spotctl pose 0.7 0 0.3 -p 15         # hand pose in spot/body, metres + degrees (roll -r, pitch -p, yaw -y)
         spotctl nudge -z 0.05 -p 10          # relative to the current hand pose (max 0.15 m / 30 deg per call)
         spotctl go half-down | save mypose | poses
         spotctl vel 0.3 0 0 -t 1.0           # base velocity for 1 s, then auto-stop (max 0.5 m/s, 3 s)
         spotctl stop                         # halt everything, stay standing
         spotctl estop | recover              # gentle E-stop / the recovery sequence that works
stop:    tmux kill-session -t spotd            (or Ctrl-C if in the foreground)
log:     ~/andrew_ws/spotd.log                 every command with timing and result
```

`spotctl` needs no ROS environment. `spotd.sh` sets FastDDS + domain 22 itself.

Notes learned 2026-09-18/19:
- After `estop` (gentle) the motor state stays ERROR; `poweron` fails until a `poweroff` is sent first.
  `recover` does estop-release -> poweroff -> poweron -> stand. Unplug shore power first.
- `pose` waits up to 8 s for TF convergence (2 cm, 3 deg) and reports the error; `--nowait` returns at once.
- Workspace clamp for `pose`: x 0.30..1.10, y -0.55..0.55, z -0.25..0.75 (body frame); `--force` overrides.
- Only one process should publish arm poses at a time: do not run `arm_pose_follower.py` and `spotctl pose`
  against each other.
