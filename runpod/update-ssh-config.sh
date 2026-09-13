#!/usr/bin/env bash
# Refresh the `segaff-dev` Host entry in ~/.ssh/config with the dev pod's
# current IP/port (they change on pod stop/start or recreation).
# Usage: bash runpod/update-ssh-config.sh [pod-id]   (run on your local machine)
set -euo pipefail

POD_NAME="segaffordance-dev"
HOST_ALIAS="segaff-dev"

POD_ID="${1:-$(runpodctl pod list | python3 -c "
import json, sys
pods = json.load(sys.stdin)
print(next(p['id'] for p in pods if p['name'] == '$POD_NAME'))
")}"

SSH_INFO="$(runpodctl ssh info "$POD_ID")" HOST_ALIAS="$HOST_ALIAS" python3 <<'PY'
import json, os, re, sys

alias = os.environ["HOST_ALIAS"]
info = json.loads(os.environ["SSH_INFO"])
if "ip" not in info:
    sys.exit(f"pod not ready for SSH yet: {info}")

entry = f"""Host {alias}
    HostName {info['ip']}
    Port {info['port']}
    User root
    IdentityFile {info['ssh_key']['path']}
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    ServerAliveInterval 60
"""

path = os.path.expanduser("~/.ssh/config")
config = open(path).read() if os.path.exists(path) else ""
block = re.compile(rf"^Host {re.escape(alias)}\n(?:^[ \t]+.*\n?)*", re.M)
config = block.sub("", config).rstrip("\n")
config = (config + "\n\n" if config else "") + entry
# Atomic replace: a plain open(path, "w") truncates first, and any ssh that starts during the
# write sees a half-written file ("no argument after keyword port"). With several sessions running
# pod commands every few minutes, that race hit real commands on 2026-09-13.
tmp = path + ".tmp"
with open(tmp, "w") as f:
    f.write(config + "\n")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
print(f"updated {path}: {alias} -> {info['ip']}:{info['port']}")
PY
