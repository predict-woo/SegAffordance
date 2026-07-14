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
open(path, "w").write(config)
print(f"updated {path}: {alias} -> {info['ip']}:{info['port']}")
PY
