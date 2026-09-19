#!/usr/bin/env bash
# Respawn wrapper: if spotd dies (e.g. a native crash), bring it back within a second so `spotctl stop` keeps working.
cd "$(dirname "$0")"
while true; do
  echo "$(date -Is) spotd_loop: starting spotd" >> spotd.out
  bash ./spotd.sh >> spotd.out 2>&1
  echo "$(date -Is) spotd_loop: spotd exited ($?), restarting in 1 s" >> spotd.out
  sleep 1
done
