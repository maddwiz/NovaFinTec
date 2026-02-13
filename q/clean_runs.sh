#!/usr/bin/env bash
set -e
mkdir -p runs
# keep only these canonical folders
for d in runs/*; do
  base="$(basename "$d")"
  case "$base" in
    IWM|RSP|LQD_TR|HYG_TR) : ;;   # keep
    *) rm -rf "$d" ;;             # delete any other run folders
  esac
done
