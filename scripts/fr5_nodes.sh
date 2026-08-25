#!/usr/bin/env bash
#
# Shared node bookkeeping for the FR5 stack. Sourced by stop_all.sh and
# start_teleop.sh; not meant to be run directly.
#
# Why this reads /proc instead of using pkill
# -------------------------------------------
# `pkill -f us_diff_ik` matches **any** command line containing that string —
# including the shell that is running the pkill, because the pattern is sitting
# in its own argv. On 2026-08-25 that killed the calling shell before the
# signal went out, a node survived, and a second copy was started on top of it.
# Two differential-IK loops then published conflicting joint velocities at
# 100 Hz each into one servo node and the arm shook.
#
# Reading /proc and comparing against our own PID cannot make that mistake, and
# it lets us report exactly which processes were found before anything is
# signalled.

WORKSPACE="${WORKSPACE:-$HOME/FR5-for-RUS}"

# Match on installed executable paths, so only this workspace's nodes are
# touched. A ROS node from another checkout, or somebody else's session, is not
# ours to kill.
FR5_PATTERNS=(
  "$WORKSPACE/install/fr5_control/lib/fr5_control/us_servo"
  "$WORKSPACE/install/fr5_control/lib/fr5_control/telemetry_bridge"
  "$WORKSPACE/install/fr5_control/lib/fr5_control/fr5_servo_joint_control"
  "$WORKSPACE/install/fr5_ik/lib/fr5_ik/us_diff_ik"
  "$WORKSPACE/install/fr5_ik/lib/fr5_ik/freespace_two_twist"
  "$WORKSPACE/install/touch_teleop/lib/touch_teleop/touch_twist"
  "$WORKSPACE/install/fr5_vision/lib/fr5_vision/us_frame"
  "$WORKSPACE/install/keyboard_teleop/lib/keyboard_teleop/keyboard"
  "ros2 launch fr5_launch"
)

#: PIDs of this script and everything that spawned it. Never signalled.
_own_pids() {
  local pid=$$
  while [[ -n "$pid" && "$pid" != "0" && "$pid" != "1" ]]; do
    printf '%s\n' "$pid"
    pid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null)
  done
}

# Prints "PID<TAB>short description" for every FR5 process now running.
fr5_find() {
  local -a own
  mapfile -t own < <(_own_pids)

  local d pid cmd pattern skip o
  for d in /proc/[0-9]*; do
    pid=${d#/proc/}
    skip=0
    for o in "${own[@]}"; do [[ "$pid" == "$o" ]] && skip=1 && break; done
    [[ $skip -eq 1 ]] && continue

    # A process can exit between the glob and this read. Redirection failure is
    # reported by the shell, not by tr, so the whole thing has to be wrapped —
    # otherwise cleanup output is peppered with harmless errors, and output
    # nobody trusts is output nobody reads.
    cmd=$( { tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null )
    [[ -z "$cmd" ]] && continue

    for pattern in "${FR5_PATTERNS[@]}"; do
      if [[ "$cmd" == *"$pattern"* ]]; then
        printf '%s\t%s\n' "$pid" "$(basename "${pattern%% *}")"
        break
      fi
    done
  done
}

# Stops every FR5 process and does not return until they are gone.
#
# TERM first so nodes can shut their hardware links down cleanly, then KILL for
# anything still standing. The wait is the point: starting a new stack on top of
# a dying one is how duplicates happen.
fr5_stop_all() {
  local -a rows
  mapfile -t rows < <(fr5_find)

  if [[ ${#rows[@]} -eq 0 ]]; then
    echo "  실행 중인 FR5 노드 없음"
    return 0
  fi

  echo "  ${#rows[@]} 개 발견:"
  local row pid name
  for row in "${rows[@]}"; do
    pid=${row%%$'\t'*}; name=${row#*$'\t'}
    printf '    %-8s %s\n' "$pid" "$name"
  done

  for row in "${rows[@]}"; do kill -TERM "${row%%$'\t'*}" 2>/dev/null || true; done

  local waited=0
  while (( waited < 60 )); do
    mapfile -t rows < <(fr5_find)
    [[ ${#rows[@]} -eq 0 ]] && { echo "  정상 종료됨"; return 0; }
    sleep 0.25
    (( waited++ ))
  done

  echo "  15 초 내 종료 안 됨 — SIGKILL"
  for row in "${rows[@]}"; do kill -9 "${row%%$'\t'*}" 2>/dev/null || true; done
  sleep 1

  mapfile -t rows < <(fr5_find)
  if [[ ${#rows[@]} -eq 0 ]]; then
    echo "  강제 종료됨"
    return 0
  fi
  echo "  종료 실패: ${rows[*]%%$'\t'*}" >&2
  return 1
}
