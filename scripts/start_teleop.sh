#!/usr/bin/env bash
#
# Start a teleoperation session from a known-clean state.
#
#   ./scripts/start_teleop.sh                          # real robot, teleop on
#   ./scripts/start_teleop.sh backend:=mock            # mock robot
#   ./scripts/start_teleop.sh freespace:=false         # contact-stage limits
#
# Every argument is passed through to us_phase0.launch.py.
#
# Why it always stops first
# ------------------------
# A leftover node is invisible until it misbehaves. Two differential-IK nodes
# publishing joint velocities into one servo look exactly like a tuning problem
# from the operator's seat — the arm shakes and nothing in the logs says why.
# Starting from a verified-empty state costs a couple of seconds and removes
# that entire failure mode.
#
# The force bridge is not started here. It is a separate concern (it needs
# dialout for the PX6D) and belongs in its own terminal so its log is readable:
#
#   sg dialout -c "ros2 run fr5_control telemetry_bridge \
#     --ros-args -p bridge.px6d_port:=/dev/ttyACM0"
set -uo pipefail

cd "$(dirname "$0")/.."
WORKSPACE="$(pwd)"
export WORKSPACE
# shellcheck source=fr5_nodes.sh
source "$WORKSPACE/scripts/fr5_nodes.sh"

echo "1/3  기존 노드 정리"
if ! fr5_stop_all; then
  echo >&2
  echo "정리에 실패해 기동하지 않는다. 남은 프로세스를 직접 확인하라." >&2
  exit 1
fi

echo
echo "2/3  환경"
if [[ ! -f "$WORKSPACE/install/setup.bash" ]]; then
  echo "install/setup.bash 이 없다. colcon build 를 먼저 하라." >&2
  exit 1
fi

# ROS 의 setup 스크립트는 nounset 을 견디지 못한다 (AMENT_TRACE_SETUP_FILES 등을
# 정의 없이 참조한다). 우리 코드에는 -u 를 유지하되 source 구간에서만 푼다.
set +u
if [[ -z "${ROS_DISTRO:-}" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi
# shellcheck disable=SC1091
source "$WORKSPACE/install/setup.bash"
set -u
echo "  ROS_DISTRO=$ROS_DISTRO"
echo "  workspace=$WORKSPACE"

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  ARGS=(backend:=fairino teleop:=true)
fi

echo
echo "3/3  기동:  ros2 launch fr5_launch us_phase0.launch.py ${ARGS[*]}"
echo "     Ctrl-C 로 종료. 종료 후 남는 것이 있으면 ./scripts/stop_all.sh"
echo
exec ros2 launch fr5_launch us_phase0.launch.py "${ARGS[@]}"
