#!/usr/bin/env bash
#
# 한 세션을 통째로: 정리 → 센서 송출 → teleop.
#
#   ./scripts/start_session.sh                       # freespace, 실로봇
#   ./scripts/start_session.sh backend:=mock         # mock
#   ./scripts/start_session.sh freespace:=false      # 접촉용 상한
#   ./scripts/start_session.sh --px6d /dev/ttyACM1   # 센서 포트 지정
#   ./scripts/start_session.sh --no-px6d             # 센서 없이
#   ./scripts/start_session.sh --no-gui              # GUI 없이
#   ./scripts/start_session.sh --dry-run             # 무엇을 할지만 출력
#
# `--` 로 시작하지 않는 인자는 us_phase0.launch.py 로 넘어간다.
#
# 왜 하나로 묶는가
# ----------------
# 세 가지를 각각 띄우면 하나를 빠뜨리거나 두 번 띄우기 쉽다. 2026-08-25 에 실로봇에
# 제어 스택이 두 벌 올라간 적이 있는데, 조작석에서는 구별되지 않고 로그에도 원인이
# 남지 않는다. 한 명령이 소유하면 그 창을 닫는 것이 곧 세션의 끝이 된다.
#
# **종료할 때도 정리한다.** Ctrl-C 든 비정상 종료든, 나가기 전에 이 워크스페이스의
# 노드가 남아 있지 않은 것을 확인한다. 오늘 겪은 문제는 전부 "남은 것" 에서 나왔다.
#
# GUI 도 함께 띄운다. SSH 셸에는 DISPLAY 가 없으므로 teleop_gui/scripts/run-on-console.sh
# 가 좌석의 X 서버를 찾아 붙여 준다 — 창은 이 터미널이 아니라 본체 화면에 뜬다.
#
# GUI 는 setsid 로 **자기 프로세스 그룹**에 띄운다. Electron 은 자식 프로세스를 여럿
# 만들고(zygote·gpu·renderer), 부모만 죽이면 그것들이 남는다. 그룹째 신호를 보내면
# 한 번에 내려간다.
set -uo pipefail

cd "$(dirname "$0")/.."
WORKSPACE="$(pwd)"
export WORKSPACE
# shellcheck source=fr5_nodes.sh
source "$WORKSPACE/scripts/fr5_nodes.sh"

PX6D_PORT="/dev/ttyACM0"
USE_PX6D=1
USE_GUI=1
DRY_RUN=0
LAUNCH_ARGS=()

while (( $# )); do
  case "$1" in
    --px6d)     PX6D_PORT="${2:-}"; shift 2 ;;
    --no-px6d)  USE_PX6D=0; shift ;;
    --no-gui)   USE_GUI=0; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  sed -n '2,30p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)          LAUNCH_ARGS+=("$1"); shift ;;
  esac
done

if [[ ${#LAUNCH_ARGS[@]} -eq 0 ]]; then
  LAUNCH_ARGS=(backend:=fairino teleop:=true freespace:=true)
fi

BRIDGE_LOG="$WORKSPACE/log/telemetry_bridge.log"
GUI_LOG="$WORKSPACE/log/teleop_gui.log"
BRIDGE_PID=""
GUI_PGID=""

# 나가는 길은 하나뿐이다. Ctrl-C 든 오류든 여기를 지난다.
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  echo
  echo "세션 종료 — 정리"
  if [[ -n "$GUI_PGID" ]]; then
    # 음수 PID = 프로세스 그룹 전체. Electron 의 자식들까지 함께 내려간다.
    kill -TERM -- "-$GUI_PGID" 2>/dev/null || true
  fi
  if [[ -n "$BRIDGE_PID" ]]; then
    kill -TERM "$BRIDGE_PID" 2>/dev/null || true
  fi
  fr5_stop_all || true
  if [[ -n "$GUI_PGID" ]] && kill -0 -- "-$GUI_PGID" 2>/dev/null; then
    sleep 1
    kill -9 -- "-$GUI_PGID" 2>/dev/null || true
  fi
  exit "$code"
}
trap cleanup EXIT INT TERM

echo "1/5  기존 노드 정리"
if ! fr5_stop_all; then
  echo "정리에 실패해 기동하지 않는다. 남은 프로세스를 직접 확인하라." >&2
  exit 1
fi

echo
echo "2/5  환경"
if [[ ! -f "$WORKSPACE/install/setup.bash" ]]; then
  echo "install/setup.bash 이 없다. colcon build 를 먼저 하라." >&2
  exit 1
fi
# ROS 의 setup 스크립트는 nounset 을 견디지 못한다. source 구간에서만 푼다.
set +u
if [[ -z "${ROS_DISTRO:-}" ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
fi
# shellcheck disable=SC1091
source "$WORKSPACE/install/setup.bash"
set -u
echo "  ROS_DISTRO=$ROS_DISTRO · workspace=$WORKSPACE"

echo
echo "3/5  센서 송출 (telemetry_bridge)"
BRIDGE_ARGS=(run fr5_control telemetry_bridge)
if (( USE_PX6D )); then
  if [[ -e "$PX6D_PORT" ]]; then
    BRIDGE_ARGS+=(--ros-args -p "bridge.px6d_port:=$PX6D_PORT")
    echo "  PX6D 직결: $PX6D_PORT"
  else
    # 센서가 없다고 세션을 막지 않는다. 다만 조용히 넘어가지도 않는다 —
    # 힘이 안 보이는 이유를 나중에 찾게 하면 안 된다.
    echo "  ⚠ $PX6D_PORT 없음 — 센서 없이 계속한다 (힘 값은 안 나온다)"
    USE_PX6D=0
  fi
else
  echo "  PX6D 생략 (--no-px6d)"
fi

# 시리얼은 dialout 그룹이다. 그룹 목록(`id -nG`)이 아니라 **실제 접근 권한**으로
# 판정한다 — 그룹이 계정에 붙어 있어도 지금 셸의 자격증명에는 없을 수 있고, 그
# 차이가 "왜 여기서는 되는데 저기서는 안 되나" 로 나타난다.
#
# sg 로 감싸면 브리지가 그 자식이 되어 PID 추적이 한 단계 멀어지므로, 필요할 때만
# 쓴다. 그래도 놓치는 경우는 종료 시 fr5_stop_all 이 경로로 잡는다.
BRIDGE_CMD=(ros2 "${BRIDGE_ARGS[@]}")
NEEDS_SG=0
if (( USE_PX6D )) && { [[ ! -r "$PX6D_PORT" ]] || [[ ! -w "$PX6D_PORT" ]]; }; then
  NEEDS_SG=1
  echo "  (권한 없음 — sg dialout 로 감싼다)"
fi

GUI_DIR="$WORKSPACE/teleop_gui"
GUI_RUNNER="$GUI_DIR/scripts/run-on-console.sh"
echo
echo "4/5  GUI"
if (( USE_GUI )); then
  if [[ ! -d "$GUI_DIR/node_modules" ]]; then
    echo "  ⚠ teleop_gui/node_modules 없음 — GUI 생략 (npm install 필요)"
    USE_GUI=0
  elif [[ ! -x "$GUI_RUNNER" ]]; then
    echo "  ⚠ $GUI_RUNNER 없음 — GUI 생략"
    USE_GUI=0
  else
    echo "  본체 화면(${DISPLAY_OVERRIDE:-:1})에 띄운다 · 로그 $GUI_LOG"
  fi
else
  echo "  생략 (--no-gui)"
fi

echo
echo "5/5  teleop:  ros2 launch fr5_launch us_phase0.launch.py ${LAUNCH_ARGS[*]}"

if (( DRY_RUN )); then
  echo
  echo "--- dry run, 실행하지 않는다 ---"
  if (( NEEDS_SG )); then
    echo "  브리지: sg dialout -c \"${BRIDGE_CMD[*]}\"   > $BRIDGE_LOG"
  else
    echo "  브리지: ${BRIDGE_CMD[*]}   > $BRIDGE_LOG"
  fi
  if (( USE_GUI )); then
    echo "  GUI:    setsid bash $GUI_RUNNER built   > $GUI_LOG"
  fi
  echo "  teleop: ros2 launch fr5_launch us_phase0.launch.py ${LAUNCH_ARGS[*]}"
  BRIDGE_PID=""
  GUI_PGID=""
  trap - EXIT INT TERM
  exit 0
fi

mkdir -p "$(dirname "$BRIDGE_LOG")"
: > "$BRIDGE_LOG"
if (( NEEDS_SG )); then
  sg dialout -c "$(printf '%q ' "${BRIDGE_CMD[@]}")" >>"$BRIDGE_LOG" 2>&1 &
else
  "${BRIDGE_CMD[@]}" >>"$BRIDGE_LOG" 2>&1 &
fi
BRIDGE_PID=$!

# 브리지가 실제로 소켓을 열었는지 확인하고 넘어간다. 안 뜬 채로 teleop 을 띄우면
# 조작 중에 힘이 안 보이는 이유를 찾게 된다.
for _ in $(seq 40); do
  grep -q "브리지 대기" "$BRIDGE_LOG" 2>/dev/null && break
  sleep 0.25
done
if grep -q "브리지 대기" "$BRIDGE_LOG" 2>/dev/null; then
  grep -E "PX6D 스트리밍 시작|PX6D 읽기 실패|브리지 대기" "$BRIDGE_LOG" | sed 's/^.*\]: /  /' | head -3
else
  echo "  ⚠ 브리지가 10 초 안에 안 떴다 — $BRIDGE_LOG 를 볼 것"
fi

if (( USE_GUI )); then
  mkdir -p "$(dirname "$GUI_LOG")"
  : > "$GUI_LOG"
  # setsid 로 새 프로세스 그룹을 만든다. run-on-console.sh 는 마지막에 exec 하므로
  # 빌드가 끝나면 그 프로세스가 곧 Electron 이 되고, 자식들(zygote·gpu·renderer)이
  # 같은 그룹에 딸린다. 종료할 때 그룹째 신호를 보내면 한 번에 내려간다.
  #
  # `$!` 를 쓰면 안 된다 — setsid 는 fork 한 뒤 **즉시 빠지므로** 그 PID 는 그룹
  # 리더가 아니고, 몇 밀리초 뒤에는 존재하지도 않는다. 새 세션의 리더는 자기 PID 가
  # 곧 PGID 이므로, 자식이 직접 알려 주게 한다.
  GUI_PGID_FILE=$(mktemp)
  setsid bash -c 'echo $$ >"$1"; exec bash "$2" built' _ \
    "$GUI_PGID_FILE" "$GUI_RUNNER" >>"$GUI_LOG" 2>&1 &
  for _ in $(seq 40); do
    [[ -s "$GUI_PGID_FILE" ]] && break
    sleep 0.1
  done
  GUI_PGID=$(cat "$GUI_PGID_FILE" 2>/dev/null || true)
  rm -f "$GUI_PGID_FILE"
  if [[ -n "$GUI_PGID" ]]; then
    echo "  GUI 기동 중 (빌드 포함, 수 초) — PGID $GUI_PGID"
  else
    echo "  ⚠ GUI PGID 를 못 잡았다 — 종료 시 수동으로 내려야 할 수 있다"
  fi
fi

echo
echo "  브리지 로그: $BRIDGE_LOG"
(( USE_GUI )) && echo "  GUI 로그:    $GUI_LOG"
echo "  Ctrl-C 로 세션 전체 종료 (브리지·GUI 포함, 종료 후 정리 확인)"
echo
ros2 launch fr5_launch us_phase0.launch.py "${LAUNCH_ARGS[@]}" &
LAUNCH_PID=$!
wait "$LAUNCH_PID"
