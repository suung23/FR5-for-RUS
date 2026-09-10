#!/usr/bin/env bash
#
# 한 세션을 통째로: 정리 → 센서 송출 → teleop.
#
#   ./scripts/start_session.sh                       # ★ 기본: 접촉 전환 + 힘 유지
#   ./scripts/start_session.sh backend:=mock         # mock
#   ./scripts/start_session.sh freespace:=false      # 접촉용 상한
#   ./scripts/start_session.sh contact_probing:=false  # ⚠️ 접촉 전환·힘 유지 끔
#   ./scripts/start_session.sh --px6d /dev/ttyACM1   # 센서 포트 지정
#   ./scripts/start_session.sh --no-px6d             # 센서 없이
#   ./scripts/start_session.sh --no-gui              # GUI 없이
#   ./scripts/start_session.sh --no-us               # 초음파 없이 (영상 패널은 비어 있다)
#   ./scripts/start_session.sh --no-ap               # AP 는 이미 붙여 두었을 때
#   ./scripts/start_session.sh --calib               # 교정 모드 (teleop 없이)
#   ./scripts/start_session.sh --dry-run             # 무엇을 할지만 출력
#
# `--` 로 시작하지 않는 인자는 us_phase0.launch.py 로 넘어간다.
#
# 기본 커멘드가 하는 일 (2026-08-31)
# ----------------------------------
# 인자 없이 띄우면 이렇게 돈다:
#
#   접근    freespace 상한 150 mm/s · 0.9 rad/s. 조작자가 여섯 축을 다 쥔다.
#   ↓       접촉력 ‖F‖ 가 2 N 을 20 ms 넘으면
#   접촉    상한 10 mm/s · 0.2 rad/s 로 내려가고, **로봇이 z 를 잡는다** —
#           목표 ‖F‖ 3.0 ± 0.05 N. 나머지 다섯 축은 0 이다.
#           ⚠️ 목표가 진입 문턱보다 높다. 닿는 순간부터 조작자가 아무 지령도 주지
#           않는 동안 로봇이 스스로 3 N 까지 파고든다 (처음 약 0.22 mm/s, 붙으며 감속).
#   ↓       ‖F‖ 가 0.3 N 아래로 0.5 s 지속되면
#   접근    상한과 z 가 조작자에게 돌아온다. 세션을 다시 띄울 필요가 없다.
#
# 값은 전부 probe.yaml 에 있고 기동할 때 아래 5/5 에 찍힌다. 화면에 찍힌 수와
# 로봇의 거동이 다르면 그것이 곧 버그다 — 눈으로 맞춰 볼 수 있게 찍는다.
#
# ⚠️ **힘 유지는 교정이 유효할 때만 열린다** (contact_control.require_valid_calibration).
# 교정이 없으면 접촉 판정 자체가 보류되고 접근 상한을 유지한다 — 보상 전 렌치에는
# 마운트·프로브 자중 10 N 이 자세에 따라 실려 있어 2 N 문턱과 구별되지 않는다.
# 먼저 `--calib` 로 전자영점과 다자세 중력을 마쳐라.
#
# contact_probing:=false 는 이 층을 통째로 끈다. 전자저울 검증처럼 의도적으로 문턱을
# 넘겨 누르면서 teleop 을 계속해야 하는 절차 전용이며, 그때는 힘 유지도 함께 꺼진다.
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
# 초음파 영상. `us_frame_node` 가 프로브의 **유일한 클라이언트**이고, 그것이 내는
# /us/image 를 브리지가 GUI 로, capture_sweep·정책 노드가 각자 구독한다. 프로브는
# 클라이언트를 하나만 받으므로 이 노드를 여기서 소유해 두 번 뜨는 일을 막는다.
USE_US=1
US_PROBE="c10ur"
US_HOST="192.168.1.1"
# AP 접속까지 할 것인가. NetworkManager 는 polkit 인증을 요구하므로 tty/SSH 세션에서는
# 거부된다 — 그 경우 안내만 하고 세션은 계속 간다 (영상이 없다고 teleop 을 막을 이유가 없다).
CONNECT_AP=1
DRY_RUN=0
# 교정 모드. 브리지와 GUI 만 띄우고 제어 스택은 띄우지 않는다.
#
# 중력 식별은 조작자가 로봇을 손으로 여러 자세에 옮기며 한다. 그러려면 드래그
# 모드를 켜야 하고, 그동안 us_servo 가 ServoJ 를 쏘고 있으면 서로 싸운다. 그렇다고
# 제어 스택을 끄면 관절각 토픽이 끊겨 자세를 알 수 없다 — 자세를 모르면 중력 식별이
# 성립하지 않는다. 그래서 이 모드에서는 브리지가 컨트롤러에서 관절각을 **읽기로만**
# 가져온다 (bridge.robot_ip).
CALIB=0
ROBOT_IP="192.168.58.3"
LAUNCH_ARGS=()

while (( $# )); do
  case "$1" in
    --px6d)     PX6D_PORT="${2:-}"; shift 2 ;;
    --no-px6d)  USE_PX6D=0; shift ;;
    --no-us)    USE_US=0; shift ;;
    --no-ap)    CONNECT_AP=0; shift ;;
    --probe)    US_PROBE="${2:-c10ur}"; shift 2 ;;
    --us-host)  US_HOST="${2:-192.168.1.1}"; shift 2 ;;
    --no-gui)   USE_GUI=0; shift ;;
    --calib)    CALIB=1; shift ;;
    --ip)       ROBOT_IP="${2:-}"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    # 도움말은 헤더의 사용법 절 그대로다. 접촉 전환·힘 유지 설명이 그 안에 있으므로
    # 범위를 늘려 함께 나오게 한다 — 기본 커멘드가 무엇을 하는지가 도움말에 없으면
    # `-h` 를 본 조작자는 그것을 모른 채 로봇을 띄운다.
    -h|--help)  sed -n '2,55p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *)          LAUNCH_ARGS+=("$1"); shift ;;
  esac
done

# 세션 기본값. 조작자가 준 인자는 **덮어쓰는 것이 아니라 위에 얹는다.**
#
# 예전에는 인자가 하나라도 있으면 이 목록을 통째로 버렸다. 그래서
# `start_session.sh contact_probing:=false` 가 backend 를 launch 기본값인 mock 으로,
# teleop 을 false 로 돌려놓았고, 조작자는 실로봇을 몬다고 생각하면서 아무것도 안
# 뜬 상태를 보게 됐다. 인자 하나를 바꾸려고 나머지를 다 잃는 것은 인자가 아니라
# 함정이다.
SESSION_DEFAULTS=(backend:=fairino teleop:=true freespace:=true)
for default in "${SESSION_DEFAULTS[@]}"; do
  key="${default%%:=*}"
  supplied=0
  for given in ${LAUNCH_ARGS[@]+"${LAUNCH_ARGS[@]}"}; do
    [[ "${given%%:=*}" == "$key" ]] && supplied=1 && break
  done
  (( supplied )) || LAUNCH_ARGS+=("$default")
done

BRIDGE_LOG="$WORKSPACE/log/telemetry_bridge.log"
GUI_LOG="$WORKSPACE/log/teleop_gui.log"
BRIDGE_PID=""
GUI_PGID=""
US_PID=""

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
  if [[ -n "$US_PID" ]]; then
    kill -TERM "$US_PID" 2>/dev/null || true
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
# probe.yaml 을 반드시 넘긴다. 브리지는 렌치 보상에 적층 기하(장착각·레버암·
# 플랜지→센서 회전)를 쓰는데, 그 값들은 probe.yaml 에서 유도된다. 예전에는 이
# 파일을 안 넘겨서 노드 기본값으로만 돌았고, 레버암이 0 인 채로 오래 갔다 —
# 축방향 압축에서는 r ∥ f 라 차이가 0 이어서 화면상 정상으로 보인다.
#
# 이제 브리지는 값이 없으면 기동을 거부한다. 조용히 틀린 기하로 도는 것보다 낫다.
PROBE_YAML="$WORKSPACE/install/fr5_control/share/fr5_control/config/probe.yaml"
if [[ ! -f "$PROBE_YAML" ]]; then
  echo "  $PROBE_YAML 이 없다. colcon build 를 다시 하라." >&2
  exit 1
fi
BRIDGE_ARGS=(run fr5_control telemetry_bridge --ros-args --params-file "$PROBE_YAML")
if (( USE_PX6D )); then
  if [[ -e "$PX6D_PORT" ]]; then
    BRIDGE_ARGS+=(-p "bridge.px6d_port:=$PX6D_PORT")
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

if (( CALIB )); then
  # --ros-args 는 위 params-file 과 함께 이미 붙어 있다.
  BRIDGE_ARGS+=(-p "bridge.robot_ip:=$ROBOT_IP")
  echo "  로봇 읽기 전용: $ROBOT_IP  (명령은 보내지 않는다)"
fi

# 시리얼은 dialout 그룹이다. 그룹 목록(`id -nG`)이 아니라 **실제 접근 권한**으로
# 판정한다 — 그룹이 계정에 붙어 있어도 지금 셸의 자격증명에는 없을 수 있고, 그
# 차이가 "왜 여기서는 되는데 저기서는 안 되나" 로 나타난다.
#
# sg 로 감싸면 브리지가 그 자식이 되어 PID 추적이 한 단계 멀어지므로, 필요할 때만
# 쓴다. 그래도 놓치는 경우는 종료 시 fr5_stop_all 이 경로로 잡는다.
BRIDGE_CMD=(ros2 "${BRIDGE_ARGS[@]}")

# sg 는 setuid-root(newgrp)다. 동적 링커는 setuid 실행 파일의 환경에서 LD_* 를
# 지우므로, 감싼 안쪽 셸에는 LD_LIBRARY_PATH 가 **비어서** 도착한다. 그러면 rclpy
# 가 librcl_action.so 를 못 찾고 브리지는 import 단계에서 죽는다 — 2026-09-03 에
# 로그가 `ImportError: librcl_action.so` 트레이스백만 남기고 끝난 것이 이것이다.
# 소켓이 안 열리니 GUI 는 NO TELEMETRY 를 보여주고, 원인은 브리지 코드 어디에도
# 없다.
#
# 다른 ROS 변수(AMENT_PREFIX_PATH·PYTHONPATH·ROS_DISTRO)는 그대로 넘어오므로 이
# 하나만 되돌린다. setup.bash 를 안쪽에서 다시 source 하지 않는 이유는 sg -c 가
# /bin/sh 로 실행돼 bash 전용 setup 스크립트를 못 읽기 때문이다.
sg_wrap() {
  printf 'LD_LIBRARY_PATH=%q ' "${LD_LIBRARY_PATH:-}"
  printf '%q ' "$@"
}
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

# 접촉 전환과 힘 유지를 **probe.yaml 에서 읽어** 찍는다.
#
# 여기에 값을 적어 넣지 않는 이유: 적어 넣으면 파일이 바뀐 뒤에도 화면은 옛 수를
# 계속 말하고, 조작자는 로봇이 실제로 쓰는 값이 아니라 이 스크립트가 기억하는 값을
# 보게 된다. 화면에 찍힌 수와 거동이 다르면 그것이 곧 버그여야 한다.
print_contact_summary() {
  local enabled="$1"
  python3 - "$PROBE_YAML" "$enabled" <<'PY'
import sys

import yaml

path, enabled = sys.argv[1], sys.argv[2] == "1"

# 한 줄씩 바로 찍지 않고 모아서 찍는다. 예전에는 print 를 그대로 흘려보내고
# 실패하면 뒤에 "probe.yaml 을 읽지 못했다" 를 덧붙였는데, 중간에 키가 하나
# 사라지면 **앞부분은 이미 화면에 나간 상태**에서 그 문장이 붙었다. 조작자는
# 접촉 문턱은 읽었고 힘 유지 줄은 못 읽은 채 "읽지 못했다" 를 보게 되고, 무엇을
# 못 읽었는지는 아무 데도 안 적힌다. 2026-09-02 에 safety 의 힘 한계 키가
# max_normal_force_n → max_contact_force_n 으로 바뀐 뒤 실제로 그 상태였다.
out = []


def need(group, name, path_label):
    """없으면 어느 키가 없는지 말하고 끝낸다. KeyError 트레이스백은 답이 아니다."""
    try:
        return group[name]
    except (KeyError, TypeError):
        print(f"  (probe.yaml 에 {path_label} 이 없다 — 스크립트와 설정이 어긋났다. "
              "노드 기동 로그에서 실제 값을 확인하라)")
        sys.exit(0)


try:
    params = yaml.safe_load(open(path))["/**"]["ros__parameters"]
except (OSError, KeyError, yaml.YAMLError) as exc:
    print(f"  (probe.yaml 을 읽지 못했다: {exc} — 노드 기동 로그에서 확인하라)")
    sys.exit(0)

teleop = params.get("teleop", {})
control = params.get("contact_control", {})
sensor = params.get("ft_sensor", {})
safety = params.get("safety", {})

# 2026-09-02 이후 이름. 법선력이 아니라 접촉력(‖F‖)에 걸리는 한계라서 이름이 바뀌었다.
warn = need(safety, "warn_contact_force_n", "safety.warn_contact_force_n")
limit = need(safety, "max_contact_force_n", "safety.max_contact_force_n")

if not enabled:
    out.append("  ⚠️ 접촉 전환·힘 유지 꺼짐 (contact_probing:=false) — 힘이 얼마가 되든")
    out.append(f"     접근 상한을 유지한다. 남는 안전층은 힘 한계 {limit:.1f} N 뿐이다.")
    print("\n".join(out))
    sys.exit(0)

mode = sensor.get("contact_force_mode", "magnitude")
label = "‖F‖" if mode == "magnitude" else "F_n"
enter = need(teleop, "contact_probing_force_n", "teleop.contact_probing_force_n")
release = teleop.get("contact_probing_release_n", 0.0)
target = need(control, "target_force_n", "contact_control.target_force_n")
band = need(control, "deadband_n", "contact_control.deadband_n")
b_z = need(control, "admittance_b_z", "contact_control.admittance_b_z")

out.append(f"  접촉 판정   {label} ≥ {enter:.1f} N "
           f"({teleop.get('contact_probing_confirm_s', 0.0) * 1000:.0f} ms 연속)")
out.append(f"  힘 유지     {target:.1f} ± {band:.2f} N   "
           f"(경고 {warn:.1f} · 한계 {limit:.1f} N)")
# 접근 속도까지 찍는다. B_z 는 거동을 가장 크게 바꾸는 값인데 이름만으로는
# 무엇을 뜻하는지 안 보인다 — 조작자가 읽을 수 있는 단위는 mm/s 다.
# 진입 문턱에서 목표까지 남은 오차가 첫 속도를 정한다.
out.append(f"  접근 속도   B_z {b_z:.0f} N·s/m → 진입 직후 "
           f"{1000.0 * (target - enter) / b_z:.2f} mm/s (붙으며 감속)")
if release > 0.0:
    out.append(f"  접근 복귀   {label} ≤ {release:.1f} N "
               f"({teleop.get('contact_probing_release_confirm_s', 0.5):.1f} s 연속)")
else:
    out.append("  접근 복귀   없음 — 단방향 전환이다. 되돌리려면 세션을 새로 시작한다")

# 유지 밴드의 아래끝이 이탈 문턱 아래로 내려가면, 힘을 정상적으로 잡고 있는 동안에도
# 이탈 조건이 성립한다. 노드도 기동할 때 경고하지만, 여기가 먼저 눈에 든다.
if release > 0.0 and target - band <= release:
    out.append(f"  ⚠️ 유지 밴드 아래끝 {target - band:.2f} N 이 복귀 문턱 {release:.2f} N "
               "이하다 — 힘을 잡는 중에 모드가 오간다")
# probe.yaml 에 없으면 us_diff_ik 의 선언 기본값(True)이 쓰인다.
if not control.get("require_valid_calibration", True):
    out.append("  ⚠️ 교정 게이트가 꺼져 있다 — 보상 안 된 값으로 힘을 잡는다")
else:
    out.append("  (교정이 유효할 때만 열린다 — 아니면 접근 상한을 유지한다)")
print("\n".join(out))
PY
}

# contact_probing 인자를 읽는다. SESSION_DEFAULTS 가 채우지 않는 인자라 기본은 launch
# 쪽 기본값(true)이다.
CONTACT_PROBING_ON=1
for given in ${LAUNCH_ARGS[@]+"${LAUNCH_ARGS[@]}"}; do
  if [[ "${given%%:=*}" == "contact_probing" ]]; then
    case "${given#*:=}" in
      false|False|FALSE|0|no) CONTACT_PROBING_ON=0 ;;
    esac
  fi
done

echo
if (( CALIB )); then
  echo "5/5  교정 모드 — 제어 스택을 띄우지 않는다 (로봇에 명령을 보내지 않는다)"
else
  echo "5/5  teleop:  ros2 launch fr5_launch us_phase0.launch.py ${LAUNCH_ARGS[*]}"
  print_contact_summary "$CONTACT_PROBING_ON"
fi

if (( DRY_RUN )); then
  echo
  echo "--- dry run, 실행하지 않는다 ---"
  if (( USE_US )); then
    echo "  초음파: ros2 run fr5_vision us_frame_node --ros-args -p us.probe:=$US_PROBE -p us.host:=$US_HOST"
    (( CONNECT_AP )) && echo "          AP: python3 imu_bench/host/probe_wifi_linux.py"
  fi
  if (( NEEDS_SG )); then
    echo "  브리지: sg dialout -c \"$(sg_wrap "${BRIDGE_CMD[@]}")\"   > $BRIDGE_LOG"
  else
    echo "  브리지: ${BRIDGE_CMD[*]}   > $BRIDGE_LOG"
  fi
  if (( USE_GUI )); then
    echo "  GUI:    setsid bash $GUI_RUNNER built   > $GUI_LOG"
  fi
  if (( CALIB )); then
    echo "  teleop: (띄우지 않는다 — 교정 모드)"
  else
    echo "  teleop: ros2 launch fr5_launch us_phase0.launch.py ${LAUNCH_ARGS[*]}"
  fi
  BRIDGE_PID=""
  GUI_PGID=""
  trap - EXIT INT TERM
  exit 0
fi

mkdir -p "$(dirname "$BRIDGE_LOG")"
# 초음파를 브리지보다 **먼저** 띄운다. 순서가 기능을 가르지는 않지만(ROS 는 늦게 붙어도
# 된다), 프레임이 안 올 때 브리지 로그가 아니라 이 노드 로그를 먼저 보게 된다.
if (( USE_US )); then
  echo
  echo "초음파  us_frame_node (probe=$US_PROBE host=$US_HOST)"
  if (( CONNECT_AP )); then
    # **프로브가 없으면 여기서 멈춘다.** 예전에는 경고만 하고 계속 갔는데, 그러면
    # 영상 없는 콘솔이 떠서 조작자가 한참 쓴 뒤에야 알아차린다 — 그리고 그때는
    # 이미 세션을 다시 띄워야 한다. 영상이 목적인 세션이면 영상부터 세운다.
    # 영상 없이 쓸 작정이면 `--no-us` 로 그 뜻을 밝히면 된다.
    if /usr/bin/python3 "$WORKSPACE/imu_bench/host/probe_wifi_linux.py"; then
      echo "  AP 접속됨"
    else
      echo >&2
      echo "✗ 프로브 AP 에 붙지 못했다 — 세션을 띄우지 않는다." >&2
      echo "  · 프로브 배터리 전원을 켜고 USB 는 뽑은 상태인지" >&2
      echo "  · NetworkManager 가 polkit 인증을 요구하므로, 이 PC 화면에 로그인한" >&2
      echo "    터미널에서 실행하고 있는지 (tty/SSH 세션은 거부된다)" >&2
      echo "  영상 없이 진행하려면:  $0 --no-us $*" >&2
      exit 1
    fi
  fi
  US_LOG="$(mktemp -t us_frame_node.XXXXXX.log)"
  ros2 run fr5_vision us_frame_node --ros-args \
    -p us.probe:="$US_PROBE" -p us.host:="$US_HOST" >>"$US_LOG" 2>&1 &
  US_PID=$!
  echo "  로그: $US_LOG"
fi

: > "$BRIDGE_LOG"
if (( NEEDS_SG )); then
  sg dialout -c "$(sg_wrap "${BRIDGE_CMD[@]}")" >>"$BRIDGE_LOG" 2>&1 &
else
  "${BRIDGE_CMD[@]}" >>"$BRIDGE_LOG" 2>&1 &
fi
BRIDGE_PID=$!

# 브리지가 실제로 소켓을 열었는지 확인하고 넘어간다. 안 뜬 채로 teleop 을 띄우면
# 조작 중에 힘이 안 보이는 이유를 찾게 된다.
for _ in $(seq 40); do
  grep -q "브리지 대기" "$BRIDGE_LOG" 2>/dev/null && break
  # 죽었으면 10 초를 다 기다릴 이유가 없다. import 실패는 1 초 안에 끝난다.
  kill -0 "$BRIDGE_PID" 2>/dev/null || break
  sleep 0.25
done
if grep -q "브리지 대기" "$BRIDGE_LOG" 2>/dev/null; then
  # 센서 판정은 소켓보다 조금 늦게 나온다 (기동 로그상 100~200 ms). 여기서 안
  # 기다리면 아래 grep 이 소켓 줄만 잡고, 조작자는 힘이 붙었는지 모른 채 넘어간다.
  if (( USE_PX6D )); then
    for _ in $(seq 20); do
      grep -qE "PX6D 스트리밍 시작|PX6D 읽기 실패" "$BRIDGE_LOG" 2>/dev/null && break
      sleep 0.25
    done
  fi
  grep -E "PX6D 스트리밍 시작|PX6D 읽기 실패|브리지 대기" "$BRIDGE_LOG" | sed 's/^.*\]: /  /' | head -3
else
  # "로그를 봐라" 로 끝내지 않는다. 브리지가 못 뜨는 이유는 거의 항상 로그
  # 마지막 몇 줄에 그대로 있고, 그것을 여기서 보여 주지 않으면 조작자는 세션이
  # 이미 teleop 으로 넘어간 뒤에 힘이 없다는 사실부터 발견한다.
  if kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "  ⚠ 브리지가 10 초 안에 소켓을 안 열었다 (프로세스는 살아 있다)"
  else
    echo "  ⚠ 브리지가 기동 중 죽었다"
  fi
  echo "     $BRIDGE_LOG 마지막 줄:"
  tail -5 "$BRIDGE_LOG" 2>/dev/null | sed 's/^/       /'
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
if (( CALIB )); then
  echo
  echo "  ── 교정 모드 ──────────────────────────────────────────────"
  echo "  제어 스택은 띄우지 않는다. 로봇은 티치펜던트의 드래그 모드로 옮겨라."
  echo "  GUI 왼쪽 Calibration 페이지에서 진행한다:"
  echo "    1) 전자 영점 — 프로브가 아무것도 안 닿은 채 정지, 3~5 초"
  echo "    2) 다자세 중력 — 손으로 12 자세 이상, 서로 충분히 다르게"
  echo "  이 세션은 로봇에 **명령을 보내지 않는다.** Ctrl-C 로 종료."
  echo "  ───────────────────────────────────────────────────────────"
  echo
  # 브리지가 소유자다. 종료될 때까지 기다린다.
  wait "$BRIDGE_PID"
else
  ros2 launch fr5_launch us_phase0.launch.py "${LAUNCH_ARGS[@]}" &
  LAUNCH_PID=$!
  wait "$LAUNCH_PID"
fi
