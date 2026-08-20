#!/usr/bin/env bash
# 실시간 GUI 뷰어. 인자는 그대로 imu_gui.py 로 전달된다.
#
#   ./scripts/gui.sh              # 보기만
#   ./scripts/gui.sh --log        # 시작과 동시에 로깅 (imu_bench/logs)
#   ./scripts/gui.sh --no-mag     # 6축 IMU-only 로 비교
set -euo pipefail
source "$(dirname "$0")/env.sh"

# 에이전트/서비스 셸에는 DISPLAY 가 없다. 이 PC 의 X 소켓을 찾아 붙는다.
if [ -z "${DISPLAY:-}" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
  for sock in /tmp/.X11-unix/X*; do
    [ -e "$sock" ] || continue
    export DISPLAY=":${sock##*/X}"
    break
  done
fi
echo "DISPLAY=${DISPLAY:-<none>}"

run_serial "DISPLAY='${DISPLAY:-}' python3 '$IMU_BENCH_ROOT/host/imu_gui.py' --port '$IMU_PORT' $*"
