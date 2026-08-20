#!/usr/bin/env bash
# 툴체인 경로와 시리얼 접근 규약을 한 곳에 모아둔다. 다른 스크립트가 source 한다.
#
# 시리얼: /dev/ttyACM* 는 root:dialout 0660 이고 이 셸은 dialout 을 상속받지 못할 수
# 있다. chmod 는 USB 재열거(업로드·리셋마다 발생)마다 초기화되므로, 항상 sg 로 감싼다.

IMU_BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export IMU_BENCH_ROOT

# arduino-cli 와, 코어가 요구하는 python / adafruit-nrfutil 심링크
export PATH="$HOME/.local/bin:$IMU_BENCH_ROOT/.toolchain/bin:$PATH"

: "${IMU_PORT:=/dev/ttyACM0}"
export IMU_PORT

FQBN="Seeeduino:nrf52:xiaonRF52840Sense"
export FQBN

# dialout 그룹으로 감싸 실행한다. 이미 그룹에 속해 있으면 그대로 실행.
run_serial() {
  if id -nG | tr ' ' '\n' | grep -qx dialout; then
    bash -c "$*"
  else
    sg dialout -c "PATH=$PATH; $*"
  fi
}
