#!/usr/bin/env bash
# 이 벤치를 처음 쓰는 PC에서 한 번 실행한다. 이미 설치된 항목은 건너뛴다.
#
# 설치 대상
#   ~/.local/bin/arduino-cli          : 빌드/업로드 프론트엔드
#   Seeeduino:nrf52 코어              : XIAO nRF52840 (TinyUSB 계열 — 펌웨어가 요구)
#   Adafruit BNO08x / VL53L0X / ...   : 센서 라이브러리
#   imu_bench/.toolchain/             : adafruit-nrfutil venv + `python` 심링크
#
# 마지막 두 개가 왜 필요한지:
#   * Seeeduino nrf52 코어는 빌드 중 `python` 실행파일을 직접 부른다. 요즘 배포판엔
#     python3 만 있어서 심링크가 없으면 "exec: python: not found" 로 빌드가 깨진다.
#   * 업로드는 adafruit-nrfutil 로 하는데 리눅스에선 코어가 번들하지 않는다.
set -euo pipefail
source "$(dirname "$0")/env.sh"

BOARD_URL="https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json"

if ! command -v arduino-cli >/dev/null 2>&1; then
  echo "== arduino-cli 설치 =="
  mkdir -p "$HOME/.local/bin"
  curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh \
    | BINDIR="$HOME/.local/bin" sh
fi

echo "== 보드 인덱스 =="
arduino-cli config init --overwrite >/dev/null
arduino-cli config add board_manager.additional_urls "$BOARD_URL"
arduino-cli core update-index

echo "== 보드 코어 =="
arduino-cli core install Seeeduino:nrf52

echo "== 센서 라이브러리 =="
arduino-cli lib install "Adafruit BNO08x" "Adafruit_VL53L0X" \
                        "Adafruit BusIO" "Adafruit Unified Sensor"

echo "== .toolchain (adafruit-nrfutil + python 심링크) =="
TC="$IMU_BENCH_ROOT/.toolchain"
mkdir -p "$TC/bin"
[ -x "$TC/venv/bin/adafruit-nrfutil" ] || {
  python3 -m venv "$TC/venv"
  "$TC/venv/bin/pip" install -q --upgrade pip
  "$TC/venv/bin/pip" install -q adafruit-nrfutil
}
ln -sf "$(command -v python3)"            "$TC/bin/python"
ln -sf "$TC/venv/bin/adafruit-nrfutil"    "$TC/bin/adafruit-nrfutil"

echo "== 호스트 파이썬 의존성 =="
python3 -c "import serial, numpy" 2>/dev/null \
  || echo "  ! pyserial / numpy 가 없습니다: pip install pyserial numpy"

echo
echo "완료. 시리얼 접근 권한이 아직 없다면 한 번만:"
echo "  sudo usermod -aG dialout \$USER"
