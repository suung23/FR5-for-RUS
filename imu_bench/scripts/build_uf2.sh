#!/usr/bin/env bash
# IMU 펌웨어를 빌드해 UF2 로 만든다 — Windows 수집 노트북(Smart App Control 켜짐, 서명 없는 툴체인 실행 불가)에
# 파일 복사만으로 올리기 위해. 리눅스에서 flash.sh 와 같은 툴체인(setup_toolchain.sh) 을 쓴다.
#
#   ./scripts/build_uf2.sh                 -> firmware/umi_device_hardware/umi_device_hardware.uf2  (git 으로 나른다)
#   ./scripts/build_uf2.sh --no-commit-dir -> .build/imu_only/umi_device_hardware.uf2 에만 둔다
#
# Windows 에서:  python imu_bench\host\flash_win.py --uf2 imu_bench\firmware\umi_device_hardware\umi_device_hardware.uf2
#
# 보드가 리눅스 PC 에 꽂혀 있다면 그냥 ./scripts/flash.sh 가 더 짧다. 이 스크립트는 보드가 Windows 쪽에 있을 때 쓴다.
set -euo pipefail
source "$(dirname "$0")/env.sh"

SKETCH="$IMU_BENCH_ROOT/firmware/umi_device_hardware"
OUT="$IMU_BENCH_ROOT/.build/imu_only"
mkdir -p "$OUT"
echo "== compile imu_only =="
arduino-cli compile --fqbn "$FQBN" --build-property 'compiler.cpp.extra_flags=-DUMI_IMU_ONLY=1' \
  --output-dir "$OUT" "$SKETCH"

HEX="$OUT/umi_device_hardware.ino.hex"
UF2="$OUT/umi_device_hardware.uf2"
echo "== hex -> uf2 =="
python3 "$IMU_BENCH_ROOT/host/hex2uf2.py" "$HEX" -o "$UF2"

if [ "${1:-}" != "--no-commit-dir" ]; then
  cp "$UF2" "$SKETCH/umi_device_hardware.uf2"
  TAG="$(sed -n 's/^#define FW_TAG "\(.*\)"/\1/p' "$SKETCH/umi_device_hardware.ino")"
  echo "== $SKETCH/umi_device_hardware.uf2  (FW_TAG $TAG) — git add/commit/push 해서 Windows 로 =="
fi
echo "done."
