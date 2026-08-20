#!/usr/bin/env bash
# 펌웨어를 빌드해 XIAO nRF52840 Sense 에 올린다.
#
#   ./scripts/flash.sh              -> IMU 전용 빌드 (BNO085 만 있는 이 벤치 구성)
#   ./scripts/flash.sh i2c_scan     -> I2C 스캐너 (버스에 뭐가 붙었는지 확인용)
#   ./scripts/flash.sh full         -> ToF/홀 포함 원본 구성
set -euo pipefail
source "$(dirname "$0")/env.sh"

TARGET="${1:-imu_only}"
case "$TARGET" in
  imu_only) SKETCH="$IMU_BENCH_ROOT/firmware/umi_device_hardware"
            EXTRA='compiler.cpp.extra_flags=-DUMI_IMU_ONLY=1' ;;
  full)     SKETCH="$IMU_BENCH_ROOT/firmware/umi_device_hardware"
            EXTRA='compiler.cpp.extra_flags=' ;;
  i2c_scan) SKETCH="$IMU_BENCH_ROOT/firmware/i2c_scan"
            EXTRA='compiler.cpp.extra_flags=' ;;
  *) echo "unknown target: $TARGET (imu_only | full | i2c_scan)" >&2; exit 2 ;;
esac

OUT="$IMU_BENCH_ROOT/.build/$TARGET"
mkdir -p "$OUT"
echo "== compile $TARGET =="
arduino-cli compile --fqbn "$FQBN" --build-property "$EXTRA" --output-dir "$OUT" "$SKETCH"

# arduino-cli 는 이 코어에서 1200bps touch 를 스스로 걸지 못한다("Touch disabled").
# 수동으로 걸어 부트로더로 넘긴다: 성공하면 USB PID 가 0x8045 -> 0x0045 로 바뀐다.
echo "== bootloader touch =="
run_serial "python3 - <<'PY'
import serial, time
try:
    s = serial.Serial('$IMU_PORT', 1200); s.dtr = False; time.sleep(0.1); s.close()
    print('  1200bps touch sent')
except Exception as e:
    print('  touch skipped:', e)
PY"
sleep 4

echo "== upload =="
run_serial "arduino-cli upload --fqbn '$FQBN' --input-dir '$OUT' -p '$IMU_PORT' '$SKETCH'"
sleep 3
echo "done."
