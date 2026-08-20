#!/usr/bin/env bash
# 뷰어를 dialout 권한으로 실행한다. 인자는 그대로 imu_fusion_view.py 로 전달된다.
#
#   ./scripts/run.sh --calibrate-mag 20   # 자력계 보정 (8자 모션)
#   ./scripts/run.sh                      # 9축 값 + 퓨전
#   ./scripts/run.sh --raw                # 원시 레코드 덤프
#   ./scripts/run.sh --no-mag             # 6축 IMU-only 로 비교
set -euo pipefail
source "$(dirname "$0")/env.sh"
run_serial "python3 '$IMU_BENCH_ROOT/host/imu_fusion_view.py' --port '$IMU_PORT' $*"
