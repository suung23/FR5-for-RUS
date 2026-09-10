#!/usr/bin/env python3
"""⏳ imu.sensor_to_probe (R_SP) 를 세션 하나로 잰다 (2026-09-08, README §3).

    python scripts/calibrate_sensor_to_probe.py <session_dir> [--set imu.still_gyro_sd=...]

수집 절차 (한 세션, 30 s):
    [정지 1 s] → 프로브 **+x (lateral)** 축 둘레로 오른손 양의 방향으로 크게 (≥ 45°) 돌렸다 되돌린다 → [정지 1 s]
             → **+y (elevational)** 축 둘레로 같은 식 → [정지 1 s] → **+z (beam)** 축 둘레로 같은 식 → [정지 1 s]
    "되돌리는" 반회전은 방향 판정을 흐리므로, 갈 때는 빠르게·올 때는 천천히 돌리거나 되돌림 사이에 정지를 둔다.
    프로브 축 규약은 docs/FRAMES_AND_SE2.md §1 (x lateral, y elevational, z beam).

동작: 정지 판정으로 이동 구간을 나누고, 누적 회전각이 큰 순서로 세 구간을 골라 **시간 순**으로 x, y, z 에
대응시킨다. 각 구간의 자이로 주축이 R_SP 의 행이다 (rus_policy.calibration.estimate_sensor_to_probe).
출력은 그대로 configs/policy_default.yaml 의 imu.sensor_to_probe 에 붙여 넣을 수 있는 형식이다.
검산: 정지 상태에서 빔축(+z) 을 중력에 맞추면 R_SP·a_S ≈ (0, 0, −g) 여야 한다 (--gravity-check).
"""

from __future__ import annotations

import argparse

import numpy as np

from _common import add_config_args, config_from_args

from rus_policy.calibration import estimate_sensor_to_probe
from rus_policy.imu_labels import move_segments, still_mask
from rus_policy.session import load_session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("session_dir")
    parser.add_argument("--n-segments", type=int, default=3, help="회전 구간 수 (x, y, z)")
    parser.add_argument("--gravity-check", action="store_true",
                        help="첫 정지 구간에서 R_SP·a_S 를 출력 (빔축을 중력에 맞춘 세션이면 (0,0,−g) 근처)")
    args = parser.parse_args()
    cfg = config_from_args(args)

    s = load_session(args.session_dir)
    imu = s.imu
    dt = float(np.median(np.diff(imu.t_dev)))
    still = still_mask(imu.t_dev, imu.gyr, imu.acc, cfg.imu, dt=dt)
    segs, stills = move_segments(imu.t_dev, still, cfg.imu)
    if len(segs) < args.n_segments:
        print(f"이동 구간이 {len(segs)} 개뿐입니다 (필요 {args.n_segments}). 정지를 더 길게 두거나 임계를 조정하십시오.")
        return 1
    # 누적 회전각 (∫|ω| dt) 가 큰 구간 n 개 → 시간 순.
    # np.trapz 는 NumPy 2.0 에서 제거됐다 (np.trapezoid 로 이름만 바뀜) — 두 쪽 다 받는다.
    integrate = getattr(np, "trapezoid", None) or np.trapz
    rot = []
    for sg in segs:
        sl = slice(sg.a1, sg.b0 + 1)
        w = np.linalg.norm(imu.gyr[sl], axis=1)
        rot.append(float(integrate(w, imu.t_dev[sl])) if w.size > 1 else 0.0)
    order = sorted(np.argsort(rot)[-args.n_segments:])
    chosen = [segs[i] for i in order]
    print("회전 구간 (시간 순 → x, y, z):")
    for name, i, sg in zip("xyz", order, chosen):
        print(f"  {name}: t = {imu.t_dev[sg.a1] - imu.t_dev[0]:.2f}–{imu.t_dev[sg.b0] - imu.t_dev[0]:.2f} s, "
              f"누적 회전 {np.degrees(rot[i]):.0f}°")
    gyr_segments = [imu.gyr[sg.a1:sg.b0 + 1] for sg in chosen]
    est = estimate_sensor_to_probe(gyr_segments)
    print("\n원시 주축 사이 각도 (xy, yz, zx) [deg]: " + " ".join(f"{v:.1f}" for v in est.angles_deg)
          + "   (90 에서 10° 이상 벗어나면 회전이 한 축에 실리지 않은 것)")
    print(f"투영 잔차 {est.residual_deg:.2f}°, det {est.det:+.3f}")
    print("\nimu:\n  sensor_to_probe:")
    for row in est.as_yaml_rows():
        print(f"    - [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
    if args.gravity_check and stills:
        a0, a1 = stills[0]
        a_S = imu.acc[a0:a1 + 1].mean(axis=0)
        a_P = est.R_sp @ a_S
        print(f"\n중력 검산 (첫 정지): R_SP·a_S = ({a_P[0]:+.2f}, {a_P[1]:+.2f}, {a_P[2]:+.2f}) m/s²"
              "   빔축을 아래로 둔 세션이면 (0, 0, −9.8) 근처여야 한다 (센서가 재는 것은 −g 방향의 반력)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
