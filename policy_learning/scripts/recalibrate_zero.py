#!/usr/bin/env python3
"""전원을 껐다 켠 뒤 실린 힘을 없앤다 — 전기 영점만 다시 잡는다.

    ./scripts/start_session.sh --calib --no-us --no-ap     # 다른 창에서 먼저
    python3 scripts/recalibrate_zero.py                    # 공중에서 자세 1~3 곳, Enter

**다자세 중력 교정을 다시 할 필요는 없다.** 하드웨어가 그대로면 질량·무게중심·장착 회전은
그대로다. 전원을 껐다 켜면 바뀌는 것은 스트레인게이지의 **전기 영점**뿐이고, 그것은 센서
프레임의 상수다. 센서→프로브 회전은 자세와 무관한 고정 회전이므로 프로브 프레임에서도
상수로 나타난다 — 자세를 바꿔도 같은 값이 따라다니고, 상수 하나만 고치면 전부 사라진다.

어디를 고치는가
---------------
`compensate()` 는 이렇게 접힌다::

    external = (raw − bias.bias) − (predict(g_s) − bias.bias) = raw − predict(g_s)

즉 **`bias.bias` 는 상쇄되어 사라진다.** 자세를 아는 경로에서 실제로 값을 바꾸는 것은
중력 모델 안의 `residual_bias` 하나뿐이다. GUI 의 "전자 영점" 은 `bias.bias` 에 들어가므로
그것만 눌러서는 보상된 렌치가 꿈쩍도 하지 않는다 (2026-09-11 "zero 버튼도 안 먹혀").
이 스크립트는 `residual_bias` 를 고친다.

작업 자세 영점은 **지운다.** 옛 영점으로 잰 값이라 그대로 두면 이번 보정과 겹쳐 두 번 빠진다.
고친 뒤 실제 시작 자세에서 다시 잡아라 (런북 C).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

import _common  # noqa: F401

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "fr5_control", "fr5_control"))

from fr5_control.wrench_frames import (FrameRegistration, wrench_rotate,  # noqa: E402
                                       wrench_translate)

DEFAULT_PROFILE = os.path.expanduser("~/.ros/fr5_px6d_calibration.json")


def _registration(reg: dict) -> FrameRegistration:
    return FrameRegistration(
        mounting_angle_deg=float(reg["mounting_angle_deg"]),
        axial_flip=bool(reg["axial_flip"]),
        r_sensor_to_probe_m=np.asarray(reg["r_sensor_to_probe_m"], float),
        flange_to_sensor_rpy=np.asarray(reg["flange_to_sensor_rpy"], float),
    )


def sensor_shift(observed_probe: np.ndarray, reg: FrameRegistration) -> np.ndarray:
    """프로브 프레임에서 본 상수 오차 → 센서 프레임의 영점 이동.

    ``compensate`` 의 뒤쪽 두 단계를 거꾸로 되돌린다::

        contact_probe = wrench_translate(wrench_rotate(R_ps, external_sensor), r_sp)

    순수 함수라 로봇 없이 시험할 수 있다.
    """
    rot = reg.rotation_probe_from_sensor()
    r_sp = np.asarray(reg.r_sensor_to_probe_m, float)
    back = wrench_translate(np.asarray(observed_probe, float).reshape(6), -r_sp)
    return wrench_rotate(rot.T, back)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--namespace", default="/fr5_right")
    p.add_argument("--profile", default=DEFAULT_PROFILE)
    p.add_argument("--seconds", type=float, default=2.0, help="자세마다 모을 시간")
    p.add_argument("--min-force", type=float, default=0.3,
                   help="이보다 작으면 고칠 것이 없다고 보고 멈춘다")
    p.add_argument("--yes", action="store_true", help="확인 없이 적용한다")
    args = p.parse_args()

    if not os.path.exists(args.profile):
        print(f"교정 파일이 없다: {args.profile} — 먼저 중력 교정을 하라")
        return 1
    prof = json.load(open(args.profile))
    if not prof.get("gravity"):
        print("중력 모델이 없다 — 전기 영점만으로는 보상할 수 없다. 다자세 중력 교정을 하라")
        return 1
    reg = _registration(prof["registration"])

    import rclpy
    from geometry_msgs.msg import WrenchStamped
    from rclpy.node import Node

    class Probe(Node):
        def __init__(self):
            super().__init__("recalibrate_zero")
            self.w = None
            self.frame = ""
            self.create_subscription(WrenchStamped, f"{args.namespace}/wrench_px6d", self._w, 10)

        def _w(self, m):
            f, t = m.wrench.force, m.wrench.torque
            self.w = np.array([f.x, f.y, f.z, t.x, t.y, t.z])
            self.frame = m.header.frame_id

    rclpy.init()
    node = Probe()
    for _ in range(100):
        if node.w is not None:
            break
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.w is None:
        print(f"렌치가 안 온다 — {args.namespace}/wrench_px6d 에 아무도 publish 하지 않는다.")
        print("  ./scripts/start_session.sh --calib --no-us --no-ap 를 먼저 띄우십시오")
        rclpy.shutdown()
        return 1
    if not node.frame.endswith("_probe"):
        print(f"⚠️ 보상 전 렌치다 (frame_id={node.frame!r}) — 중력 교정이 먼저다")
        rclpy.shutdown()
        return 1

    tare = np.asarray(prof.get("workingTare") or [0.0] * 6, float)
    print("**아무것도 닿지 않은 상태**로 Enter. 자세를 바꿔 두세 번 하면 더 정확하다. q 로 끝.\n")
    means = []
    try:
        while True:
            print(f"  [{len(means) + 1}] 수집 중 {args.seconds:.0f} s …", end="", flush=True)
            got = []
            t_end = node.get_clock().now().nanoseconds / 1e9 + args.seconds
            while node.get_clock().now().nanoseconds / 1e9 < t_end:
                rclpy.spin_once(node, timeout_sec=0.05)
                if node.w is not None:
                    got.append(node.w.copy())
            if not got:
                print(" 표본 없음")
                break
            m = np.mean(got, axis=0)
            # 기록된 값은 작업 영점이 이미 빠진 뒤다. 그 영점은 옛 전기 영점으로 잰 것이라
            # 지울 것이므로, 여기서 되돌려 놓고 계산한다.
            means.append(m + tare)
            print(f" ‖F‖={np.linalg.norm(m[:3]):.2f} N  (작업영점 되돌리면 "
                  f"{np.linalg.norm(means[-1][:3]):.2f} N)")
            if input("   Enter=자세 추가 / q=계산 > ").strip().lower() == "q":
                break
    except (EOFError, KeyboardInterrupt):
        pass
    rclpy.shutdown()

    if not means:
        print("표본이 없다.")
        return 1
    obs = np.mean(means, axis=0)
    if len(means) > 1:
        spread = np.linalg.norm(np.std(means, axis=0)[:3])
        print(f"\n자세 {len(means)} 곳 · 자세 사이 힘 산포 {spread:.2f} N "
              f"(이만큼은 상수가 아니라 자세에 따른 몫이라 영점으로 안 지워진다)")
    if np.linalg.norm(obs[:3]) < args.min_force:
        print(f"\n공중 힘이 {np.linalg.norm(obs[:3]):.2f} N 뿐이다 — 고칠 것이 없다.")
        return 0

    shift = sensor_shift(obs, reg)
    g = prof["gravity"]
    old_rb = np.asarray(g["residual_bias"], float)
    new_rb = old_rb + shift
    print("\n공중에서 읽힌 값 (프로브 프레임, 작업영점 되돌림)")
    print(f"  F = ({obs[0]:+.2f}, {obs[1]:+.2f}, {obs[2]:+.2f}) N   ‖F‖ {np.linalg.norm(obs[:3]):.2f}")
    print("\n residual_bias 를 이만큼 옮긴다 (센서 프레임)")
    print(f"  Δ = ({shift[0]:+.2f}, {shift[1]:+.2f}, {shift[2]:+.2f}, "
          f"{shift[3]:+.3f}, {shift[4]:+.3f}, {shift[5]:+.3f})")
    print(f"  질량 {g['mass_kg'] * 1000:.0f} g · 무게중심 · 장착 회전은 **그대로 둔다**")
    if prof.get("workingTare"):
        print(f"  작업 자세 영점(‖{np.linalg.norm(tare[:3]):.2f} N‖)은 지운다 — 옛 영점으로 잰 값이다")

    if not args.yes:
        if input("\n적용할까요? (y) > ").strip().lower() != "y":
            print("적용하지 않았다.")
            return 0

    shutil.copy(args.profile, args.profile + ".prev")
    g["residual_bias"] = new_rb.tolist()
    prof["bias"]["bias"] = new_rb.tolist()      # 두 값은 같게 유지한다 (refit 규약)
    prof["bias"]["reason"] = "전원 재기동 뒤 전기 영점만 다시 잡음 (recalibrate_zero.py)"
    prof["workingTare"] = None
    prof["tareFlangeZ"] = None
    prof["tarePoseDeg"] = []
    prof["mountingNote"] = (str(prof.get("mountingNote", "")) +
                            f" · 전기영점 재설정 {np.linalg.norm(obs[:3]):.2f} N "
                            f"(자세 {len(means)}) recalibrate_zero.py")
    with open(args.profile, "w", encoding="utf-8") as fh:
        json.dump(prof, fh, ensure_ascii=False, indent=2)
    print(f"\n저장했다 → {args.profile}   (직전 파일은 {args.profile}.prev)")
    print("브리지가 아직 옛 값을 들고 있다. 세션을 다시 띄워야 반영된다:")
    print("  ./scripts/stop_all.sh && ./scripts/start_session.sh")
    print("그 뒤 공중에서 ‖F‖ 가 1 N 아래인지 확인하고, 시작 자세에서 작업 자세 영점을 잡아라.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
