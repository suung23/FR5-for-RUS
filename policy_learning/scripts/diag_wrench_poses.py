#!/usr/bin/env python3
"""자세를 바꿔가며 공중에서 잔차 렌치를 재고, 영점으로 고쳐지는 종류인지 가른다.

    ./scripts/start_session.sh --calib --no-us --no-ap    # 다른 창에서 먼저
    python3 scripts/diag_wrench_poses.py                  # 프로브 전원 불필요

렌치를 내는 것은 telemetry_bridge 다. **교정 모드**로 띄우는 이유는 이 진단이 손으로
자세를 바꾸는 절차여서다 — 제어 스택이 함께 떠 있으면 us_servo 가 ServoJ 를 쏘아
드래그 모드와 싸운다. 교정 모드에는 제어 스택이 없으므로 자세(ee_wrt_base)도 안 온다.
자세는 참고로 찍을 뿐 판정에는 쓰지 않으니 없어도 된다.

**아무것도 닿지 않은 상태**로 여러 자세에서 표본을 찍고, 잔차를 둘로 쪼갠다:

  상수 성분 (자세 평균)        → **영점(tare)이 지운다.**
  그 주위 산포                → **영점으로 안 지워진다.** 영점은 상수를 빼는 것이라
                               잰 자세에서 멀어지면 뺀 만큼이 오차로 돌아온다.

조작자가 정해야 하는 것은 원인의 이름이 아니라 **영점을 잡고 진행해도 되는가** 이고,
그 답은 남는 몫의 크기가 힘 대역에 비해 작은가로 바로 나온다.

이 장비 실측(자세 75 개, 2026-09-11): 상수 0.83 N · 자세 성분 중앙 0.42 N, 최대 1.2 N.
즉 **영점을 잡으면 대부분 사라지고 0.4 N 남짓이 남는다** — 3 N 목표에 비해 작다.

자세를 손으로(드래그 모드) 또는 teleop 으로 바꾸고, 매번 Enter 를 친다. q 로 끝낸다.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import _common  # noqa: F401


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--namespace", default="/fr5_right")
    args = p.parse_args()

    import rclpy
    from geometry_msgs.msg import Pose, WrenchStamped
    from rclpy.node import Node

    class Probe(Node):
        def __init__(self):
            super().__init__("diag_wrench_poses")
            self.w = None
            self.frame = ""
            self.pose = None
            self.create_subscription(WrenchStamped, f"{args.namespace}/wrench_px6d", self._w, 10)
            self.create_subscription(Pose, f"{args.namespace}/ee_wrt_base", self._p, 10)

        def _w(self, m):
            f = m.wrench.force
            self.w = np.array([f.x, f.y, f.z])
            self.frame = m.header.frame_id

        def _p(self, m):
            o = m.orientation
            n = math.sqrt(o.w**2 + o.x**2 + o.y**2 + o.z**2) or 1.0
            self.pose = (o.w/n, o.x/n, o.y/n, o.z/n)

    rclpy.init()
    node = Probe()
    # 자세는 **있으면 찍고 없으면 넘어간다.** ee_wrt_base 를 내는 것은 제어 스택인데,
    # 이 진단은 손으로 자세를 바꿔야 하므로 제어 스택이 없는 교정 모드에서 돌린다 —
    # 즉 자세가 없는 것이 이 스크립트의 **정상 사용법**이다. 예전에는 두 토픽을 같이
    # 기다렸다가 자세 없이 진행한 뒤 flange_z 를 읽다 TypeError 로 죽었고, 그 값은
    # 화면에 찍히기만 할 뿐 판정에는 쓰이지도 않는다.
    for _ in range(100):
        if node.w is not None:
            break
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.w is None:
        print(f"렌치가 안 온다 — {args.namespace}/wrench_px6d 에 아무도 publish 하지 않는다.")
        print("   이 토픽을 내는 것은 telemetry_bridge 다. 세션이 떠 있는지 확인하라:")
        print("     ros2 topic list | grep wrench")
        print("   안 떠 있으면 **교정 모드**로 띄운다 (제어 스택 없이 브리지·GUI 만 —")
        print("   손으로 자세를 바꿔야 하므로 이 진단에는 이 모드가 맞다):")
        print("     ./scripts/start_session.sh --calib --no-us --no-ap")
        rclpy.shutdown(); return 1
    if not node.frame.endswith("_probe"):
        print(f"⚠️ 보상 전 렌치다 (frame_id={node.frame!r}) — 중력 교정이 먼저다")
        rclpy.shutdown(); return 1

    print("아무것도 닿지 않은 상태로 자세를 바꿔가며 Enter. q 로 끝냅니다.\n")
    rows = []
    try:
        while True:
            for _ in range(20):
                rclpy.spin_once(node, timeout_sec=0.02)
            w, q = node.w.copy(), node.pose
            fz = 1 - 2*(q[1]**2 + q[2]**2) if q is not None else float("nan")
            mag = float(np.linalg.norm(w))
            shown = f"  flange_z={fz:+.2f}" if q is not None else ""
            print(f"  [{len(rows)+1:2d}] ‖F‖={mag:5.2f} N  "
                  f"F=({w[0]:+5.2f},{w[1]:+5.2f},{w[2]:+5.2f}){shown}", end="")
            if input("   Enter=기록 / q=끝 > ").strip().lower() == "q":
                break
            rows.append((w, mag, fz))
    except (EOFError, KeyboardInterrupt):
        pass
    rclpy.shutdown()

    if len(rows) < 3:
        print("\n표본이 3 개 미만이라 판정할 수 없습니다 — 자세를 셋 이상에서 재십시오.")
        return 0

    W = np.stack([r[0] for r in rows])
    mags = np.array([r[1] for r in rows])

    # 잔차를 **영점으로 지워지는 것**과 **안 지워지는 것**으로 쪼갠다.
    #
    # 예전에는 "전기 영점 오차 / 질량·질량중심 오차" 둘 중 하나로 가르려 했다.
    # 이 장비의 실측(자세 75 개)에 그 기준을 돌리면 200 번 중 200 번 "섞여 있다" 가
    # 나온다 — 즉 가르지 못한다. 실제로도 둘 중 하나가 아니라 **둘 다** 있다:
    # 상수 0.83 N 에 자세 따라 도는 0.42 N 이 얹혀 있다.
    #
    # 조작자가 정해야 하는 것은 원인의 이름이 아니라 **영점을 잡고 진행해도 되는가** 다.
    # 그 답은 평균 벡터(영점이 지우는 몫)와 그 주위 산포(남는 몫)로 바로 나온다.
    const = W.mean(axis=0)                       # 영점이 지우는 몫
    resid = W - const
    left = np.linalg.norm(resid, axis=1)         # 영점 뒤에도 남는 몫
    print(f"\n표본 {len(rows)}")
    print(f"  지금 읽히는 ‖F‖      평균 {mags.mean():.2f} N   최대 {mags.max():.2f} N")
    print(f"  상수 성분           ({const[0]:+.2f}, {const[1]:+.2f}, {const[2]:+.2f})"
          f"   ‖·‖ {np.linalg.norm(const):.2f} N   ← 영점이 지운다")
    print(f"  자세 따라 도는 성분   중앙 {np.median(left):.2f} N   최대 {left.max():.2f} N"
          f"   ← 영점으로 안 지워진다")
    print("\n  (이 장비 실측 기준값: 상수 0.83 N · 자세 성분 중앙 0.42 N — "
          "자세 75 개, 2026-09-11)\n")

    y = float(np.median(left))
    if mags.mean() < 0.3:
        print("  → 잔차가 애초에 작다. 그대로 진행해도 된다.")
    elif y < 0.5:
        print(f"  → **영점을 잡으면 {np.linalg.norm(const):.2f} N 이 사라지고 "
              f"{y:.2f} N 이 남는다.**")
        print("     남는 몫이 5 N 안전 대역에 비해 작다. **영점 잡고 진행해도 된다** —")
        print("     단 영점은 **실제 시작 자세에서, 실행 직전에** 잡아야 한다. 영점은")
        print("     상수를 빼는 것이라 잰 자세에서 멀어질수록 뺀 만큼이 오차로 돌아온다.")
    elif y < 1.0:
        print(f"  → 영점이 {np.linalg.norm(const):.2f} N 을 지우지만 {y:.2f} N 이 남는다.")
        print("     쓸 수는 있으나 여유가 크지 않다. 시작 자세에서 영점을 잡고,")
        print("     정책의 회전 범위를 좁게 유지하라 (영점 자세에서 ±30° 안).")
    else:
        print(f"  → 영점을 잡아도 {y:.2f} N 이 남는다. **이건 영점 문제가 아니다.**")
        print("     중력 모델이 이 자세 영역을 설명하지 못하고 있다:")
        print("       python3 phantom_stiffness/refit_calibration.py --count 75 \\")
        print("               --down-cone-deg 60 --min-poses 12 --dry-run")
        print("     그래도 안 되면 GUI Calibration 에서 다자세 중력을 다시 잡아라.")
        print("     ⚠️ 자세를 **많이** 모아라. 자세가 적으면 적합이 그 자세들에만 맞춰지고")
        print("        (in-sample) rms 는 좋아 보이지만 다른 자세에서 두 배로 틀어진다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
