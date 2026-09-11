#!/usr/bin/env python3
"""자세를 바꿔가며 공중에서 잔차 렌치를 재고, 영점으로 고쳐지는 종류인지 가른다.

    python3 scripts/diag_wrench_poses.py          # 프로브 전원 불필요

**아무것도 닿지 않은 상태**로 여러 자세에서 표본을 찍는다. 잔차의 성질이 원인을 가른다:

  벡터가 자세와 무관하게 일정   → 센서 전기 영점 오차. **영점(tare)으로 정확히 고쳐진다.**
  크기는 일정, 방향이 회전      → 질량·질량중심 추정 오차. **영점으로는 못 고친다** —
                                 한 자세에서만 맞고 나머지가 틀어진다. 정책이 프로브를
                                 회전시키는 실험에서는 회전할 때마다 잔차가 따라 돌아
                                 힘 제어기가 없는 힘을 쫓는다. 중력 교정을 다시 해야 한다.

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
    for _ in range(100):
        if node.w is not None and node.pose is not None:
            break
        rclpy.spin_once(node, timeout_sec=0.05)
    if node.w is None:
        print(f"렌치가 안 온다 — {args.namespace}/wrench_px6d")
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
            fz = 1 - 2*(q[1]**2 + q[2]**2)
            mag = float(np.linalg.norm(w))
            print(f"  [{len(rows)+1:2d}] ‖F‖={mag:5.2f} N  "
                  f"F=({w[0]:+5.2f},{w[1]:+5.2f},{w[2]:+5.2f})  flange_z={fz:+.2f}", end="")
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
    print(f"\n표본 {len(rows)}")
    print(f"  ‖F‖        평균 {mags.mean():.2f} N   표준편차 {mags.std():.3f}")
    print(f"  벡터 성분   std = ({W[:,0].std():.3f}, {W[:,1].std():.3f}, {W[:,2].std():.3f}) N")
    # 크기는 일정한데 벡터가 흔들리면 방향이 돈 것이다
    vec_spread = float(np.linalg.norm(W.std(axis=0)))
    print(f"  벡터 산포   {vec_spread:.3f} N   (크기 산포 {mags.std():.3f} N)\n")

    if mags.mean() < 0.3:
        print("  → 잔차가 애초에 작다. 보상이 잘 되고 있다.")
    elif vec_spread < 0.2 and mags.std() < 0.2:
        print("  → **벡터가 자세와 무관하게 일정하다 = 전기 영점 오차.**")
        print("     GUI 의 작업 자세 영점(tare)으로 정확히 고쳐진다. 0 으로 보정해도 된다.")
    elif vec_spread > 2 * max(mags.std(), 0.05):
        print("  → **크기는 일정한데 방향이 자세 따라 돈다 = 질량·질량중심 추정 오차.**")
        print("     영점으로는 못 고친다 — 한 자세에서만 맞고 나머지가 틀어진다.")
        print("     정책이 프로브를 회전시키므로 회전할 때마다 잔차가 따라 돈다.")
        print("     → 중력 교정(다자세)을 다시 하십시오.")
    else:
        print("  → 섞여 있다. 전기 영점과 질량 오차가 함께 있을 수 있다.")
        print("     중력 교정을 다시 하고, 그 뒤에도 남으면 그때 영점을 얹으십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
