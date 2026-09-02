#!/usr/bin/env python3
"""힘 축이 무엇을 지령받고 무엇을 달성하는지 한 줄로 본다.

    python3 monitor_force.py

"안 움직인다" 에는 세 가지가 있고 이 화면이 그것을 가른다.

* **지령이 없다** — 조절기가 v_z = 0 을 낸다. 밴드 안이거나, 접촉 프로빙이 아니거나,
  교정이 막혔거나. 사유 문자열이 어느 쪽인지 말한다.
* **지령은 있는데 안 간다** — v_z 는 0 이 아닌데 달성 오차 e_z 가 그만큼 크다.
  특이점 근처이거나 관절이 한계에 걸린 것이다.
* **가고 있는데 느리다** — v_z 가 작다. admittance B_z 가 오차에 비해 크다.
"""
from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray, String

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--namespace", default="/fr5_right")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    rclpy.init()
    node = Node("monitor_force")
    st = {"d": None, "reason": "(아직 없음)", "mode": ""}

    node.create_subscription(Float32MultiArray, "/diag/force_regulation",
                             lambda m: st.update(d=list(m.data)), 10)
    node.create_subscription(String, "/diag/force_regulation_reason",
                             lambda m: st.update(reason=m.data), 10)
    node.create_subscription(String, f"{args.namespace}/probing_mode",
                             lambda m: st.update(mode=m.data), LATCHED)

    print(f"{'모드':<22}{'힘':>7}{'목표':>7}{'지령 v_z':>11}{'달성오차 e_z':>13}   사유")
    print("─" * 96)
    last = None
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.2)
            d = st["d"]
            if d is None or len(d) < 5:
                continue
            f, tgt, v_z, e_z, probing = d[:5]
            line = (f"{st['mode'] or '?':<22}{f:>6.3f}N{tgt:>6.1f}N"
                    f"{v_z * 1000:>9.3f}mm/s{e_z * 1000:>11.3f}mm/s   {st['reason']}")
            if line != last:
                last = line
                print(line, flush=True)
    except KeyboardInterrupt:
        print()
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
