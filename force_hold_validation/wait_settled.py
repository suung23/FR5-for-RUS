#!/usr/bin/env python3
"""힘이 목표에 앉을 때까지 기다린다. 조작자의 Enter 를 대신한다.

    python3 wait_settled.py --target 1.0            # 앉으면 0, 시간 초과면 1

`param set` 뒤에 고정 시간을 자는 것과 다르다. 팬텀이 무르면 수렴이 느리고
단단하면 빠른데, 고정 대기는 그 차이를 모른다 — 여기서는 **실제로 밴드 안에
연속으로 머무는 것**을 보고 넘어간다. 접촉이 끊어졌으면 기다리지 않고 실패한다.
"""
from __future__ import annotations

import argparse
import sys
import time

import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
PROBING = ("contact_probing", "contact_probing_inplane")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--tolerance", type=float, default=0.08,
                    help="이 안에 들어오면 앉은 것으로 본다 [N]")
    ap.add_argument("--hold-s", type=float, default=2.0,
                    help="연속으로 머물러야 하는 시간 [s]")
    ap.add_argument("--timeout-s", type=float, default=45.0)
    ap.add_argument("--await-contact", action="store_true",
                    help="접촉 전이면 실패하지 않고 조작자가 누르기를 기다린다")
    ap.add_argument("--contact-timeout-s", type=float, default=120.0,
                    help="접촉을 기다리는 최대 시간 [s]")
    ap.add_argument("--namespace", default="/fr5_right")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    rclpy.init()
    node = Node("wait_settled")
    state = {"f": None, "mode": "", "inside_since": None}

    def on_wrench(msg: WrenchStamped) -> None:
        w = msg.wrench.force
        state["f"] = (w.x ** 2 + w.y ** 2 + w.z ** 2) ** 0.5

    node.create_subscription(WrenchStamped, f"{args.namespace}/wrench_px6d",
                             on_wrench, 50)
    node.create_subscription(String, f"{args.namespace}/probing_mode",
                             lambda m: state.update(mode=m.data), LATCHED)

    started = time.time()
    contact_started = started
    last_print = 0.0
    rc = 1
    try:
        while rclpy.ok() and time.time() - started < args.timeout_s:
            rclpy.spin_once(node, timeout_sec=0.05)
            f = state["f"]
            if f is None:
                continue
            if state["mode"] and state["mode"] not in PROBING:
                if not args.await_contact:
                    print(f"\n  ✗ 접촉이 끊겼다 (모드 {state['mode']}). 다시 접촉시켜라.")
                    rc = 2
                    break
                # 접촉을 기다리는 동안은 정착 시계를 멈춰 둔다. 조작자가 프로브를
                # 얹는 시간이 정착 판정에 섞이면, 아직 눌리지도 않은 힘을 "앉았다"
                # 로 읽을 수 있다.
                started = time.time()
                state["inside_since"] = None
                if time.time() - contact_started > args.contact_timeout_s:
                    print(f"\n  ✗ {args.contact_timeout_s:.0f} s 안에 접촉이 안 잡혔다.")
                    rc = 2
                    break
                if time.time() - last_print > 0.5:
                    last_print = time.time()
                    print(f"\r  … 접촉을 기다린다 — 프로브를 팬텀에 눌러라 "
                          f"(지금 {f:.3f} N, 문턱 이상이면 잡힌다)      ",
                          end="", flush=True)
                continue
            now = time.time()
            if abs(f - args.target) <= args.tolerance:
                if state["inside_since"] is None:
                    state["inside_since"] = now
                if now - state["inside_since"] >= args.hold_s:
                    print(f"\r  ✓ {f:.3f} N 에 앉았다 (목표 {args.target})        ")
                    rc = 0
                    break
            else:
                state["inside_since"] = None
            if now - last_print > 0.5:
                last_print = now
                print(f"\r  … {f:.3f} N → {args.target} N 로 이동 중 "
                      f"({now - started:.0f}/{args.timeout_s:.0f} s)", end="", flush=True)
        else:
            print(f"\n  ✗ {args.timeout_s:.0f} s 안에 앉지 않았다.")
    except KeyboardInterrupt:
        rc = 130
    node.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
