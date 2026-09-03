#!/usr/bin/env python3
"""정속 힘 유지 검증 — 실행 하나를 기록한다.

    source ~/FR5-for-RUS/install/setup.bash
    python3 capture_force_hold.py --label A_t1p0 --run-type hold --seconds 60
    python3 capture_force_hold.py --label B_t1p0 --run-type disturbance --step-ml 150

제어 스택이 쓰는 렌치를 **그대로** 기록한다. 여기서 다시 보상하거나 걸러내지
않는다 — 두 벌이 있으면 언젠가 갈라지고, 그때 어느 쪽이 로봇이 본 값인지 알 수
없다.

키:  i 주사기 주입 · w 주사기 회수 · s 안정됨 · space 표시 · u 취소 · q 저장 후 종료
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import rclpy
from rclpy.node import Node

from fh.capture import KEYS, ForceHoldCapture, KeyReader


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--label", required=True, help="실행 이름. 파일 이름이 된다")
    parser.add_argument("--run-type", required=True,
                        choices=("hold", "disturbance", "safety", "stiffness"))
    parser.add_argument("--namespace", default="/fr5_right")
    parser.add_argument("--ik-node", default="/us_diff_ik_node",
                        help="지령 파라미터를 읽을 노드")
    parser.add_argument("--normal-force-sign", type=float, default=-1.0)
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="0 이면 q 를 누를 때까지")
    parser.add_argument("--step-ml", type=float, default=None,
                        help="주사기 한 번의 부피 [mL]. 분석이 그대로 기록한다")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--overwrite", action="store_true",
                        help="같은 이름의 캡처가 있어도 덮어쓴다")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    # 같은 이름이 이미 있으면 **찍기 전에** 멈춘다.
    #
    # save() 는 "w" 로 연다. 그래서 회차 번호를 재활용해 유지 구간만 다시 돌면
    # 이전 회차의 같은 대역이 조용히 사라진다 — 30 초를 다시 받는 대신 30 초를
    # 잃는다. 2026-09-04 에 4.0 N 만 깨끗한 유지였는데, 그 하나를 덮어쓰면
    # 그날 쓸 수 있는 유일한 유지 구간이 없어진다.
    #
    # 로봇을 이미 접촉시킨 뒤에 알게 되면 늦으므로 인자 검사 단계에서 본다.
    existing = os.path.join(args.out_dir, f"{args.label}_samples.csv")
    if os.path.exists(existing) and not args.overwrite:
        print(f"✗ {existing} 이 이미 있다. 덮어쓰지 않는다.", file=sys.stderr)
        print("  다른 회차 이름을 쓰거나(예: run_repeat.sh 1h), "
              "정말 버릴 것이면 --overwrite 를 준다.", file=sys.stderr)
        return 2

    rclpy.init()
    node = Node("force_hold_capture")
    capture = ForceHoldCapture(node, args.namespace, args.normal_force_sign,
                               args.label, args.run_type, args.out_dir, args.step_ml)

    commanded = capture.read_commanded(node, args.ik_node)
    if not commanded.get("parameters_read"):
        # 멈추지는 않는다 — 기록은 여전히 가치가 있다. 다만 무엇을 지시받았는지
        # 모르는 채로 남았다는 것이 파일에 적힌다.
        print(f"⚠️ 지령 파라미터를 못 읽었다: {commanded.get('reason')}")
        print("   그대로 기록하지만, 분석은 목표 힘이 없으면 이 실행을 제외한다.")
    else:
        print(f"지령: 목표 {commanded.get('target_force_n')} ± "
              f"{commanded.get('deadband_n')} N · 진입 "
              f"{commanded.get('contact_probing_force_n')} · 한계 "
              f"{commanded.get('max_contact_force_n')} N "
              f"({commanded.get('contact_force_mode')})")

    print(f"\n[{args.label}] {args.run_type} — "
          + ("q 로 종료" if args.seconds <= 0 else f"{args.seconds:.0f} s 또는 q"))
    print("  " + " · ".join(f"{k if k != ' ' else 'space'}={v}" for k, v in KEYS.items())
          + " · u=취소 · q=저장 후 종료\n")

    started = time.time()
    marks = 0
    try:
        with KeyReader() as keys:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.02)
                key = keys.get()
                if key == "q":
                    break
                if key == "u":
                    print("  취소" if capture.undo() else "  취소할 표시가 없다")
                elif key in KEYS:
                    marks += 1
                    print(f"  [{time.time() - started:6.1f} s] {capture.mark(key)}")
                if args.seconds > 0 and time.time() - started >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass

    samples, meta = capture.save(commanded)
    print(f"\n표본 {len(capture.rows)}개 · 표시 {marks}개")
    print(f"  {samples}\n  {meta}")
    if not capture.rows:
        print("⚠️ 표본이 없다 — wrench_px6d 가 오지 않았다. 브리지가 떠 있는가")

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
