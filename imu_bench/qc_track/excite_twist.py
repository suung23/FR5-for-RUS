#!/usr/bin/env python3
"""(선택) 스크립트 자극 — desired_twist 에 사인을 실어 로봇이 스스로 흔들게 한다.

    python3 excite_twist.py --axis tilt --dry-run
    python3 excite_twist.py --axis roll

**왜 필요한가.** teleop 만으로는 상호상관 실효 지연 하나밖에 못 낸다. 그 값은
그 운동의 스펙트럼에 딸린 값이라 조건이 바뀌면 바뀐다. 순수지연 L 과 1 차 지체
T 를 가르려면 **구간마다 한 주파수**로 흔들어 위상·이득 곡선을 얻어야 하고,
그건 손으로는 못 한다 (Exp-Latency §3.1).

IMU 는 T ~ 0 이어야 정상이다. T 가 크게 나오면 어딘가에서 저역통과가 걸린
것이고, 그건 펌웨어나 퓨전 설정에서 찾을 일이지 센서 탓이 아니다.

**안전.**
  · freespace 로 띄운 상태에서, 프로브가 아무것도 안 닿는 자세에서만 돌린다.
  · 진폭은 protocol 의 예산(첨두 각속도 0.30 rad/s, 속도 0.030 m/s)으로 묶는다.
    probe.yaml 의 freespace 상한(1.5 rad/s, 0.15 m/s)의 1/5 다.
  · E-stop 을 손에 두고, twist 발행자가 이것 하나뿐인지 확인한다 — teleop 이
    같이 돌면 두 발행자가 번갈아 나가 팔이 덜컹거린다.
  · Ctrl-C 하면 0 twist 를 몇 번 보내고 끝낸다 (워치독이 잡기 전에 세운다).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol                                                 # noqa: E402
import qc_common as qc                                          # noqa: E402

# 축 -> twist 성분. protocol.FREQS 의 축 정의와 짝을 이룬다.
#   tilt  : base +y 둘레 각속도  -> 침투축이 base x 쪽으로 기운다 (분석기 tip_x)
#   roll  : base +z 둘레 각속도  -> 침투축(거의 연직) 둘레 회전   (분석기 spin)
#   trans : base +x 병진
AXIS_SLOT = {"tilt": ("angular", 1), "roll": ("angular", 2), "trans": ("linear", 0)}


def make_exciter(robot, axis, rate_hz):
    """rclpy 를 **여기서** import 한다 — --dry-run 은 ROS 없는 기계에서도 돌아야
    계획을 먼저 볼 수 있다. 모듈 최상단에서 받으면 그게 안 된다."""
    from geometry_msgs.msg import Twist
    from rclpy.node import Node

    class Exciter(Node):
        def __init__(self):
            super().__init__(f"qc_excite_{axis}")
            self.pub = self.create_publisher(Twist, qc.topics(robot)["twist"], 10)
            self.axis = axis
            self.rate = rate_hz
            self.segments = []

        def publish(self, value):
            m = Twist()
            field, idx = AXIS_SLOT[self.axis]
            setattr(getattr(m, field), "xyz"[idx], float(value))
            self.pub.publish(m)

        def stop(self, n=5):
            for _ in range(n):
                self.publish(0.0)
                time.sleep(1.0 / self.rate)

        def run(self):
            """구간마다: 정지(LEAD) -> 사인 -> 정지(LEAD) -> 정착.

            사인의 앞뒤를 raised-cosine 으로 연다. 갑자기 진폭을 세우면 그 계단이
            **모든 주파수를 때려** 록인 잔차가 커진다. 러너와 시뮬레이터가 같은 창을
            쓰도록 protocol.raised_cosine 하나만 쓴다.
            """
            dt = 1.0 / self.rate
            for seg in protocol.plan_sweep(self.axis):
                f = seg["freq_hz"]
                amp_rate = seg["peak_rate"]          # rad/s 또는 m/s 첨두
                lead = protocol.LEAD_S
                dur = seg["seconds"] + 2 * lead
                t0 = time.time()
                self.get_logger().info(
                    f"{f:.2f} Hz  진폭 {seg.get('amp_deg', seg.get('amp_mm')):.1f}"
                    f" {seg['unit']}  첨두 {amp_rate:.3f}  ({seg['seconds']:.0f} s, cap={seg['cap']})")
                k = 0
                while True:
                    t = time.time() - t0
                    if t >= dur:
                        break
                    env = (protocol.raised_cosine(min(t / lead, 1.0))
                           * protocol.raised_cosine(min((dur - t) / lead, 1.0)))
                    # 위치가 A sin(wt) 이면 속도는 A w cos(wt) 다. 자극은 twist(속도)이므로
                    # cos 을 낸다 — 그래야 변위가 사인이 되고 위상 기준이 분석기와 맞는다.
                    self.publish(amp_rate * env * np.cos(2 * np.pi * f * (t - lead)))
                    k += 1
                    time.sleep(max(0.0, (k + 1) * dt - (time.time() - t0)))
                self.segments.append({"freq_hz": f, "t_start": t0 + lead,
                                      "t_stop": t0 + dur - lead, "cap": seg["cap"],
                                      "amp": seg.get("amp_deg", seg.get("amp_mm")),
                                      "unit": seg["unit"]})
                self.stop(2)
                time.sleep(protocol.SETTLE_S)



    return Exciter()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True, choices=sorted(AXIS_SLOT))
    ap.add_argument("--robot", default=qc.ROBOT)
    ap.add_argument("--rate", type=float, default=100.0, help="발행 주기 [Hz]")
    ap.add_argument("--raw", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        total = protocol.sweep_seconds(args.axis)
        print(f"축 {args.axis}   총 {total:.0f} s")
        for seg in protocol.plan_sweep(args.axis):
            print(f"  {seg['freq_hz']:>5.2f} Hz  "
                  f"진폭 {seg.get('amp_deg', seg.get('amp_mm')):>6.2f} {seg['unit']}"
                  f"  {seg['seconds']:>5.0f} s  첨두 {seg['peak_rate']:.3f}"
                  f"  (묶은 것: {seg['cap']})")
        return 0

    if args.raw:
        qc.use_raw_dir(args.raw)
    import rclpy
    rclpy.init()
    qc.install_shutdown_handlers()
    node = make_exciter(args.robot, args.axis, args.rate)
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().warn("중단 — twist 0 을 보내고 끝낸다")
    finally:
        node.stop()
        # 구간 시각을 program.json 에 남긴다. 분석기의 록인이 이걸 읽는다.
        prog = qc.load_json(qc.PROGRAM_JSON, {}) or {}
        blocks = [b for b in prog.get("blocks", []) if b.get("block") != f"sweep_{args.axis}"]
        blocks.append({"block": f"sweep_{args.axis}", "segments": node.segments})
        prog["blocks"] = blocks
        qc.dump_json(qc.PROGRAM_JSON, prog)
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
