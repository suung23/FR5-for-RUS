#!/usr/bin/env python3
"""teleop 축 대응이 조작 중에 얼마나 틀어지는지 잰다 (2026-08-21).

빌드가 필요 없다. ROS 를 source 한 셸에서 그냥 돌린다::

    source /opt/ros/jazzy/setup.bash && source install/setup.bash
    python3 fr5_contorl/touch_teleop/tools/teleop_frame_audit.py

무엇을 재는가
-------------
teleop 경로는 **body → body** 다.

    v_base = R_base_probe · A · R_stylus^T · v_world

``touch_twist_node`` 가 손 속도를 **스타일러스 body 프레임**으로 투영하고
(``vel_body = R^T · vel_world``), ``us_diff_ik_node`` 는 그 twist 를 **프로브 body
프레임**으로 해석한다 (``dls_solver.solve``). 그래서 "손을 오른쪽으로 밀면 로봇이
어디로 가는가" 는 두 자세의 **상대 관계**가 정한다.

이 관계는 rate control 에서 저절로 유지되지 않는다. 스타일러스 자세는 장치가 주는
절대값이지만, 프로브 자세는 **지령 각속도의 적분**이고 그 각속도는 도중에

    angular_scale(1.2 배) · 데드존(0.05 rad/s 이하 버림) · 클램프 · 필터 · DLS 감쇠

를 통과한다. 어느 하나도 1:1 이 아니므로 두 자세는 반드시 벌어지고, **벌어진 각도는
손을 돌린 만큼 누적된다.** 그것이 "초반에는 잘 되다가 시간이 지나면 어색해진다" 다.

그래서 이 스크립트는 딱 하나를 본다: **"스타일러스 +X 를 밀었을 때 로봇 base 에서
가리키는 방향"이 시작할 때에 비해 몇 도 돌아갔는가.**

읽는 법
-------
* 표류각이 시간에 따라 단조 증가 → body 프레임 누적 어긋남이 원인이다.
  데드맨을 놓았다 다시 잡아도 안 줄면 확정이다 (현재 코드에는 재정렬이 없다).
* 표류각은 작은데 조작이 어색 → 축 대응 자체가 틀린 것이다. 맨 위 "시작 매핑" 을
  보라. ``tip_roll_deg`` 는 회전축뿐 아니라 **병진축에도 같이 걸린다.**
* 스타일러스 X 이동범위가 장치 한계에 붙어 있다 → 손이 워크스페이스 끝에 가 있다.
  이 경우 X 만 안 먹는 것처럼 느껴진다. 데드맨 놓고 손을 가운데로 옮겨 다시 잡으면
  즉시 낫는다.

``/touch/<side>/stylus_pose`` 는 **데드맨을 누르고 있는 동안에만** 발행된다.
버튼을 잡은 채로 평소처럼 조작하면 된다.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation

PROBE_AXES = ("+x (lateral)", "+y (elevational)", "+z (침투)")


def mapping_matrix(tip_roll_deg: float) -> np.ndarray:
    """``touch_twist_node`` 의 A = Rz(tip_roll) · diag(+1, -1, -1) 을 그대로 재현한다."""
    angle = math.radians(tip_roll_deg)
    rot_z = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rot_z @ np.diag([1.0, -1.0, -1.0])


def describe(direction: np.ndarray) -> str:
    """프로브 프레임 단위벡터를 사람이 읽을 수 있게."""
    index = int(np.argmax(np.abs(direction)))
    sign = "+" if direction[index] > 0 else "-"
    share = abs(direction[index]) / max(np.linalg.norm(direction), 1e-9)
    return f"{sign}{PROBE_AXES[index]}  (그 축 성분 {share * 100:.0f}%)"


class FrameAudit(Node):
    """스타일러스 자세와 프로브 자세를 함께 받아 축 대응의 표류를 잰다."""

    def __init__(
        self, side: str, tip_roll_deg: float, period: float, engage_gap_s: float
    ) -> None:
        super().__init__("teleop_frame_audit")
        self.mapping = mapping_matrix(tip_roll_deg)
        self.tip_roll_deg = tip_roll_deg

        self.rot_stylus = None
        self.rot_probe = None
        self.stylus_xyz = []
        self.reference_dir = None
        self.samples = 0
        self.worst_drift = 0.0

        # us_diff_ik_node 의 latched 모드를 여기서도 그대로 재현한다. 노드 코드를
        # import 하지 않는 이유는, 같은 모듈을 쓰면 "코드가 자기 자신과 일치한다" 를
        # 확인할 뿐이기 때문이다. 여기서는 **결과**를 독립적으로 다시 계산한다.
        self.latched = None
        self.latched_ref = None
        self.worst_latched = 0.0
        self.engage_count = 0
        self.last_stylus_ns = None
        self.engage_gap_s = engage_gap_s

        self.create_subscription(
            PoseStamped, f"/touch/{side}/stylus_pose", self._on_stylus, 10
        )
        self.create_subscription(Pose, f"/fr5_{side}/ee_wrt_base", self._on_probe, 10)
        self.create_timer(period, self._report)

        self.get_logger().info(
            f"감시 시작 — /touch/{side}/stylus_pose · /fr5_{side}/ee_wrt_base · "
            f"tip_roll_deg={tip_roll_deg:.0f}"
        )
        self.get_logger().info("데드맨(버튼 1)을 잡고 평소처럼 조작하라. Ctrl+C 로 종료.")

    # -- 입력 ------------------------------------------------------------

    @staticmethod
    def _rotation(orientation) -> np.ndarray:
        return Rotation.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_matrix()

    def _on_stylus(self, msg: PoseStamped) -> None:
        now_ns = self.get_clock().now().nanoseconds
        self.rot_stylus = self._rotation(msg.pose.orientation)
        position = msg.pose.position
        self.stylus_xyz.append([position.x, position.y, position.z])

        # stylus_pose 는 데드맨을 잡고 있는 동안에만 나온다 — 공백 뒤 첫 메시지가
        # 곧 재파지다. 노드와 같은 판정이다.
        gap = None if self.last_stylus_ns is None else (now_ns - self.last_stylus_ns) / 1e9
        self.last_stylus_ns = now_ns
        if (gap is None or gap > self.engage_gap_s) and self.rot_probe is not None:
            self.engage_count += 1
            if self.latched is None:
                self.latched = self.rot_probe @ self.mapping @ self.rot_stylus.T
                self.latched_ref = self.latched @ np.array([1.0, 0.0, 0.0])
                self.get_logger().info(
                    f"병진 기준 고정 (파지 {self.engage_count}회차) — "
                    f"이후 latched 표류는 0 이어야 한다"
                )
            else:
                self.get_logger().info(f"재파지 {self.engage_count}회차")

        self._accumulate()

    def _on_probe(self, msg: Pose) -> None:
        self.rot_probe = self._rotation(msg.orientation)

    # -- 측정 ------------------------------------------------------------

    def _push_x_in_base(self) -> np.ndarray | None:
        """스타일러스 world +X 밀기가 로봇 base 에서 가리키는 방향."""
        if self.rot_stylus is None or self.rot_probe is None:
            return None
        # v_base = R_base_probe · A · R_stylus^T · e_x
        return self.rot_probe @ self.mapping @ self.rot_stylus.T @ np.array([1.0, 0.0, 0.0])

    def _push_x_latched(self) -> np.ndarray | None:
        """latched 모드에서 스타일러스 +X 밀기가 base 에서 가리키는 방향.

        ``v_base = R_latch · R_stylus · Aᵀ · v_cmd`` 이고 ``v_cmd = A · R_stylusᵀ · e_x``
        이므로 결과는 ``R_latch · e_x`` — **시간 의존 항이 없다.** 그래도 식을 접지 않고
        그대로 계산한다. 접어 버리면 "0 이 나오도록 짠 계산" 이 되어 아무것도 검증하지
        못한다.
        """
        if self.latched is None or self.rot_stylus is None:
            return None
        command = self.mapping @ self.rot_stylus.T @ np.array([1.0, 0.0, 0.0])
        return self.latched @ self.rot_stylus @ self.mapping.T @ command

    def _accumulate(self) -> None:
        direction = self._push_x_in_base()
        if direction is None:
            return
        self.samples += 1
        if self.reference_dir is None:
            self.reference_dir = direction
            self._report_start()
            return
        cosine = float(np.clip(np.dot(direction, self.reference_dir), -1.0, 1.0))
        self.worst_drift = max(self.worst_drift, math.degrees(math.acos(cosine)))

        latched = self._push_x_latched()
        if latched is not None:
            cosine = float(np.clip(np.dot(latched, self.latched_ref), -1.0, 1.0))
            self.worst_latched = max(self.worst_latched, math.degrees(math.acos(cosine)))

    def _report_start(self) -> None:
        """시작 시점의 정적 매핑. 여기가 틀렸으면 표류와 무관하게 처음부터 어색하다."""
        columns = self.mapping @ np.eye(3)
        self.get_logger().info("── 시작 매핑 (A = Rz(tip_roll)·diag(+1,-1,-1)) ──")
        for axis, name in zip(range(3), ("스타일러스 +X (오른쪽)",
                                         "스타일러스 +Y (위)",
                                         "스타일러스 +Z (조작자 쪽)")):
            self.get_logger().info(f"   {name:24s} → 프로브 {describe(columns[:, axis])}")
        if abs(self.tip_roll_deg % 180.0 - 90.0) < 1e-6:
            self.get_logger().warn(
                "   tip_roll_deg 가 90 이라 병진 x·y 도 함께 맞바뀌어 있다. "
                "이 값은 2026-08-19 에 **회전** 기준으로만 확정됐다 — "
                "병진이 어색하면 0 으로 두고 다시 밀어 보라."
            )

    def _report(self) -> None:
        if self.reference_dir is None:
            self.get_logger().warn(
                "아직 두 자세가 다 안 들어왔다 — 데드맨을 잡고 있는지, "
                "us_servo_node 가 떠 있는지 확인하라.",
                throttle_duration_sec=5.0,
            )
            return

        direction = self._push_x_in_base()
        cosine = float(np.clip(np.dot(direction, self.reference_dir), -1.0, 1.0))
        drift = math.degrees(math.acos(cosine))

        span = ""
        if len(self.stylus_xyz) > 1:
            array = np.array(self.stylus_xyz)
            ranges = (array.max(axis=0) - array.min(axis=0)) * 1000.0
            span = (
                f" · 스타일러스 이동범위 "
                f"X {ranges[0]:.0f} / Y {ranges[1]:.0f} / Z {ranges[2]:.0f} mm"
            )

        # probe(옛) 표류는 **고쳐도 계속 커진다.** 그것은 rate control 에서 두 자세가
        # 벌어지는 물리 현상 자체이고, latched 매핑은 그것을 없애는 게 아니라 병진이
        # 거기 끌려가지 않게 만드는 것이다. 그래서 판정은 latched 쪽으로 한다.
        verdict = ""
        if self.latched is not None and self.worst_latched > 0.1:
            verdict = "  ← latched 인데 표류한다. 버그다"
        elif drift > 25.0:
            verdict = "  ← 두 자세는 벌어졌다 (latched 가 0 이면 병진은 무사하다)"

        latched_text = " · latched 미고정 (데드맨을 한 번 잡아라)"
        if self.latched is not None:
            latched = self._push_x_latched()
            cosine = float(np.clip(np.dot(latched, self.latched_ref), -1.0, 1.0))
            latched_text = (
                f" · latched {math.degrees(math.acos(cosine)):.3f}° "
                f"(최대 {self.worst_latched:.3f}°)"
            )

        self.get_logger().info(
            f"probe(옛) {drift:5.1f}° (최대 {self.worst_drift:5.1f}°){latched_text}"
            f" · 파지 {self.engage_count} · 표본 {self.samples}{span}{verdict}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", default="right", help="right | left (기본 right)")
    parser.add_argument(
        "--tip-roll-deg",
        type=float,
        default=90.0,
        help="touch_twist_node 의 teleop.tip_roll_deg 와 같은 값을 줘야 한다",
    )
    parser.add_argument("--period", type=float, default=1.0, help="보고 주기 [s]")
    parser.add_argument(
        "--engage-gap-s",
        type=float,
        default=0.3,
        help="stylus_pose 가 이보다 길게 끊기면 재파지로 본다 (노드 파라미터와 같은 값)",
    )
    args = parser.parse_args()

    rclpy.init()
    node = FrameAudit(args.side, args.tip_roll_deg, args.period, args.engage_gap_s)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        # Ctrl+C 든 SIGTERM 이든 같은 요약을 찍고 조용히 나간다.
        print(
            f"\n두 자세 벌어짐(옛 경로) 최대 {node.worst_drift:.1f}°"
            f" · latched 병진 표류 최대 {node.worst_latched:.3f}°"
            f" · 파지 {node.engage_count} · 표본 {node.samples}"
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
