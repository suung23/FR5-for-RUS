"""Stage 1 힘 탐색 노드 (DESIGN_NOTES §7).

    /supervisor/state        String            STAGE1A_CONTACT / STAGE1B_FOCUS 에서만 돈다
    /fr5_right/wrench        WrenchStamped     정착 판정용
    /us/quality_raw          Float32           Stage 1a 목적함수
    /us/quality_seg          Float32           Stage 1b 목적함수
    /us/valid_for_control    Bool
              │
              ▼
    /control/contact_setpoint  WrenchStamped   force.z = F_n* (모멘트는 0)
    /diag/force_search         Float32MultiArray [phase, target, F*, best_Q]
    /diag/force_curve          Float32MultiArray 힘–품질 곡선 (완료 시 1회)

탐색 알고리즘은 :mod:`fr5_control.force_search` 에 있고 이 노드는 배선만 한다.

**램프는 여기서 하지 않는다.** admittance 의 슬루 제한이 `force_slew_n_s` 로 수행하므로
여기서는 목표 레벨만 내고 정착을 기다린다.
"""
from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from fr5_control.force_search import ForceSearch, ForceSearchConfig, SearchPhase

#: 진단 배열 0번에 싣는 phase 코드. supervisor 가 이 값으로 완료/실패를 읽는다.
PHASE_CODE = {
    SearchPhase.IDLE: 0.0,
    SearchPhase.SETTLE: 1.0,
    SearchPhase.MEASURE: 2.0,
    SearchPhase.DONE: 3.0,
}
PHASE_FAILED = 4.0

STAGE_1A = "STAGE1A_CONTACT"
STAGE_1B = "STAGE1B_FOCUS"


class UsForceSearchNode(Node):
    """Stage 1a 전체 격자와 Stage 1b 국소 재탐색을 수행한다."""

    def __init__(self) -> None:
        super().__init__("us_force_search_node")

        self._declare_parameters()
        self.robot_name = self.get_parameter("robot.name").value
        self.rate_hz = float(self.get_parameter("search.rate_hz").value)
        self.normal_force_sign = float(self.get_parameter("ft_sensor.normal_force_sign").value)

        self.search = ForceSearch(
            ForceSearchConfig(
                force_min=float(self.get_parameter("setpoint.f_normal_min_n").value),
                force_max=float(self.get_parameter("setpoint.f_normal_max_n").value),
                force_step=float(self.get_parameter("search.force_step_n").value),
                refine_span=float(self.get_parameter("search.refine_span_n").value),
                refine_step=float(self.get_parameter("search.refine_step_n").value),
                settle_tolerance_n=float(self.get_parameter("search.settle_tolerance_n").value),
                settle_hold_s=float(self.get_parameter("search.settle_hold_s").value),
                settle_timeout_s=float(self.get_parameter("search.settle_timeout_s").value),
                measure_window_s=float(self.get_parameter("search.measure_window_s").value),
                min_valid_fraction=float(self.get_parameter("search.min_valid_fraction").value),
                epsilon=float(self.get_parameter("search.epsilon").value),
                early_stop_drop=float(self.get_parameter("search.early_stop_drop").value),
                early_stop_count=int(self.get_parameter("search.early_stop_count").value),
            )
        )

        self.state = ""
        self.normal_force = 0.0
        self.q_raw = 0.0
        self.q_seg = 0.0
        self.valid = False
        self.optimal_force: float | None = None

        ns = f"/{self.robot_name}"
        self.create_subscription(String, "/supervisor/state", self._on_state, 10)
        self.create_subscription(WrenchStamped, f"{ns}/wrench", self._on_wrench, 10)
        self.create_subscription(Float32, "/us/quality_raw", self._on_q_raw, 10)
        self.create_subscription(Float32, "/us/quality_seg", self._on_q_seg, 10)
        self.create_subscription(Bool, "/us/valid_for_control", self._on_valid, 10)

        self.setpoint_pub = self.create_publisher(
            WrenchStamped, "/control/contact_setpoint", 10
        )
        self.diag_pub = self.create_publisher(Float32MultiArray, "/diag/force_search", 10)
        self.curve_pub = self.create_publisher(Float32MultiArray, "/diag/force_curve", 10)

        self.create_timer(1.0 / self.rate_hz, self._loop)
        levels = len(self.search.config.coarse_levels())
        self.get_logger().info(
            f"힘 탐색 대기 · 격자 {levels}점 "
            f"({self.search.config.force_min}–{self.search.config.force_max} N) · "
            f"정착 후 {self.search.config.measure_window_s} s 측정"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("robot.name", "fr5_right")
        self.declare_parameter("search.rate_hz", 50.0)
        self.declare_parameter("search.force_step_n", 0.5)
        self.declare_parameter("search.refine_span_n", 1.0)
        self.declare_parameter("search.refine_step_n", 0.25)
        self.declare_parameter("search.settle_tolerance_n", 0.05)
        self.declare_parameter("search.settle_hold_s", 0.2)
        self.declare_parameter("search.settle_timeout_s", 3.0)
        self.declare_parameter("search.measure_window_s", 1.0)
        self.declare_parameter("search.min_valid_fraction", 0.6)
        self.declare_parameter("search.epsilon", 0.03)
        self.declare_parameter("search.early_stop_drop", 0.15)
        self.declare_parameter("search.early_stop_count", 3)
        self.declare_parameter("setpoint.f_normal_min_n", 1.0)
        self.declare_parameter("setpoint.f_normal_max_n", 7.0)
        self.declare_parameter("ft_sensor.normal_force_sign", -1.0)

    # -- 입력 ------------------------------------------------------------

    def _on_state(self, msg: String) -> None:
        """상태 진입 시 탐색을 시작한다. 1b 는 1a 결과 주변만 훑는다."""
        previous, self.state = self.state, msg.data
        if self.state == previous:
            return

        if self.state == STAGE_1A:
            self.search.start()
            self.get_logger().info("Stage 1a — 전체 격자 탐색 시작")
        elif self.state == STAGE_1B:
            center = self.optimal_force
            if center is None:
                self.get_logger().warn("1a 결과가 없다 — 1b 도 전체 격자로 돈다")
                self.search.start()
            else:
                self.search.start_refine(center)
                self.get_logger().info(f"Stage 1b — {center:.2f} N 주변 재탐색")

    def _on_wrench(self, msg: WrenchStamped) -> None:
        self.normal_force = self.normal_force_sign * msg.wrench.force.z

    def _on_q_raw(self, msg: Float32) -> None:
        self.q_raw = float(msg.data)

    def _on_q_seg(self, msg: Float32) -> None:
        self.q_seg = float(msg.data)

    def _usable(self, quality: float) -> bool:
        """표본을 평균에 넣어도 되는가.

        `valid_for_control` 이 이미 NaN 프레임을 걸러 주지만, 두 토픽이 서로 다른
        시각에 도착하므로 한 틱 어긋날 수 있다. NaN 하나가 레벨 평균을 통째로
        오염시키므로 여기서 한 번 더 막는다.
        """
        return self.valid and not math.isnan(quality)

    def _on_valid(self, msg: Bool) -> None:
        self.valid = bool(msg.data)

    # -- 루프 ------------------------------------------------------------

    def _loop(self) -> None:
        if self.state not in (STAGE_1A, STAGE_1B):
            return  # 다른 상태에서는 setpoint 를 내지 않는다

        was_active = self.search.active
        if was_active:
            # 1a 는 세그멘테이션 비의존 지표, 1b 는 세그 기반 지표 (§6.1)
            quality = self.q_raw if self.state == STAGE_1A else self.q_seg
            target = self.search.step(
                1.0 / self.rate_hz, self.normal_force, quality, self._usable(quality)
            )
            self._publish_setpoint(target)

        result = self.search.result()
        if was_active and not self.search.active and result is not None:
            self._announce(result)

        phase = PHASE_CODE[self.search.phase]
        if result is not None and result.optimal_force is None:
            phase = PHASE_FAILED
        self.diag_pub.publish(
            Float32MultiArray(
                data=[
                    phase,
                    float(self.search.target_force),
                    float(self.optimal_force if self.optimal_force is not None else -1.0),
                    float(result.best_quality) if result and result.best_quality else -1.0,
                ]
            )
        )

    def _announce(self, result) -> None:
        """탐색 종료. 힘–품질 곡선을 반드시 내보낸다 — 학습 데이터다 (§8.4)."""
        if result.optimal_force is None:
            self.get_logger().error(
                f"탐색 실패: {result.failure}. "
                f"측정 실패 사유: "
                f"{[(l.force, l.failure) for l in result.levels if not l.usable]}"
            )
        else:
            self.optimal_force = result.optimal_force
            self.get_logger().info(
                f"F* = {result.optimal_force:.2f} N (최고 품질 {result.best_quality:.3f}"
                f"{', 조기종료' if result.stopped_early else ''}) — "
                f"{len(result.levels)}개 레벨 측정"
            )
            self._publish_setpoint(result.optimal_force)

        # [force, quality, valid_fraction] 삼중항의 평탄 배열. 측정 실패는 quality=-1.
        curve: list[float] = []
        for level in result.levels:
            curve += [
                level.force,
                level.mean_quality if level.mean_quality is not None else -1.0,
                level.valid_fraction,
            ]
        self.curve_pub.publish(Float32MultiArray(data=curve))

    def _publish_setpoint(self, force: float) -> None:
        """Stage 1 은 힘만 다룬다. 모멘트 편향은 Stage 2 의 policy 몫이다 (§5.4)."""
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"{self.robot_name}_probe"
        msg.wrench.force.z = float(force)
        self.setpoint_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = UsForceSearchNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
