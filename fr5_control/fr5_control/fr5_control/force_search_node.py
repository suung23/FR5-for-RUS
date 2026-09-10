"""힘 설정값을 영상 품질의 경사 방향으로 옮기는 감독 노드 (DESIGN_NOTES §8.4).

    ros2 run fr5_control force_search_node                      # 관찰만 (기본)
    ros2 run fr5_control force_search_node --ros-args -p execute:=true

``target_force_n`` 을 고정 상수로 두는 대신 **Q 가 가장 높은 힘을 찾아 따라간다.**
설정값에 구형파 디더를 걸고, 품질 응답을 같은 위상으로 복조해 경사를 얻고, 그 방향으로
작은 걸음을 옮긴다 (극값 탐색). 판정 자체는 ``ForceSetpointAdapter`` 가 하고 — rclpy 없이
돌며 시험 14 개가 붙어 있다 — 이 노드는 **배선만** 한다: 품질을 받아 넣고, 나온 설정값을
``us_diff_ik_node`` 의 파라미터로 민다.

품질 신호
--------
``{ns}/image_quality_raw`` (Float32) 를 받는다. **Q_seg 가 아니다** — 힘 축은 Q_raw 이고
(DESIGN_NOTES §333: "Q_raw 최대화하는 최소 F_n*"), Q_seg 는 분할 기반이라 분할이 아무것도
못 내놓는 구간에서 힘을 고를 수 없다. 지금 이 토픽을 내는 것은 policy_learning 의
``run_policy.py`` 뿐이다 — 실시간 지각이 거기서만 돌기 때문이다. 값은 **Q_raw** 다.
NaN 은 "측정 안 됨" 으로 그대로 넘긴다 (게이트가 닫힌 프레임). adapter 가 세기만 하고
쓰지 않는다.

안전
----
* 설정값은 ``[f_min + a, f_max − a]`` 에 묶여 디더를 얹어도 ``[f_min, f_max]`` 를 안 넘는다.
  ``f_max`` 는 하드 한계 5.0 이 아니라 **warn 선 4.5** 를 쓴다 (§8.4 주석).
* 파라미터 set 은 ``us_diff_ik_node`` 가 ``target < warn ≤ max`` 불변식으로 검사해 거절한다.
  즉 5 N 안전 계층이 이 탐색을 구조적으로 감시한다 — 여기서 문턱을 다시 만들지 않는다.
* ``/diag/retreating`` 이 뜨면 미는 것을 멈춘다. 후퇴 중에 설정값을 옮기면 후퇴가 끝난 자리가
  달라진다.
* ``execute`` 가 거짓이면 **아무것도 밀지 않고** 무엇을 밀었을지만 로그로 남긴다. 처음 붙일 때
  디더 주기와 복조가 맞는지 이걸로 본다.
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from std_msgs.msg import Bool, Float32

from fr5_control.force_setpoint_adapter import ForceSetpointAdapter, SetpointPusher


class ForceSearchNode(Node):
    def __init__(self) -> None:
        super().__init__("force_search")
        # 제어 스택 토픽은 /{robot.name} 아래에 있다 (us_diff_ik_node · telemetry_bridge 와 동일).
        # run_policy 가 {robot_ns}/image_quality 로 Q_raw 를 내므로 여기와 같아야 한다.
        self.declare_parameter("namespace", "/fr5_right")
        self.declare_parameter("control_node", "/us_diff_ik_node")
        self.declare_parameter("quality_topic", "")
        self.declare_parameter("execute", False)
        self.declare_parameter("f_bar0_n", 3.0)
        self.declare_parameter("f_min_n", 1.0)
        # 하드 한계 5.0 이 아니라 warn 선. 적응기가 상승 금지선을 넘기는 것이 맞다 (§8.4).
        self.declare_parameter("f_max_n", 4.5)
        self.declare_parameter("dither_amplitude_n", 0.25)
        self.declare_parameter("dither_period_s", 5.0)
        self.declare_parameter("settle_s", 0.5)
        self.declare_parameter("step_max_n", 0.1)

        g = lambda n: self.get_parameter(n).value
        ns = str(g("namespace"))
        self.control_node = str(g("control_node"))
        self.execute = bool(g("execute"))
        self.adapter = ForceSetpointAdapter(
            f_bar0=float(g("f_bar0_n")),          # 탐색의 **시작값**. 고정 목표가 아니다.
            amplitude_n=float(g("dither_amplitude_n")),
            period_s=float(g("dither_period_s")),
            settle_s=float(g("settle_s")),
            step_max_n=float(g("step_max_n")),
            f_min_n=float(g("f_min_n")),
            f_max_n=float(g("f_max_n")),
        )
        self.adapter.reset(float(g("f_bar0_n")), self._now())
        self.pusher = SetpointPusher()
        self.retreating = False
        self._param_client: Optional[rclpy.client.Client] = None
        self._n_q = 0

        topic = str(g("quality_topic")) or f"{ns}/image_quality_raw"
        self.create_subscription(Float32, topic, self._on_quality, 10)
        self.create_subscription(Bool, "/diag/retreating", self._on_retreat, 10)
        self.f_bar_pub = self.create_publisher(Float32, f"{ns}/force_setpoint_bar", 10)
        self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f"{'실행' if self.execute else '관찰만 (execute:=true 로 켠다)'} · "
            f"품질 {topic} · 대상 {self.control_node}/contact_control.target_force_n\n  "
            f"{self.adapter.describe()}")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_retreat(self, msg: Bool) -> None:
        if msg.data and not self.retreating:
            self.get_logger().warn("후퇴 신호 — 설정값 이동을 멈춘다")
        self.retreating = bool(msg.data)

    def _on_quality(self, msg: Float32) -> None:
        q = float(msg.data)
        self._n_q += 1
        closed = self.adapter.update(self._now(), None if not math.isfinite(q) else q)
        if closed is not None:
            self.get_logger().info(
                f"주기 닫힘 — f_bar {self.adapter.f_bar:.3f} N  "
                + "  ".join(f"{k}={v}" for k, v in closed.items()))

    def _tick(self) -> None:
        t = self._now()
        self.f_bar_pub.publish(Float32(data=float(self.adapter.f_bar)))
        if self.retreating:
            return
        sp = self.adapter.setpoint(t)
        if not self.pusher.should_push(sp):
            return
        if not self.execute:
            self.pusher.mark(sp)
            self.get_logger().info(f"[관찰] target_force_n ← {sp:.3f} N (밀지 않음)")
            return
        if self._set_remote_parameter("contact_control.target_force_n", float(sp)):
            self.pusher.mark(sp)
        else:
            self.get_logger().warn(
                f"{self.control_node} 의 파라미터 서비스가 없다 — 제어 스택이 떠 있는가")

    def _set_remote_parameter(self, name: str, value: float) -> bool:
        """비동기로 민다. 응답을 기다리면 spin 을 잡는다 — 결과는 노드 로그가 말한다.

        거절될 수 있다는 것이 중요하다. ``us_diff_ik_node`` 가 ``target < warn ≤ max``
        를 검사하므로 안전 범위를 벗어나는 값은 저쪽에서 막힌다.
        """
        if self._param_client is None:
            self._param_client = self.create_client(
                SetParameters, f"{self.control_node}/set_parameters")
        if not self._param_client.service_is_ready():
            if not self._param_client.wait_for_service(timeout_sec=0.2):
                return False
        req = SetParameters.Request()
        req.parameters = [Parameter(
            name=name,
            value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)))]
        self._param_client.call_async(req)
        return True


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ForceSearchNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
