"""힘 설정값을 영상 품질의 경사 방향으로 옮기는 감독 노드 (DESIGN_NOTES §8.4).

    ros2 run fr5_control force_search_node

**관찰 전용이다 (2026-10-19).** 설정값에 구형파 디더를 걸고, 품질 응답을 같은 위상으로
복조해 경사를 얻고, 그 방향으로 작은 걸음을 옮긴 값 ``f_bar`` 를 **토픽으로만** 낸다
(``{ns}/force_setpoint_bar``). 판정 자체는 ``ForceSetpointAdapter`` 가 하고 — rclpy 없이
돌며 시험 14 개가 붙어 있다 — 이 노드는 **배선만** 한다: 품질을 받아 넣고, 나온 값을
알린다.

이 노드는 ``us_diff_ik_node`` 의 ``contact_control.target_force_n`` 을 **쓰지 않는다.**
2026-10-19 까지는 ``execute:=true`` 로 그 파라미터를 밀 수 있었다. 그 길은 지웠다 —
유지 힘(최소 접촉력)의 주인은 콘솔의 접촉력 창 하나다 (telemetry_bridge ``force.limits``,
fr5_control/force_limits.py). 두 주인이 같은 파라미터를 쓰면 조작자가 넣은 값이 다음
디더 주기에 조용히 덮인다. run_policy 는 여전히 ``force_setpoint_bar`` 를 "자리 잡았는가"
게이트와 로그에 쓴다 — 그 토픽은 그대로다.

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
* 로봇에 닿는 길이 없다. 이 노드가 내는 것은 Float32 토픽 하나이고, 그것을 파라미터로
  옮기는 코드는 어디에도 없다.
* ``/diag/retreating`` 이 뜨면 설정값 이동을 멈춘다 — 후퇴 중의 품질은 힘의 함수가 아니다.
"""

from __future__ import annotations

import math

import rclpy
from rclpy.executors import ExternalShutdownException
try:  # rclpy 배포판마다 위치가 다르다
    from rclpy._rclpy_pybind11 import RCLError
except ImportError:  # pragma: no cover
    RCLError = RuntimeError
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
        self.declare_parameter("f_bar0_n", 3.0)
        self.declare_parameter("f_min_n", 1.0)
        # 하드 한계 5.0 이 아니라 warn 선. 적응기가 상승 금지선을 넘기는 것이 맞다 (§8.4).
        self.declare_parameter("f_max_n", 4.5)
        self.declare_parameter("dither_amplitude_n", 0.25)
        self.declare_parameter("dither_period_s", 5.0)
        self.declare_parameter("settle_s", 0.5)
        self.declare_parameter("step_max_n", 0.1)
        # 경사 → 걸음 이득 [N / (품질단위/N)].
        #
        # 1.0 이었고, 그러면 **움직이지 않는다.** 2026-09-11 팬텀 실측에서 dQ/dF 는
        # 0.0003~0.008 /N 이었다 (3 N 부근, 디더 ±0.25 N). 이득 1.0 이면 한 주기에
        # 0.0003~0.008 N 이라 90 s 에피소드(18 주기) 동안 0.09 N 도 못 간다 — 힘이
        # 출발값에 붙어 있는 것처럼 보이고, 실제로 그랬다.
        #
        # step_max_n 주석은 "한 주기에 ≤ 0.1 N, 1 N 옮기는 데 ≥ 50 s" 를 설계 의도로
        # 말한다. 그 의도가 성립하려면 전형적 경사가 clamp 근처의 걸음을 내야 하므로
        # 이득은 0.1/0.005 = 20 이다. 이득을 올려도 **한 주기 0.1 N · f_bar ≤ f_max−진폭**
        # 두 clamp 는 그대로다 — 빨라지는 것은 한계에 닿는 속도지 한계 자체가 아니다.
        self.declare_parameter("gain_n_per_unit", 20.0)

        g = lambda n: self.get_parameter(n).value
        ns = str(g("namespace"))
        # control_node 는 호환을 위해 선언만 남긴다. 이 노드는 더 이상 그 노드의
        # 파라미터를 건드리지 않는다 (모듈 docstring).
        self.control_node = str(g("control_node"))
        self.adapter = ForceSetpointAdapter(
            f_bar0=float(g("f_bar0_n")),          # 탐색의 **시작값**. 고정 목표가 아니다.
            amplitude_n=float(g("dither_amplitude_n")),
            period_s=float(g("dither_period_s")),
            settle_s=float(g("settle_s")),
            step_max_n=float(g("step_max_n")),
            gain_n_per_unit=float(g("gain_n_per_unit")),
            f_min_n=float(g("f_min_n")),
            f_max_n=float(g("f_max_n")),
        )
        self.adapter.reset(float(g("f_bar0_n")), self._now())
        # 로그 중복 제거에만 쓴다 — 바뀐 설정값만 한 줄씩 남긴다.
        self.pusher = SetpointPusher()
        self.retreating = False
        self._n_q = 0

        topic = str(g("quality_topic")) or f"{ns}/image_quality_raw"
        self.create_subscription(Float32, topic, self._on_quality, 10)
        self.create_subscription(Bool, "/diag/retreating", self._on_retreat, 10)
        self.f_bar_pub = self.create_publisher(Float32, f"{ns}/force_setpoint_bar", 10)
        self.create_timer(0.05, self._tick)
        self.get_logger().info(
            f"관찰 전용 — 로봇 파라미터를 쓰지 않는다 · 품질 {topic} · "
            f"출력 {ns}/force_setpoint_bar\n  {self.adapter.describe()}")

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
        self.pusher.mark(sp)
        self.get_logger().info(f"[관찰] 설정값 {sp:.3f} N — 로봇에는 보내지 않는다")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ForceSearchNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # 정리는 **실패해도 조용히** 끝낸다. 종료 중에는 context 가 이미 무효일 수
        # 있고, 그때 destroy_node / shutdown 은 또 예외를 던진다. 예전에는 그 두 번째
        # 예외가 그대로 올라와 "rcl_shutdown already called" 트레이스백이 로그의
        # 마지막을 차지했고, 정작 왜 내려갔는지는 그 위에 묻혔다.
        try:
            node.destroy_node()
        except RCLError:
            pass
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except RCLError:
                pass


if __name__ == "__main__":
    main()
