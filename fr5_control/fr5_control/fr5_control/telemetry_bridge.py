"""ROS 2 → 웹소켓 텔레메트리 브리지. teleop 콘솔 GUI 가 붙는 자리다.

    ros2 run fr5_control telemetry_bridge
    ros2 run fr5_control telemetry_bridge --ros-args -p bridge.px6d_port:=/dev/ttyACM0

GUI 는 자기 힘으로 로봇 상태를 알 방법이 없다. FR5 컨트롤러는 자체 XML-RPC/UDP 만
말하고 웹소켓도 ROS 브리지도 열지 않는다. 이 노드가 그 간극을 메운다 — 이미 돌고 있는
Phase 0 스택의 토픽을 구독해 GUI 의 계약 형식(JSON)으로 밀어 준다.

**단방향이다.** 구독만 하고 발행하지 않으며, 소켓에서 들어오는 것은 전부 버린다.
로봇을 움직이는 경로는 속도 클램프와 워치독을 가진 제어 스택 하나로 유지한다.

구독:
  {ns}/joint_states   sensor_msgs/JointState     위치·속도·토크
  {ns}/ee_wrt_base    geometry_msgs/Pose         TCP 자세
  {ns}/wrench         geometry_msgs/WrenchStamped  컨트롤러 경유 F/T

**PX6D 는 컨트롤러를 거치지 않는다.** USB 변형이라 `robot_state_pkg.ft_sensor_data`
에 값이 안 들어오고, 그 결과 `{ns}/wrench` 는 보통 비어 있다. `bridge.px6d_port` 를
주면 이 노드가 센서를 직접 시리얼로 읽어 그 값을 쓴다 — 지금으로서는 그것이 유일한
실제 힘 경로다 (DESIGN_NOTES §2.1 · [[paxini-px6d-ft-sensor]]).

기동하면 어떤 소스가 살아 있는지 로그로 찍는다. GUI 가 "연결됐는데 값이 없다" 를
보여줄 때 여기부터 본다.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import rclpy
from geometry_msgs.msg import Pose, WrenchStamped
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import JointState

#: 관절이 이보다 느리면 정지로 본다 [rad/s]. 컨트롤러는 teleop 여부를 말해 주지
#: 않으므로 움직임으로 추론한다. 추론이라는 사실은 프레임에 함께 실어 보낸다.
_IDLE_VEL_RAD_S = 0.002


class TelemetryBridge(Node):
    """ROS 토픽을 모아 웹소켓으로 밀어 주는 노드."""

    def __init__(self) -> None:
        """노드를 만들고 구독·서버를 띄운다."""
        super().__init__("fr5_telemetry_bridge")

        self.declare_parameter("robot.name", "fr5_right")
        self.declare_parameter("bridge.host", "0.0.0.0")
        self.declare_parameter("bridge.port", 8765)
        self.declare_parameter("bridge.telemetry_hz", 30.0)
        self.declare_parameter("bridge.wrench_hz", 50.0)
        self.declare_parameter("bridge.px6d_port", "")
        self.declare_parameter("bridge.px6d_baud", 921600)

        self.robot_name = self.get_parameter("robot.name").value
        ns = f"/{self.robot_name}"

        self._lock = threading.Lock()
        self._joints: JointState | None = None
        self._joints_at = 0.0
        self._pose: Pose | None = None
        self._wrench: tuple[float, ...] | None = None
        self._wrench_at = 0.0
        self._wrench_source = "none"
        # PX6D 직결일 때만 채워진다. 회선 품질을 GUI 가 그대로 보여 준다 —
        # "값이 이상하다" 와 "선이 이상하다" 를 조작자가 가를 수 있어야 한다.
        self._px6d_crc = 0
        self._px6d_hz = 0.0

        sensor_qos = QoSPresetProfiles.SENSOR_DATA.value
        self.create_subscription(JointState, f"{ns}/joint_states", self._on_joints, sensor_qos)
        self.create_subscription(Pose, f"{ns}/ee_wrt_base", self._on_pose, 10)
        self.create_subscription(WrenchStamped, f"{ns}/wrench", self._on_wrench, sensor_qos)

        self.get_logger().info(
            f"구독: {ns}/joint_states · {ns}/ee_wrt_base · {ns}/wrench"
        )

        px6d_port = str(self.get_parameter("bridge.px6d_port").value)
        if px6d_port:
            self._start_px6d(px6d_port, int(self.get_parameter("bridge.px6d_baud").value))

        self._start_server()

    # -- 구독 콜백 ---------------------------------------------------------

    def _on_joints(self, msg: JointState) -> None:
        with self._lock:
            self._joints = msg
            self._joints_at = time.time()

    def _on_pose(self, msg: Pose) -> None:
        with self._lock:
            self._pose = msg

    def _on_wrench(self, msg: WrenchStamped) -> None:
        # 컨트롤러 경유 값은 PX6D 직결이 없을 때만 쓴다. 둘 다 있으면 직결이 이긴다 —
        # 우리 개체는 컨트롤러에 물려 있지 않으므로 그쪽 값은 0 이거나 남의 것이다.
        if self._wrench_source == "px6d_serial":
            return
        w = msg.wrench
        with self._lock:
            self._wrench = (w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z)
            self._wrench_at = time.time()
            self._wrench_source = "controller"

    # -- PX6D 직결 ---------------------------------------------------------

    def _start_px6d(self, port: str, baud: int) -> None:
        """센서를 시리얼로 직접 읽는 스레드를 띄운다."""
        try:
            import serial  # noqa: F401
        except ImportError:
            self.get_logger().error("pyserial 이 없어 PX6D 직결을 건너뛴다")
            return

        thread = threading.Thread(
            target=self._px6d_loop, args=(port, baud), name="px6d", daemon=True
        )
        thread.start()
        self.get_logger().info(f"PX6D 직결 시도: {port} @ {baud}")

    def _px6d_loop(self, port_name: str, baud: int) -> None:
        import serial

        from fr5_control.px6d_protocol import (
            build_command,
            build_set_rate,
            CMD_STREAM_START,
            CMD_STREAM_STOP,
            FrameParser,
            STREAM_START_DATA,
        )

        while rclpy.ok():
            try:
                with serial.Serial(port_name, baud, timeout=0.01) as port:
                    parser = FrameParser()
                    port.write(build_command(CMD_STREAM_STOP, 0x00))
                    time.sleep(0.1)
                    port.reset_input_buffer()
                    port.write(build_set_rate(1000))
                    time.sleep(0.05)
                    port.write(build_command(CMD_STREAM_START, STREAM_START_DATA))
                    port.reset_input_buffer()
                    parser.reset()

                    self.get_logger().info(f"PX6D 스트리밍 시작: {port_name}")
                    with self._lock:
                        self._wrench_source = "px6d_serial"

                    # 스트림을 켜고 여기까지 오는 사이의 재동기는 회선 오류가 아니다.
                    # 첫 프레임이 선 시점을 기준으로 삼는다 (px6d_monitor 와 같은 규약).
                    crc_base = None
                    stamps: list[float] = []

                    while rclpy.ok():
                        chunk = port.read(port.in_waiting or 1)
                        if not chunk:
                            continue
                        for frame in parser.feed(chunk):
                            if not frame.is_wrench:
                                continue
                            now = time.time()
                            if crc_base is None:
                                crc_base = parser.crc_errors
                            stamps.append(now)
                            while stamps and now - stamps[0] > 1.0:
                                stamps.pop(0)
                            with self._lock:
                                self._wrench = frame.wrench()
                                self._wrench_at = now
                                self._px6d_crc = parser.crc_errors - crc_base
                                self._px6d_hz = float(len(stamps))
            except Exception as exc:  # 케이블이 빠져도 노드는 살아 있어야 한다
                self.get_logger().warn(
                    f"PX6D 읽기 실패 ({exc}) — 2 초 뒤 재시도", throttle_duration_sec=5.0
                )
                with self._lock:
                    self._wrench_source = "none"
                time.sleep(2.0)

    # -- 프레임 조립 -------------------------------------------------------

    def telemetry_frame(self) -> dict:
        """GUI 계약의 RobotTelemetry 한 장."""
        now = time.time()
        with self._lock:
            joints = self._joints
            joints_at = self._joints_at
            pose = self._pose

        # 토픽이 끊긴 지 오래면 연결되지 않은 것으로 보고한다. 마지막 자세를 계속
        # 내보내면 GUI 는 멈춘 로봇과 살아 있는 로봇을 구별할 수 없다.
        fresh = joints is not None and (now - joints_at) < 0.5

        # 종류를 명시한다. 한 소켓에 두 종류가 흐르므로 받는 쪽이 모양으로
        # 추측하게 두면, 필드가 빠진 프레임이 다른 종류로 오인된다.
        frame: dict = {
            "type": "telemetry",
            "timestamp": int(now * 1000),
            "connected": bool(fresh),
        }
        if not fresh:
            return frame

        assert joints is not None
        frame["jointPositions"] = list(joints.position)
        if joints.velocity:
            frame["jointVelocities"] = list(joints.velocity)
            moving = any(abs(v) > _IDLE_VEL_RAD_S for v in joints.velocity)
            # 컨트롤러는 teleop 여부를 말해 주지 않는다. 움직임에서 추론한 값이며
            # safetyState 는 추론하지 않는다 — 안전 상태를 지어내면 안 된다.
            frame["robotState"] = "teleop" if moving else "idle"

        if pose is not None:
            frame["tcpPose"] = {
                "position": [pose.position.x, pose.position.y, pose.position.z],
                "quaternion": [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
            }
        return frame

    def wrench_frame(self) -> dict | None:
        """GUI 계약의 WrenchSample 한 장. 값이 신선하지 않으면 ``None``."""
        now = time.time()
        with self._lock:
            values = self._wrench
            at = self._wrench_at
            source = self._wrench_source
            crc = self._px6d_crc
            sensor_hz = self._px6d_hz
        if values is None or (now - at) > 0.5:
            return None

        sample = {
            "type": "wrench",
            "timestamp": int(now * 1000),
            "force": [values[0], values[1], values[2]],
            "torque": [values[3], values[4], values[5]],
            "source": source,
        }
        # 센서 회선 지표는 직결일 때만 뜻이 있다. 컨트롤러 경유 값에 붙이면
        # 재지 않은 것을 잰 것처럼 보인다.
        if source == "px6d_serial":
            sample["crcErrors"] = int(crc)
            sample["sensorHz"] = round(sensor_hz, 1)
        return sample

    # -- 웹소켓 ------------------------------------------------------------

    def _start_server(self) -> None:
        host = str(self.get_parameter("bridge.host").value)
        port = int(self.get_parameter("bridge.port").value)
        thread = threading.Thread(
            target=self._serve, args=(host, port), name="ws", daemon=True
        )
        thread.start()

    def _serve(self, host: str, port: int) -> None:
        try:
            import websockets
        except ImportError:
            self.get_logger().error(
                "websockets 가 없다: pip install websockets — 브리지를 열 수 없다"
            )
            return

        async def handler(connection):
            peer = getattr(connection, "remote_address", "?")
            self.get_logger().info(f"GUI 접속: {peer}")
            tele_dt = 1.0 / max(1.0, float(self.get_parameter("bridge.telemetry_hz").value))
            wrench_dt = 1.0 / max(1.0, float(self.get_parameter("bridge.wrench_hz").value))
            next_tele = next_wrench = 0.0
            try:
                while True:
                    now = time.monotonic()
                    if now >= next_tele:
                        next_tele = now + tele_dt
                        await connection.send(json.dumps(self.telemetry_frame()))
                    if now >= next_wrench:
                        next_wrench = now + wrench_dt
                        sample = self.wrench_frame()
                        if sample is not None:
                            await connection.send(json.dumps(sample))
                    await asyncio.sleep(min(tele_dt, wrench_dt) / 2.0)
            except Exception:
                pass
            finally:
                self.get_logger().info(f"GUI 연결 종료: {peer}")

        async def main() -> None:
            async with websockets.serve(handler, host, port):
                self.get_logger().info(f"텔레메트리 브리지 대기: ws://{host}:{port}")
                await asyncio.Future()

        try:
            asyncio.run(main())
        except Exception as exc:
            self.get_logger().error(f"브리지 종료: {exc}")


def main(argv=None) -> None:
    """노드를 띄우고 Ctrl-C 까지 돈다."""
    rclpy.init(args=argv)
    node = TelemetryBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
