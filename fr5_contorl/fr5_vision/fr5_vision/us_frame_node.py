#!/usr/bin/env python3
"""US 프레임그래버 노드 — 초음파 장비 AP 에서 받은 프레임을 /us/image 로 낸다.

DESIGN_NOTES §3 의 `us_frame_node`. 상류는 장비 AP(TCP 5002/5003), 하류는
`quality_raw_node` 와 `perception_node` 다.

    ros2 run fr5_vision us_frame --ros-args --params-file <probe.yaml>
    ros2 run fr5_vision us_frame --ros-args -p us.host:=127.0.0.1   # mock 상대

주의할 점 세 가지.

1. **장비로 설정 명령을 보내지 않는다.** 2026-08-06 PCAP 에서 관찰된 연결·
   keepalive·세션 바이트만 재생한다 (`us_protocol`). gain/depth/scan 같은
   장비 파라미터는 이 경로로 건드릴 수 없고, 건드려서도 안 된다.

2. **캡처 콜백에 GUI 를 넣지 않는다.** 기존 `camera_node.py` 는 타이머 콜백
   안에서 `cv2.imshow` + `waitKey(1)` 을 돌려 캡처 경로에 GUI 지연을 섞었다
   (DESIGN_NOTES §938). 영상 확인은 `ros2 run rqt_image_view rqt_image_view`
   나 tracer 뷰어처럼 별도 프로세스로 한다.

3. **여기서 나가는 것은 candidate 프레임이다.** 방향·scan conversion 이
   FrameBridge 출력과 대조 검증되기 전까지 임상 판독이나 측정에 쓰지 않는다.
   기본 `us.orientation` 이 "none" 인 이유다 — 검증 안 된 변환을 데이터
   경로에 조용히 넣지 않는다.
"""

from __future__ import annotations

import threading
import time

from fr5_vision.us_protocol import UsScannerSession
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


# np.rot90 의 k 값. tracer 뷰어의 Orientation 슬라이더(0..3)와 같은 네 가지다.
ORIENTATIONS = {
    "none": 0,
    "rot90_ccw": 1,
    "rot180": 2,
    "rot90_cw": -1,
}


class UsFrameNode(Node):
    """장비 AP 세션을 물고 완성된 프레임마다 sensor_msgs/Image 를 낸다."""

    def __init__(self) -> None:
        super().__init__("us_frame_node")

        self.declare_parameter("us.host", "192.168.1.1")
        self.declare_parameter("us.video_port", 5002)
        self.declare_parameter("us.control_port", 5003)
        self.declare_parameter("us.topic", "/us/image")
        self.declare_parameter("us.frame_id", "us_image")
        self.declare_parameter("us.orientation", "none")
        self.declare_parameter("us.reconnect_period_s", 2.0)
        self.declare_parameter("us.stale_timeout_s", 2.0)
        self.declare_parameter("us.report_period_s", 5.0)

        orientation = self.get_parameter("us.orientation").value
        if orientation not in ORIENTATIONS:
            raise ValueError(
                f"us.orientation must be one of {sorted(ORIENTATIONS)}, got {orientation!r}"
            )
        self._rotate_k = ORIENTATIONS[orientation]

        self._host = self.get_parameter("us.host").value
        self._video_port = int(self.get_parameter("us.video_port").value)
        self._control_port = int(self.get_parameter("us.control_port").value)
        self._frame_id = self.get_parameter("us.frame_id").value
        self._reconnect_period = float(self.get_parameter("us.reconnect_period_s").value)
        self._stale_timeout = float(self.get_parameter("us.stale_timeout_s").value)
        self._report_period = float(self.get_parameter("us.report_period_s").value)

        topic = self.get_parameter("us.topic").value
        # 센서 QoS(best effort). 지연된 프레임을 재전송받는 것보다 최신 프레임이
        # 중요하다 — 접촉 제어 루프가 소비하기 때문이다.
        self._publisher = self.create_publisher(Image, topic, qos_profile_sensor_data)

        self._stop = threading.Event()
        self._frames_published = 0
        self._frames_total = 0
        self._thread = threading.Thread(target=self._receive_loop, name="us_rx", daemon=True)

        self.get_logger().info(
            f"us_frame_node -> {topic} | scanner {self._host}:{self._video_port}/"
            f"{self._control_port} | orientation={orientation}"
        )
        self._thread.start()

    # -- 발행 -----------------------------------------------------------------

    def _to_image_msg(self, raw_image: np.ndarray, stamp) -> Image:
        image = raw_image if self._rotate_k == 0 else np.rot90(raw_image, self._rotate_k)
        image = np.ascontiguousarray(image)

        message = Image()
        message.header.stamp = stamp
        message.header.frame_id = self._frame_id
        message.height, message.width = image.shape
        message.encoding = "mono8"
        message.is_bigendian = 0
        message.step = image.shape[1]
        message.data = image.tobytes()
        return message

    # -- 수신 스레드 ----------------------------------------------------------

    def _receive_loop(self) -> None:
        """블로킹 select 를 executor 밖에서 돌린다.

        타이머 콜백으로 하면 select 대기가 executor 를 잡아 다른 콜백이 밀린다.
        rclpy publish 는 별도 스레드에서 호출해도 된다.
        """
        session = UsScannerSession(self._host, self._video_port, self._control_port)
        last_frame_at = time.monotonic()
        last_report_at = last_frame_at
        last_connect_attempt = 0.0
        was_active = False
        warned_stale = False

        while not self._stop.is_set():
            if not session.is_open:
                if time.monotonic() - last_connect_attempt < self._reconnect_period:
                    self._stop.wait(0.05)
                    continue
                last_connect_attempt = time.monotonic()
                try:
                    session.open()
                except OSError as error:
                    self.get_logger().warn(
                        f"scanner connect failed ({type(error).__name__}: {error}); "
                        f"retrying in {self._reconnect_period:.1f}s",
                        throttle_duration_sec=10.0,
                    )
                    continue
                self.get_logger().info(f"connected to scanner {self._host}")
                last_frame_at = time.monotonic()
                was_active = False
                warned_stale = False

            try:
                frames = session.poll(0.05)
            except (ConnectionError, OSError) as error:
                session.close()
                # 종료 중 끊김은 정상이다. 여기서 로그를 내면 이미 무효해진
                # rclpy context 에 rosout 발행을 시도해 종료 경합이 난다.
                if self._stop.is_set():
                    break
                self.get_logger().warn(f"scanner session dropped: {type(error).__name__}: {error}")
                continue

            # 블록이 다 모인 직후를 찍는다. 장비 내부 획득 시각이 아니라
            # 마지막 블록의 도착 시각이다 — 엔드투엔드 지연은 별도 실측 대상
            # (DESIGN_NOTES §2 US 프레임그래버 ⏳).
            stamp = self.get_clock().now().to_msg()
            for frame in frames:
                self._publisher.publish(self._to_image_msg(frame.as_uint8_image(), stamp))
                self._frames_published += 1
                self._frames_total += 1

            now = time.monotonic()
            if frames:
                last_frame_at = now
                warned_stale = False

            if session.scanner_active and not was_active:
                self.get_logger().info("scanner reported active stream state")
                was_active = True

            if not warned_stale and now - last_frame_at > self._stale_timeout:
                warned_stale = True
                self.get_logger().warn(
                    f"no candidate frame for {now - last_frame_at:.1f}s "
                    f"(scanner_active={session.scanner_active}); "
                    "장비가 실제 스캔/스트리밍 상태인지, Windows 앱이 AP 를 점유하고 있지 않은지 확인"
                )

            if self._report_period > 0.0 and now - last_report_at >= self._report_period:
                elapsed = now - last_report_at
                self.get_logger().info(
                    f"{self._frames_published} frames in {elapsed:.1f}s "
                    f"({self._frames_published / elapsed:.1f} Hz), total {self._frames_total}"
                )
                self._frames_published = 0
                last_report_at = now

        session.close()

    def shutdown(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UsFrameNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
