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

from fr5_vision.us_protocol import PROFILES, UsScannerSession
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
        # 프로브 프로파일. 상수와 거동이 기종마다 다르다 — C10UR 은 **스캔 시작을
        # 클라이언트가 명령**하고 프레임이 160 라인 × 512 표본이다. 예전에는 기본
        # 프로파일(SL2C)로 고정이라 C10UR 에서는 프레임이 하나도 오지 않았다.
        self.declare_parameter("us.probe", "c10ur")
        self.declare_parameter("us.video_port", 5002)
        self.declare_parameter("us.control_port", 5003)
        self.declare_parameter("us.topic", "/us/image")
        self.declare_parameter("us.frame_id", "us_image")
        self.declare_parameter("us.orientation", "none")
        self.declare_parameter("us.reconnect_period_s", 2.0)
        self.declare_parameter("us.stale_timeout_s", 2.0)
        # 프레임이 이만큼 끊기면 **세션을 다시 연다.**
        #
        # 프로브를 켠 뒤 첫 세션이 이전 세션의 잔여 상태를 물고 열리는 일이 있다:
        # `scanner_active` 가 start_scan 을 보내기도 전에 참으로 오고, 프레임은
        # 한 장 오다 만다 (2026-09-11, 재현됨). 다시 열면 깨끗하게 붙는다.
        #
        # 경고용 stale_timeout(2 s) 보다 넉넉히 잡는다 — 10 fps 스트림이 잠깐
        # 더듬는 것으로 세션을 끊으면 그게 더 나쁘다.
        self.declare_parameter("us.reconnect_after_stale_s", 8.0)
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
        self._stale_reconnect = float(self.get_parameter("us.reconnect_after_stale_s").value)
        self._report_period = float(self.get_parameter("us.report_period_s").value)

        topic = self.get_parameter("us.topic").value
        # 센서 QoS(best effort). 지연된 프레임을 재전송받는 것보다 최신 프레임이
        # 중요하다 — 접촉 제어 루프가 소비하기 때문이다.
        probe = str(self.get_parameter("us.probe").value)
        if probe not in PROFILES:
            raise RuntimeError(f"알 수 없는 프로브 {probe!r} — 있는 것: {sorted(PROFILES)}")
        self._profile = PROFILES[probe]

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
        session = UsScannerSession(self._host, self._video_port, self._control_port,
                                   profile=self._profile)
        last_frame_at = time.monotonic()
        last_report_at = last_frame_at
        last_connect_attempt = 0.0
        was_active = False
        warned_stale = False
        # 세션이 열린 뒤 아래에서 다시 잡는다. 여기서도 정의해 두는 것은 스레드 안의
        # NameError 가 조용히 스레드만 죽이기 때문이다 — 밖에서는 프레임이 안 오는
        # 것으로만 보인다.
        opened_at = time.monotonic()
        commanded_scan = False
        connect_fails = 0

        while not self._stop.is_set():
            if not session.is_open:
                if time.monotonic() - last_connect_attempt < self._reconnect_period:
                    self._stop.wait(0.05)
                    continue
                last_connect_attempt = time.monotonic()
                try:
                    session.open()
                except OSError as error:
                    connect_fails += 1
                    hint = ""
                    # 프로브는 클라이언트를 하나만 받고, **곱게 닫히지 않은 세션을 한동안
                    # 붙들고 있다.** 그 상태에서는 TCP 가 그냥 시간 초과하며, 재시도로는
                    # 절대 풀리지 않는다 — Wi-Fi 를 다시 붙여야 프로브가 세션을 버린다.
                    # 그 사실을 말해 주지 않으면 조작자는 재시도 로그만 보며 기다린다.
                    if connect_fails == 3:
                        hint = ("\n  프로브가 죽은 세션을 붙들고 있는 것으로 보인다. "
                                "재시도로는 풀리지 않는다 — Wi-Fi 를 다시 붙여라:\n"
                                "    python3 imu_bench/host/probe_wifi_linux.py --disconnect\n"
                                "    python3 imu_bench/host/probe_wifi_linux.py\n"
                                "  (다음부터는 세션을 Ctrl-C 로 끝내라. 강제 종료하면 "
                                "세션이 닫히지 않아 이 상태가 된다)")
                    self.get_logger().warn(
                        f"scanner connect failed ({type(error).__name__}: {error}); "
                        f"retrying in {self._reconnect_period:.1f}s" + hint,
                        throttle_duration_sec=10.0,
                    )
                    continue
                connect_fails = 0
                self.get_logger().info(
                    f"connected to scanner {self._host} (profile {self._profile.name})")
                last_frame_at = time.monotonic()
                opened_at = time.monotonic()
                commanded_scan = False
                was_active = False
                warned_stale = False

            # C10UR 은 클라이언트가 스캔을 명령해야 프레임이 온다. setup 바이트가
            # 다 나간 뒤(ready_after_s) 한 번만 보낸다 — 그 전에 보내면 무시된다.
            if (session.can_command_scan and not commanded_scan
                    and time.monotonic() - opened_at > self._profile.ready_after_s):
                # 우리가 명령하기 **전에** 이미 스캔 중이라고 답하면, 직전 클라이언트가
                # 스캔을 켜둔 채 떠난 것이다 (곱게 닫히지 않은 세션). 그 상태로 start 를
                # 보내면 프로브는 한 장 흘리고 만다 — 2026-09-11 에 두 번 재현됐다.
                # 먼저 끄고 켜서 상태를 확실히 만든다. 8 s 뒤 재접속으로도 풀리지만
                # 그때까지 화면이 비어 있고, 조작자는 그 이유를 알 수 없다.
                if session.scanner_active:
                    self.get_logger().warn(
                        "명령 전에 이미 스캔 상태다 — 직전 세션의 잔여로 본다. "
                        "정지 후 다시 시작한다")
                    session.stop_scan()
                    deadline = time.monotonic() + 1.5
                    while time.monotonic() < deadline and not self._stop.is_set():
                        session.poll(0.05)      # 정지 바이트가 나가도록 세션을 돌린다
                session.start_scan()
                commanded_scan = True
                self.get_logger().info("스캔 시작을 명령했다 (클라이언트 주도 프로브)")

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

            # 조용한 세션을 붙들고 있어 봐야 저절로 돌아오지 않는다 — 다시 연다.
            if self._stale_reconnect > 0.0 and now - last_frame_at > self._stale_reconnect:
                self.get_logger().warn(
                    f"{now - last_frame_at:.1f}s 동안 프레임이 없다 — 세션을 다시 연다 "
                    "(프로브가 이전 세션의 잔여 상태를 물고 열렸을 때 이렇게 풀린다)"
                )
                session.close()
                last_connect_attempt = 0.0      # 재접속 주기를 기다리지 않는다
                last_frame_at = time.monotonic()
                warned_stale = False
                continue

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
