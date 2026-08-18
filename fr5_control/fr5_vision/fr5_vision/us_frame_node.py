"""초음파 프레임 취득 노드 (DESIGN_NOTES §3.1).

    source (프레임그래버 / 비디오 파일 / 이미지 시퀀스)
              │
              ▼
    /us/image   sensor_msgs/Image, mono8

``camera_node`` 에서 왔지만 세 가지가 다르다.

1. **GUI 없음.** 이전 노드는 타이머 콜백 안에서 ``imshow`` + ``waitKey(1)`` 를 했다.
   캡처 경로에 GUI 지연이 그대로 실린다. 표시가 필요하면 ``rqt_image_view`` 를 쓴다.
2. **mono8 발행.** B-mode 는 흑백이다. RGB 로 실어 보내면 대역만 3배 쓰고
   지각 노드가 어차피 다시 흑백으로 만든다.
3. **파일 소스 지원.** 녹화된 초음파 영상으로 Stage 1 전 구간을 하드웨어 없이
   돌릴 수 있어야 한다.

지연 실측은 이 노드가 아니라 ``us_perception_node`` 가 한다 (§11.1). 다만 여기서
찍는 타임스탬프가 그 측정의 기준점이므로, **읽자마자** 찍고 변환은 그 뒤에 한다.
"""
from __future__ import annotations

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image


class UsFrameNode(Node):
    """프레임그래버 또는 녹화 파일에서 US 프레임을 읽어 발행한다."""

    def __init__(self) -> None:
        super().__init__("us_frame_node")

        self.declare_parameter("frame.source", "0")
        self.declare_parameter("frame.fps", 30.0)
        self.declare_parameter("frame.loop", True)
        self.declare_parameter("frame.width", 0)
        self.declare_parameter("frame.height", 0)
        self.declare_parameter("frame.frame_id", "us_probe")

        source = str(self.get_parameter("frame.source").value)
        fps = float(self.get_parameter("frame.fps").value)
        self.loop = bool(self.get_parameter("frame.loop").value)
        self.frame_id = str(self.get_parameter("frame.frame_id").value)

        # 숫자면 장치 인덱스, 아니면 경로. cv2 가 둘 다 받는다.
        self.source_is_device = source.isdigit()
        handle = int(source) if self.source_is_device else source

        self.capture = cv2.VideoCapture(handle)
        if not self.capture.isOpened():
            raise RuntimeError(
                f"프레임 소스를 열 수 없다: {source!r}. "
                f"장치 인덱스이거나 존재하는 파일 경로여야 한다."
            )

        if self.source_is_device:
            width = int(self.get_parameter("frame.width").value)
            height = int(self.get_parameter("frame.height").value)
            if width and height:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.capture.set(cv2.CAP_PROP_FPS, fps)

        self.bridge = CvBridge()
        self.frames = 0

        # 지각 노드와 같은 정책. 소비자가 못 따라가면 최신 프레임만 남긴다.
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.publisher = self.create_publisher(Image, "/us/image", qos)
        self.create_timer(1.0 / fps, self._tick)

        kind = "장치" if self.source_is_device else "파일"
        self.get_logger().info(f"US 프레임 소스 {kind} {source!r} · {fps:.0f} Hz · mono8")

    def _tick(self) -> None:
        ok, frame = self.capture.read()

        if not ok:
            if self.source_is_device:
                self.get_logger().warn("프레임 취득 실패", throttle_duration_sec=2.0)
                return
            if not self.loop:
                self.get_logger().info(f"파일 끝 — {self.frames} 프레임 발행 후 정지")
                self.capture.release()
                raise SystemExit(0)
            # 되감기. 지각 노드는 프레임 끊김을 감지해 시간축 상태를 버리므로,
            # 이어붙인 지점이 거짓 불연속으로 잡히지는 않는다.
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        # 타임스탬프는 읽은 직후에 찍는다 — 변환 시간이 지연 측정에 섞이면 안 된다.
        stamp = self.get_clock().now().to_msg()

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="mono8")
        except Exception as exc:
            self.get_logger().error(f"메시지 변환 실패: {exc}", throttle_duration_sec=2.0)
            return

        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)
        self.frames += 1

    def destroy_node(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            self.capture.release()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = UsFrameNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    except RuntimeError as exc:
        print(f"[us_frame_node] 기동 거부: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
