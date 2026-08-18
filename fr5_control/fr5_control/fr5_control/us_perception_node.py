"""지각 노드 — rus_perception 을 ROS 토픽으로 잇는 얇은 래퍼 (DESIGN_NOTES §3.1, §13.3).

    /us/image                sensor_msgs/Image
              │
              ▼
    /us/quality_seg          Float32            세그 기반 Q_seg
    /us/quality_raw          Float32            세그 비의존 Q_raw (점수 불가 시 NaN)
    /us/valid_for_control    Bool
    /us/rejection_reasons    String             '|' 로 이어붙인 사유 코드 (두 집합의 합집합)
    /us/control_features     Float32MultiArray  center_error 등 policy 입력 예정
    /diag/perception_latency Float32MultiArray  [전처리, 추론, 후처리, 특징, 종단] ms

**계산은 전부 rus_perception 이 한다.** 이 노드에는 제어 판단도 품질 정의도 없다.
지각 라이브러리가 ROS 를 모르는 것은 의도한 경계다 (§13.3.2) — 그래야 학습과 평가가
ROS 없이 돈다. 그 경계를 건너는 유일한 지점이 여기다.

## 두 품질함수는 같은 ROI 를 봐야 한다

`Q_raw` 와 `Q_seg` 가 서로 다른 영역을 보면 §5.3 의 진단 분기가 무너진다 — "둘 다 열화"
와 "한쪽만 열화"를 구별할 수 없게 된다. 그래서 ``Predictor.roi_for()`` 가 만든 마스크를
``compute_raw_quality`` 에 그대로 넘긴다.

## 지연은 여기서 측정된다

§11.1 이 요구하는 US 종단 지연이 이 노드의 관측값이다. 50 ms 를 넘으면 5 Hz policy
루프의 위상 여유가 위험하므로 진단 토픽으로 항상 내보낸다.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32, Float32MultiArray, String

from rus_perception.control.raw_quality import RawQualityConfig, compute_raw_quality
from rus_perception.inference.predictor import Predictor

#: 종단 지연 경고 문턱 [ms]. §11.1 — 넘으면 5 Hz policy 의 위상 여유가 위험하다.
LATENCY_WARN_MS = 50.0


class UsPerceptionNode(Node):
    """프레임 하나를 ControlState 와 Q_raw 로 바꿔 발행한다."""

    def __init__(self) -> None:
        super().__init__("us_perception_node")

        self._declare_parameters()

        checkpoint = str(self.get_parameter("perception.checkpoint").value)
        if not checkpoint or not Path(checkpoint).is_file():
            raise RuntimeError(
                f"perception.checkpoint 가 없다: {checkpoint!r}. "
                f"체크포인트 없이는 세그멘테이션이 성립하지 않는다."
            )

        self.predictor = Predictor.from_checkpoint(checkpoint)
        self.raw_config = RawQualityConfig.from_dict(
            {
                "scan_geometry": str(self.get_parameter("raw_quality.scan_geometry").value),
                "near_field_fraction": float(
                    self.get_parameter("raw_quality.near_field_fraction").value
                ),
                "far_field_fraction": float(
                    self.get_parameter("raw_quality.far_field_fraction").value
                ),
                "dark_intensity_threshold": float(
                    self.get_parameter("raw_quality.dark_intensity_threshold").value
                ),
                "shadow_relative_threshold": float(
                    self.get_parameter("raw_quality.shadow_relative_threshold").value
                ),
            }
        )

        self.bridge = CvBridge()
        self.previous_state = None
        self.frame_index = 0
        self.reset_gap_s = float(self.get_parameter("perception.reset_gap_s").value)
        self.last_frame_time = None

        # 깊이 1 + BEST_EFFORT: 추론이 30 Hz 를 못 따라가면 **가장 최근 프레임만** 남긴다.
        # 밀린 프레임을 처리해봐야 제어에는 이미 쓸모없는 과거다.
        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        self.create_subscription(Image, "/us/image", self._on_image, qos)

        self.q_seg_pub = self.create_publisher(Float32, "/us/quality_seg", 10)
        self.q_raw_pub = self.create_publisher(Float32, "/us/quality_raw", 10)
        self.valid_pub = self.create_publisher(Bool, "/us/valid_for_control", 10)
        self.reasons_pub = self.create_publisher(String, "/us/rejection_reasons", 10)
        self.features_pub = self.create_publisher(Float32MultiArray, "/us/control_features", 10)
        self.latency_pub = self.create_publisher(
            Float32MultiArray, "/diag/perception_latency", 10
        )

        self.get_logger().info(
            f"지각 노드 준비 · 체크포인트 {Path(checkpoint).name} · "
            f"scan_geometry={self.raw_config.scan_geometry}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("perception.checkpoint", "")
        # 프레임이 이만큼 끊기면 시간축 특징의 이전 상태를 버린다. 끊긴 앞뒤를 이어
        # 붙이면 warped IoU 가 거짓으로 낮게 나와 valid 가 잘못 떨어진다.
        self.declare_parameter("perception.reset_gap_s", 0.5)

        # ⏳ US 영상 기하 확정 후 갱신. 라이브러리 기본값을 그대로 노출한다.
        self.declare_parameter("raw_quality.scan_geometry", "linear")
        self.declare_parameter("raw_quality.near_field_fraction", 0.15)
        self.declare_parameter("raw_quality.far_field_fraction", 0.35)
        self.declare_parameter("raw_quality.dark_intensity_threshold", 0.10)
        self.declare_parameter("raw_quality.shadow_relative_threshold", 0.25)

    # -- 처리 ------------------------------------------------------------

    def _to_grayscale(self, msg: Image) -> np.ndarray:
        """B-mode 는 흑백이다. 컬러로 와도 강도만 남긴다. 값역은 ``[0, 1]``."""
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        return np.asarray(frame, dtype=np.float32) / 255.0

    def _on_image(self, msg: Image) -> None:
        now = self.get_clock().now()
        try:
            frame = self._to_grayscale(msg)
        except Exception as exc:
            self.get_logger().error(f"영상 변환 실패: {exc}", throttle_duration_sec=2.0)
            return

        # 프레임이 끊겼다 이어지면 시간축 비교 대상이 성립하지 않는다.
        if self.last_frame_time is not None:
            gap = (now - self.last_frame_time).nanoseconds / 1e9
            if gap > self.reset_gap_s:
                self.get_logger().warn(f"프레임 {gap:.2f} s 끊김 — 시간축 상태 초기화")
                self.previous_state = None
        self.last_frame_time = now

        try:
            state = self.predictor.predict_control_state(
                frame=frame,
                previous_state=self.previous_state,
                metadata={
                    "frame_id": str(self.frame_index),
                    "timestamp": now.nanoseconds / 1e9,
                },
            )
        except Exception as exc:
            self.get_logger().error(f"추론 실패: {exc}", throttle_duration_sec=2.0)
            return

        # 두 품질함수가 같은 영역을 보게 한다 (§5.3 진단 분기의 전제).
        raw = compute_raw_quality(
            frame, roi_mask=self.predictor.roi_for(frame.shape), config=self.raw_config
        )

        self.previous_state = state
        self.frame_index += 1
        self._publish(state, raw)

    # -- 발행 ------------------------------------------------------------

    @staticmethod
    def _finite(value) -> float:
        """``None`` 과 비유한값을 NaN 으로 정규화한다."""
        if value is None:
            return float("nan")
        number = float(value)
        return number if math.isfinite(number) else float("nan")

    def _publish(self, state, raw) -> None:
        self.q_seg_pub.publish(Float32(data=float(state.control_quality_score)))

        # 점수를 낼 수 없는 프레임은 NaN 으로 낸다. 0 으로 내면 "품질이 최악"과
        # "측정 불가"가 구별되지 않아 supervisor 가 잘못된 재탐색을 건다.
        self.q_raw_pub.publish(Float32(data=self._finite(raw.score)))

        # 접촉이 나쁘면 세그도 믿을 수 없다. 두 게이트를 모두 통과해야 유효로 본다.
        self.valid_pub.publish(
            Bool(data=bool(state.valid_for_control and raw.score is not None))
        )

        # 두 집합은 교집합이 없도록 라이브러리가 강제한다 (§10.3). 원시 사유를 앞에
        # 두어, supervisor 가 가장 앞 단계로 되돌릴 때 접촉 문제가 먼저 보이게 한다.
        reasons = list(raw.rejection_reasons) + list(state.rejection_reasons)
        self.reasons_pub.publish(String(data="|".join(reasons)))

        # policy 가 생기기 전까지의 임시 형식. us_interfaces 가 들어오면 대체된다.
        self.features_pub.publish(
            Float32MultiArray(
                data=[
                    self._finite(state.center_error_x),
                    self._finite(state.center_error_y),
                    self._finite(state.mask_area_ratio),
                    self._finite(state.orientation_degrees),
                    self._finite(state.lumen_surrounding_contrast),
                    self._finite(state.temporal_stability_score),
                ]
            )
        )

        # §11.1 이 요구하는 종단 지연 실측값.
        self.latency_pub.publish(
            Float32MultiArray(
                data=[
                    float(state.preprocessing_latency_ms),
                    float(state.inference_latency_ms),
                    float(state.postprocessing_latency_ms),
                    float(state.control_feature_latency_ms),
                    float(state.end_to_end_latency_ms),
                ]
            )
        )
        if state.end_to_end_latency_ms > LATENCY_WARN_MS:
            self.get_logger().warn(
                f"종단 지연 {state.end_to_end_latency_ms:.1f} ms > {LATENCY_WARN_MS:.0f} ms — "
                f"5 Hz policy 루프의 위상 여유가 위험하다 (§11.1)",
                throttle_duration_sec=5.0,
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = UsPerceptionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f"[us_perception_node] 기동 거부: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
