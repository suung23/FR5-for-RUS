#!/usr/bin/env python3
"""학습된 U-Net 을 실시간 영상에 돌려 방광 마스크를 낸다 — **화면용이다.**

    # ROS 를 먼저, venv 를 나중에 (런북 §4 와 같은 순서)
    source /opt/ros/jazzy/setup.bash
    source ~/FR5-for-RUS/install/setup.bash
    source ~/FR5-for-RUS/Unet_seg/.venv/bin/activate

    cd policy_learning
    python3 scripts/run_segmentation.py                    # /us/image → /us/seg/*

GUI 의 `Bladder segmentation` 패널이 이 노드가 내는 것을 그린다. 브리지는 중계만 한다.

무엇을 내는가
-------------
| 토픽 | 형식 | 내용 |
|---|---|---|
| `/us/seg/bmode` | `Image` mono8 256² | **망이 실제로 본 그림** |
| `/us/seg/mask`  | `Image` mono8 256² | 그 그림 위의 마스크 (0/255) |
| `/us/seg/state` | `String` (JSON)    | `ControlState` 요약 — Q_seg · 유효성 · 기하 |

세 가지를 왜 이렇게 나눴는지
----------------------------

1. **B-mode 를 같이 낸다.** 브리지가 GUI 로 보내는 부채꼴 그림(`/us/image` →
   `fr5_vision.scan_convert`)과 망에 들어가는 그림(`rus_policy.bmode.BmodeConverter`
   → 256² 레터박스)은 **서로 다른 변환** 이다. 마스크를 브리지 쪽 그림에 얹으면
   경계가 실제와 어긋난 채 그럴듯하게 보인다 — 그림도 같이 내서 짝을 맞춘다.
   두 메시지의 `header.stamp` 이 같으면 같은 프레임이다.

2. **마스크를 색칠하지 않는다.** 나가는 것은 0/255 이진 마스크고, 채우기·윤곽선·
   투명도는 GUI 가 정한다. 조작자가 마스크를 껐다 켜서 경계가 실제 루멘 가장자리를
   따라가는지 눈으로 확인할 수 있어야 하는데, 미리 태워 넣은 그림으로는 그것을 못 한다.

3. **지각은 한 번만 돈다.** `UnetPerception.step_detailed` 하나로 특징 벡터와
   마스크를 함께 얻는다. 화면용 경로를 따로 짜면 정책이 보는 전처리와 갈라진다.

**제어에 쓰지 않는다.** `{ns}/image_quality` 를 내지 않는다 — 그것은 `run_policy.py`
의 것이고, 발행자가 둘이 되면 힘 탐색(DESIGN_NOTES §8.4)이 어느 쪽 Q 를 보는지
알 수 없게 된다. 이 노드는 구독만 하고 로봇에 닿는 토픽에는 아무것도 쓰지 않는다.

⚠️ `run_policy.py` 와 같이 띄우면 U-Net 이 **두 벌** 돈다. RTX 4090 에서는 여유가
있지만(프레임당 수 ms), 지각이 느려지면 정책의 관측 창이 오래된 프레임으로 채워진다
— 그럴 때는 `--max-hz` 를 낮춰라. 이 노드가 찍는 `지각 N ms/장` 이 그 판단 근거다.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from typing import Optional

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32, String
except ImportError:                                   # ROS 없이 --help 는 되게 한다
    rclpy = None
    Node = object

from _common import add_config_args, config_from_args

from rus_policy.bmode import BmodeConverter
from rus_policy.view_quality import view_quality
from rus_policy.perception import TOKEN_NAMES, apply_frame_transform, build_backend


def _f(value) -> Optional[float]:
    """JSON 으로 나갈 실수. 유한하지 않으면 None — "0" 과 "측정 안 됨" 은 다른 사실이다."""
    if value is None:
        return None
    v = float(value)
    return v if math.isfinite(v) else None


class SegmentationNode(Node):
    """`/us/image` 를 받아 U-Net 을 돌리고 마스크를 낸다. 화면용이다."""

    def __init__(self, args, cfg, backend):
        super().__init__("rus_segmentation")
        self.args, self.cfg, self.backend = args, cfg, backend
        self.conv: Optional[BmodeConverter] = None
        self.seq = 0
        self.n_img = self.n_skipped = 0
        self.next_run_at = 0.0
        self.last_frame_at = 0.0
        self.dt_ms = float("nan")
        self._warned_shape = False
        self._min_period = 1.0 / max(args.max_hz, 1e-6)
        # 도착 주기가 상한과 **같을 때** 지터로 절반이 날아가는 것을 막는다. 8 fps 원본에
        # 8 Hz 상한을 걸면 프레임이 1 ms 일찍 오는 것만으로 버려지고, 그다음은 늦어서
        # 통과하므로 실효 4.5 Hz 가 된다 (2026-09-11 mock 으로 재현). 다음 허용 시각을
        # "지금" 이 아니라 직전 허용 시각에서 재고, 그 앞에 여유를 조금 둔다.
        self._slack = 0.2 * self._min_period

        # 브리지·GUI 는 최신 그림만 쓴다. 늦은 프레임을 재전송받는 것보다 최신이 낫고,
        # `us_frame_node` 와 같은 규약이라 구독 쪽이 헷갈리지 않는다.
        self.bmode_pub = self.create_publisher(Image, args.bmode_topic, qos_profile_sensor_data)
        self.mask_pub = self.create_publisher(Image, args.mask_topic, qos_profile_sensor_data)
        self.state_pub = self.create_publisher(String, args.state_topic, qos_profile_sensor_data)
        # 품질 두 값. **러너가 없을 때만** 낸다 — 아래 _publish_quality 참조.
        self.q_seg_topic = f"{args.robot_namespace}/image_quality_seg"
        self.q_raw_topic = f"{args.robot_namespace}/image_quality_raw"
        self.q_seg_pub = self.create_publisher(Float32, self.q_seg_topic, 10)
        self.q_raw_pub = self.create_publisher(Float32, self.q_raw_topic, 10)
        self._yielded = False
        # us_frame_node 는 BEST_EFFORT 로 낸다. 기본 QoS 로 구독하면 프레임이 하나도 오지 않는다.
        self.create_subscription(Image, args.image_topic, self._on_image, qos_profile_sensor_data)

        self.get_logger().info(
            f"방광 세그멘테이션 — {args.image_topic} → {args.bmode_topic} · {args.mask_topic} · "
            f"{args.state_topic} | 상한 {args.max_hz:.0f} Hz | "
            f"체크포인트 {cfg.resolve(cfg.paths.unet_checkpoint)} ({backend.checkpoint_id})")

    def _publish_quality(self, vec, bmode) -> None:
        """텔레옵 중에도 화면에 Q 가 뜨도록 낸다 — 단 **러너가 없을 때만**.

        Q 를 내는 주체는 원래 run_policy 하나였다. 그래야 화면의 값과 정책이 보는 값이
        같기 때문이다. 그런데 러너는 에피소드가 도는 동안에만 살아 있어서, 조작자가 자세를
        잡는 내내 눈금이 비어 있었다 — 정작 그때가 Q 를 보고 싶은 때다.

        그래서 여기서도 내되, **발행자가 우리뿐일 때만** 낸다. 러너가 뜨면 이 노드는 곧바로
        입을 닫는다. 같은 토픽에 둘이 쓰면 값이 번갈아 들어가 화면이 두 값 사이를 튄다 —
        2026-09-12 에 desired_twist 에서 그것 때문에 로봇이 멈췄다. 발행자 등록 수로 판정하는
        것이 여기서는 옳다: 러너의 발행자는 러너가 살아 있는 동안에만 있고, 살아 있으면 늘 낸다.
        """
        alone = self.count_publishers(self.q_seg_topic) <= 1
        if not alone:
            if not self._yielded:
                self._yielded = True
                self.get_logger().info("러너가 품질을 내기 시작했다 — 이 노드는 품질 발행을 멈춘다")
            return
        if self._yielded:
            self._yielded = False
            self.get_logger().info("러너가 사라졌다 — 품질 발행을 다시 맡는다")
        self.q_seg_pub.publish(Float32(data=float(view_quality(vec)[0])))
        self.q_raw_pub.publish(Float32(data=float(self.backend.raw_quality(bmode))))

    # -- 발행 -----------------------------------------------------------------

    def _image_msg(self, image: np.ndarray, stamp) -> Image:
        image = np.ascontiguousarray(image.astype(np.uint8))
        msg = Image()
        msg.header.stamp = stamp
        # 마스크와 B-mode 가 같은 그림임을 프레임 이름으로도 말해 둔다. 짝은 stamp 로 맞춘다.
        msg.header.frame_id = "us_seg"
        msg.height, msg.width = image.shape
        msg.encoding = "mono8"
        msg.is_bigendian = 0
        msg.step = image.shape[1]
        msg.data = image.tobytes()
        return msg

    def _state_payload(self, cs, q: float, token: int, e_px: float) -> dict:
        """`ControlState` 중 조작자가 읽을 것만. 마스크 자체는 여기 넣지 않는다."""
        has_mask = cs.centroid_x_px is not None and cs.mask_area_px > 0
        return {
            "seq": self.seq,
            "checkpointId": self.backend.checkpoint_id,
            # Q_seg 는 "방광을 찾았다" 를 전제로 한 점수다. 못 찾았으면 그 사실이 먼저다.
            "quality": _f(q),
            "validForControl": bool(cs.valid_for_control),
            "rejectionReasons": list(cs.rejection_reasons),
            "hasMask": bool(has_mask),
            "maskAreaPx": int(cs.mask_area_px),
            "maskAreaRatio": _f(cs.mask_area_ratio),
            "centroidPx": [_f(cs.centroid_x_px), _f(cs.centroid_y_px)] if has_mask else None,
            "centerError": [_f(cs.center_error_x), _f(cs.center_error_y)] if has_mask else None,
            # 빔축은 ROI(실측 부채꼴)의 가로 중심이지 그림의 중심이 아니다. ê 는 이것을 기준으로 잰다.
            "beamAxisPx": _f(self.backend.beam_axis_px),
            "eHatPx": _f(e_px),
            "token": TOKEN_NAMES[token] if 0 <= token < len(TOKEN_NAMES) else None,
            "segmentationConfidence": _f(cs.segmentation_confidence),
            "lumenContrast": _f(cs.lumen_surrounding_contrast),
            "borderContactRatio": _f(cs.border_contact_ratio),
            "largestComponentRatio": _f(cs.largest_component_ratio),
            "temporalWarpedIou": _f(cs.temporal_warped_iou),
            "centroidJump": _f(cs.normalized_centroid_jump),
            "maskThreshold": _f(cs.mask_threshold),
            "roiMode": str(cs.roi_mode),
            # 이 노드 안에서 잰 벽시계다. 장비 지연(≈200 ms)은 여기 들어 있지 않다.
            "perceptionMs": _f(self.dt_ms),
            "skipped": int(self.n_skipped),
        }

    # -- 수신 -----------------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        buf = np.frombuffer(bytes(msg.data), np.uint8)
        if buf.size != msg.height * msg.width:
            if not self._warned_shape:
                self._warned_shape = True
                self.get_logger().error(
                    f"영상 형식 예상 밖: {msg.width}×{msg.height} {msg.encoding!r} — 버린다")
            return
        raw = buf.reshape(msg.height, msg.width)
        now = time.monotonic()

        # 영상이 끊겼다 돌아오면 시간 의존 특징을 이어 붙이면 안 된다 — 끊긴 구간을
        # 건너뛴 것이 되어 warped IoU·centroid jump 가 거짓으로 좋아진다 (run_policy 와 같은 규약).
        if self.last_frame_at and (now - self.last_frame_at) > self.args.gap_reset_s:
            self.get_logger().warn(
                f"영상이 {now - self.last_frame_at:.1f}s 끊겼다 돌아왔다 — 지각 스트림을 다시 시작한다")
            self.backend.reset()
        self.last_frame_at = now

        # 상한을 넘겨 들어오는 프레임은 버린다. 그림이 부드러워지는 것보다 지각이
        # 밀리지 않는 것이 중요하다 — 특히 정책이 같이 돌 때.
        if now + self._slack < self.next_run_at:
            self.n_skipped += 1
            return
        # max(now, ...) 이라 영상이 끊겼다 돌아와도 밀린 만큼 몰아서 돌지 않는다.
        self.next_run_at = max(now, self.next_run_at) + self._min_period

        if self.conv is None:                            # 첫 프레임에서 변환기 확정
            self.conv = BmodeConverter({}, raw.shape, out_size=max(self.cfg.perception.frame_size))
            self.get_logger().info(
                f"B-mode 변환: {json.dumps(self.conv.describe(), ensure_ascii=False)}")

        t0 = time.time()
        bmode = apply_frame_transform(self.conv(raw)[None], self.cfg.perception.frame_transform)[0]
        vec, q, token, e_px, cs = self.backend.step_detailed(bmode)
        self.dt_ms = (time.time() - t0) * 1000.0
        self.seq += 1
        self.n_img += 1

        if cs.binary_mask is None:
            # 설정이 마스크를 버리게 되어 있으면(control.keep_binary_mask=false) 그릴 것이 없다.
            # 조용히 빈 화면을 보내는 대신 이유를 말한다.
            self.get_logger().error(
                "ControlState 에 binary_mask 가 없다 — U-Net 설정의 control.keep_binary_mask 를 켜라",
                throttle_duration_sec=30.0)
            return

        mask = (np.asarray(cs.binary_mask) > 0).astype(np.uint8) * 255
        # 두 그림은 **같은 stamp** 로 나간다. 브리지가 그것으로 짝을 맞춘다 — 짝이 어긋난
        # 마스크는 어긋난 줄 모르고 그려지므로, 시각이 다른 것은 붙이지 않는다.
        stamp = self.get_clock().now().to_msg()
        self.bmode_pub.publish(self._image_msg(bmode, stamp))
        self.mask_pub.publish(self._image_msg(mask, stamp))
        self.state_pub.publish(String(data=json.dumps(
            self._state_payload(cs, q, token, e_px), ensure_ascii=False)))
        self._publish_quality(vec, bmode)

        if self.n_img % self.args.report_every == 0:
            self.get_logger().info(
                f"프레임 {self.n_img} 장 · 지각 {self.dt_ms:.0f} ms/장 · 건너뜀 {self.n_skipped} · "
                f"Q={q:.3f} {'유효' if cs.valid_for_control else '무효'} "
                f"면적 {100.0 * cs.mask_area_ratio:.1f}%")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--robot-namespace", default="/fr5_right",
                   help="품질을 낼 네임스페이스 — GUI 가 보는 곳과 같아야 한다")
    p.add_argument("--image-topic", default="/us/image")
    p.add_argument("--bmode-topic", default="/us/seg/bmode")
    p.add_argument("--mask-topic", default="/us/seg/mask")
    p.add_argument("--state-topic", default="/us/seg/state")
    p.add_argument("--max-hz", type=float, default=15.0,
                   help="추론 상한. 프로브가 이보다 빠르면 프레임을 버린다 (기본 15)")
    p.add_argument("--gap-reset-s", type=float, default=1.0,
                   help="이만큼 영상이 끊기면 지각의 시간 상태를 버린다")
    p.add_argument("--report-every", type=int, default=50, help="이 프레임마다 한 줄 찍는다")
    args = p.parse_args()

    cfg = config_from_args(args)
    if rclpy is None:
        raise SystemExit(
            "ROS 2 (rclpy) 를 찾을 수 없습니다.\n"
            "  source /opt/ros/jazzy/setup.bash && source ~/FR5-for-RUS/install/setup.bash 를\n"
            "  **먼저** 하고 그다음 Unet_seg/.venv 를 활성화하십시오 (런북 §4).")

    backend = build_backend(cfg)
    if backend is None:
        raise SystemExit(
            "perception.backend 가 none 입니다 — 그릴 마스크가 없습니다.\n"
            "  --set perception.backend=unet 로 켜거나 설정 파일을 확인하십시오.")

    rclpy.init()
    node = SegmentationNode(args, cfg, backend)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n프레임 {node.n_img} 장 처리, {node.n_skipped} 장 건너뜀")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
