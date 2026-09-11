#!/usr/bin/env python3
"""프레임 한 장을 떠서 지각이 왜 아무것도 못 내는지 본다.

    python3 scripts/diag_frame.py --out /tmp/diag

Q_seg 와 Q_raw 가 **둘 다** not measured 면 모델이 아니라 그 앞이 문제다 — 영상 기하,
방향, 또는 ROI. 이 스크립트는 원본과 B-mode 를 그림으로 떨구고, Q_raw 가 왜 None 인지
(``rejection_reasons``) 와 U-Net 이 마스크를 내는지를 인쇄한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _common  # noqa: F401

from rus_policy.bmode import BmodeConverter
from rus_policy.config import load_config
from rus_policy.perception import apply_frame_transform, build_backend


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-topic", default="/us/image")
    p.add_argument("--out", default="/tmp/diag")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    cfg = load_config(args.config) if args.config else load_config()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    grabbed = {}

    class Grab(Node):
        def __init__(self):
            super().__init__("diag_frame")
            self.create_subscription(Image, args.image_topic, self.cb, qos_profile_sensor_data)

        def cb(self, m):
            if grabbed:
                return
            a = np.frombuffer(m.data, np.uint8)
            if a.size != m.height * m.width:
                print(f"⚠️ 영상 형식이 예상 밖: {m.width}×{m.height} {m.encoding!r} "
                      f"step={m.step} bytes={a.size}")
                return
            grabbed["raw"] = a.reshape(m.height, m.width).copy()
            grabbed["shape"] = (m.height, m.width)
            grabbed["encoding"] = m.encoding

    rclpy.init()
    n = Grab()
    for _ in range(200):
        if grabbed:
            break
        rclpy.spin_once(n, timeout_sec=0.05)
    rclpy.shutdown()
    if not grabbed:
        print(f"프레임을 못 받았다 — {args.image_topic} 가 나오고 있는가")
        return 1

    raw = grabbed["raw"]
    print(f"원본 {raw.shape} {grabbed['encoding']}  값 {raw.min()}~{raw.max()} 평균 {raw.mean():.1f}")

    conv = BmodeConverter({}, raw.shape, out_size=max(cfg.perception.frame_size))
    print(f"B-mode 변환: {json.dumps(conv.describe(), ensure_ascii=False)}")
    bm = apply_frame_transform(conv(raw)[None], cfg.perception.frame_transform)[0]
    print(f"B-mode {bm.shape}  값 {bm.min()}~{bm.max()} 평균 {bm.mean():.1f}")

    from PIL import Image as PILImage
    PILImage.fromarray(raw).save(out / "raw.png")
    PILImage.fromarray(bm).save(out / "bmode.png")
    print(f"그림: {out/'raw.png'} · {out/'bmode.png'}")

    backend = build_backend(cfg)
    if backend is None:
        print("⚠️ perception.backend = none — 지각이 꺼져 있다. 그래서 둘 다 못 잰다")
        return 0

    # Q_raw 가 왜 None 인지가 핵심이다
    from rus_perception.control.raw_quality import compute_raw_quality
    r = compute_raw_quality(np.asarray(bm, np.float32) / 255.0, roi_mask=backend.roi)
    print(f"\nQ_raw  score={r.score}")
    print(f"  거부 사유      {list(r.rejection_reasons)}")
    print(f"  어두운 A-line  {r.dark_a_line_ratio:.2f}   음영 A-line {r.shadowed_a_line_ratio:.2f}")
    print(f"  ROI            {'있음 ' + str(np.asarray(backend.roi).shape) if backend.roi is not None else '없음'}")
    if backend.roi is not None:
        roi = np.asarray(backend.roi)
        print(f"  ROI 참 비율    {roi.mean():.2%}   B-mode 와 모양 일치: {roi.shape == bm.shape}")

    vec, q_seg, tok, e = backend.step(bm)
    print(f"\nQ_seg  {q_seg}")
    from rus_policy.perception import STATE_FEATURE_NAMES
    for k in ("has_mask", "valid_for_control", "area_ratio", "largest_component_ratio",
              "segmentation_confidence"):
        print(f"  {k:26s} {vec[STATE_FEATURE_NAMES.index(k)]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
