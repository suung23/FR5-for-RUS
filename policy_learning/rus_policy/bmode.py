"""candidate 프레임 → 학습용 B-mode. 세션 메타의 프레임 형상·부채꼴 기하에 따라 필요한 변환만 한다 (2026-09-09).

두 종류의 세션이 있다:
  * SL-2C (Wi-Fi, 256×256): 뷰어가 이미 scan conversion 한 B-mode candidate. 방향만 미검증 → `frame_transform` 만 적용.
  * C10UR (Wi-Fi 원시, 160×512): **극좌표** (행 = A-line, 열 = 깊이 표본).
    (2026-09-09 저녁까지 320×256 으로 적혀 있었다 — 같은 바이트 수라 읽히지만 라인이 두 행으로 쪼개진 잘못된 배치다.) U-Net·Q_seg 는 부채꼴 B-mode 를 기대하므로
    `imu_bench/host/us_scan_convert.py` 로 부채꼴을 만든 뒤 정방형 레터박스로 `perception.frame_size` 에 맞춘다.
    기하는 `session.meta.json` 의 `us.fan_geometry` (GUI 가 기록) 를 쓰고, 없으면 기본값 (R 59 mm / 28° / 220 mm).
    비등방 리사이즈는 하지 않는다 — 부채꼴의 원 기하(Q_raw fan 반경) 를 지키기 위해.

`Q_raw` 는 A-line 을 원하므로 극좌표 원본이 오히려 맞다 — 그쪽은 이 변환을 거치지 않고 원본을 쓴다 (⏳ 연결 예정).
"""

from __future__ import annotations

import sys
from typing import Any, Optional

import numpy as np

from .config import IMU_BENCH_ROOT

_HOST = IMU_BENCH_ROOT / "host"
if str(_HOST) not in sys.path:
    sys.path.insert(0, str(_HOST))


def is_polar_session(meta: dict[str, Any], frame_shape: tuple[int, int]) -> bool:
    us = meta.get("us") or {}
    if us.get("fan_geometry"):
        return True
    return str(us.get("probe", "")).lower() == "c10ur" or tuple(frame_shape) in ((160, 512), (320, 256))


def fan_geometry_from_meta(meta: dict[str, Any]):
    from us_scan_convert import FanGeometry

    g = (meta.get("us") or {}).get("fan_geometry") or {}
    return FanGeometry(radius_mm=float(g.get("radius_mm", 59.0)), half_angle_deg=float(g.get("half_angle_deg", 28.0)),
                       depth_mm=float(g.get("depth_mm", 220.0)), flip_lines=bool(g.get("flip_lines", False)))


def letterbox_square(img: np.ndarray, size: int, pad_value: int = 0) -> np.ndarray:
    """(H, W) → (size, size): 등방 축소 후 검정 패딩 (imu_bench/host/us_screen_capture.letterbox_resize 와 같은 규약)."""
    h, w = img.shape
    scale = size / float(max(h, w))
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    try:
        import cv2
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        small = np.asarray(Image.fromarray(img).resize((sw, sh), Image.BILINEAR), dtype=np.uint8)
    out = np.full((size, size), pad_value, np.uint8)
    oy, ox = (size - sh) // 2, (size - sw) // 2
    out[oy:oy + sh, ox:ox + sw] = small[:size - oy, :size - ox]
    return out


class BmodeConverter:
    """세션 하나에 대한 candidate → B-mode 변환기 (필요 없으면 항등)."""

    def __init__(self, meta: dict[str, Any], frame_shape: tuple[int, int], out_size: int = 256,
                 supersample: int = 2):
        self.polar = is_polar_session(meta, frame_shape)
        self.out_size = int(out_size)
        self.supersample = int(supersample)
        self.frame_shape = tuple(frame_shape)
        self._conv = None
        if self.polar:
            from us_scan_convert import ScanConverter

            self.geometry = fan_geometry_from_meta(meta)
            # 부채꼴을 supersample × out_size 높이로 만든 뒤 (폭 ≈ 1.16 × 높이) 면적 평균으로 정방형 레터박스 —
            # 512 깊이 표본이 256 px 로 줄 때 앨리어싱 없이 평균되게 한다. (2026-09-09 낮에 본 "모아레" 의 진짜 원인은
            # 320×256 오배치였다 — 160×512 로 고친 뒤에는 supersample 없이도 줄무늬가 없다. 품질을 위해 2× 는 유지.)
            self._conv = ScanConverter(self.geometry, frame_shape[0], frame_shape[1],
                                       out_h=self.supersample * self.out_size)

    def describe(self) -> dict[str, Any]:
        if not self.polar:
            return {"mode": "identity", "frame_shape": list(self.frame_shape)}
        return {"mode": "scan_convert+letterbox", "frame_shape": list(self.frame_shape), **self._conv.meta(),
                "out_size": self.out_size, "supersample": self.supersample}

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if not self.polar:
            return frame
        fan = self._conv.convert(np.ascontiguousarray(frame, dtype=np.uint8))
        return letterbox_square(fan, self.out_size)

    def convert_all(self, frames: np.ndarray, progress: bool = False) -> np.ndarray:
        if not self.polar:
            return np.asarray(frames)
        n = len(frames)
        out = np.empty((n, self.out_size, self.out_size), np.uint8)
        it = range(n)
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(it, desc="scan-convert", unit="frame")
            except ImportError:
                pass
        for i in it:
            out[i] = self(frames[i])
        return out
