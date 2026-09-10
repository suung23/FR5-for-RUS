"""극좌표 candidate 프레임을 부채꼴 B-mode 로 바꾼다 (ROS 쪽 사본).

⚠️ **`imu_bench/host/us_scan_convert.py` 와 같은 코드다.** 두 벌이 있는 이유는 실행
환경이 갈리기 때문이다 — 그쪽은 Windows 수집기가 ROS 없이 임포트하고
(`session_to_images.py` · `policy_learning/rus_policy/bmode.py` 는 `fr5_vision` 을
sys.path 에 넣지 않는다), 이쪽은 `telemetry_bridge` 가 설치된 패키지에서 쓴다.

두 사본이 갈라지면 화면과 저장 데이터의 기하가 조용히 달라진다. 그래서
`test_scan_convert_matches_imu_bench.py` 가 같은 입력에 같은 출력을 내는지 고정한다.
기하를 고칠 일이 있으면 **양쪽을 함께** 고치고 그 테스트로 확인할 것.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

try:
    import cv2
    _HAVE_CV2 = True
except Exception:  # noqa: BLE001
    cv2 = None
    _HAVE_CV2 = False


@dataclass(frozen=True)
class FanGeometry:
    radius_mm: float = 59.0          # transducer 곡률 반경 (apex → 첫 표본)
    half_angle_deg: float = 28.0     # 섹터 반각
    depth_mm: float = 220.0          # 첫 표본 → 마지막 표본
    flip_lines: bool = False         # 라인 순서 반전 (좌우 뒤집힘 검증용)

    def as_dict(self) -> dict:
        return asdict(self)


class ScanConverter:
    """기하·입력/출력 크기별로 remap 테이블을 한 번 만들고 재사용한다 (cv2.remap 이면 프레임당 ~1 ms)."""

    def __init__(self, geometry: FanGeometry, n_lines: int, n_samples: int, out_h: int = 512,
                 out_w: Optional[int] = None):
        self.geo = geometry
        self.n_lines = int(n_lines)
        self.n_samples = int(n_samples)
        R, D, th = geometry.radius_mm, geometry.depth_mm, np.radians(geometry.half_angle_deg)
        y_min, y_max = R * np.cos(th), R + D                           # apex 기준 세로 범위 (mm)
        x_half = (R + D) * np.sin(th)
        self.mm_per_px = (y_max - y_min) / float(out_h)
        if out_w is None:
            out_w = int(round(2 * x_half / self.mm_per_px))
        self.out_h, self.out_w = int(out_h), int(out_w)
        ys = y_min + (np.arange(self.out_h) + 0.5) * self.mm_per_px
        xs = (np.arange(self.out_w) + 0.5) * self.mm_per_px - self.out_w * self.mm_per_px / 2.0
        X, Y = np.meshgrid(xs, ys)
        r = np.hypot(X, Y)
        phi = np.arctan2(X, Y)                                          # 0 = 정면, ± 반각
        sample = (r - R) / D * (self.n_samples - 1)                     # 깊이 표본 인덱스 (열)
        line = (phi + th) / (2 * th) * (self.n_lines - 1)               # 라인 인덱스 (행)
        if geometry.flip_lines:
            line = (self.n_lines - 1) - line
        inside = (r >= R) & (r <= R + D) & (np.abs(phi) <= th)
        self.map_x = np.where(inside, sample, -1).astype(np.float32)    # remap: x = 열(표본), y = 행(라인)
        self.map_y = np.where(inside, line, -1).astype(np.float32)
        self.inside = inside

    def convert(self, polar: np.ndarray) -> np.ndarray:
        if polar.shape != (self.n_lines, self.n_samples):
            raise ValueError(f"polar 는 ({self.n_lines}, {self.n_samples}) 여야 합니다: {polar.shape}")
        if _HAVE_CV2:
            return cv2.remap(polar, self.map_x, self.map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # numpy 최근접 대체
        li = np.clip(np.rint(self.map_y), 0, self.n_lines - 1).astype(int)
        si = np.clip(np.rint(self.map_x), 0, self.n_samples - 1).astype(int)
        out = polar[li, si]
        out[~self.inside] = 0
        return out.astype(np.uint8)

    def meta(self) -> dict:
        return {**self.geo.as_dict(), "n_lines": self.n_lines, "n_samples": self.n_samples,
                "out_hw": [self.out_h, self.out_w], "mm_per_px": self.mm_per_px,
                "apex_xy_px": [self.out_w / 2.0, -self.geo.radius_mm * np.cos(np.radians(self.geo.half_angle_deg)) / self.mm_per_px],
                "note": "candidate: 라인 순서·각도 미검증 (2026-09-09 뷰어 화면 실측 R59/28°/220mm)"}


_CACHE: dict = {}


def polar_to_fan(polar: np.ndarray, geometry: FanGeometry = FanGeometry(), out_h: int = 512) -> np.ndarray:
    key = (geometry, polar.shape, out_h)
    conv = _CACHE.get(key)
    if conv is None:
        conv = _CACHE[key] = ScanConverter(geometry, polar.shape[0], polar.shape[1], out_h)
    return conv.convert(np.ascontiguousarray(polar, dtype=np.uint8))
