"""극좌표 candidate (라인 × 깊이 표본) → 부채꼴 B-mode (scan conversion). 표시·후처리용.

C10UR 의 Wi-Fi 프레임은 320 라인 × 256 깊이 표본이다 (`us_protocol.C10UR`). 행 = A-line, 행 시작 = 근거리.
부채꼴 기하는 2026-09-09 뷰어(WirelessUSG) 화면에서 실측했다 — apex·옆변 직선 적합:

    반경 R ≈ 59 mm (뷰어 프로브 종류 "凸阵R60" 과 일치), 반각 ≈ 28°, 깊이 220 mm (뷰어 D:220mm)

⚠️ 라인 순서(0 번이 좌측인지 우측인지) 와 정확한 각도는 candidate 다 — 좌우가 뒤집혀 있을 수 있다. 팬텀에서
검증되기 전까지 저장은 극좌표 원본 그대로 하고, 이 변환은 표시와 후처리에만 쓴다. 기하는 세션 메타에 남긴다.

    fan = polar_to_fan(polar, FanGeometry())        # (H, W) uint8, 검정 패딩
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
