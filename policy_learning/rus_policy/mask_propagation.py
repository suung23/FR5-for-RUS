"""직전에 **확신 있게 본** 마스크를 광류로 옮겨 지금 프레임을 메운다.

왜 필요한가 (2026-09-14 실측)
-----------------------------
경계가 흐린 프레임에서 모델이 통째로 놓친다. 사람은 같은 프레임을 두 번 그려도 Dice
0.865 로 일치하니 애매해서가 아니다 (모델은 그 프레임에서 0.301). 그런데 방광은 0.1 초
만에 사라지지 않는다. 또렷한 이웃 프레임의 마스크를 옮겨 오면 놓친 프레임이 살아난다
(실측: 모델 0.000 → 전파 0.814, 다섯 장 평균 0.422 → 0.646).

같은 43 장을 **학습에** 넣어 보았으나 2,308 장 속에 묻혀 효과가 없었다 (0.776 → 0.780).
그래서 학습이 아니라 **실행 시점**에 쓴다.

무엇을 하지 않는가
------------------
* 모델이 이미 본 프레임은 건드리지 않는다. 메우는 것은 **빈 출력**뿐이다.
* 접촉이 끊기면 즉시 버린다. 커플링이 없으면 방광이 찍힐 수 없고, 지금 오탐 0/18 이라는
  성질은 그 구간에서 나온다 — 전파가 그것을 깨면 얻는 것보다 잃는 것이 크다.
* 오래된 것은 안 쓴다. 유지 상한을 넘기면 "예전에 봤다" 는 근거가 아니라 관성이다.
* 광류가 미덥지 않으면(움직임이 크거나 면적이 크게 변하면) 버린다.

전파로 채운 프레임은 **표시해서 돌려준다.** 실제로 본 것과 추정한 것을 뒤섞으면, 판정도
정책도 무엇을 근거로 움직였는지 되짚을 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PropagationConfig:
    """전파를 허용하는 조건. 전부 보수적으로 잡는다."""

    enabled: bool = True
    #: 이 시간 안에 본 마스크만 옮긴다 [s]. 10 fps 에서 0.5 s = 다섯 프레임.
    max_age_s: float = 0.5
    #: 옮긴 마스크가 이만큼은 돼야 쓴다 (면적비). 조각은 버린다.
    min_area: float = 0.004
    #: 원본 대비 면적이 이 범위를 벗어나면 광류를 못 믿는다.
    area_ratio_band: tuple = (0.5, 1.8)
    #: 평균 광류 크기가 이 값을 넘으면 장면이 너무 바뀐 것이다 [px].
    max_flow_px: float = 12.0
    #: 저장할 때 요구하는 확신. 이보다 낮으면 애초에 근거로 삼지 않는다.
    min_store_prob: float = 0.99
    #: 전파 마스크에 실어 주는 확률. 1.0 이 아닌 이유는 아래 주석 참조.
    fill_prob: float = 0.80


@dataclass
class PropagationDecision:
    """한 프레임의 판정."""

    #: 전파를 썼는가.
    used: bool
    #: 왜 썼는지 / 왜 안 썼는지. 기록과 화면에 그대로 나간다.
    reason: str
    #: 전파한 마스크. ``used`` 가 참일 때만 채워진다.
    mask: Optional[np.ndarray] = None


class MaskPropagator:
    """확신 있는 마스크를 하나 들고 있다가, 빈 프레임에 옮겨 준다.

    광류 계산은 밖에서 주입한다 (``flow_fn``). 그래야 이 판정을 OpenCV 없이 시험할 수
    있고, 광류 구현을 바꿔도 조건이 그대로 남는다.
    """

    def __init__(self, config: PropagationConfig | None = None, flow_fn=None) -> None:
        self.cfg = config or PropagationConfig()
        self._flow_fn = flow_fn
        self.reset()

    def reset(self) -> None:
        self._frame: Optional[np.ndarray] = None
        self._mask: Optional[np.ndarray] = None
        self._t: float = 0.0

    def observe(self, frame: np.ndarray, mask: np.ndarray, prob_max: float, t: float) -> None:
        """모델이 스스로 본 프레임. 확신이 충분하면 근거로 저장한다."""
        if not self.cfg.enabled:
            return
        if mask.mean() >= self.cfg.min_area and prob_max >= self.cfg.min_store_prob:
            self._frame, self._mask, self._t = np.asarray(frame), np.asarray(mask), t

    def propagate(self, frame: np.ndarray, t: float, coupled: bool) -> PropagationDecision:
        """빈 출력을 메울 수 있는지 판정한다."""
        if not self.cfg.enabled:
            return PropagationDecision(False, "")
        if self._mask is None:
            return PropagationDecision(False, "근거 없음")
        if not coupled:
            # 접촉이 끊겼다. 여기서 마스크를 만들면 오탐이 된다.
            self.reset()
            return PropagationDecision(False, "접촉 끊김 — 근거를 버린다")
        age = t - self._t
        if age > self.cfg.max_age_s:
            self.reset()
            return PropagationDecision(False, f"근거가 {age:.2f} s 로 오래됐다")
        if self._flow_fn is None:
            return PropagationDecision(False, "광류 없음")

        flow = self._flow_fn(self._frame, np.asarray(frame))
        if flow is None:
            return PropagationDecision(False, "광류 실패")
        mag = float(np.hypot(flow[..., 0], flow[..., 1]).mean())
        if mag > self.cfg.max_flow_px:
            return PropagationDecision(False, f"장면이 너무 움직였다 ({mag:.1f} px)")

        moved = _warp(self._mask, flow)
        area = float(moved.mean())
        if area < self.cfg.min_area:
            return PropagationDecision(False, "옮긴 마스크가 너무 작다")
        ratio = area / max(float(self._mask.mean()), 1e-9)
        lo, hi = self.cfg.area_ratio_band
        if not (lo <= ratio <= hi):
            return PropagationDecision(False, f"면적이 {ratio:.2f} 배로 변했다")
        return PropagationDecision(True, f"직전 {age:.2f} s 전 마스크를 옮겼다", moved)


def _warp(mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """광류로 마스크를 옮긴다. OpenCV 가 있으면 remap, 없으면 최근접 인덱싱."""
    h, w = mask.shape
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    mx = gx - flow[..., 0]
    my = gy - flow[..., 1]
    try:
        import cv2
        return cv2.remap((mask.astype(np.uint8)) * 255, mx.astype(np.float32),
                         my.astype(np.float32), cv2.INTER_NEAREST) > 127
    except ImportError:
        xi = np.clip(np.rint(mx), 0, w - 1).astype(int)
        yi = np.clip(np.rint(my), 0, h - 1).astype(int)
        return mask[yi, xi]


def farneback_flow(src: np.ndarray, dst: np.ndarray):
    """기본 광류. 실측에서 이 파라미터로 놓친 프레임이 0.000 → 0.814 가 됐다."""
    import cv2
    return cv2.calcOpticalFlowFarneback(
        np.asarray(src, np.uint8), np.asarray(dst, np.uint8), None,
        0.5, 3, 21, 3, 5, 1.2, 0)
