"""지령 프레임 ↔ 실제 프레임 — **재서 알아내는** 사상.

    fm = FrameMap.load("runs/frame_map.json")      # 없으면 None
    cmd = fm.to_command(desired_realized)          # 내고 싶은 움직임 → 낼 지령

무엇이 어긋나는가
-----------------
러너는 자세를 ``{ns}/ee_wrt_base`` 에서 읽는다 — 로봇 컨트롤러의 TCP 다. 지령은
``{ns}/desired_twist`` 로 내는데, 그것을 받는 솔버는 **자기 FK 가 계산한 프로브 프레임**
으로 읽는다 (dls_solver.rotation_base_probe). 두 프레임이 같다는 보장이 없다.

2026-09-14 실측 (본 시험 1,077 표본, 지령 회전과 실제 회전을 맞춤):

    최적 직교 사상 (지령 → 실제)      →  z 축 둘레 **90.2°**
      -0.003  -1.000  +0.000
      +1.000  -0.003  -0.000
      +0.000  +0.000  +1.000

``thx`` 를 지령하면 로봇은 ``thy`` 로 돈다. 이것을 모른 채 오차를 줄이려 하면 90° 돌아간
방향으로 나가므로 **단조 발산한다** — 그날 기준 자세 복귀가 24 → 70 mm 로 멀어졌다.

왜 규칙 제어기는 멀쩡했나
-------------------------
``axis_probe`` 가 "지령 축 → 영상 변화" 를 통째로 쟀기 때문이다. 프레임이 돌아가 있어도
그 관계 안에 이미 들어 있다. **프레임을 유도하지 않고 잰 것이 살렸다.** 이 파일도 같은
방식이다 — 규약을 추론하지 않고 기록에서 잰다 (scripts/measure_frame_map.py).

쓰는 곳
-------
기준 자세 복귀(rus_policy.pose_return). 거기서는 "기저 프레임의 목표 자세" 라는 **유도된**
양을 쓰므로 프레임이 맞아야 한다. 규칙 제어기는 쓰지 않는다 — 이득이 이미 지령 축 기준이다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

#: 잔차가 이보다 크면 순수 회전으로 보기 어렵다 — 쓰기 전에 경고한다.
MAX_RESIDUAL = 0.8


@dataclass
class FrameMap:
    """``실제 = R · 지령``. 직교행렬이므로 역은 전치다."""

    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    residual: float = 0.0
    n_samples: int = 0
    source: str = ""

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arccos(np.clip((np.trace(self.R) - 1.0) / 2.0, -1.0, 1.0))))

    @property
    def is_identity(self) -> bool:
        return bool(np.allclose(self.R, np.eye(3), atol=1e-3))

    def to_command(self, desired: np.ndarray) -> np.ndarray:
        """내고 싶은 **실제** 움직임 → 그것을 내려면 줘야 할 지령.

        ``실제 = R · 지령`` 이므로 ``지령 = Rᵀ · 실제`` 다.
        """
        return self.R.T @ np.asarray(desired, float)

    # ---- 파일 ---------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> Optional["FrameMap"]:
        p = Path(path)
        if not p.is_file():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(R=np.array(d["R"], float), residual=float(d.get("residual", 0.0)),
                   n_samples=int(d.get("n_samples", 0)), source=str(d.get("source", str(p))))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "R": [[float(x) for x in row] for row in self.R],
            "residual": self.residual, "n_samples": self.n_samples,
            "angle_deg": self.angle_deg, "source": self.source,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def describe(self) -> str:
        if self.is_identity:
            return "지령 프레임 = 실제 프레임 (사상 없음)"
        return (f"지령→실제 프레임 사상 {self.angle_deg:.1f}° "
                f"(잔차 {self.residual:.2f} · 표본 {self.n_samples} · {self.source})")


def fit(commanded: np.ndarray, realized: np.ndarray) -> FrameMap:
    """``실제 ≈ R · 지령`` 을 만족하는 **직교** R (Procrustes).

    직교로 묶는 이유: 프레임 차이는 회전이지 늘임이 아니다. 최소제곱으로 자유롭게 풀면
    잡음과 축별 추종 열화가 이득 차이로 흡수되어, 사상이 아니라 이득표가 된다.
    """
    C = np.asarray(commanded, float)
    V = np.asarray(realized, float)
    if C.shape != V.shape or C.ndim != 2 or C.shape[0] != 3:
        raise ValueError(f"(3, N) 두 벌이어야 한다: {C.shape} vs {V.shape}")
    if C.shape[1] < 20:
        raise ValueError(f"표본이 너무 적다 ({C.shape[1]}) — 사상을 믿을 수 없다")
    U, _, Vt = np.linalg.svd(V @ C.T)
    R = U @ Vt
    if np.linalg.det(R) < 0:                  # 반사가 아니라 회전이어야 한다
        U = U.copy()
        U[:, -1] *= -1.0
        R = U @ Vt
    res = float(np.linalg.norm(V - R @ C) / (np.linalg.norm(V) + 1e-12))
    return FrameMap(R=R, residual=res, n_samples=int(C.shape[1]))
