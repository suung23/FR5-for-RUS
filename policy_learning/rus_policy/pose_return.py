"""시작 자세로 되돌아가기 — 짝지어진 조건 비교의 전제.

    tgt = PoseTarget.parse("-0.4826,-0.3490,0.4156,0.707,0,0,0.707")
    vel6, err = tgt.step(p_now, q_now, max_mm_s=3.0, max_deg_s=3.0)

왜 필요한가
-----------
한 자세에서 여러 조건을 비교하려면 **셋이 같은 자리에서 출발해야** 한다. 그런데 에피소드가
끝나면 로봇은 이미 옮겨져 있다 — 구조상 그럴 수밖에 없다 (2026-09-14 본 시험 ep0001 은
50 mm 를 옮겨 놓고 끝났다). 조작자가 손으로 되돌리면 그 오차가 조건 차이에 섞인다.

얼마나 정확해야 하는가는 실측이 말한다. 첫 실행 자세에서 벗어난 거리와 마스크 보임:

    0 mm → 94 %   ·   9 mm → 90 %   ·   29 mm → 44 %   ·   82 mm → 1 %

**10 mm 안이면 난이도가 같다고 볼 수 있고, 30 mm 면 다른 과제가 된다.**

빔축(z)은 되돌리지 않는다
-------------------------
z 는 힘 조절기의 것이다. 여기서 z 를 밀면 두 주체가 같은 축을 다투고, 그 축은 둘 중
누구의 의도도 아니게 된다. 접촉 깊이는 설정된 힘이 정하는 것이지 기록된 좌표가 정하는
것이 아니다 — 팬텀 용적이 바뀌면 같은 힘이 다른 z 를 낸다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

#: 여기 안에 들어오면 같은 자리로 본다. 실측 근거는 위 도크스트링.
TOL_MM = 3.0
TOL_DEG = 1.5


def _rot(q: Sequence[float]) -> np.ndarray:
    """단위 사원수 (w, x, y, z) → 3×3 회전행렬 (기저 → 프로브 프레임의 역)."""
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], float)


def _mul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """사원수 곱 (w, x, y, z)."""
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], float)


def _conj(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([w, -x, -y, -z], float)


def _axis_angle_deg(q: Sequence[float]) -> np.ndarray:
    """사원수 → 회전벡터 [도]. 짧은 쪽으로 돈다 (q ≡ −q)."""
    q = np.asarray(q, float)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.zeros(3)
    q = q / n
    if q[0] < 0:            # 180° 넘게 돌지 않는다
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-9:
        return np.zeros(3)
    return np.degrees(2.0 * np.arctan2(s, q[0])) * (q[1:] / s)


@dataclass
class PoseTarget:
    """기저 프레임의 목표 자세. 위치 [m], 사원수 (w, x, y, z)."""

    p: np.ndarray
    q: np.ndarray
    tol_mm: float = TOL_MM
    tol_deg: float = TOL_DEG

    @classmethod
    def parse(cls, text: str, **kw) -> "PoseTarget":
        """``x,y,z,qw,qx,qy,qz`` — 기록(decisions.csv)과 같은 순서·단위다."""
        v = [float(s) for s in str(text).replace(" ", "").split(",") if s]
        if len(v) != 7:
            raise ValueError(
                f"자세는 x,y,z,qw,qx,qy,qz 일곱 개여야 한다 (받은 것 {len(v)} 개): {text!r}")
        return cls(p=np.array(v[:3], float), q=np.array(v[3:], float), **kw)

    def error(self, p_now, q_now) -> tuple[float, float]:
        """(위치 오차 [mm], 자세 오차 [°]). **빔축 성분은 빼고 잰다** — z 는 힘의 것이다."""
        e_probe, r_deg = self._error_vectors(p_now, q_now)
        return float(np.linalg.norm(e_probe[:2]) * 1000.0), float(np.linalg.norm(r_deg))

    def _error_vectors(self, p_now, q_now):
        R = _rot(q_now)                       # 프로브 → 기저
        e_base = np.asarray(self.p, float) - np.asarray(p_now, float)
        e_probe = R.T @ e_base                # 기저 → 프로브
        # 프로브 프레임에서의 회전 오차. q_err = q_now⁻¹ ⊗ q_target
        r_deg = _axis_angle_deg(_mul(_conj(q_now), self.q))
        return e_probe, r_deg

    def reached(self, p_now, q_now) -> bool:
        d_mm, d_deg = self.error(p_now, q_now)
        return d_mm <= self.tol_mm and d_deg <= self.tol_deg

    def step(self, p_now, q_now, *, max_mm_s: float, max_deg_s: float,
             k_p: float = 1.5) -> tuple[np.ndarray, tuple[float, float]]:
        """프로브 프레임의 vel6 [mm/s, °/s] 과 (위치 오차 mm, 자세 오차 °).

        빔축 병진은 언제나 0 이다 — 힘 조절기가 그 축을 쥐고 있다.
        """
        e_probe, r_deg = self._error_vectors(p_now, q_now)
        vel = np.zeros(6, float)
        vel[:2] = np.clip(k_p * e_probe[:2] * 1000.0, -max_mm_s, max_mm_s)
        vel[3:] = np.clip(k_p * r_deg, -max_deg_s, max_deg_s)
        d_mm = float(np.linalg.norm(e_probe[:2]) * 1000.0)
        return vel, (d_mm, float(np.linalg.norm(r_deg)))


def parse_optional(text: Optional[str], **kw) -> Optional[PoseTarget]:
    """빈 값이면 None. 인자를 안 주는 것이 정상 경로다 (자세의 첫 에피소드)."""
    if not text:
        return None
    return PoseTarget.parse(text, **kw)
