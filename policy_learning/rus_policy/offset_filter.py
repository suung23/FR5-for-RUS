"""§3.9 오프셋 필터 — elevational 오프셋 d 를 학습하지 않고 **추정**한다 (2026-09-08 채택).

왜 필터인가 (POLICY_LEARNING_MATH §3.4–3.9, 2026-09-08 검토):
  * ±d 대칭 아래에서 Q̂(o, A) 는 반사실 (같은 o 에 다른 A) 을 본 적이 없어 후보를 순위 매길 근거가 없다.
  * Q_seg 의 면적 플라토는 최적점 근방에서 |d| 에 대해 상수라 애초에 목적함수가 평탄하다.
  * 앙상블 히스테리시스는 Q̂ 가 무정보일 때 첫 무작위 모드에 잠긴다.
  세 문제가 모두 "d 를 관측의 함수로 배우려 한다" 에서 나온다. d 는 스칼라 하나고, 그 동역학과 관측
  모델은 알려져 있다. 그러면 이것은 대칭 지도에서의 1 차원 국소화 문제이고 표준 해법이 있다.

모델 (단위 mm, 시간은 policy tick):
    상태   d      = 프로브 − 목표 (최대 단면) 의 elevational (+y) 오프셋.  보조 상태 R (방광 반축), A₀ (최대 면적)
    예측   d_t    = d_{t−1} + u_t + w,     u_t = 실현된 프로브 elevational 변위 (IMU / J q̇ 평균, §1.2.1 Ã_{t−1})
                                           w ~ N(0, σ_u²),  σ_u = max(floor, c·τ^1.5)   — 라벨 σ 와 같은 식
    관측   y_t    = A₀·h(|d_t|; R) + v,   h = sqrt(1 − (d/R)²)  (타원체 단면),  v ~ N(0, σ_q²)
           y 는 dataset.py 의 Q_area (세션 최대로 정규화한 마스크 면적) 와 같은 양이다.
    사후   격자 (d × R × A₀) 위의 이산 분포.  n_d ≈ 121, n_R = 6, n_A = 3 → 2,178 칸. 연산량은 없다.

행동 규칙 (두 줄):
    Var[d] 작음                 → u* = −E[d]                       수렴: 평균으로 간다
    이봉 (양쪽 부호에 질량)       → u* = −sign(d̂_major)·Δ_probe       큰 봉 쪽 가설을 시험한다
        Δ_probe = min{ Δ ∈ [Δ_min, Δ_max] : |h(|d̂₁ − sΔ|) − h(|d̂₂ − sΔ|)| ≥ probe_snr · σ_q }
    Δ_max 로도 두 가설을 못 가르면 → hold.  두 가설 모두 면적이 못 보는 사각지대 안이다 (서보잉 §5.2).
    이것이 설계가 원래 요구한 "가보고 고친다" 이고, 정보 가치가 규칙에 명시적으로 들어 있다.

이 모듈이 내는 것은 elevational 지령 u* (mm, chunk 순변위) 와 모드 부호뿐이다. ACT 는 (v_x, ω_z) 와
면내 스타일을 맡고, ``ensemble.TemporalEnsembler.step(forced_mode=…)`` 에 이 모듈의 부호를 준다.

한계 (설계 문서 §3.9 "한계" 그대로):
  * y 는 압박 변형 (∂A/∂F_n) 과 호흡에 오염된다. σ_q 와 프로세스 잡음에 넣는다. ⏳ 실측.
  * (d, φ) 2 차원 확장 (ω_z) 은 아직 없다.
  * h 의 단조성은 "최대 단면이 목표" 라는 정의에서 나온다. Q_seg 플라토가 아니라 면적 자체를 쓰는 이유다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .config import OffsetFilterConfig

__all__ = ["OffsetFilter", "OffsetFilterConfig", "FilterAction", "ellipsoid_h", "replay"]


def ellipsoid_h(d: np.ndarray, R: np.ndarray) -> np.ndarray:
    """정규화 단면 h(|d|; R) = sqrt(max(0, 1 − (d/R)²)). 브로드캐스트."""
    x = 1.0 - (np.asarray(d, float) / np.asarray(R, float)) ** 2
    return np.sqrt(np.maximum(x, 0.0))


@dataclass
class FilterAction:
    u_mm: float              # elevational chunk 순변위 지령 (프로브 +y)
    mode: str                # converge | probe | hold
    sign: int                # ensembler forced_mode 로 넘길 부호 (0 = hold)
    d_mean: float
    d_std: float
    p_neg: float
    p_pos: float
    d_major: float
    d_minor: float
    probe_mm: float          # probe 모드에서 쓴 Δ (아니면 0)
    separation: float        # 탐침이 두 가설 사이에 만드는 기대 |Δh|

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class OffsetFilter:
    def __init__(self, cfg: Optional[OffsetFilterConfig] = None):
        self.cfg = cfg or OffsetFilterConfig()
        c = self.cfg
        n = int(round(2 * c.d_range_mm / c.d_step_mm)) + 1
        self.d = np.linspace(-c.d_range_mm, c.d_range_mm, n)
        self.R = np.asarray(c.R_grid_mm, float)
        self.A0 = np.asarray(c.A0_grid, float)
        if self.R.ndim != 1 or self.A0.ndim != 1 or self.R.size == 0 or self.A0.size == 0:
            raise ValueError("offset_filter.R_grid_mm / A0_grid 는 비어 있지 않은 1 차원 목록이어야 합니다")
        # h[d, R] 와 기대 관측 A0·h  → (n_d, n_R, n_A)
        self.h = ellipsoid_h(self.d[:, None], self.R[None, :])
        self.y_exp = self.A0[None, None, :] * self.h[:, :, None]
        self.reset()

    # ------------------------------------------------------------------ 상태
    def reset(self, prior: Optional[np.ndarray] = None) -> None:
        """균일 사전분포 (또는 주어진 (n_d,) 분포를 R·A₀ 에 대해 균일하게 확장)."""
        n_d, n_R, n_A = self.d.size, self.R.size, self.A0.size
        if prior is None:
            p = np.ones((n_d, n_R, n_A), float)
        else:
            pr = np.asarray(prior, float)
            if pr.shape != (n_d,):
                raise ValueError(f"prior 는 (n_d,)={n_d} 여야 합니다: {pr.shape}")
            p = np.maximum(pr, 0.0)[:, None, None] * np.ones((1, n_R, n_A))
        self.p = p / p.sum()
        self.n_updates = 0
        self.n_predicts = 0

    @property
    def marginal_d(self) -> np.ndarray:
        return self.p.sum(axis=(1, 2))

    @property
    def marginal_R(self) -> np.ndarray:
        return self.p.sum(axis=(0, 2))

    @property
    def marginal_A0(self) -> np.ndarray:
        return self.p.sum(axis=(0, 1))

    def stats(self) -> dict[str, float]:
        pd_ = self.marginal_d
        mean = float((pd_ * self.d).sum())
        std = float(np.sqrt(max((pd_ * (self.d - mean) ** 2).sum(), 0.0)))
        neg, pos = self.d < 0, self.d > 0
        p_neg, p_pos = float(pd_[neg].sum()), float(pd_[pos].sum())
        d_neg = float((pd_[neg] * self.d[neg]).sum() / p_neg) if p_neg > 1e-12 else 0.0
        d_pos = float((pd_[pos] * self.d[pos]).sum() / p_pos) if p_pos > 1e-12 else 0.0
        return {"mean": mean, "std": std, "p_neg": p_neg, "p_pos": p_pos, "d_neg": d_neg, "d_pos": d_pos,
                "map": float(self.d[int(np.argmax(pd_))]),
                "R_mean": float((self.marginal_R * self.R).sum()),
                "A0_mean": float((self.marginal_A0 * self.A0).sum())}

    # ------------------------------------------------------------------ 예측 · 갱신
    def process_sigma(self, tau_s: Optional[float] = None) -> float:
        c = self.cfg
        tau = c.process_tau_s if tau_s is None else float(tau_s)
        return max(c.process_floor_mm, c.process_coeff_mm * max(tau, 0.0) ** 1.5)

    def predict(self, u_mm: float, sigma_mm: Optional[float] = None, tau_s: Optional[float] = None) -> None:
        """d ← d + u + w.  u 는 실현된 프로브 변위 (지령이 아니다, L14)."""
        sig = self.process_sigma(tau_s) if sigma_mm is None else max(float(sigma_mm), 1e-6)
        # 전이 행렬 T[new, old] = N(d_new; d_old + u, σ²), 열 정규화 (격자 밖으로 나가는 질량은 끝에 남긴다)
        diff = self.d[:, None] - (self.d[None, :] + float(u_mm))
        T = np.exp(-0.5 * (diff / sig) ** 2)
        T /= np.maximum(T.sum(axis=0, keepdims=True), 1e-300)
        n_d = self.d.size
        flat = self.p.reshape(n_d, -1)
        self.p = (T @ flat).reshape(self.p.shape)
        self.p /= self.p.sum()
        self.n_predicts += 1

    def update(self, y: Optional[float], sigma_q: Optional[float] = None) -> None:
        """y = 정규화 면적 (dataset.Q_area 와 같은 정의). None 이면 건너뛴다."""
        if y is None or not np.isfinite(y):
            return
        sq = self.cfg.sigma_q if sigma_q is None else max(float(sigma_q), 1e-6)
        ll = -0.5 * ((float(y) - self.y_exp) / sq) ** 2
        ll -= ll.max()
        self.p = self.p * np.exp(ll)
        s = self.p.sum()
        if not np.isfinite(s) or s <= 0:
            self.reset()
            return
        self.p /= s
        self.n_updates += 1

    # ------------------------------------------------------------------ 행동
    def _separation(self, d1: float, d2: float, delta: float, sign: float) -> float:
        R, A0 = self.stats()["R_mean"], self.stats()["A0_mean"]
        return float(A0 * abs(ellipsoid_h(abs(d1 + sign * delta), R) - ellipsoid_h(abs(d2 + sign * delta), R)))

    def action(self) -> FilterAction:
        c = self.cfg
        st = self.stats()
        mean, std = st["mean"], st["std"]
        bimodal = st["p_neg"] >= c.bimodal_min_mass and st["p_pos"] >= c.bimodal_min_mass
        if std < c.converge_std_mm or not bimodal:
            u = float(np.clip(-mean, -c.max_probe_mm, c.max_probe_mm))
            sign = int(np.sign(u)) if abs(u) >= 0.5 * c.min_probe_mm else 0
            return FilterAction(u_mm=u, mode="converge", sign=sign, d_mean=mean, d_std=std,
                                p_neg=st["p_neg"], p_pos=st["p_pos"], d_major=mean, d_minor=mean,
                                probe_mm=0.0, separation=0.0)
        # 이봉: 큰 봉 쪽 가설을 시험한다.  u = −sign(d_major)·Δ  (그 가설이 맞으면 |d| 가 줄고, 틀리면 커진다)
        if st["p_pos"] >= st["p_neg"]:
            d_major, d_minor = st["d_pos"], st["d_neg"]
        else:
            d_major, d_minor = st["d_neg"], st["d_pos"]
        s = -float(np.sign(d_major)) if d_major != 0 else -1.0
        need = c.probe_snr * c.sigma_q
        deltas = np.arange(c.min_probe_mm, c.max_probe_mm + 1e-9, c.d_step_mm)
        chosen, sep = None, 0.0
        for delta in deltas:
            sep = self._separation(d_major, d_minor, float(delta), s)
            if sep >= need:
                chosen = float(delta)
                break
        if chosen is None:
            return FilterAction(u_mm=0.0, mode="hold", sign=0, d_mean=mean, d_std=std,
                                p_neg=st["p_neg"], p_pos=st["p_pos"], d_major=d_major, d_minor=d_minor,
                                probe_mm=0.0, separation=sep)
        u = s * chosen
        return FilterAction(u_mm=u, mode="probe", sign=int(np.sign(u)), d_mean=mean, d_std=std,
                            p_neg=st["p_neg"], p_pos=st["p_pos"], d_major=d_major, d_minor=d_minor,
                            probe_mm=chosen, separation=sep)

    def forced_mode(self) -> int:
        """ensemble.TemporalEnsembler.step(forced_mode=…) 에 줄 부호."""
        return self.action().sign


# --------------------------------------------------------------------------- 오프라인 재생
def replay(filter_: OffsetFilter, u_seq: np.ndarray, y_seq: np.ndarray,
           tau_seq: Optional[np.ndarray] = None) -> list[dict[str, Any]]:
    """실현 변위열 u_t 와 면적열 y_t (chunk 격자) 로 필터를 재생한다 — 실세션 검증용 (§8-6 대체).

    각 tick: predict(u_t) → update(y_t) → stats/action 기록.  u 의 첫 원소는 0 이어도 된다.
    """
    u_seq = np.asarray(u_seq, float)
    y_seq = np.asarray(y_seq, float)
    if u_seq.shape != y_seq.shape:
        raise ValueError("u_seq 와 y_seq 의 길이가 다릅니다")
    out = []
    for t in range(u_seq.size):
        tau = None if tau_seq is None else float(tau_seq[t])
        if t > 0 or u_seq[t] != 0.0:
            filter_.predict(float(u_seq[t]), tau_s=tau)
        filter_.update(None if not np.isfinite(y_seq[t]) else float(y_seq[t]))
        row = {"t": t, "u": float(u_seq[t]), "y": float(y_seq[t]), **filter_.stats()}
        row.update({f"act_{k}": v for k, v in filter_.action().as_dict().items()})
        out.append(row)
    return out
