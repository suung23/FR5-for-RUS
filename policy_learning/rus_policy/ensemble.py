"""실행시 chunk 혼합 — temporal ensembling 이 이봉분포를 0 으로 뭉개지 않게 한다.

``docs/POLICY_LEARNING_MATH.md`` §3.5 (L12 · A7) 의 구현.

ACT 의 표준 temporal ensembling 은 겹치는 chunk 예측들의 **지수가중 평균**이다:

    a_t  =  Σ_i w_i · A^{(t−i)}[i]  /  Σ_i w_i ,      w_i = exp(−m·i)

`i` 는 예측의 나이 (0 = 가장 오래된 것, ACT 원본과 같은 방향 — 오래된 예측에 큰 가중).
관측 잡음에 대해서는 이것이 이득이지만, (3.1) 의 행동분포가 **이봉**이면 평균 연산자가
`+d` 와 `−d` 를 상쇄해 정확히 0 을 만든다 — L6 의 실패가 학습이 아니라 **실행시에** 재현된다.
`Q̂` 모드 선택 (§3.3) 은 이봉성을 *살리려고* 넣은 것이라 그 위에 평균을 얹으면 서로 지운다.

§3.5 의 처방 세 개를 모드로 제공한다:

    mean         표준 ACT. 대조군 — 이봉 입력에서 0 으로 붕괴하는 것을 보이는 데 쓴다
    cluster      처방 2. 평균 **전에** a_y 부호로 클러스터링하고 커밋된 모드 안에서만 평균
    hysteresis   처방 3. cluster + 모드 전환을 chunk 경계로 제한하고 점수 차 임계를 건다

처방 1 (모드 일관성 보너스) 은 혼합이 아니라 **선택** 단계라 여기가 아니라
``model.ActPolicy.select_action`` 의 ``gamma``·``prev_dy`` 에 있다. 세 처방은 배타적이지
않으므로 1 + 2 또는 1 + 3 으로 겹쳐 쓰는 것이 정상이다.

**2026-09-08 — 모드의 출처.** §3.9 의 오프셋 필터 (``offset_filter.py``) 를 채택하면 커밋 모드는
Q̂ 점수가 아니라 필터의 MAP 부호다. ``step(forced_mode=…)`` 로 넘긴다. 사후분포가 매 tick 실현
운동과 면적 관측으로 갱신되므로, Q̂ 점수가 무정보일 때 히스테리시스가 첫 무작위 모드에 **잠기는**
실패 (§9-6) 가 구조적으로 생기지 않는다. ``forced_mode=0`` 은 "필터가 아직 모름" 이고, 그때는
기존 점수 규칙으로 돌아간다.

진단 (§3.5 "진단을 반드시 발행하십시오"): 윈도우 내 `a_y` **부호 전환율**. 이 실패는
조용하다 — 로봇이 제자리에 있을 뿐이고 `Q_seg` 도 나빠지지 않는다. 🟡 초당 1 회를 넘으면
모드 진동이다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

SIGN_AXIS_Y = 1          # (v_x, v_y, ω_z) 중 면외 축
MODES = ("mean", "cluster", "hysteresis", "none")


@dataclass
class EnsembleConfig:
    mode: str = "cluster"           # mean | cluster | hysteresis | none  (§3.5)
    m: float = 0.01                 # ACT 지수가중 w_i = exp(−m·i) 🟡 원본 기본값
    sign_axis: int = SIGN_AXIS_Y    # 부호 모드를 정의하는 축
    deadband_mm: float = 0.2        # chunk 순변위가 이보다 작으면 "모드 없음" (0)
    switch_margin: float = 0.0      # 처방 3: 도전 모드가 커밋 모드를 이 차이 이상 넘어야 전환
    boundary_only: bool = False     # 처방 3: chunk 경계 tick 에서만 전환 허용
    flip_window_s: float = 3.0      # 부호 전환율을 재는 창
    flip_rate_warn_hz: float = 1.0  # 🟡 §3.5 — 초당 1 회 초과면 모드 진동

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"ensemble.mode 는 {'|'.join(MODES)}: {self.mode!r}")


@dataclass
class _Entry:
    tick: int                # 발행 tick
    chunk: np.ndarray        # (k, 3) 스텝 행동
    score: float             # Q̂ 합 등 (없으면 0)
    mode: int                # −1 | 0 | +1


def chunk_mode(chunk: np.ndarray, axis: int, deadband: float, dt: float) -> int:
    """chunk 의 부호 모드. 순변위 (Σ a·dt) 가 deadband 를 넘을 때만 ±1."""
    net = float(np.sum(chunk[:, axis]) * dt)
    if abs(net) < deadband:
        return 0
    return 1 if net > 0 else -1


class TemporalEnsembler:
    """겹치는 chunk 예측을 tick 마다 하나의 행동으로 합친다.

    사용:  매 tick 에 ``push(chunk, score)`` 후 ``step()``. ``step`` 이 이번 tick 의 지령
    ``a`` (3,) 와 진단 dict 를 돌려준다. chunk 는 (k, 3) 스텝 행동이고 행 `i` 는 발행 tick
    으로부터 `i` tick 뒤의 행동이다 (``model.PolicyOutput.a_hat`` 과 같은 규약).
    """

    def __init__(self, k: int, dt: float, cfg: Optional[EnsembleConfig] = None):
        self.k = int(k)
        self.dt = float(dt)
        self.cfg = cfg or EnsembleConfig()
        self.tick = 0
        self.entries: deque[_Entry] = deque()
        self.committed: int = 0          # 현재 커밋된 모드
        self.committed_since: int = 0    # 커밋 tick (경계 판정용)
        self.history: list[tuple[int, int]] = []   # (tick, 방출 행동의 부호)
        self.n_flips = 0

    # ------------------------------------------------------------------ 입력
    def push(self, chunk: np.ndarray, score: float = 0.0) -> None:
        c = np.asarray(chunk, float)
        if c.shape != (self.k, 3):
            raise ValueError(f"chunk 는 (k,3)={self.k, 3} 이어야 합니다: {c.shape}")
        self.entries.append(_Entry(tick=self.tick, chunk=c, score=float(score),
                                   mode=chunk_mode(c, self.cfg.sign_axis, self.cfg.deadband_mm, self.dt)))
        while self.entries and self.tick - self.entries[0].tick >= self.k:
            self.entries.popleft()

    # ------------------------------------------------------------------ 혼합
    def _alive(self) -> tuple[list[_Entry], np.ndarray, np.ndarray]:
        """(살아있는 항목, 각 항목이 이번 tick 에 주는 행동 (n,3), ACT 가중 (n,))."""
        alive = [e for e in self.entries if 0 <= self.tick - e.tick < self.k]
        alive.sort(key=lambda e: e.tick)                     # 오래된 것 먼저 (ACT 규약)
        acts = np.array([e.chunk[self.tick - e.tick] for e in alive]) if alive else np.zeros((0, 3))
        w = np.exp(-self.cfg.m * np.arange(len(alive), dtype=float))
        return alive, acts, w

    def _mode_scores(self, alive: list[_Entry], w: np.ndarray) -> dict[int, float]:
        """모드별 가중 점수. score 가 전부 0 이면 가중 합 (= 표 수) 이 기준이 된다."""
        out: dict[int, float] = {}
        use_score = any(e.score != 0.0 for e in alive)
        for e, wi in zip(alive, w):
            if e.mode == 0:
                continue
            out[e.mode] = out.get(e.mode, 0.0) + wi * (e.score if use_score else 1.0)
        return out

    def _select_mode(self, alive: list[_Entry], w: np.ndarray) -> int:
        """커밋 모드를 갱신한다 (처방 2·3)."""
        scores = self._mode_scores(alive, w)
        if not scores:
            return self.committed
        best = max(scores, key=lambda mo: scores[mo])
        if self.committed == 0:
            return best
        if best == self.committed:
            return self.committed
        # 전환 조건 — 처방 3
        if self.cfg.boundary_only and (self.tick - self.committed_since) % self.k != 0:
            return self.committed
        margin = scores[best] - scores.get(self.committed, 0.0)
        return best if margin > self.cfg.switch_margin else self.committed

    def step(self, forced_mode: Optional[int] = None) -> tuple[np.ndarray, dict[str, Any]]:
        """forced_mode: 외부(오프셋 필터) 가 정한 모드 ±1. None/0 이면 점수 규칙 (처방 2·3)."""
        alive, acts, w = self._alive()
        if not alive:
            self.tick += 1
            return np.zeros(3), {"n_alive": 0, "mode": self.committed, "flip_rate_hz": self.flip_rate_hz}

        if self.cfg.mode == "none":
            a = acts[-1]                                       # 최신 chunk 만 (혼합 없음)
            used = 1
        elif self.cfg.mode == "mean":
            a = (w[:, None] * acts).sum(0) / w.sum()           # 표준 ACT
            used = len(alive)
        else:                                                  # cluster | hysteresis
            if forced_mode is not None and int(forced_mode) != 0:
                new_mode = int(np.sign(forced_mode))
            else:
                new_mode = self._select_mode(alive, w)
            if new_mode != self.committed:
                self.committed, self.committed_since = new_mode, self.tick
            keep = np.array([e.mode == self.committed or e.mode == 0 for e in alive], bool)
            if not keep.any():
                keep = np.ones(len(alive), bool)
            a = (w[keep, None] * acts[keep]).sum(0) / w[keep].sum()
            used = int(keep.sum())

        s = int(np.sign(a[self.cfg.sign_axis])) if abs(a[self.cfg.sign_axis]) > 1e-9 else 0
        if self.cfg.mode in ("none", "mean") and s != 0 and s != self.committed:
            # 이 모드들은 모드를 명시적으로 고르지 않는다. 그래도 **방출된 행동의 부호가 사실상
            # 커밋된 모드**이고, 처방 1 (§3.5) 의 "직전 커밋 모드" 는 그것을 가리킨다.
            self.committed, self.committed_since = s, self.tick
        if s != 0:
            if self.history and self.history[-1][1] != 0 and self.history[-1][1] != s:
                self.n_flips += 1
            self.history.append((self.tick, s))
        diag = {"n_alive": len(alive), "n_used": used, "mode": self.committed,
                "a_y": float(a[self.cfg.sign_axis]), "flip_rate_hz": self.flip_rate_hz}
        self.tick += 1
        return a, diag

    # ------------------------------------------------------------------ 진단
    @property
    def flip_rate_hz(self) -> float:
        """최근 ``flip_window_s`` 안의 부호 전환율 [Hz]. 🟡 1 Hz 초과 = 모드 진동 (§3.5)."""
        win = max(1, int(round(self.cfg.flip_window_s / self.dt)))
        recent = [s for t, s in self.history if self.tick - t < win and s != 0]
        if len(recent) < 2:
            return 0.0
        flips = sum(1 for a, b in zip(recent[:-1], recent[1:]) if a != b)
        return flips / (len(recent) * self.dt)

    @property
    def oscillating(self) -> bool:
        return self.flip_rate_hz > self.cfg.flip_rate_warn_hz


# --------------------------------------------------------------------------- A7 실험용
def bimodal_policy(rng: np.random.RandomState, k: int, dt: float, d_mm: float, vx_mm_s: float = 0.0,
                   p_plus: float = 0.5) -> np.ndarray:
    """학습 없이 (3.1) 의 이봉 행동분포를 내는 policy — §8-7 의 "합성 이봉 policy".

    chunk 전체에 걸쳐 순변위가 ±d 가 되는 등속 행동. 부호는 확률 ``p_plus`` 로 +.
    """
    sign = 1.0 if rng.rand() < p_plus else -1.0
    a = np.zeros((k, 3))
    a[:, 0] = vx_mm_s
    a[:, SIGN_AXIS_Y] = sign * d_mm / (k * dt)
    return a
