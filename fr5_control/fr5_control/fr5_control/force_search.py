"""Stage 1 접촉력 격자 탐색 (DESIGN_NOTES §7).

위치 탐색이 아니라 **힘 탐색**이다. 탐색 변수는 스칼라 ``F_n`` 하나이고, 각 레벨에서
홀드한 뒤 **구간 평균** ``Q̄(F)`` 로 판정한다. 선택 규칙은 argmax 가 아니다:

    F* = min{ F : Q̄(F) ≥ (1−ε)·max Q̄ }

"최댓값과 통계적으로 구분되지 않는 가장 작은 힘". 환자 부담을 줄이는 방향으로 의도적으로
편향되어 있다 (쟁점 3).

## 정착과 측정을 분리한다

감쇠 지배형 admittance 를 강성 ``k`` 인 조직에 물리면 ``τ = B_z / k`` 다.
``B_z = 1000 N·s/m``, 연부조직 ``k ≈ 2000 N/m`` 이면 τ = 0.5 s, 95% 정착에 1.5 s.
홀드 창을 1.0 s 로 두고 그 전체를 평균하면 **아직 도달하지 않은 힘에서 품질을 재게 된다.**

그래서 각 레벨은 두 구간으로 나뉜다:

    SETTLE   |F_n − F*| < tol 이 settle_hold 만큼 지속될 때까지 (상한 settle_timeout)
    MEASURE  measure_window 동안 품질 표본 수집

정착에 실패한 레벨은 점수를 주지 않고 **측정 실패**로 기록한다.

ROS 에 의존하지 않는다. 램프 자체는 admittance 의 슬루 제한이 수행하므로 여기서는
목표 레벨만 내면 된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "SearchPhase",
    "ForceSearchConfig",
    "LevelResult",
    "SearchResult",
    "ForceSearch",
]


class SearchPhase(Enum):
    """탐색기의 내부 단계."""

    IDLE = "idle"
    SETTLE = "settle"
    MEASURE = "measure"
    DONE = "done"


@dataclass
class ForceSearchConfig:
    """격자와 판정 기준 (DESIGN_NOTES §7.2).

    Attributes:
        force_min: 격자 하한 [N]. 접촉을 확인할 수 있는 최소.
        force_max: 격자 상한 [N]. 안전 한계와 같아야 한다.
        force_step: 격자 간격 [N].
        refine_span: Stage 1b 재탐색 폭 [N]. 1a 결과 ±이 값.
        refine_step: Stage 1b 격자 간격 [N].
        settle_tolerance_n: 정착 판정 허용 오차 [N].
        settle_hold_s: 허용 오차 안에 머물러야 하는 시간 [s].
        settle_timeout_s: 정착 대기 상한 [s]. 초과하면 측정 실패.
        measure_window_s: 품질 표본 수집 시간 [s].
        min_valid_fraction: 창 안에서 유효 프레임이 차지해야 할 최소 비율.
        epsilon: 품질 동률 허용치. `(1−ε)·max` 이상이면 최댓값과 구분하지 않는다.
        early_stop_drop: 진행 최댓값 대비 이만큼 낮으면 열화로 본다.
        early_stop_count: 열화가 연속 이만큼이면 상승을 멈춘다.
    """

    force_min: float = 1.0
    force_max: float = 7.0
    force_step: float = 0.5
    refine_span: float = 1.0
    refine_step: float = 0.25
    settle_tolerance_n: float = 0.05
    settle_hold_s: float = 0.2
    settle_timeout_s: float = 3.0
    measure_window_s: float = 1.0
    min_valid_fraction: float = 0.6
    epsilon: float = 0.03
    early_stop_drop: float = 0.15
    early_stop_count: int = 3

    def __post_init__(self) -> None:
        if self.force_min <= 0 or self.force_max <= self.force_min:
            raise ValueError("force_min < force_max 이고 둘 다 양수여야 한다")
        if self.force_step <= 0 or self.refine_step <= 0:
            raise ValueError("격자 간격은 0보다 커야 한다")
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError(f"epsilon 은 [0, 1) 이어야 한다, {self.epsilon}")
        if not 0.0 < self.min_valid_fraction <= 1.0:
            raise ValueError("min_valid_fraction 은 (0, 1] 이어야 한다")
        if self.measure_window_s <= 0 or self.settle_timeout_s <= 0:
            raise ValueError("시간 창은 0보다 커야 한다")

    def coarse_levels(self) -> list[float]:
        """Stage 1a 전체 격자."""
        return self._levels(self.force_min, self.force_max, self.force_step)

    def refine_levels(self, center: float) -> list[float]:
        """Stage 1b 국소 격자. 전체 재탐색은 시간 낭비다."""
        low = max(self.force_min, center - self.refine_span)
        high = min(self.force_max, center + self.refine_span)
        return self._levels(low, high, self.refine_step)

    @staticmethod
    def _levels(low: float, high: float, step: float) -> list[float]:
        levels, value = [], low
        # 부동소수 누적 대신 곱셈으로 만들어 마지막 레벨이 새지 않게 한다.
        count = int(round((high - low) / step))
        for index in range(count + 1):
            value = low + index * step
            if value <= high + 1e-9:
                levels.append(round(value, 6))
        return levels


@dataclass
class LevelResult:
    """한 힘 레벨의 측정 결과."""

    force: float
    mean_quality: float | None      # None 이면 측정 실패
    valid_fraction: float
    samples: int
    failure: str | None = None      # "settle_timeout" | "low_valid_fraction"

    @property
    def usable(self) -> bool:
        return self.mean_quality is not None


@dataclass
class SearchResult:
    """탐색 종료 시 산출물.

    ``levels`` 는 힘–품질 곡선 그 자체다. **버리지 말고 에피소드에 기록해야 한다** —
    Stage 2 에서 policy 가 `F_n*` 를 이어받으려면 이 곡선이 학습 데이터다 (§8.4).
    """

    optimal_force: float | None
    best_quality: float | None
    levels: list[LevelResult] = field(default_factory=list)
    stopped_early: bool = False
    failure: str | None = None      # "no_usable_level"


class ForceSearch:
    """격자를 훑으며 힘–품질 곡선을 만들고 최소 힘을 고른다.

    시간은 밖에서 ``dt`` 로 주입한다. ROS 타이머든 시험 코드든 같은 방식으로 돈다.
    """

    def __init__(self, config: ForceSearchConfig | None = None) -> None:
        self.config = config or ForceSearchConfig()
        self._phase = SearchPhase.IDLE
        self._levels: list[float] = []
        self._index = 0
        self._elapsed = 0.0
        self._within_tolerance = 0.0
        self._sum_quality = 0.0
        self._valid_samples = 0
        self._total_samples = 0
        self._results: list[LevelResult] = []
        self._result: SearchResult | None = None
        self._degraded_streak = 0
        self._stopped_early = False

    # -- 상태 조회 -------------------------------------------------------

    @property
    def phase(self) -> SearchPhase:
        return self._phase

    @property
    def active(self) -> bool:
        return self._phase in (SearchPhase.SETTLE, SearchPhase.MEASURE)

    @property
    def target_force(self) -> float:
        """지금 지령해야 할 setpoint. 탐색 중이 아니면 0."""
        if not self._levels or self._index >= len(self._levels):
            return 0.0
        return self._levels[self._index]

    def result(self) -> SearchResult | None:
        """완료 시 결과, 그 외에는 ``None``."""
        return self._result

    # -- 진행 -----------------------------------------------------------

    def start(self, levels: list[float] | None = None) -> None:
        """탐색 시작. ``levels`` 를 주지 않으면 전체 격자(Stage 1a)."""
        self.__init__(self.config)  # 내부 상태 전부 초기화
        self._levels = list(levels) if levels else self.config.coarse_levels()
        if not self._levels:
            raise ValueError("탐색할 힘 레벨이 없다")
        self._phase = SearchPhase.SETTLE

    def start_refine(self, center: float) -> None:
        """Stage 1b — 1a 결과 주변만 촘촘히 다시 훑는다."""
        self.start(self.config.refine_levels(center))

    def step(self, dt: float, force_now: float, quality: float, valid: bool) -> float:
        """한 틱 진행하고 지령할 힘 setpoint 를 돌려준다.

        Args:
            dt: 경과 시간 [s].
            force_now: 측정 접촉 법선력 [N].
            quality: 목적함수 표본 (Stage 1a 는 ``Q_raw``, 1b 는 ``Q_seg``).
            valid: 이 표본을 평균에 넣어도 되는가.
        """
        if not self.active:
            return self.target_force

        cfg = self.config
        target = self._levels[self._index]
        self._elapsed += dt

        if self._phase is SearchPhase.SETTLE:
            if abs(force_now - target) <= cfg.settle_tolerance_n:
                self._within_tolerance += dt
            else:
                self._within_tolerance = 0.0

            if self._within_tolerance >= cfg.settle_hold_s:
                self._enter_measure()
            elif self._elapsed >= cfg.settle_timeout_s:
                self._close_level(failure="settle_timeout")
            return target

        # MEASURE
        self._total_samples += 1
        if valid:
            self._valid_samples += 1
            self._sum_quality += float(quality)
        if self._elapsed >= cfg.measure_window_s:
            self._close_level()
        return target

    # -- 내부 -----------------------------------------------------------

    def _enter_measure(self) -> None:
        self._phase = SearchPhase.MEASURE
        self._elapsed = 0.0
        self._sum_quality = 0.0
        self._valid_samples = 0
        self._total_samples = 0

    def _close_level(self, failure: str | None = None) -> None:
        """현재 레벨을 마감하고 다음으로 넘어가거나 탐색을 끝낸다."""
        cfg = self.config
        force = self._levels[self._index]
        fraction = (
            self._valid_samples / self._total_samples if self._total_samples else 0.0
        )

        if failure is None and fraction < cfg.min_valid_fraction:
            failure = "low_valid_fraction"

        mean = (
            self._sum_quality / self._valid_samples
            if failure is None and self._valid_samples
            else None
        )
        self._results.append(
            LevelResult(
                force=force,
                mean_quality=mean,
                valid_fraction=fraction,
                samples=self._total_samples,
                failure=failure,
            )
        )

        if mean is not None and self._is_degrading(mean):
            self._degraded_streak += 1
        elif mean is not None:
            self._degraded_streak = 0

        if self._degraded_streak >= cfg.early_stop_count:
            self._stopped_early = True
            self._finish()
            return

        self._index += 1
        if self._index >= len(self._levels):
            self._finish()
            return

        self._phase = SearchPhase.SETTLE
        self._elapsed = 0.0
        self._within_tolerance = 0.0

    def _is_degrading(self, mean: float) -> bool:
        """진행 최댓값 대비 충분히 낮아졌는가 (조기 종료 판정)."""
        best = self._running_best()
        if best is None or best <= 0.0:
            return False
        return mean < best * (1.0 - self.config.early_stop_drop)

    def _running_best(self) -> float | None:
        usable = [r.mean_quality for r in self._results if r.usable]
        return max(usable) if usable else None

    def _finish(self) -> None:
        """선택 규칙을 적용한다 — argmax 가 아니라 동률 구간의 왼쪽 끝."""
        self._phase = SearchPhase.DONE
        usable = [r for r in self._results if r.usable]

        if not usable:
            self._result = SearchResult(
                optimal_force=None,
                best_quality=None,
                levels=list(self._results),
                stopped_early=self._stopped_early,
                failure="no_usable_level",
            )
            return

        best = max(r.mean_quality for r in usable)
        threshold = best * (1.0 - self.config.epsilon)
        # 격자 순서대로 훑으므로 첫 번째로 문턱을 넘는 레벨이 곧 최소 힘이다.
        optimal = min(r.force for r in usable if r.mean_quality >= threshold)

        self._result = SearchResult(
            optimal_force=optimal,
            best_quality=best,
            levels=list(self._results),
            stopped_early=self._stopped_early,
        )
