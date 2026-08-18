"""Stage 1 힘 탐색의 단위 시험 (DESIGN_NOTES §7).

여기서 지키는 성질은 **argmax 가 아니라 최소 힘을 고른다**는 것이다 (쟁점 3).
이게 깨지면 환자를 필요 이상으로 누르게 된다.
"""
import pytest

from fr5_control.force_search import (
    ForceSearch,
    ForceSearchConfig,
    SearchPhase,
)

DT = 0.01  # 100 Hz


def config(**kwargs) -> ForceSearchConfig:
    base = dict(
        force_min=1.0, force_max=3.0, force_step=1.0,
        settle_hold_s=0.02, settle_timeout_s=0.5, measure_window_s=0.1,
        epsilon=0.03, min_valid_fraction=0.6,
    )
    base.update(kwargs)
    return ForceSearchConfig(**base)


def run(search: ForceSearch, curve, *, valid=True, settle=True, ticks=100000):
    """정착이 즉시 되는 이상적인 플랜트로 탐색을 끝까지 돌린다.

    Args:
        curve: 힘 → 품질 매핑.
        settle: False 면 힘이 목표에 도달하지 않아 정착 시한이 만료된다.
    """
    search.start()
    for _ in range(ticks):
        if not search.active:
            break
        target = search.target_force
        measured = target if settle else target + 10.0
        search.step(DT, measured, curve.get(target, 0.0), valid)
    return search.result()


# --------------------------------------------------------------------------
# 선택 규칙 — 쟁점 3
# --------------------------------------------------------------------------

def test_picks_smallest_force_within_epsilon_of_the_best():
    """1 N 과 3 N 의 품질이 동률이면 **1 N** 을 골라야 한다."""
    search = ForceSearch(config())
    result = run(search, {1.0: 0.88, 2.0: 0.90, 3.0: 0.89})

    assert result.best_quality == pytest.approx(0.90)
    assert result.optimal_force == pytest.approx(1.0)  # argmax 는 2.0


def test_rejects_level_below_epsilon_threshold():
    """동률 구간 밖이면 최소 힘이 되지 못한다."""
    search = ForceSearch(config())
    result = run(search, {1.0: 0.50, 2.0: 0.90, 3.0: 0.89})

    assert result.optimal_force == pytest.approx(2.0)


def test_epsilon_zero_is_strict_argmax_tie_break():
    search = ForceSearch(config(epsilon=0.0))
    result = run(search, {1.0: 0.90, 2.0: 0.90, 3.0: 0.80})

    assert result.optimal_force == pytest.approx(1.0)


def test_documented_example_saves_one_newton():
    """DESIGN_NOTES 도판 04 의 곡선을 그대로 재현한다."""
    curve = {
        1.0: 0.22, 1.5: 0.34, 2.0: 0.48, 2.5: 0.63, 3.0: 0.75, 3.5: 0.83,
        4.0: 0.87, 4.5: 0.885, 5.0: 0.89, 5.5: 0.885, 6.0: 0.87,
        6.5: 0.84, 7.0: 0.79,
    }
    search = ForceSearch(
        config(force_min=1.0, force_max=7.0, force_step=0.5, early_stop_count=99)
    )
    result = run(search, curve)

    assert result.best_quality == pytest.approx(0.89)
    assert result.optimal_force == pytest.approx(4.0)  # argmax 5.0 대비 1.0 N 절약


# --------------------------------------------------------------------------
# 정착과 측정의 분리
# --------------------------------------------------------------------------

def test_measurement_only_starts_after_settling():
    """정착 전에는 MEASURE 로 넘어가지 않는다.

    이 분리가 없으면 아직 도달하지 않은 힘에서 품질을 재게 된다.
    """
    search = ForceSearch(config(settle_hold_s=0.2))
    search.start()

    for _ in range(10):  # 힘이 목표에서 멀다
        search.step(DT, 99.0, 0.9, True)
    assert search.phase is SearchPhase.SETTLE

    for _ in range(30):  # 목표에 도달
        search.step(DT, search.target_force, 0.9, True)
    assert search.phase is SearchPhase.MEASURE


def test_settle_timeout_marks_level_unusable():
    """정착하지 못한 레벨은 점수를 주지 않는다."""
    search = ForceSearch(config(settle_timeout_s=0.1))
    result = run(search, {1.0: 0.9, 2.0: 0.9, 3.0: 0.9}, settle=False)

    assert all(not level.usable for level in result.levels)
    assert all(level.failure == "settle_timeout" for level in result.levels)
    assert result.optimal_force is None
    assert result.failure == "no_usable_level"


def test_settle_streak_resets_when_force_leaves_tolerance():
    """허용 오차를 벗어나면 정착 카운터가 초기화된다."""
    search = ForceSearch(config(settle_hold_s=0.1))
    search.start()

    for _ in range(9):
        search.step(DT, search.target_force, 0.9, True)
    search.step(DT, search.target_force + 5.0, 0.9, True)  # 이탈
    for _ in range(5):
        search.step(DT, search.target_force, 0.9, True)

    assert search.phase is SearchPhase.SETTLE


# --------------------------------------------------------------------------
# 유효 프레임 비율
# --------------------------------------------------------------------------

def test_level_fails_when_too_few_valid_frames():
    """유효 프레임이 60% 미만이면 측정 실패로 친다 (§6.3)."""
    search = ForceSearch(config())
    result = run(search, {1.0: 0.9, 2.0: 0.9, 3.0: 0.9}, valid=False)

    assert all(level.failure == "low_valid_fraction" for level in result.levels)
    assert result.optimal_force is None


def test_invalid_samples_excluded_from_the_mean():
    """무효 표본은 평균을 오염시키지 않아야 한다."""
    search = ForceSearch(config(measure_window_s=0.1))
    search.start([1.0])  # 단일 레벨만 명시
    while search.phase is SearchPhase.SETTLE:
        search.step(DT, 1.0, 0.0, True)

    # 창 경계는 부동소수 누적에 한 틱 민감하므로 틱 수를 세지 않고 끝날 때까지 돈다.
    index = 0
    while search.active:
        # 30% 는 무효이고 품질이 0 — 평균에 들어가면 0.8 밑으로 떨어진다
        valid = index % 10 >= 3
        search.step(DT, 1.0, 0.8 if valid else 0.0, valid)
        index += 1

    level = search.result().levels[0]
    assert level.mean_quality == pytest.approx(0.8)
    assert level.valid_fraction >= 0.6


# --------------------------------------------------------------------------
# 조기 종료와 격자
# --------------------------------------------------------------------------

def test_early_stop_prevents_unnecessary_pressure():
    """품질이 계속 떨어지면 상한까지 올리지 않는다."""
    curve = {1.0: 0.9, 2.0: 0.7, 3.0: 0.5, 4.0: 0.3, 5.0: 0.1}
    search = ForceSearch(
        config(force_max=5.0, early_stop_drop=0.15, early_stop_count=2)
    )
    result = run(search, curve)

    assert result.stopped_early
    assert max(level.force for level in result.levels) < 5.0
    assert result.optimal_force == pytest.approx(1.0)


def test_refine_levels_stay_inside_the_safe_range():
    cfg = config(force_min=1.0, force_max=7.0, refine_span=1.0, refine_step=0.25)

    assert min(cfg.refine_levels(1.0)) >= 1.0
    assert max(cfg.refine_levels(7.0)) <= 7.0


def test_coarse_grid_includes_both_endpoints():
    cfg = config(force_min=1.0, force_max=7.0, force_step=0.5)
    levels = cfg.coarse_levels()

    assert levels[0] == pytest.approx(1.0)
    assert levels[-1] == pytest.approx(7.0)
    assert len(levels) == 13


def test_force_curve_is_retained_for_training():
    """힘–품질 곡선은 Stage 2 의 학습 데이터다. 버리면 안 된다 (§8.4)."""
    search = ForceSearch(config())
    result = run(search, {1.0: 0.5, 2.0: 0.8, 3.0: 0.9})

    assert [level.force for level in result.levels] == [1.0, 2.0, 3.0]
    assert [level.mean_quality for level in result.levels] == pytest.approx([0.5, 0.8, 0.9])


def test_rejects_invalid_config():
    with pytest.raises(ValueError):
        ForceSearchConfig(force_min=5.0, force_max=1.0)
    with pytest.raises(ValueError):
        ForceSearchConfig(epsilon=1.5)
