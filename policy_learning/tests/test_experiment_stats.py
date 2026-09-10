"""파일럿 통계 — Wilson 구간과 본 시험 규모 계산."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_experiment import sample_size_two_proportions, wilson
from run_experiment import build_plan


def test_wilson_never_leaves_the_unit_interval():
    """n=6 에서 정규근사는 음수 하한을 낸다. 파일럿 규모가 정확히 그 크기다.

    p=0 · p=1 에서 Wilson 의 끝은 해석적으로 정확히 0 · 1 이라 부동소수점이 1e-9 만큼
    모자란다 — 구간이 관측 비율을 담는지는 그 허용오차로 본다.
    """
    for k in range(7):
        p, lo, hi = wilson(k, 6)
        assert 0.0 <= lo <= p + 1e-9 and p - 1e-9 <= hi <= 1.0


def test_wilson_handles_the_extremes():
    p, lo, hi = wilson(0, 6)
    assert p == 0.0 and lo == 0.0 and hi > 0.0        # 0/6 이 "절대 없다" 가 되면 안 된다
    p, lo, hi = wilson(6, 6)
    assert p == 1.0 and hi == pytest.approx(1.0) and lo < 1.0
    assert all(x != x for x in wilson(0, 0))          # n=0 은 NaN


def test_sample_size_grows_as_the_gap_shrinks():
    near = sample_size_two_proportions(0.30, 0.40)
    far = sample_size_two_proportions(0.30, 0.70)
    assert near > far > 0


def test_sample_size_is_undefined_without_a_gap():
    assert sample_size_two_proportions(0.3, 0.3) == -1
    assert sample_size_two_proportions(0.0, 0.5) == -1


def test_sample_size_matches_a_known_case():
    """p=0.30 vs 0.60, α=0.05 양측, 검정력 80 % → 팔당 40 대."""
    n = sample_size_two_proportions(0.30, 0.60)
    assert 35 <= n <= 48, n


# ---- 계획 ---------------------------------------------------------------

def test_plan_is_balanced_and_shuffled_within_pose():
    plan = build_plan(6, ["hold", "placebo", "policy"], 1, seed=0)
    assert len(plan) == 18
    for cond in ("hold", "placebo", "policy"):
        assert sum(e["condition"] == cond for e in plan) == 6
    for pose in range(1, 7):
        block = [e["condition"] for e in plan if e["pose"] == pose]
        assert sorted(block) == ["hold", "placebo", "policy"]     # 자세마다 한 번씩
    orders = {tuple(e["condition"] for e in plan if e["pose"] == pose) for pose in range(1, 7)}
    assert len(orders) > 1        # 순서가 자세마다 같으면 조건 효과에 시간 효과가 붙는다


def test_plan_is_reproducible_from_the_seed():
    a = build_plan(4, ["hold", "policy"], 1, seed=7)
    b = build_plan(4, ["hold", "policy"], 1, seed=7)
    assert [e["condition"] for e in a] == [e["condition"] for e in b]
    c = build_plan(4, ["hold", "policy"], 1, seed=8)
    assert [e["condition"] for e in a] != [e["condition"] for e in c]


def test_main_study_plan_is_100_episodes():
    plan = build_plan(25, ["hold", "placebo", "policy", "expert"], 1, seed=0)
    assert len(plan) == 100 and all(e["status"] == "pending" for e in plan)
