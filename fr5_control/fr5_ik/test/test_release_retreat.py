"""Stop 뒤에 프로브가 눌린 채 남지 않는가 — 이탈 후퇴 판정.

2026-09-12: Stop 은 régime 만 풀었고 프로브는 3.39 N 으로 눌린 채 남았다. 그 상태에서
조작자의 지령은 접촉에 맞서고, 서보의 정지 시 지령 선행분이 정책 전 0.50·0.56° 에서
Stop 뒤 2.87·2.77·1.77·3.58° 로 뛴다 (밴드 5°). 화면에는 아무 오류도 없다.
"""

import pytest

from fr5_ik.release_retreat import ReleaseRetreat


def _rr(**kw) -> ReleaseRetreat:
    args = dict(speed_m_s=0.005, until_force_n=0.2, max_s=3.0)
    args.update(kw)
    return ReleaseRetreat(**args)


def test_arms_only_when_the_probe_is_still_pressed():
    rr = _rr()
    assert rr.arm(3.39) and rr.active, "눌린 채 놓았는데 무장하지 않았다"

    rr = _rr()
    assert not rr.arm(0.05), "이미 떨어진 프로브를 더 물릴 이유가 없다"
    assert not rr.active


def test_disabled_never_arms():
    rr = _rr(enabled=False)
    assert not rr.arm(3.39) and not rr.active


def test_retreats_along_minus_z_until_the_force_drops():
    rr = _rr()
    rr.arm(3.39)
    d = rr.update(elapsed_s=0.5, force_n=2.0, wrench_fresh=True, operator_cmd_max=0.0)
    assert d.v_z == pytest.approx(-0.005) and not d.finished

    d = rr.update(elapsed_s=1.0, force_n=0.1, wrench_fresh=True, operator_cmd_max=0.0)
    assert d.v_z is None and d.finished and "종료" in d.reason
    assert not rr.active, "끝난 뒤에도 무장이 남으면 다음 주기에 또 물러난다"


def test_the_operator_wins():
    """조작자가 스타일러스를 움직이면 그 순간 접는다. 이것은 편의이지 안전장치가 아니다."""
    rr = _rr()
    rr.arm(3.39)
    d = rr.update(elapsed_s=0.2, force_n=3.0, wrench_fresh=True, operator_cmd_max=0.01)
    assert d.v_z is None and d.finished and "조작자" in d.reason
    assert not rr.active


def test_stale_wrench_does_not_end_the_retreat_but_the_clock_does():
    """wrench 가 죽은 채로 '힘이 0 이다' 를 믿으면 영원히 물러난다."""
    rr = _rr()
    rr.arm(3.39)
    d = rr.update(elapsed_s=0.5, force_n=0.0, wrench_fresh=False, operator_cmd_max=0.0)
    assert d.v_z == pytest.approx(-0.005), "낡은 wrench 로 종료를 판정했다"

    d = rr.update(elapsed_s=3.5, force_n=0.0, wrench_fresh=False, operator_cmd_max=0.0)
    assert d.v_z is None and d.finished and "상한" in d.reason


def test_travel_is_bounded_by_the_time_limit():
    """상한 시간 × 속도 = 최대 이동거리. 워치독 후퇴와 같은 15 mm 다."""
    rr = _rr()
    assert rr.speed_m_s * rr.max_s == pytest.approx(0.015)


def test_not_armed_means_the_operator_twist_passes_through():
    rr = _rr()
    d = rr.update(elapsed_s=0.0, force_n=3.0, wrench_fresh=True, operator_cmd_max=0.0)
    assert d.v_z is None and not d.finished and d.reason == ""


@pytest.mark.parametrize("bad", [
    dict(speed_m_s=0.0), dict(max_s=0.0), dict(until_force_n=-0.1),
])
def test_rejects_impossible_settings(bad):
    with pytest.raises(ValueError):
        _rr(**bad)
