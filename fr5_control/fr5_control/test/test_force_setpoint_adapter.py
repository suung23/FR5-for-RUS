"""force_setpoint_adapter 회귀 테스트 — 디더 위상·복조·clamp·재탐색 트리거.

이 루프는 힘 설정값을 **스스로** 옮긴다. 틀리면 두 방향으로 위험하다 — 경사 부호가
뒤집히면 품질이 나쁜 쪽으로 기어가고, clamp 가 새면 디더를 얹은 설정값이 안전 한계를
넘는다. 둘 다 팬텀에서 느리게 드러나는 종류라 여기서 잡는다.
"""

import math

from fr5_control.force_setpoint_adapter import ForceSetpointAdapter, quality_degraded
import pytest

HZ = 8.0                       # US 프레임률 실측 (probe.yaml us: 8 fps)
PERIOD = 5.0
HALF = PERIOD / 2.0


def _concave(f):
    """§8.4 가 전제하는 단봉 Q(F). 최적 3.0 N."""
    return 1.0 - 0.1 * (f - 3.0) ** 2


def _run(adapter, quality, periods, hz=HZ, t0=0.0):
    """품질을 setpoint 의 함수(또는 t 의 함수)로 주며 주기 수만큼 돌린다.

    ``quality(f, t)`` 를 호출하고, 주기가 닫힌 판정 dict 들을 돌려준다. 마지막 표본은
    정확히 ``periods`` 번째 주기의 첫 표본이라 마지막 주기도 닫힌다.
    """
    n_total = int(round(periods * adapter.period_s * hz))
    results = []
    for n in range(n_total + 1):
        t = t0 + n / hz
        q = quality(adapter.setpoint(t), t)
        r = adapter.update(t, q)
        if r is not None:
            results.append(r)
    return results


def test_setpoint_alternates_with_half_period():
    """앞 반주기 +a, 뒤 반주기 −a, 경계는 뒤쪽 반주기에 속한다."""
    ad = ForceSetpointAdapter(3.0)
    assert ad.period_s == PERIOD and ad.amplitude_n == 0.25
    for t in (0.0, 1.0, 2.499):
        assert ad.setpoint(t) == pytest.approx(3.25)
    for t in (2.5, 3.7, 4.999):
        assert ad.setpoint(t) == pytest.approx(2.75)
    assert ad.setpoint(5.0) == pytest.approx(3.25)          # 한 바퀴 돌아 처음으로
    assert ad.setpoint(7.5) == pytest.approx(2.75)


def test_observation_carries_no_dither_in_f_bar():
    """관측의 f_bar 는 주기 내내 하나의 값이고, 디더는 별도 채널로만 나간다."""
    ad = ForceSetpointAdapter(3.0)
    seen = set()
    for n in range(80):
        t = n / HZ
        obs = ad.observation(t)
        seen.add(obs["f_bar"])
        expected = math.copysign(1.0, ad.setpoint(t) - obs["f_bar"])
        assert obs["dither_sign"] == pytest.approx(expected)
        assert obs["dither_sin"] ** 2 + obs["dither_cos"] ** 2 == pytest.approx(1.0)
    assert seen == {3.0}
    assert ad.observation(PERIOD / 4.0)["dither_sin"] == pytest.approx(1.0)
    assert ad.observation(0.0)["dither_cos"] == pytest.approx(1.0)


def test_concave_quality_converges_to_optimum_within_step_limit():
    """단봉 Q 에서 2.0 → 3.0 N 으로 올라가되 한 주기에 step_max 를 넘지 않는다."""
    ad = ForceSetpointAdapter(2.0)
    prev = ad.f_bar
    for r in _run(ad, lambda f, t: _concave(f), periods=15):
        assert not r["measurement_failed"]
        assert abs(r["f_bar"] - prev) <= ad.step_max + 1e-12
        assert r["f_bar"] >= prev                            # 경사 부호가 맞다
        prev = r["f_bar"]
    assert ad.n_updates == 15 and ad.n_failed == 0
    assert abs(ad.f_bar - 3.0) < 0.15


def test_flat_quality_leaves_f_bar_exactly_unchanged():
    """평탄한 Q 는 경사 0 — f_bar 가 한 비트도 안 움직인다."""
    ad = ForceSetpointAdapter(2.0)
    results = _run(ad, lambda f, t: 0.7, periods=6)
    assert len(results) == 6
    assert all(r["step_n"] == 0.0 and r["gradient"] == 0.0 for r in results)
    assert ad.f_bar == 2.0


def test_all_none_samples_mark_measurement_failed():
    """측정 안 된 표본만 들어오면 실패로 세고 f_bar 는 그대로다."""
    ad = ForceSetpointAdapter(2.0)
    results = _run(ad, lambda f, t: None, periods=3)
    assert len(results) == 3
    assert all(r["measurement_failed"] and r["q_plus"] is None and r["step_n"] == 0.0
               for r in results)
    assert ad.f_bar == 2.0
    assert ad.n_failed == 3 and ad.n_updates == 0


def test_too_few_samples_per_half_is_a_failure():
    """1 Hz 로는 반주기에 유효 표본이 2 개뿐이라 min_samples_per_half(4) 에 못 미친다."""
    ad = ForceSetpointAdapter(2.0)
    results = _run(ad, lambda f, t: _concave(f), periods=2, hz=1.0)
    assert len(results) == 2
    assert all(r["measurement_failed"] for r in results)
    assert results[0]["n_plus"] == 2 and results[0]["n_minus"] == 2
    assert ad.f_bar == 2.0 and ad.n_failed == 2


def test_samples_inside_settle_window_are_excluded():
    """전환 직후 settle_s 안의 표본은 버린다 — 그 구간만 높은 Q 는 경사를 못 만든다."""
    ad = ForceSetpointAdapter(3.0, settle_s=0.5)

    def q_high_only_in_plus_settle(f, t):
        tau = t % HALF
        return 1.0 if (ad.dither_sign(t) > 0 and tau < 0.5) else 0.0

    results = _run(ad, q_high_only_in_plus_settle, periods=3)
    assert all(not r["measurement_failed"] for r in results)
    assert all(r["gradient"] == 0.0 for r in results)
    assert ad.f_bar == 3.0
    assert ad.n_discarded == 3 * 2 * 4 + 1          # 반주기마다 4 표본 + 마지막 경계 표본
    assert results[0]["n_plus"] == 16 and results[0]["n_minus"] == 16


def test_dithered_setpoint_never_leaves_force_limits():
    """위로 계속 밀어도 setpoint ≤ f_max, 아래로 밀어도 ≥ f_min."""
    up = ForceSetpointAdapter(3.0, f_min_n=1.0, f_max_n=5.0)
    _run(up, lambda f, t: f, periods=40)                  # 경사 +1 — 계속 올라간다
    assert up.f_bar == pytest.approx(4.75)
    assert max(up.setpoint(n / 100.0) for n in range(2000)) <= 5.0 + 1e-12

    down = ForceSetpointAdapter(3.0, f_min_n=1.0, f_max_n=5.0)
    _run(down, lambda f, t: -f, periods=40)               # 경사 −1 — 계속 내려간다
    assert down.f_bar == pytest.approx(1.25)
    assert min(down.setpoint(n / 100.0) for n in range(2000)) >= 1.0 - 1e-12


def test_initial_f_bar_is_clamped_so_dither_fits():
    """Stage 1 이 한계 끝값을 주면 디더가 들어갈 만큼 안쪽으로 옮긴다."""
    assert ForceSetpointAdapter(1.0).f_bar == pytest.approx(1.25)
    assert ForceSetpointAdapter(9.0).f_bar == pytest.approx(4.75)


def test_time_going_backwards_is_rejected_and_reset_restarts():
    """시간이 뒤로 가면 ValueError, reset 은 위상과 f_bar 를 새로 잡는다."""
    ad = ForceSetpointAdapter(2.0)
    _run(ad, lambda f, t: _concave(f), periods=2)
    assert ad.f_bar > 2.0 and ad.n_updates == 2
    with pytest.raises(ValueError):
        ad.update(1.0, 0.5)
    ad.update(10.0, 0.5)                                   # 같은 시각은 허용된다

    ad.reset(3.0, t0=100.0)
    assert ad.f_bar == 3.0
    assert ad.n_updates == 0 and ad.n_failed == 0
    assert ad.dither_sign(100.0) == 1.0 and ad.dither_sign(102.5) == -1.0
    assert ad.setpoint(100.0) == pytest.approx(3.25)
    with pytest.raises(ValueError):
        ad.update(99.0, 0.5)                               # t0 앞도 "뒤로 간 것" 이다
    assert ad.update(100.0, 0.5) is None


def test_period_closes_only_once_per_wrap():
    """닫힘 판정은 위상이 감길 때 한 번이고, 그 사이 표본은 None 을 돌려준다."""
    ad = ForceSetpointAdapter(2.0)
    n_none = 0
    n_closed = 0
    for n in range(int(2 * PERIOD * HZ) + 1):
        r = ad.update(n / HZ, 0.5)
        if r is None:
            n_none += 1
        else:
            n_closed += 1
    assert n_closed == 2
    assert n_none == int(2 * PERIOD * HZ) + 1 - 2


def test_invalid_parameters_are_rejected_at_construction():
    """완화가 반주기를 다 먹거나 한계 안에 디더 자리가 없으면 만들 때 막는다."""
    with pytest.raises(ValueError):
        ForceSetpointAdapter(3.0, settle_s=2.5)
    with pytest.raises(ValueError):
        ForceSetpointAdapter(3.0, amplitude_n=0.0)
    with pytest.raises(ValueError):
        ForceSetpointAdapter(3.0, f_min_n=2.9, f_max_n=3.1, amplitude_n=0.25)
    with pytest.raises(ValueError):
        ForceSetpointAdapter(3.0, min_samples_per_half=0)


def test_quality_degraded_boundaries():
    """25 % 열화는 기준의 75 % **미만**에서 참이고, 경계값은 거짓이다."""
    assert quality_degraded(1.0, 0.7499)
    assert not quality_degraded(1.0, 0.75)
    assert not quality_degraded(1.0, 0.76)
    assert not quality_degraded(1.0, 1.0)
    assert quality_degraded(1.0, 0.0)
    assert quality_degraded(1.0, 0.49, fraction=0.5)
    assert not quality_degraded(1.0, 0.5, fraction=0.5)
    assert quality_degraded(1.0, 0.999, fraction=0.0)     # fraction 0 은 "조금이라도"
    assert not quality_degraded(0.0, 0.0)                  # 기준이 0 이면 열화할 것이 없다


def test_replaying_a_recording_is_deterministic():
    """벽시계를 안 읽으므로 같은 입력은 같은 결과를 낸다."""
    runs = []
    for _ in range(2):
        ad = ForceSetpointAdapter(2.0)
        runs.append([r["f_bar"] for r in _run(ad, lambda f, t: _concave(f), periods=5)])
    assert runs[0] == runs[1]
