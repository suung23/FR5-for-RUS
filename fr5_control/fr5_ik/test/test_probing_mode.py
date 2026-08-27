"""probing_mode 회귀 테스트 — 언제 조이고, 언제 조이지 않는가.

이 전환은 로봇의 속도 상한을 15 배 낮춘다. 틀리는 두 방향의 무게가 다르다.

  * **늦게 조인다** — 자유공간 속도(150 mm/s)로 조직을 미는 시간이 길어진다.
  * **일찍 조인다** — 접근이 느려진다. 성가시지만 위험하지 않다.

그래서 확인 창이 짧고, 잡음 한 번에는 안 넘어가되 진짜 접촉은 놓치지 않는지를
양쪽으로 고정한다.
"""
from fr5_ik.probing_mode import APPROACH, CONTACT_PROBING, ProbingModeSwitch
import pytest

DT = 0.01          # 100 Hz. us_diff_ik 제어 주기다.
ENTER = 8.0        # probe.yaml teleop.contact_probing_force_n


def _feed(sw, force, seconds, dt=DT):
    """일정 힘을 일정 시간 넣는다."""
    mode = sw.mode
    for _ in range(int(round(seconds / dt))):
        mode = sw.update(force, dt)
    return mode


def test_starts_in_approach():
    """세션은 접근 모드로 시작한다."""
    assert ProbingModeSwitch(ENTER).mode == APPROACH


def test_crossing_the_threshold_switches_to_contact_probing():
    """문턱을 넘고 확인 창을 채우면 접촉 프로빙으로 간다."""
    sw = ProbingModeSwitch(ENTER)
    assert _feed(sw, 8.5, 0.1) == CONTACT_PROBING
    assert sw.in_contact_probing


def test_exactly_at_the_threshold_counts_as_crossing():
    """8.0 은 '넘었다' 다. 부등호 방향이 뒤집히면 조이는 시점이 늦어진다."""
    sw = ProbingModeSwitch(ENTER)
    assert _feed(sw, 8.0, 0.1) == CONTACT_PROBING


def test_just_below_threshold_stays_in_approach():
    """7.9 N 은 아직 접근이다. 문턱은 문턱이다."""
    sw = ProbingModeSwitch(ENTER)
    assert _feed(sw, 7.9, 2.0) == APPROACH


def test_short_spike_does_not_switch():
    """충격 스파이크 한 번으로 상한이 바뀌면 접근 중에 로봇이 갑자기 굼떠진다."""
    sw = ProbingModeSwitch(ENTER, confirm_s=0.02)
    _feed(sw, 40.0, 0.01)
    assert sw.mode == APPROACH


def test_confirm_window_must_be_continuous():
    """넘었다 내려갔다를 반복하면 누적되지 않는다."""
    sw = ProbingModeSwitch(ENTER, confirm_s=0.02)
    for _ in range(50):
        _feed(sw, 9.0, 0.01)
        _feed(sw, 0.0, 0.01)
    assert sw.mode == APPROACH


def test_confirm_window_is_short_enough_to_catch_real_contact():
    """실제 접촉은 놓치면 안 된다. 기본 창은 20 ms — 100 Hz 에서 두 표본이다."""
    sw = ProbingModeSwitch(ENTER)
    assert _feed(sw, 8.2, 0.03) == CONTACT_PROBING


def test_switch_is_one_way():
    """힘이 0 으로 떨어져도 접근 모드로 안 돌아간다.

    프로브를 살짝 떼는 것은 접촉 작업의 일부다. 그때마다 상한이 15 배로 뛰면
    같은 손동작에 로봇 반응이 달라진다.
    """
    sw = ProbingModeSwitch(ENTER)
    _feed(sw, 9.0, 0.1)
    assert _feed(sw, 0.0, 5.0) == CONTACT_PROBING
    assert _feed(sw, -3.0, 5.0) == CONTACT_PROBING


def test_tension_never_switches():
    """인장(음수)은 아무리 커도 접촉이 아니다."""
    sw = ProbingModeSwitch(ENTER)
    assert _feed(sw, -50.0, 2.0) == APPROACH


def test_unloaded_bias_does_not_switch():
    """자중이 실린 무부하 값(1.5 N 안팎)으로는 안 넘어간다."""
    sw = ProbingModeSwitch(ENTER)
    for bias in (0.9, 1.2, 1.6):
        sw.reset()
        assert _feed(sw, bias, 3.0) == APPROACH


def test_reset_returns_to_approach():
    """새 세션을 시작할 때만 접근으로 되돌린다."""
    sw = ProbingModeSwitch(ENTER)
    _feed(sw, 9.0, 0.1)
    sw.reset()
    assert sw.mode == APPROACH


def test_confirm_progress_reports_the_window():
    """확인 창의 진행률. 표본 수로 세므로 스텝 단위로 확인한다."""
    sw = ProbingModeSwitch(ENTER, confirm_s=0.05)
    assert sw.confirm_progress == 0.0
    for _ in range(2):                      # 2 x 10 ms = 20 ms
        sw.update(9.0, DT)
    assert sw.confirm_progress == pytest.approx(0.4)
    for _ in range(3):                      # 누적 50 ms — 창을 채운다
        sw.update(9.0, DT)
    assert sw.confirm_progress == 1.0
    assert sw.in_contact_probing


def test_zero_confirm_switches_on_the_first_sample():
    """확인을 끄면 표본 하나로 전환한다 — 되감기 시험용."""
    sw = ProbingModeSwitch(ENTER, confirm_s=0.0)
    assert sw.update(8.1, DT) == CONTACT_PROBING


def test_rejects_nonsense_configuration():
    """0 이하 문턱은 기동하자마자 접촉으로 잡힌다. 만들 때 막는다."""
    with pytest.raises(ValueError):
        ProbingModeSwitch(0.0)
    with pytest.raises(ValueError):
        ProbingModeSwitch(-1.0)
    with pytest.raises(ValueError):
        ProbingModeSwitch(ENTER, confirm_s=-0.01)


def test_replay_is_deterministic():
    """벽시계를 안 읽으므로 같은 입력은 같은 결과를 낸다."""
    trace = [(1.2, DT)] * 200 + [(8.6, DT)] * 10 + [(2.0, DT)] * 200
    runs = []
    for _ in range(2):
        sw = ProbingModeSwitch(ENTER)
        runs.append([sw.update(f, dt) for f, dt in trace])
    assert runs[0] == runs[1]
