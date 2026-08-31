"""probing_mode 회귀 테스트 — 언제 조이고, 언제 조이지 않고, 언제 푸는가.

이 전환은 로봇의 속도 상한을 15 배 낮춘다. 틀리는 두 방향의 무게가 다르다.

  * **늦게 조인다** — 자유공간 속도(150 mm/s)로 조직을 미는 시간이 길어진다.
  * **일찍 조인다** — 접근이 느려진다. 성가시지만 위험하지 않다.

그래서 진입 확인 창이 짧고, 잡음 한 번에는 안 넘어가되 진짜 접촉은 놓치지 않는지를
양쪽으로 고정한다.

이탈은 반대다. **푸는 쪽이 위험한 방향** 이므로 창이 길고, 문턱이 진입보다 낮다.
2026-08-31 에 진입 문턱이 8 N 에서 한 자릿수로 내려오면서 열린 길이며, 열지 않았다면
세션 초반의 스침 한 번으로 남은 세션 전체가 10 mm/s 에 묶인다.
"""
from fr5_ik.probing_mode import APPROACH, CONTACT_PROBING, ProbingModeSwitch
import pytest

DT = 0.01          # 100 Hz. us_diff_ik 제어 주기다.
ENTER = 2.0        # probe.yaml teleop.contact_probing_force_n
RELEASE = 0.3      # probe.yaml teleop.contact_probing_release_n
RELEASE_S = 0.5    # probe.yaml teleop.contact_probing_release_confirm_s
TARGET = 3.0       # probe.yaml contact_control.target_force_n
BAND = 0.5         # probe.yaml contact_control.deadband_n

#: 확실히 접촉인 힘. 문턱에 **상대적으로** 쓴다 — 절대값을 박아 두면 문턱이 움직일
#: 때마다 시험이 조용히 다른 것을 재게 된다 (8 → 1 → 2 N 을 이미 겪었다).
PRESS = ENTER * 2.0


def _feed(sw, force, seconds, dt=DT):
    """일정 힘을 일정 시간 넣는다."""
    mode = sw.mode
    for _ in range(int(round(seconds / dt))):
        mode = sw.update(force, dt)
    return mode


def _reversible(**kwargs):
    """probe.yaml 기본값 그대로의 전환기."""
    options = dict(
        enter_force_n=ENTER,
        release_force_n=RELEASE,
        release_confirm_s=RELEASE_S,
    )
    options.update(kwargs)
    return ProbingModeSwitch(**options)


def test_starts_in_approach():
    """세션은 접근 모드로 시작한다."""
    assert ProbingModeSwitch(ENTER).mode == APPROACH


def test_crossing_the_threshold_switches_to_contact_probing():
    """문턱을 넘고 확인 창을 채우면 접촉 프로빙으로 간다."""
    sw = ProbingModeSwitch(ENTER)
    assert _feed(sw, ENTER * 1.5, 0.1) == CONTACT_PROBING
    assert sw.in_contact_probing


def test_exactly_at_the_threshold_counts_as_crossing():
    """문턱 그 값은 '넘었다' 다. 부등호가 뒤집히면 조이는 시점이 늦어진다."""
    sw = ProbingModeSwitch(ENTER)
    assert _feed(sw, ENTER, 0.1) == CONTACT_PROBING


def test_just_below_threshold_stays_in_approach():
    """문턱의 90% 는 아직 접근이다. 문턱은 문턱이다."""
    sw = ProbingModeSwitch(ENTER)
    assert _feed(sw, ENTER * 0.9, 2.0) == APPROACH


def test_short_spike_does_not_switch():
    """충격 스파이크 한 번으로 상한이 바뀌면 접근 중에 로봇이 갑자기 굼떠진다."""
    sw = ProbingModeSwitch(ENTER, confirm_s=0.02)
    _feed(sw, 40.0, 0.01)
    assert sw.mode == APPROACH


def test_confirm_window_must_be_continuous():
    """넘었다 내려갔다를 반복하면 누적되지 않는다."""
    sw = ProbingModeSwitch(ENTER, confirm_s=0.02)
    for _ in range(50):
        _feed(sw, PRESS, 0.01)
        _feed(sw, 0.0, 0.01)
    assert sw.mode == APPROACH


def test_confirm_window_is_short_enough_to_catch_real_contact():
    """실제 접촉은 놓치면 안 된다. 기본 창은 20 ms — 100 Hz 에서 두 표본이다."""
    sw = ProbingModeSwitch(ENTER)
    assert _feed(sw, ENTER * 1.2, 0.03) == CONTACT_PROBING


def test_sensor_noise_does_not_switch():
    """문턱은 잡음 위에 있어야 한다.

    PX6D 는 축당 ±0.05 N 이고 보상 잔차가 0.2 N 남짓이다 — 프로브가 위·옆을 볼 때는
    −0.32 N (SD 0.10), 최대 |e| 0.47 N 까지 간다. 그 크기로는 못 넘는다. 넘는다면
    세션이 시작하자마자 접촉 상한에 묶인다. 여기 값들은 절대 크기라 문턱을 따라
    움직이지 않는다 — 잡음은 문턱이 어디든 그 크기다.
    """
    sw = _reversible()
    for level in (0.05, 0.2, 0.35, 0.9):
        sw.reset(unlatch=True)
        assert _feed(sw, level, 3.0) == APPROACH


def test_tension_never_switches():
    """인장(음수)은 아무리 커도 접촉이 아니다.

    ``us_diff_ik`` 가 ``‖F‖`` 에 ``F_n`` 의 부호를 얹어 넣기 때문에, 당겨지는 동안
    이 판정기가 보는 값은 음수다. 크기를 그대로 넣으면 여기서 접촉으로 잡힌다.
    """
    sw = ProbingModeSwitch(ENTER)
    assert _feed(sw, -50.0, 2.0) == APPROACH


# -- 되돌림 -----------------------------------------------------------------


def test_one_way_by_default():
    """이탈 문턱을 안 주면 예전 단방향 거동 그대로다."""
    sw = ProbingModeSwitch(ENTER)
    assert not sw.reversible
    _feed(sw, PRESS, 0.1)
    assert _feed(sw, 0.0, 5.0) == CONTACT_PROBING
    assert _feed(sw, -3.0, 5.0) == CONTACT_PROBING


def test_release_returns_to_approach():
    """힘이 이탈 문턱 아래로 확인 창만큼 지속되면 접근으로 되돌아간다."""
    sw = _reversible()
    assert _feed(sw, PRESS, 0.1) == CONTACT_PROBING
    assert _feed(sw, 0.0, RELEASE_S + 0.1) == APPROACH


def test_release_needs_the_full_window():
    """창을 다 채우기 전에는 안 푼다. 짧게 뜨는 것은 이탈이 아니다."""
    sw = _reversible()
    _feed(sw, PRESS, 0.1)
    assert _feed(sw, 0.0, RELEASE_S - 0.1) == CONTACT_PROBING


def test_release_window_must_be_continuous():
    """한 표본이라도 문턱 위로 올라오면 이탈 창은 처음부터 다시 센다.

    누르는 중의 순간적인 힘 감소로 접근 속도(150 mm/s)가 되살아나는 것이 이 판정에서
    가장 위험한 오작동이다.
    """
    sw = _reversible()
    _feed(sw, PRESS, 0.1)
    for _ in range(20):
        _feed(sw, 0.0, 0.4)      # 창(0.5 s)에 못 미치는 이탈
        _feed(sw, PRESS, 0.01)     # 다시 눌린다
    assert sw.mode == CONTACT_PROBING


def test_holding_at_target_never_releases():
    """유지 밴드 안에서 도는 동안에는 절대 안 푼다.

    목표 3.0 ± 0.5 의 아래끝은 2.5 이고 이탈 문턱은 0.3 이다. 이 여유가 없으면 힘을
    정상적으로 잡고 있는 중에 모드가 접촉과 접근을 오간다.
    """
    sw = _reversible()
    _feed(sw, PRESS, 0.1)
    for force in (TARGET - BAND, TARGET, TARGET + BAND, TARGET - BAND * 0.8):
        assert _feed(sw, force, 5.0) == CONTACT_PROBING


def test_release_then_reenter():
    """풀린 뒤 다시 누르면 다시 잡힌다. 세션을 새로 시작할 필요가 없다."""
    sw = _reversible()
    _feed(sw, PRESS, 0.1)
    assert _feed(sw, 0.0, RELEASE_S + 0.1) == APPROACH
    assert _feed(sw, PRESS, 0.1) == CONTACT_PROBING


def test_latch_survives_release():
    """걸쇠는 모드와 다르다. 되돌아가도 '닿은 적이 있다' 는 남는다."""
    sw = _reversible()
    assert not sw.has_contacted
    _feed(sw, PRESS, 0.1)
    assert sw.has_contacted
    _feed(sw, 0.0, RELEASE_S + 0.1)
    assert sw.mode == APPROACH
    assert sw.has_contacted


def test_release_threshold_must_be_below_enter():
    """이력이 없으면 문턱 근처에서 모드가 떨리고, 그 떨림이 곧 속도 상한의 on/off 다."""
    with pytest.raises(ValueError):
        ProbingModeSwitch(ENTER, release_force_n=ENTER)
    with pytest.raises(ValueError):
        ProbingModeSwitch(ENTER, release_force_n=ENTER + 0.1)
    with pytest.raises(ValueError):
        ProbingModeSwitch(ENTER, release_force_n=RELEASE, release_confirm_s=-0.1)


def test_reset_returns_to_approach():
    """새 세션을 시작할 때만 접근으로 되돌린다."""
    sw = ProbingModeSwitch(ENTER)
    _feed(sw, PRESS, 0.1)
    sw.reset()
    assert sw.mode == APPROACH
    assert sw.has_contacted          # 걸쇠는 기본적으로 남는다
    sw.reset(unlatch=True)
    assert not sw.has_contacted


def test_confirm_progress_reports_the_window():
    """확인 창의 진행률. 표본 수로 세므로 스텝 단위로 확인한다."""
    sw = ProbingModeSwitch(ENTER, confirm_s=0.05)
    assert sw.confirm_progress == 0.0
    for _ in range(2):                      # 2 x 10 ms = 20 ms
        sw.update(PRESS, DT)
    assert sw.confirm_progress == pytest.approx(0.4)
    for _ in range(3):                      # 누적 50 ms — 창을 채운다
        sw.update(PRESS, DT)
    assert sw.confirm_progress == 1.0
    assert sw.in_contact_probing


def test_confirm_progress_tracks_the_release_window_after_contact():
    """접촉 뒤에는 같은 자리가 이탈 창의 진행률을 말한다."""
    sw = _reversible()
    _feed(sw, PRESS, 0.1)
    assert sw.confirm_progress == 0.0
    _feed(sw, 0.0, 0.25)                    # 창(0.5 s)의 절반
    assert sw.confirm_progress == pytest.approx(0.5)


def test_confirm_progress_is_full_when_there_is_no_way_back():
    """단방향 설정에서는 접촉에 들어선 순간 기다릴 창이 없다."""
    sw = ProbingModeSwitch(ENTER)
    _feed(sw, PRESS, 0.1)
    assert sw.confirm_progress == 1.0


def test_zero_confirm_switches_on_the_first_sample():
    """확인을 끄면 표본 하나로 전환한다 — 되감기 시험용."""
    sw = ProbingModeSwitch(ENTER, confirm_s=0.0)
    assert sw.update(ENTER * 1.1, DT) == CONTACT_PROBING


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
    trace = (
        [(0.2, DT)] * 200 + [(1.6, DT)] * 10 + [(2.0, DT)] * 200
        + [(0.0, DT)] * 100 + [(1.4, DT)] * 50
    )
    runs = []
    for _ in range(2):
        sw = _reversible()
        runs.append([sw.update(f, dt) for f, dt in trace])
    assert runs[0] == runs[1]
