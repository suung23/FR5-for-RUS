"""contact_state 회귀 테스트 — 문턱·이력·걸쇠.

이 판정이 teleop 종결과 freespace 해제를 함께 정하므로, 틀리면 두 가지가 동시에
틀린다. 특히 **접촉을 놓치는 쪽**이 위험하다 — 접촉했는데 접근으로 남으면
150 mm/s 클램프가 걸린 채 조직을 민다.
"""

from fr5_control.contact_state import APPROACH, CONTACT, ContactDetector
import pytest

DT = 0.001                     # 1 kHz. PX6D 실측 주기다.


def _feed(det, f_z, seconds, dt=DT):
    """일정 힘을 일정 시간 넣는다. 마지막 상태를 돌려준다."""
    state = det.state
    for _ in range(int(round(seconds / dt))):
        state = det.update(f_z, dt)
    return state


def test_sign_convention_turns_compression_into_positive_normal_force():
    """probe.yaml 기본값(-1.0)에서 F_z = -7 은 F_n = +7 이다."""
    det = ContactDetector()
    assert det.normal_force(-7.0) == pytest.approx(7.0)
    assert det.normal_force(+7.0) == pytest.approx(-7.0)


def test_positive_sign_convention_flips_it():
    """§4.4 검증에서 부호가 반대로 나오면 파라미터만 바꾼다."""
    det = ContactDetector(normal_force_sign=+1.0)
    assert det.normal_force(+7.0) == pytest.approx(7.0)


def test_bias_is_subtracted_from_normal_force():
    """상수 오프셋은 F_n 에서 빠진다 — 자중의 일부를 지우는 자리다."""
    det = ContactDetector(bias_n=1.4)
    assert det.normal_force(-7.0) == pytest.approx(5.6)


def test_starts_in_approach_with_freespace_allowed():
    """처음은 접근이고 freespace 가 허용된다."""
    det = ContactDetector()
    assert det.state == APPROACH
    assert det.freespace_allowed


def test_crossing_the_threshold_enters_contact():
    """F_n 이 문턱을 넘고 확인 시간을 채우면 접촉이다."""
    det = ContactDetector()
    assert _feed(det, -7.5, 0.02) == CONTACT
    assert not det.freespace_allowed


def test_just_below_threshold_stays_in_approach():
    """6.9 N 은 접촉이 아니다 — 문턱은 문턱이다."""
    det = ContactDetector()
    assert _feed(det, -6.9, 1.0) == APPROACH
    assert det.freespace_allowed


def test_short_spike_does_not_trigger_contact():
    """충격 스파이크 한 번으로 teleop 이 끝나면 안 된다."""
    det = ContactDetector(confirm_s=0.005)
    _feed(det, -20.0, 0.002)                   # 2 ms — 확인 시간 미만
    assert det.state == APPROACH
    assert not det.latched


def test_confirm_window_must_be_continuous():
    """넘었다 내려갔다를 반복하면 누적되지 않는다."""
    det = ContactDetector(confirm_s=0.005)
    for _ in range(50):
        _feed(det, -20.0, 0.003)               # 3 ms 넘고
        _feed(det, 0.0, 0.001)                 # 1 ms 내려가면 처음부터
    assert det.state == APPROACH


def test_hysteresis_keeps_contact_between_the_two_thresholds():
    """진입 후 5 N 으로 떨어져도 접촉이다 — 이탈 문턱은 0.2 N 이다."""
    det = ContactDetector()
    _feed(det, -7.5, 0.02)
    assert _feed(det, -5.0, 1.0) == CONTACT


def test_release_requires_dropping_below_release_threshold():
    """이탈 문턱 아래로 내려가야 접근으로 돌아온다."""
    det = ContactDetector()
    _feed(det, -7.5, 0.02)
    assert _feed(det, -0.1, 0.2) == APPROACH


def test_release_is_slower_than_entry():
    """누르는 중의 순간적 힘 감소로 freespace 가 되살아나면 안 된다."""
    det = ContactDetector(release_confirm_s=0.050)
    _feed(det, -7.5, 0.02)
    assert _feed(det, 0.0, 0.020) == CONTACT   # 20 ms 로는 부족
    assert _feed(det, 0.0, 0.040) == APPROACH  # 누적 60 ms


def test_release_window_must_also_be_continuous():
    """이탈 확인도 끊기면 처음부터 다시 센다."""
    det = ContactDetector(release_confirm_s=0.050)
    _feed(det, -7.5, 0.02)
    for _ in range(20):
        _feed(det, 0.0, 0.030)                 # 30 ms 떨어졌다가
        _feed(det, -5.0, 0.001)                # 다시 눌리면 처음부터
    assert det.state == CONTACT


def test_latch_survives_returning_to_approach():
    """프로브를 떼도 teleop 은 되살아나지 않는다."""
    det = ContactDetector()
    _feed(det, -7.5, 0.02)
    _feed(det, 0.0, 0.2)
    assert det.state == APPROACH
    assert det.latched
    assert not det.freespace_allowed


def test_reset_keeps_the_latch_by_default():
    """상태만 되돌리는 것이 기본이다 — 접촉 이력을 조용히 지우지 않는다."""
    det = ContactDetector()
    _feed(det, -7.5, 0.02)
    det.reset()
    assert det.state == APPROACH
    assert det.latched


def test_reset_can_unlatch_explicitly():
    """새 시행을 시작할 때만 걸쇠를 명시적으로 푼다."""
    det = ContactDetector()
    _feed(det, -7.5, 0.02)
    det.reset(unlatch=True)
    assert det.freespace_allowed


def test_tension_never_triggers_contact():
    """규약 -1.0 에서 F_z 양수는 인장이다. 아무리 커도 접촉이 아니다."""
    det = ContactDetector()
    assert _feed(det, +50.0, 1.0) == APPROACH
    assert not det.latched


def test_unloaded_bias_alone_does_not_trigger():
    """2026-08-24 무부하 실측 -1.2 ~ -1.6 N 은 접촉이 아니다."""
    det = ContactDetector()
    for f_z in (-1.2, -1.4, -1.6):
        det.reset(unlatch=True)
        assert _feed(det, f_z, 2.0) == APPROACH


def test_release_threshold_must_be_below_entry():
    """이력이 없으면 문턱 근처에서 freespace 가 on/off 로 떤다."""
    with pytest.raises(ValueError):
        ContactDetector(enter_n=7.0, release_n=7.0)
    with pytest.raises(ValueError):
        ContactDetector(enter_n=7.0, release_n=8.0)


def test_negative_confirm_time_is_rejected():
    """확인 시간이 음수면 만들 때 막는다."""
    with pytest.raises(ValueError):
        ContactDetector(confirm_s=-0.001)


def test_zero_confirm_time_triggers_on_first_sample():
    """확인을 끄면 표본 하나로 바로 넘어간다 — 되감기 시험에서 쓴다."""
    det = ContactDetector(confirm_s=0.0)
    assert det.update(-7.5, DT) == CONTACT


def test_describe_mentions_state_and_force():
    """요약 한 줄에 상태·걸쇠·힘이 모두 들어간다."""
    det = ContactDetector()
    _feed(det, -7.5, 0.02)
    text = det.describe()
    assert "접촉" in text and "걸쇠" in text and "7.5" in text


def test_replaying_a_recording_is_deterministic():
    """벽시계를 안 읽으므로 같은 입력은 같은 결과를 낸다."""
    trace = [(-0.5, DT)] * 500 + [(-8.0, DT)] * 50 + [(-0.1, DT)] * 500
    runs = []
    for _ in range(2):
        det = ContactDetector()
        runs.append([det.update(f, dt) for f, dt in trace])
    assert runs[0] == runs[1]
