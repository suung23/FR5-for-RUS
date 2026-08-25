"""px6d_axis_id 의 판정 로직 회귀 테스트 — 기준선 빼기와 축 지목.

시리얼과 사람 손은 여기서 다루지 않는다. 대신 표본이 들어왔을 때 **어느 채널을
어느 부호로 지목하는가** 를 고정한다. 이 판정이 곧 ``normal_force_sign`` 이 되고,
부호가 뒤집히면 admittance 가 발산한다 (DESIGN_NOTES §4.4).
"""

from fr5_control.px6d_axis_id import format_report, interpret, summarise
from fr5_control.px6d_protocol import AXIS_ORDER
import pytest


def _const(values, n=50):
    """같은 값이 n 번 들어온 구간."""
    return [tuple(values)] * n


def test_baseline_is_subtracted():
    """기준선이 0 이 아니어도 이탈만 남아야 한다 — 자중이 실린 상태가 정상이다."""
    base = _const((0.0, 0.0, -1.5, 0.0, 0.0, 0.0))
    hold = _const((0.0, 0.0, -6.5, 0.0, 0.0, 0.0))
    deltas, winner, _ = summarise(base, hold)
    assert winner == "Fz"
    assert deltas[2] == pytest.approx(-5.0)
    assert all(d == pytest.approx(0.0) for i, d in enumerate(deltas) if i != 2)


def test_sign_follows_the_larger_excursion_not_the_last_sample():
    """누르고 놓는 반동이 반대 부호로 잡혀도 부호가 뒤집히면 안 된다."""
    base = _const((0.0,) * 6)
    hold = _const((0.0, 0.0, -5.0, 0.0, 0.0, 0.0), 40) + \
        _const((0.0, 0.0, +1.0, 0.0, 0.0, 0.0), 10)
    deltas, winner, _ = summarise(base, hold)
    assert winner == "Fz"
    assert deltas[2] == pytest.approx(-5.0)


def test_positive_and_negative_pushes_are_distinguished():
    """반대 방향으로 눌렀으면 부호도 반대여야 한다."""
    base = _const((0.0,) * 6)
    up, _, _ = summarise(base, _const((0.0, 0.0, +4.0, 0.0, 0.0, 0.0)))
    down, _, _ = summarise(base, _const((0.0, 0.0, -4.0, 0.0, 0.0, 0.0)))
    assert up[2] > 0 and down[2] < 0


def test_force_and_moment_do_not_compete_for_separation():
    """단위가 다르므로 분리비는 같은 종류 안에서만 잰다.

    힘 5 N 과 모멘트 0.05 N·m 를 한 줄로 비교하면 모멘트 시행이 늘 '힘에 밀린' 것으로
    보인다. 그러면 멀쩡한 시행이 분리 실패로 잡힌다.
    """
    base = _const((0.0,) * 6)
    hold = _const((0.1, 0.05, 0.05, 0.50, 0.02, 0.01))     # Mx 가 주인공
    deltas, winner, ratio = summarise(base, hold)
    assert winner == "Mx"
    assert ratio == pytest.approx(0.50 / 0.02)             # My 와 겨룬다, Fx 가 아니라


def test_separation_is_infinite_when_others_are_silent():
    """다른 축이 조용하면 분리는 완전하다 — 0 으로 나누지 않는다."""
    base = _const((0.0,) * 6)
    deltas, winner, ratio = summarise(base, _const((5.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    assert winner == "Fx"
    assert ratio == float("inf")


def test_empty_samples_do_not_raise():
    """표본이 없으면 조용히 비운다 — 중단된 시행이 예외로 전체를 날리면 안 된다."""
    assert summarise([], []) == (None, None, None)
    assert summarise(_const((0.0,) * 6), []) == (None, None, None)


def _result(trial, deltas, winner, separation=99.0):
    """run_trial 이 돌려주는 모양의 결과 하나."""
    return {"trial": trial, "instruction": "", "n_baseline": 10, "n_hold": 10,
            "deltas": list(deltas), "winner": winner, "separation": separation}


def test_interpret_reads_the_sign_convention_off_the_z_trial():
    """압축이 Fz 음수로 잡히면 normal_force_sign 은 -1 이어야 한다."""
    res = [_result("Fz+", (0.0, 0.0, -5.0, 0.0, 0.0, 0.0), "Fz")]
    text = " ".join(interpret(res))
    assert "음수" in text
    assert "-1.0" in text


def test_interpret_flags_positive_compression_as_plus_one():
    """압축이 Fz 양수면 부호 규약이 +1 이다 — 현재 가정과 반대인 경우."""
    res = [_result("Fz+", (0.0, 0.0, +5.0, 0.0, 0.0, 0.0), "Fz")]
    text = " ".join(interpret(res))
    assert "양수" in text
    assert "+1.0" in text


def test_interpret_warns_when_penetration_axis_is_not_fz():
    """침투축이 Fz 가 아니면 배정 자체를 의심해야 한다."""
    res = [_result("Fz+", (5.0, 0.0, 0.0, 0.0, 0.0, 0.0), "Fx")]
    text = " ".join(interpret(res))
    assert "AXIS_ORDER" in text


def test_interpret_warns_on_weak_separation():
    """축이 갈리지 않은 시행은 조용히 통과시키지 않는다."""
    res = [_result("Fx+", (5.0, 4.0, 0.0, 0.0, 0.0, 0.0), "Fx", separation=1.25)]
    assert any("분리비" in line for line in interpret(res))


def test_interpret_warns_when_two_trials_pick_the_same_channel():
    """두 방향이 한 채널을 지목하면 배정을 확정할 수 없다."""
    res = [_result("Fx+", (5.0, 0.0, 0.0, 0.0, 0.0, 0.0), "Fx"),
           _result("Fy+", (4.0, 0.0, 0.0, 0.0, 0.0, 0.0), "Fx")]
    assert any("같은 채널" in line for line in interpret(res))


def test_report_has_one_line_per_trial_plus_header():
    """표 모양이 흔들리면 눈으로 읽는 판단이 어긋난다."""
    res = [_result("Fz+", (0.0, 0.0, -5.0, 0.0, 0.0, 0.0), "Fz"),
           _result("Fx+", (5.0, 0.0, 0.0, 0.0, 0.0, 0.0), "Fx")]
    lines = format_report(res).split("\n")
    assert len(lines) == len(res) + 2                      # 머리글 + 구분선
    for axis in AXIS_ORDER:
        assert axis in lines[0]


def test_report_survives_a_trial_with_no_samples():
    """중단된 시행이 표 전체를 깨뜨리면 안 된다."""
    res = [{"trial": "Fy+", "instruction": "", "n_baseline": 0, "n_hold": 0,
            "deltas": None, "winner": None, "separation": None}]
    assert "표본 없음" in format_report(res)
