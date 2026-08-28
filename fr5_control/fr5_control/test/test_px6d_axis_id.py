"""px6d_axis_id 의 판정 로직 회귀 테스트 — 기준선 빼기와 축 지목.

시리얼과 사람 손은 여기서 다루지 않는다. 대신 표본이 들어왔을 때 **어느 채널을
어느 부호로 지목하는가** 를 고정한다. 이 판정이 곧 ``normal_force_sign`` 이 되고,
부호가 뒤집히면 admittance 가 발산한다 (DESIGN_NOTES §4.4).
"""

import math

import pytest

from fr5_control.px6d_axis_id import (
    PROBE_TRIALS,
    SENSOR_TRIALS,
    TRIALS,
    format_report,
    interpret,
    lateral_angle_deg,
    mounting_angle_deg,
    pair_orthogonality,
    summarise,
)
from fr5_control.px6d_protocol import AXIS_ORDER


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


# -- 2 단계 절차: 센서 기준 → 프로브 각도 -------------------------------


def _lateral(key, angle_deg, magnitude=2.0, axial=0.0):
    """가로면에서 주어진 각도로 민 시행 하나를 합성한다."""
    radians = math.radians(angle_deg)
    return {
        "trial": key,
        "deltas": [
            magnitude * math.cos(radians),
            magnitude * math.sin(radians),
            axial,
            0.0,
            0.0,
            0.0,
        ],
        "winner": "Fx",
        "separation": 9.0,
    }


def test_lateral_angle_reads_the_push_direction():
    for angle in (0.0, 45.0, 90.0, 135.0, -90.0):
        measured, magnitude, leak = lateral_angle_deg(_lateral("Sx+", angle)["deltas"])
        assert measured == pytest.approx(angle, abs=1e-9)
        assert magnitude == pytest.approx(2.0)
        assert leak == pytest.approx(0.0)


def test_lateral_angle_reports_axial_leak():
    """가로로 민다고 했는데 축이 밀리면 각도를 믿을 수 없다."""
    _, _, leak = lateral_angle_deg(_lateral("Sx+", 0.0, magnitude=2.0, axial=1.0)["deltas"])
    assert leak == pytest.approx(0.5)


def test_lateral_angle_gives_up_on_a_push_that_did_not_happen():
    assert lateral_angle_deg([0.0] * 6)[0] is None
    assert lateral_angle_deg(None)[0] is None


def test_mounting_angle_is_the_probe_direction_in_the_channel_frame():
    """설정이 요구하는 것은 **채널 프레임에서 본 프로브 +x** 다.

    센서는 민 방향의 반대로 보고하므로 반응각에서 180° 를 뺀 것이 실제 방향이다.
    """
    angle, _ = mounting_angle_deg([_lateral("Px+", -137.0)])
    assert angle == pytest.approx(43.0, abs=0.1)


def test_mounting_angle_ignores_the_printed_marking():
    """인쇄 화살표가 어디를 향하든 채널 프레임 각도는 그대로다.

    옛 구현은 Sx+ 와의 **차이** 를 냈는데, 그것은 인쇄 기준 각도라 데이터가 사는
    프레임과 다르다. 인쇄가 30° 틀어져 있어도 결과가 흔들리면 안 된다.
    """
    without = mounting_angle_deg([_lateral("Px+", -137.0)])[0]
    with_marking = mounting_angle_deg(
        [_lateral("Px+", -137.0), _lateral("Sx+", -147.0)]
    )[0]
    assert with_marking == pytest.approx(without)


def test_mounting_angle_reports_the_marking_relative_value_as_a_note_only():
    _, notes = mounting_angle_deg([_lateral("Px+", -137.0), _lateral("Sx+", -177.1)])
    text = " ".join(notes)
    assert "+40.1°" in text or "+40.0°" in text
    assert "설정에 넣지 않는다" in text


def test_mounting_angle_cross_checks_against_the_short_side():
    _, notes = mounting_angle_deg([_lateral("Px+", -137.0), _lateral("Py+", -47.4)])
    text = " ".join(notes)
    assert "교차확인" in text
    assert "서로를 확인한다" in text


def test_mounting_angle_flags_a_failed_cross_check():
    _, notes = mounting_angle_deg([_lateral("Px+", -137.0), _lateral("Py+", -90.0)])
    assert any(n.startswith("⚠️ 교차확인 실패") for n in notes)


def test_mounting_angle_flags_axial_leak():
    _, notes = mounting_angle_deg([_lateral("Px+", -137.0, axial=30.0)])
    assert any("누출" in note for note in notes)


def test_mounting_angle_always_states_the_unresolved_180():
    """180° 애매함은 남는다. 남는다는 사실을 매번 말해야 한다."""
    _, notes = mounting_angle_deg([_lateral("Px+", -137.0)])
    assert any("180°" in note for note in notes)


def test_mounting_angle_needs_the_probe_push():
    angle, notes = mounting_angle_deg([_lateral("Sx+", 0.0)])
    assert angle is None
    assert any("Px+" in note for note in notes)


def test_stages_are_disjoint_and_cover_the_old_trial_list():
    assert not {t[0] for t in SENSOR_TRIALS} & {t[0] for t in PROBE_TRIALS}
    assert TRIALS == SENSOR_TRIALS + PROBE_TRIALS


def test_sensor_stage_starts_with_the_shared_axial_push():
    """축은 센서와 프로브가 공유하므로 마운트 회전과 무관하게 먼저 확정된다."""
    assert SENSOR_TRIALS[0][0] == "Sz+"


def test_interpret_reads_the_normal_sign_from_the_sensor_stage():
    results = [{
        "trial": "Sz+",
        "deltas": [0.0, 0.0, -4.2, 0.0, 0.0, 0.0],
        "winner": "Fz",
        "separation": float("inf"),
    }]
    text = " ".join(interpret(results))
    assert "normal_force_sign" in text
    assert "-1.0" in text


def test_orthogonality_accepts_a_proper_ninety():
    delta, note = pair_orthogonality(
        [_lateral("Sx+", -177.1), _lateral("Sy+", -95.7)], "Sx+", "Sy+"
    )
    assert delta == pytest.approx(81.4, abs=0.1)
    assert not note.startswith("⚠️")


def test_orthogonality_catches_a_push_in_the_wrong_direction():
    """각도 차만 보면 드러나지 않는 오류 — 차는 언제나 계산되니까."""
    delta, note = pair_orthogonality(
        [_lateral("Px+", 87.9), _lateral("Py+", -47.4)], "Px+", "Py+"
    )
    assert delta == pytest.approx(-135.3, abs=0.1)
    assert note.startswith("⚠️")


def test_orthogonality_points_at_the_weaker_push():
    _, note = pair_orthogonality(
        [_lateral("Px+", 0.0, magnitude=4.8), _lateral("Py+", 135.0, magnitude=27.9)],
        "Px+",
        "Py+",
    )
    assert "Px+ 부터 다시" in note


def test_orthogonality_accepts_either_handedness():
    """+y 가 +90 쪽이든 -90 쪽이든 직교는 직교다. 손잡이는 여기서 안 가른다."""
    for angle in (90.0, -90.0):
        _, note = pair_orthogonality(
            [_lateral("Sx+", 0.0), _lateral("Sy+", angle)], "Sx+", "Sy+"
        )
        assert not note.startswith("⚠️")


def test_orthogonality_is_silent_when_a_trial_is_missing():
    delta, note = pair_orthogonality([_lateral("Sx+", 0.0)], "Sx+", "Sy+")
    assert delta is None and note is None


def test_interpret_withholds_the_mounting_angle_when_the_pair_is_broken():
    """직교성이 깨졌으면 각도를 내되 믿지 말라고 말해야 한다."""
    results = [
        _lateral("Sx+", -177.1),
        _lateral("Sy+", -95.7),
        _lateral("Px+", 87.9),
        _lateral("Py+", -47.4),
    ]
    text = " ".join(interpret(results))
    assert "믿지 마라" in text
