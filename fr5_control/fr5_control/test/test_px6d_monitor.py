"""px6d_monitor 의 표시 계산 회귀 테스트 — 막대와 폭 정렬.

시리얼은 여기서 다루지 않는다 (그건 test_px6d_protocol 의 몫이다). 대신 값이
터미널에 닿기 직전의 두 가지를 고정한다: 0 을 가운데 둔 막대의 위치·부호와,
한글이 섞인 표의 칸 맞춤이다. 이 둘이 틀리면 숫자는 맞으면서 화면이 거짓말을
한다 — 축을 눌러 배정을 확인하는 용도라 특히 위험하다.
"""

from fr5_control.px6d_monitor import (
    _BAR_HALF,
    _bar,
    _display_width,
    _pad,
    _rate,
    _render,
    _spans,
    _units,
)
from fr5_control.px6d_protocol import AXIS_ORDER
import pytest


def test_bar_zero_is_centred_tick():
    """0 은 가운데 눈금 하나뿐이다 — 채워진 칸이 없어야 한다."""
    bar = _bar(0.0, 10.0)
    assert len(bar) == 2 * _BAR_HALF + 1
    assert bar[_BAR_HALF] == "┼"
    assert "█" not in bar


@pytest.mark.parametrize("value", [2.5, 5.0, 9.9])
def test_bar_positive_fills_right_half_only(value):
    """양수는 가운데에서 오른쪽으로만 찬다."""
    bar = _bar(value, 10.0)
    assert "█" not in bar[:_BAR_HALF]
    assert "█" in bar[_BAR_HALF:]


@pytest.mark.parametrize("value", [-2.5, -5.0, -9.9])
def test_bar_negative_fills_left_half_only(value):
    """음수는 가운데에서 왼쪽으로만 찬다. 부호가 뒤집히면 축 실측이 어긋난다."""
    bar = _bar(value, 10.0)
    assert "█" in bar[: _BAR_HALF + 1]
    assert "█" not in bar[_BAR_HALF + 1:]


def test_bar_is_symmetric_about_zero():
    """같은 크기의 양·음수는 가운데를 기준으로 거울상이다."""
    assert _bar(3.0, 10.0) == _bar(-3.0, 10.0)[::-1]


def test_bar_marks_clipping_at_the_ends():
    """기준값을 넘으면 잘렸다는 표시가 끝에 남는다."""
    assert _bar(99.0, 10.0)[-1] == "»"
    assert _bar(-99.0, 10.0)[0] == "«"


def test_bar_width_is_fixed_across_magnitudes():
    """폭이 흔들리면 제자리 갱신에서 표가 출렁인다."""
    widths = {len(_bar(v, 10.0)) for v in (-99.0, -1.0, 0.0, 1.0, 99.0)}
    assert widths == {2 * _BAR_HALF + 1}


def test_bar_with_nonpositive_span_degrades_to_tick():
    """기준값이 0 이면 나눗셈 대신 눈금만 — 0 으로 나누지 않는다."""
    assert _bar(1.0, 0.0) == _bar(0.0, 10.0)


def test_spans_split_force_and_torque():
    """앞 3 축은 힘, 뒤 3 축은 모멘트 기준을 쓴다."""
    assert _spans(10.0, 0.5) == (10.0, 10.0, 10.0, 0.5, 0.5, 0.5)


def test_units_match_axis_order():
    """단위는 축 수와 같고 폭이 고르다 — 표가 어긋나지 않게."""
    units = _units()
    assert len(units) == len(AXIS_ORDER)
    assert len({len(u) for u in units}) == 1


def test_display_width_counts_hangul_as_two_cells():
    """한글은 두 칸이다. len 으로 재면 머리글이 밀린다."""
    assert _display_width("축") == 2
    assert _display_width("Fx") == 2
    assert _display_width("프레임 3") == 8


def test_pad_uses_display_width_not_len():
    """채움도 표시 폭 기준이어야 머리글이 데이터 칸 위에 선다."""
    assert _display_width(_pad("축", 10)) == 10
    assert _display_width(_pad("값", 10, right=True)) == 10
    assert _pad("값", 10, right=True).endswith("값")


def _sample_lines(values):
    """표 한 장을 그려 돌려준다. 아래 정렬 검사들이 함께 쓴다."""
    peaks = [(v - 0.05, v + 0.05) for v in values] if values else [(0.0, 0.0)] * len(AXIS_ORDER)
    return _render(values, peaks, _spans(10.0, 0.5), _units(), "1000.0 Hz")


def test_render_emits_one_line_per_axis_plus_header_and_status():
    """줄 수가 흔들리면 제자리 갱신이 이전 화면을 지우지 못하고 흘러내린다."""
    lines = _sample_lines((0.28, 0.99, -1.34, 0.014, 0.005, 0.011))
    assert len(lines) == len(AXIS_ORDER) + 2
    for name, line in zip(AXIS_ORDER, lines[1:]):
        assert line.lstrip().startswith(name)
    assert "1000.0 Hz" in lines[-1]


def test_render_header_column_lines_up_with_data():
    """머리글의 peak 칸이 데이터의 peak 칸과 같은 자리에서 시작한다."""
    lines = _sample_lines((0.28, 0.99, -1.34, 0.014, 0.005, 0.011))
    header_peak = _display_width(lines[0].split("peak")[0])
    bar = _bar(0.28, 10.0)
    data_field = _display_width(lines[1].split(bar)[0]) + len(bar) + 2
    assert header_peak == data_field


def test_render_without_frames_still_names_every_axis():
    """프레임이 오기 전에도 표 모양이 유지돼야 화면이 튀지 않는다."""
    lines = _sample_lines(None)
    assert len(lines) == len(AXIS_ORDER) + 2
    for name, line in zip(AXIS_ORDER, lines[1:]):
        assert name in line


def test_rate_needs_two_samples():
    """표본이 하나면 주기를 못 잰다 — 0 으로 나누지 않고 폭도 유지한다."""
    assert _rate([]).strip() == "—"
    assert _rate([1.0]).strip() == "—"
    assert len(_rate([1.0])) == len(_rate([1.0, 1.001]))


def test_rate_uses_sample_spacing_not_window_length():
    """창이 덜 찬 시작 직후에도 값이 튀지 않아야 한다."""
    stamps = [i / 1000.0 for i in range(50)]     # 1 kHz 로 50 개, 0.05 초치
    assert float(_rate(stamps)) == pytest.approx(1000.0)


def test_rate_handles_identical_timestamps():
    """시각이 같으면 나눗셈 대신 미측정으로 물러난다."""
    assert _rate([2.0, 2.0, 2.0]).strip() == "—"
