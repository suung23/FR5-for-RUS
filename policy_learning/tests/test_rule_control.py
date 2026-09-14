"""규칙 기반 제어기 — 학습된 정책 없이 찾고 가운데에 둔다.

가상의 팬텀 위에서 폐루프로 돌린다. 팬텀은 2026-09-12 실측을 따른다:
x 병진이 중심을 0.00708 /mm 로 옮기고, y 병진이 면적을 바꾼다.
"""

import numpy as np
import pytest

from rus_policy.axis_gains import AxisGains
from rus_policy.rule_control import FIND, HOLD, RuleController

G_X, G_Y_AREA = 0.00708, 0.00161


def gains(**kw) -> AxisGains:
    d = dict(centroid_per_unit={"x": G_X, "y": 0.00031},
             area_per_unit={"x": -0.00001, "y": G_Y_AREA},
             r_centroid={"x": 1.00, "y": 0.82}, source="테스트")
    d.update(kw)
    return AxisGains(**d)


class Phantom:
    """x 는 중심을, y 는 면적을 바꾼다. 면적은 y_peak 에서 최대인 산이다."""

    def __init__(self, dx=0.10, y=0.0, y_peak=8.0, width=12.0, top=0.045):
        self.dx, self.y, self.y_peak, self.width, self.top = dx, y, y_peak, width, top

    def step(self, vel, dt):
        self.dx += vel[0] * dt * G_X
        self.y += vel[1] * dt
        return self.observe()

    def observe(self):
        area = self.top * float(np.exp(-((self.y - self.y_peak) / self.width) ** 2))
        return (area > 0.004), self.dx, area


def run(c: RuleController, ph: Phantom, seconds=180.0, hz=5.0):
    t, dt, traj = 0.0, 1.0 / hz, []
    seen, dx, area = ph.observe()
    for _ in range(int(seconds * hz)):
        vel, _ = c.update(t, seen, dx, area)
        seen, dx, area = ph.step(vel, dt)
        traj.append((t, c.phase, dx, area, ph.y))
        t += dt
    return traj


def test_finds_the_area_peak_then_centres():
    c, ph = RuleController(gains()), Phantom(dx=0.10, y=0.0)
    traj = run(c, ph, seconds=240.0)
    assert any(p == FIND for _, p, _, _, _ in traj)
    assert c.phase == HOLD, "찾기에서 못 빠져나왔다"
    assert abs(traj[-1][2]) <= c.deadband + 0.01, f"중심이 안 맞았다: {traj[-1][2]:+.3f}"
    assert traj[-1][3] > traj[0][3], "면적이 나아지지 않았다"


def test_centring_moves_the_right_way():
    """부호가 뒤집혀 있으면 중심이 더 벌어진다 — 규칙 제어의 유일한 실패 방식이다."""
    for start in (+0.12, -0.12):
        c, ph = RuleController(gains()), Phantom(dx=start, y=8.0)
        traj = run(c, ph, seconds=90.0)
        assert abs(traj[-1][2]) < abs(start), (start, traj[-1][2])


def test_deadband_stops_chasing_noise():
    """1 s 중심 잡음 표준편차가 0.014 다 — 불감대 안에서는 지령이 0 이어야 한다."""
    c = RuleController(gains())
    c.phase = HOLD
    vel, note = c.update(0.0, True, 0.02, 0.03)
    assert not np.any(vel), note


def test_goes_back_to_find_when_the_bladder_is_lost():
    c = RuleController(gains())
    c.phase = HOLD
    c.update(0.0, True, 0.0, 0.03)
    c.update(1.0, False, float("nan"), 0.0)
    assert c.phase == FIND


def test_beam_axis_is_never_commanded():
    """빔 방향은 admittance 와 force_search 의 몫이다."""
    c, ph = RuleController(gains()), Phantom(dx=0.10, y=0.0)
    for _, _, _, _, _ in run(c, ph, seconds=60.0):
        pass
    c2, ph2 = RuleController(gains()), Phantom(dx=0.10, y=0.0)
    t = 0.0
    seen, dx, area = ph2.observe()
    for _ in range(300):
        vel, _ = c2.update(t, seen, dx, area)
        assert vel[2] == 0.0 and not np.any(vel[3:]), vel
        seen, dx, area = ph2.step(vel, 0.2)
        t += 0.2


def test_refuses_to_run_without_a_measured_gain():
    """이득이 없으면 부호를 모른다 — 짐작으로 움직이면 안 된다."""
    with pytest.raises(ValueError, match="정렬 축이 없다"):
        RuleController(AxisGains())
    with pytest.raises(ValueError, match="정렬 축이 없다"):
        RuleController(gains(centroid_per_unit={"x": 0.0001}, r_centroid={"x": 1.0}))


def test_low_correlation_gain_is_not_trusted():
    """구간이 적으면 기울기가 흔들린다 — 상관이 낮은 축은 쓰지 않는다."""
    with pytest.raises(ValueError):
        RuleController(gains(centroid_per_unit={"x": G_X}, r_centroid={"x": 0.3}))


# --- 훑기 (2026-09-14 회귀 방지) ----------------------------------------------

def test_finds_a_bladder_that_starts_outside_the_view():
    """방광이 **아예 안 보이는** 데서 시작해도 찾아야 한다.

    2026-09-14 실기에서 못 찾은 자리다. 오르막 타기는 한 걸음 앞이 0 인지만 보는데
    방광이 시야 밖이면 어느 쪽도 0 이라 "나아짐" 이 없어 ±4 mm 를 영원히 왕복했다.
    훑기는 경사가 필요 없다.
    """
    c = RuleController(gains(), find_half_range=25.0)
    ph = Phantom(dx=0.02, y=0.0, y_peak=18.0, width=6.0)   # 18 mm 밖, 좁은 산
    assert ph.observe()[2] < 0.005, "시작부터 보이면 시험이 성립하지 않는다"
    traj = run(c, ph, seconds=240.0)
    assert c.phase == HOLD, f"못 찾았다 (마지막 {traj[-1]})"
    assert traj[-1][3] > 0.02, f"찾긴 했는데 면적이 작다: {traj[-1][3]:.4f}"


def test_sweep_returns_to_the_best_spot_not_the_end():
    """훑은 뒤 **최대였던 자리**로 돌아가야 한다 — 끝에 서 있으면 아무 소용이 없다."""
    from rus_policy.rule_control import SweepScan
    s = SweepScan(half_range=20.0, rate=3.0)
    t, peak = 0.0, -7.0
    for _ in range(3000):
        v = s.update(t, 0.04 * float(np.exp(-((s.pos - peak) / 5.0) ** 2)))
        t += 0.05
        if s.done:
            break
    assert s.done, "훑기가 안 끝났다"
    assert abs(s.pos - peak) < 1.0, f"최대 지점 {peak} 이 아니라 {s.pos:.1f} 에 섰다"
    assert s.found


def test_sweep_reports_not_found_on_an_empty_phantom():
    """아무것도 없으면 '못 봤다' 로 남아야 한다 — 0 을 최댓값으로 삼으면 안 된다."""
    from rus_policy.rule_control import SweepScan
    s = SweepScan(half_range=10.0, rate=3.0, min_value=0.005)
    t = 0.0
    for _ in range(3000):
        s.update(t, 0.0)
        t += 0.05
        if s.done:
            break
    assert s.done and not s.found


def test_keeps_sweeping_when_nothing_is_found():
    """한 바퀴 훑고 못 찾으면 다시 훑는다 — 멈춰 서 있으면 안 된다."""
    c = RuleController(gains(), find_half_range=8.0)
    ph = Phantom(dx=0.0, y=0.0, y_peak=500.0)              # 닿지 않는 곳에 산
    traj = run(c, ph, seconds=120.0)
    assert c.phase == FIND
    ys = [y for _, _, _, _, y in traj]
    assert max(ys) - min(ys) > 8.0, "훑기를 멈췄다"


# --- 병진 + 회전 함께 (2026-09-14) --------------------------------------------

class TwoAxisPhantom:
    """y 병진과 θx 회전이 **각각** 면적을 바꾼다. 회전 쪽 산이 더 높다."""

    def __init__(self, y=0.0, thx=0.0):
        self.y, self.thx, self.dx = y, thx, 0.02

    def step(self, vel, dt):
        self.y += vel[1] * dt
        self.thx += vel[3] * dt
        return self.observe()

    def observe(self):
        a_y = 0.010 * float(np.exp(-((self.y - 6.0) / 5.0) ** 2))
        a_r = 0.040 * float(np.exp(-((self.thx - 5.0) / 3.0) ** 2))
        area = max(a_y, a_r)
        return (area > 0.004), self.dx, area


def test_sweeps_translation_and_rotation_then_takes_the_better():
    """두 축을 다 훑고 면적이 더 컸던 쪽으로 간다."""
    c = RuleController(gains(centroid_per_unit={"x": G_X, "thy": 0.004},
                             area_per_unit={"y": G_Y_AREA, "thx": 0.002},
                             r_centroid={"x": 1.00, "thy": 0.9}),
                       find_axes=("y", "thx"), find_half_range=15.0,
                       find_half_range_deg=12.0, max_deg_s=3.0)
    assert c.find_axis_list == ["y", "thx"]
    ph = TwoAxisPhantom()
    t, dt = 0.0, 0.2
    seen, dx, area = ph.observe()
    for _ in range(1500):
        vel, _ = c.update(t, seen, dx, area)
        seen, dx, area = ph.step(vel, dt)
        t += dt
        if c.phase == HOLD:
            break
    assert c.phase == HOLD, c.last_note
    assert area > 0.02, f"낮은 산(병진)에 머물렀다: {area:.4f}"


def test_rotation_axis_uses_degree_limits_not_mm():
    """회전 축을 고르면 상한과 단위가 도로 바뀌어야 한다 — 섞이면 조용히 틀린다."""
    c = RuleController(gains(centroid_per_unit={"thy": 0.02},
                             area_per_unit={"y": G_Y_AREA},
                             r_centroid={"thy": 0.95}),
                       max_mm_s=3.0, max_deg_s=8.0)
    assert c.align_axis == "thy"
    assert c.max_rate == 8.0, "병진 상한으로 회전을 자르고 있다"
    c.phase = HOLD
    # 0.30 / 0.02 = 15° 필요 · k_p 0.4 → 6 °/s (상한 아래라 그대로 나간다)
    vel, note = c.update(0.0, True, 0.30, 0.03)
    assert vel[0] == 0.0 and vel[1] == 0.0, "병진으로 나갔다"
    assert vel[4] == pytest.approx(-6.0), vel            # θy 자리, °/s
    assert "°" in note, note
    # 0.60 → 30° 필요 · 12 °/s → 도 단위 상한 8 에서 잘려야 한다.
    # 병진 상한(3)으로 잘렸다면 3 이 나온다 — 그게 고치려던 버그다.
    c.phase = HOLD
    vel, _ = c.update(1.0, True, 0.60, 0.03)
    assert vel[4] == pytest.approx(-8.0), vel


def test_unknown_find_axis_is_refused():
    with pytest.raises(ValueError, match="모르는 찾기 축"):
        RuleController(gains(), find_axes=("y", "굴림"))


# --- 중심 보정이 시야를 잃을 때 (2026-09-14 실기) -----------------------------

class CoupledPhantom:
    """정렬 축이 **면적도 함께** 바꾼다 — thy 의 실측 면적 이득 +0.00206 /° 를 흉내낸다.

    그날 실기에서 중심 보정이 9° 돌아가며 면적을 0.041 → 0 으로 만들었고, 찾기 축
    (thx·thz)만으로는 되돌아올 수 없어 나머지 90 s 를 헤맸다.
    """

    G_THY, AREA_PER_DEG = 0.01913, 0.00206

    def __init__(self, dx=-0.09, thy=0.0, area0=0.041):
        self.dx, self.thy, self.area0 = dx, thy, area0

    def step(self, vel, dt):
        self.thy += vel[4] * dt
        self.dx += vel[4] * dt * self.G_THY
        return self.observe()

    def observe(self):
        area = max(0.0, self.area0 - abs(self.thy) * self.AREA_PER_DEG * 2.2)
        return (area > 0.004), self.dx, area


def _rot_gains():
    return AxisGains(centroid_per_unit={"thy": CoupledPhantom.G_THY, "thx": -0.00612},
                     area_per_unit={"thx": -0.00189, "thz": 0.00401, "thy": 0.00206},
                     r_centroid={"thy": 0.78}, sign_agreement={"thy": 0.83},
                     source="2026-09-14 실측")


def test_centering_never_destroys_the_view():
    """면적이 무너지면 **되돌아가야** 한다 — 중심을 맞추려다 시야를 잃으면 아무 소용 없다."""
    c = RuleController(_rot_gains(), max_deg_s=3.0, find_axes=("thx",))
    c.phase = HOLD
    ph = CoupledPhantom()
    t, dt, worst = 0.0, 0.2, 1.0
    seen, dx, area = ph.observe()
    for _ in range(600):
        vel, _ = c.update(t, seen, dx, area)
        seen, dx, area = ph.step(vel, dt)
        worst = min(worst, area)
        t += dt
    assert worst > 0.004, f"방광을 놓쳤다 (최저 면적 {worst:.4f})"
    assert abs(ph.thy) <= c.align_max_deg + 1.0, f"{ph.thy:.1f}° 까지 돌아갔다"


def test_alignment_excursion_is_capped():
    """중심이 안 맞아도 누적 변위가 상한을 넘으면 멈춘다."""
    c = RuleController(_rot_gains(), max_deg_s=3.0, align_max_deg=5.0)
    c.phase = HOLD
    t = 0.0
    for _ in range(400):
        vel, note = c.update(t, True, -0.5, 0.04)     # 중심이 끝까지 안 맞는 상황
        t += 0.2
    assert abs(c.align_pos) <= 5.0 + 1e-6, c.align_pos
    assert "상한" in note, note


def test_cap_is_in_the_axis_unit():
    """병진 축이면 mm 상한, 회전 축이면 도 상한을 써야 한다."""
    rot = RuleController(_rot_gains(), align_max_mm=20.0, align_max_deg=8.0)
    assert rot.align_axis == "thy" and rot.align_max == 8.0
    tr = RuleController(gains(), align_max_mm=20.0, align_max_deg=8.0)
    assert tr.align_axis == "x" and tr.align_max == 20.0


# --- 빔축 회전(thz)은 찾기에 쓰지 않는다 (2026-09-14 조작자 지적) ---------------

def test_beam_axis_rotation_is_never_chosen_automatically():
    """thz 는 이득이 가장 좋아 보여도 고르면 안 된다.

    실측(probe_rot)에서 thz 는 부호 일치 100 %, 면적 이득 +0.00401/° 로 세 축 중 최대다.
    그런데 그것은 방광을 찾고 있어서가 아니라 **절단면이 갈리고 있어서**다 — 빔축 둘레
    회전은 영상면을 자기 축으로 돌려 횡단면을 사면으로 만든다.
    """
    from rus_policy.axis_gains import EXCLUDED_AXES, AxisGains
    g = AxisGains(
        centroid_per_unit={"thx": -0.00612, "thy": 0.01913, "thz": 0.01518},
        area_per_unit={"thx": -0.00189, "thy": 0.00206, "thz": 0.00401},
        sign_agreement={"thx": 0.33, "thy": 0.83, "thz": 1.0},
        r_centroid={"thx": -0.79, "thy": 0.78, "thz": 0.70})
    assert "thz" in EXCLUDED_AXES
    assert g.search_axis() == "thx", "면적 이득이 가장 큰 thz 를 골랐다"
    assert g.centering_axis() == "thy"


def test_beam_axis_rotation_wins_on_gain_if_not_excluded():
    """제외가 없으면 실제로 thz 가 뽑힌다 — 위 시험이 헛돌지 않음을 보인다."""
    from rus_policy.axis_gains import EXCLUDED_AXES
    import rus_policy.axis_gains as ag
    from rus_policy.axis_gains import AxisGains
    g = AxisGains(
        centroid_per_unit={"thx": -0.00612, "thy": 0.01913, "thz": 0.01518},
        area_per_unit={"thx": -0.00189, "thy": 0.00206, "thz": 0.00401},
        sign_agreement={"thx": 0.33, "thy": 0.83, "thz": 1.0})
    old = ag.EXCLUDED_AXES
    try:
        ag.EXCLUDED_AXES = ()
        assert g.search_axis() == "thz"
    finally:
        ag.EXCLUDED_AXES = old


def test_beam_axis_rotation_is_refused_on_the_command_line():
    """자동 선택만 막으면 --rule-find-axes 로 되살아난다.

    2026-09-14 의 세 실행이 실제로 `--rule-find-axes thx,thz` 로 돌았다.
    """
    import pytest
    from rus_policy.axis_gains import AxisGains
    from rus_policy.rule_control import RuleController
    g = AxisGains(
        centroid_per_unit={"thx": -0.00612, "thy": 0.01913, "thz": 0.01518},
        area_per_unit={"thx": -0.00189, "thy": 0.00206, "thz": 0.00401},
        sign_agreement={"thx": 0.33, "thy": 0.83, "thz": 1.0})
    with pytest.raises(ValueError, match="빔축"):
        RuleController(g, find_axes=("thx", "thz"))
    RuleController(g, find_axes=("thx",))        # 남는 축은 그대로 된다


def test_axis_probe_still_measures_the_excluded_axis():
    """고르지 못하게 할 뿐, 재는 것은 막지 않는다 — 왜 제외하는지도 데이터로 남아야 한다."""
    from rus_policy.axis_probe import DEFAULT_ORDER
    assert "thz" in DEFAULT_ORDER


def test_translation_find_axes_need_no_gains():
    """찾기는 훑기라 이득이 필요 없다 — 회전으로 잰 이득 파일만 있어도 병진을 훑는다.

    정렬은 다르다. 거기서는 centroid_per_unit 이 있어야 몇 mm 를 가야 할지 나온다.
    """
    from rus_policy.axis_gains import AxisGains
    from rus_policy.rule_control import RuleController
    g = AxisGains(centroid_per_unit={"thy": 0.01913}, area_per_unit={"thx": -0.00189},
                  sign_agreement={"thy": 0.83})
    c = RuleController(g, find_axes=("thx", "thy", "x", "y"))
    assert c.find_axis_list == ["thx", "thy", "x", "y"]
    assert c.align_axis == "thy"


def test_runner_opens_the_lateral_gate_for_translation_axes():
    """병진을 쓰면 제어 스택의 통로를 켜야 한다 — 안 켜면 지령이 조용히 버려진다.

    2026-09-12 probe2: x 병진이 0.00708/mm·r=1.00 이라는 이득을 냈는데 로봇은 그 구간에
    0.07 mm 움직였다 (지령 11.8). gate_axes 가 정책 régime 에서 병진을 버린다.
    """
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    fn = re.search(r"def _enable_lateral_if_needed.*?\n    def ", src, re.S).group(0)
    assert 'LATERAL_PARAM = "contact_control.policy_lateral"' in src
    assert "self.LATERAL_PARAM" in fn
    assert "SystemExit" in fn, "켜지 못했는데도 그냥 도는 경로가 있다"
    assert "_enable_lateral_if_needed()" in src, "부르는 데가 없다"


def test_lateral_gate_is_opened_after_construction_not_inside_it():
    """__init__ 안에서 spin 하면 구독 콜백이 반쯤 지어진 객체로 들어온다.

    2026-09-14: _on_twist_echo 가 아직 정의되지 않은 _last_sent 를 읽고 AttributeError
    로 죽었다. ROS 노드의 생성자에서 응답을 기다리면 언제든 같은 일이 난다.
    """
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    init = re.search(r"class PolicyRunner.*?\n    # -- 되먹임", src, re.S).group(0)
    assert "self._enable_lateral_if_needed()" not in init, "생성자 안에서 부르고 있다"
    main = re.search(r"\ndef main\(\).*", src, re.S).group(0)
    assert "node._enable_lateral_if_needed()" in main, "main 에서 부르지 않는다"


# --- 못 찾으면 범위를 넓힌다 (2026-09-14 attempt_005) --------------------------

def test_sweep_range_widens_when_nothing_is_found():
    """한 바퀴 다 훑고 못 봤다는 것은 방광이 그 범위 밖이라는 증거다.

    2026-09-14 attempt_005: 180 s 동안 ±12° 를 왕복하며 마스크 1 % 로 끝났다. 프로브
    시작 자세가 첫 실행에서 82 mm 떨어져 있었고 ±12° 로는 닿을 수 없었다.
    """
    # find_oop_scale=1 — 넓히기 자체를 재는 시험이라 면외 축소를 섞지 않는다
    c = RuleController(gains(), find_half_range=8.0, find_widen=2.0, find_max_range_mm=40.0,
                       find_oop_scale=1.0)
    ph = Phantom(dx=0.0, y=0.0, y_peak=30.0, width=4.0)     # 30 mm 밖 — ±8 로는 못 닿는다
    assert ph.observe()[2] < 0.005
    traj = run(c, ph, seconds=600.0)
    assert c.phase == HOLD, f"넓혔는데도 못 찾았다 (반경 {c.find_range_now:.1f})"
    assert c.find_range_now == 8.0, "찾은 뒤에도 넓은 반경을 들고 있다"
    ys = [y for _, _, _, _, y in traj]
    assert max(ys) > 20.0, f"멀리까지 안 갔다 (최대 {max(ys):.1f} mm)"


def test_widening_stops_at_the_cap():
    """무한히 넓히면 로봇이 작업공간을 벗어난다 — 상한에서 멈추고 그렇게 말한다."""
    c = RuleController(gains(), find_half_range=10.0, find_widen=2.0, find_max_range_mm=20.0)
    assert c._widen_range() is True and c.find_range_now == 20.0
    assert c._widen_range() is False and c.find_range_now == 20.0
    ph = Phantom(dx=0.0, y=0.0, y_peak=5000.0)              # 절대 못 찾는다
    notes, t, dt = [], 0.0, 0.2
    seen, dx, area = ph.observe()
    for _ in range(1500):
        vel, note = c.update(t, seen, dx, area)
        notes.append(note)
        seen, dx, area = ph.step(vel, dt)
        t += dt
    assert any("옮겨야" in n for n in notes), "상한에 닿았다고 말하지 않는다"
    assert c.find_range_now == 20.0


def test_rotation_and_translation_caps_are_separate():
    """회전은 도, 병진은 mm 로 각각 묶인다 — 한쪽 상한으로 다른 쪽을 자르면 안 된다."""
    c = RuleController(gains(centroid_per_unit={"thy": 0.019}, area_per_unit={"thx": 0.002},
                             sign_agreement={"thy": 0.9}),
                       find_axes=("thx", "y"),
                       find_half_range=10.0, find_half_range_deg=12.0, find_widen=10.0,
                       find_max_range_mm=40.0, find_max_range_deg=30.0)
    c._widen_range()
    assert c.find_range_now == 40.0 and c.find_range_now_deg == 30.0


# --- Q 의 항별 부족분이 움직임을 고른다 (2026-09-14 조작자 지시) -----------------

def _q_gains():
    return AxisGains(centroid_per_unit={"thy": 0.01913}, area_per_unit={"thx": -0.00189},
                     sign_agreement={"thy": 0.83})


def _parts(**kw):
    from rus_policy.view_quality import WEIGHTS
    p = {k: 1.0 for k in WEIGHTS}
    p.update(kw)
    return p


def test_target_met_means_stand_still():
    """Q_seg 가 합격선을 넘으면 더 건드리지 않는다 — 움직임 자체가 위험이다."""
    c = RuleController(_q_gains(), find_axes=("thx",), q_target=0.90)
    c.phase = HOLD
    vel, note = c.update(0.0, True, 0.20, 0.06, q_view=0.93, q_parts=_parts())
    assert not np.any(vel), note
    assert "합격" in note, note


def test_centroid_deficit_drives_the_align_axis():
    """중심 때문에 낮으면 정렬 축을 돌린다."""
    c = RuleController(_q_gains(), find_axes=("thx",), q_target=0.90)
    c.phase = HOLD
    vel, note = c.update(0.0, True, -0.20, 0.06, q_view=0.80,
                         q_parts=_parts(centering=0.3))
    from rus_policy.rule_control import AXIS_INDEX
    assert vel[AXIS_INDEX["thy"]] != 0.0, note
    assert c.phase == HOLD, "중심 때문인데 찾기로 갔다"


def test_area_deficit_drives_a_resweep_not_centring():
    """면적 때문에 낮으면 정렬로는 못 고친다 — 다시 훑어야 한다."""
    c = RuleController(_q_gains(), find_axes=("thx",), q_target=0.90,
                       find_half_range_deg=12.0, refine_fraction=0.5)
    c.phase = HOLD
    vel, note = c.update(0.0, True, 0.0, 0.01, q_view=0.70,
                         q_parts=_parts(area=0.1, presence=0.5))
    assert c.phase == FIND, note
    assert "면적/유무" in note, note
    assert c.find_range_now_deg == 6.0, "넓게 훑고 있다 — 보고 있는 것을 놓친다"


def test_the_bigger_deficit_wins():
    """둘 다 모자라면 **가중 손실이 큰 쪽**을 먼저 고친다."""
    c = RuleController(_q_gains(), find_axes=("thx",), q_target=0.90)
    c.phase = HOLD
    # centering 가중치 13.8 % · area 20.7 % → 같은 비율이면 area 쪽 손실이 크다
    c.update(0.0, True, -0.20, 0.03, q_view=0.6, q_parts=_parts(centering=0.5, area=0.5))
    assert c.phase == FIND, "면적 손실이 더 큰데 정렬을 골랐다"


def test_unfixable_deficit_is_named_not_chased():
    """대비·신뢰도는 이 제어기가 못 고친다 — 헛돌지 말고 무엇인지 말해야 한다."""
    c = RuleController(_q_gains(), find_axes=("thx",), q_target=0.90)
    c.phase = HOLD
    vel, note = c.update(0.0, True, 0.0, 0.06, q_view=0.85,
                         q_parts=_parts(contrast=0.0, confidence=0.5))
    assert not np.any(vel), note
    assert "force_search" in note or "U-Net" in note, note


def test_resweep_has_a_cooldown():
    """한 번 훑어도 안 나아지는 상황이 있다 — 찾기와 유지를 오가면 안 된다."""
    c = RuleController(_q_gains(), find_axes=("thx",), q_target=0.90,
                       refine_cooldown_s=20.0)
    c.phase = HOLD
    c.update(0.0, True, 0.0, 0.01, q_view=0.70, q_parts=_parts(presence=0.4))
    assert c.phase == FIND
    c.phase = HOLD
    _, note = c.update(5.0, True, 0.0, 0.01, q_view=0.70, q_parts=_parts(presence=0.4))
    assert c.phase == HOLD and "대기" in note, note


def test_losses_are_weighted_not_raw():
    """항마다 가중치가 다르다 — 같은 0.5 라도 presence 가 centering 보다 3 배 아프다."""
    L = RuleController.q_losses(_parts(presence=0.5, centering=0.5))
    assert L["presence"] > 2.5 * L["centering"], L


# --- 면외는 좁게, 찾으면 바로 멈춘다 (2026-09-14 조작자 지시) -------------------

def test_out_of_plane_axes_sweep_a_smaller_range():
    """면외 기울임은 조금만 가도 절단면이 바뀐다 — 멀리 가면 방광을 지나친다."""
    c = RuleController(_q_gains(), find_axes=("thx", "thy"),
                       find_half_range_deg=12.0, find_oop_scale=0.5)
    assert c.align_axis == "thy"
    assert c.is_oop("thx") and not c.is_oop("thy")
    rng = {a: sc.half_range for a, sc in c._scans}
    assert rng["thy"] == 12.0, rng          # 면내 — 그대로
    assert rng["thx"] == 6.0, rng           # 면외 — 절반


def test_sweep_stops_as_soon_as_the_view_is_good_enough():
    """최댓값을 찾는 것이 목적이 아니다 — 쓸 만한 자리에 닿으면 거기서 멈춘다."""
    from rus_policy.rule_control import SweepScan
    s = SweepScan(half_range=30.0, rate=3.0, good_enough=0.90)
    t = 0.0
    for _ in range(4000):
        # 위치가 −20 을 넘으면 좋은 뷰 (끝인 +30 까지 안 가도 된다)
        s.update(t, 0.95 if s.pos > -20.0 else 0.1)
        t += 0.05
        if s.done:
            break
    assert s.done, "안 멈췄다"
    assert s.pos < -15.0, f"합격선을 지나 {s.pos:.1f} 까지 갔다"


def test_remaining_axes_are_skipped_once_good_enough():
    """한 축에서 이미 합격했으면 남은 축은 안 훑는다 — 좋은 자리를 떠날 이유가 없다."""
    c = RuleController(_q_gains(), find_axes=("thx", "thy"), q_target=0.90,
                       find_half_range_deg=12.0)
    t = 0.0
    for _ in range(2000):
        c.update(t, True, 0.0, 0.06, q_view=0.95, q_parts=_parts())
        t += 0.2
        if c.phase == HOLD:
            break
    assert c.phase == HOLD, c.last_note
    assert c._scan_i == 0, f"첫 축에서 합격했는데 {c._scan_i + 1} 번째 축까지 갔다"
    assert t < 40.0, f"{t:.0f} s 나 걸렸다"


def test_the_stop_line_reaches_a_sweep_already_running():
    """Q 가 처음 들어온 순간, 돌고 있는 훑기에도 정지선이 들어가야 한다."""
    c = RuleController(_q_gains(), find_axes=("thx",), q_target=0.90)
    assert all(sc.good_enough == float("inf") for _, sc in c._scans)
    c.update(0.0, True, 0.0, 0.06, q_view=0.95, q_parts=_parts())
    assert all(sc.good_enough == 0.90 for _, sc in c._scans)


# --- 멀리 가면 스스로 돌아온다 (2026-09-14 rule_v3 교착) ------------------------

def test_it_comes_home_instead_of_freezing_far_away():
    """러너는 30° 에서 지령을 끊는다 — 그 전에 스스로 돌아와야 한다.

    rule_v3 attempt_003: 112 s 에 30° 에 닿았고, 지령이 끊기니 되돌아올 수단도 없어
    남은 68 s 를 마스크 1 % 로 보냈다. 멈추는 것과 돌아오는 것은 다르다.
    """
    from rus_policy.rule_control import HOME, AXIS_INDEX
    c = RuleController(_q_gains(), find_axes=("thx",), max_excursion_deg=10.0,
                       max_deg_s=3.0)
    c.axis_pos = {"thx": 14.0, "thy": 3.0}
    assert c.excursion_deg() == 17.0
    t, dt = 0.0, 0.2
    for _ in range(400):
        vel, note = c.update(t, False, float("nan"), 0.0)
        t += dt
        if c.phase != HOME:
            break
    assert c.phase == FIND, note
    assert c.excursion_deg() <= 2.0, c.axis_pos


def test_home_drives_the_axis_that_is_furthest_out():
    from rus_policy.rule_control import HOME, AXIS_INDEX
    c = RuleController(_q_gains(), find_axes=("thx",), max_excursion_deg=5.0, max_deg_s=3.0)
    c.axis_pos = {"thx": 2.0, "thy": -9.0}
    vel, _ = c.update(0.0, False, float("nan"), 0.0)
    assert c.phase == HOME
    assert vel[AXIS_INDEX["thy"]] > 0, vel     # −9 → 양의 속도로 되돌린다
    assert vel[AXIS_INDEX["thx"]] == 0.0, "두 축을 동시에 움직인다"


def test_widening_respects_the_rotation_budget():
    """한 바퀴 도는 것만으로 복귀 상한에 닿으면 훑기가 복귀에 잡아먹힌다."""
    c = RuleController(_q_gains(), find_axes=("thx", "thy"),
                       find_half_range_deg=12.0, find_oop_scale=0.5,
                       find_widen=3.0, find_max_range_deg=90.0,
                       max_excursion_deg=24.0)
    for _ in range(6):
        c._widen_range()
    share = 0.5 + 1.0                          # thx(면외) + thy(면내)
    assert c.find_range_now_deg <= 0.8 * 24.0 / share + 1e-6, c.find_range_now_deg


def test_runner_lets_the_controller_come_home():
    """러너의 too_far 가 복귀 지령까지 끊으면 안 된다."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    blk = re.search(r'if self\.condition == "rule" and too_far.*?\n\s*if self\.condition in',
                    src, re.S).group(0)
    assert "too_far = False" in blk, blk


# --- 힘이 움직이는 동안은 면적을 쫓지 않는다 (2026-09-14 조작자 지시) -----------

def test_area_deficit_is_left_to_z_while_the_force_is_moving():
    """용적이 바뀌면 z 가 파고들거나 물러나고, 그 동안 면적이 변한다.

    그것은 z 가 이미 고치고 있는 것이다 — 여기서 훑기를 시작하면 같은 오차를 두 축이
    서로 모르게 쫓는다. 조작자가 주사기로 물을 넣고 빼는 시험에서 그대로 일어난다.
    """
    c = RuleController(_q_gains(), find_axes=("thx",), q_target=0.90)
    c.phase = HOLD
    vel, note = c.update(0.0, True, 0.0, 0.02, q_view=0.70,
                         q_parts=_parts(area=0.1, presence=0.6), force_settled=False)
    assert c.phase == HOLD, "힘이 움직이는데 훑기로 갔다"
    assert not np.any(vel), vel
    assert "힘이 아직" in note, note


def test_centring_still_runs_while_the_force_is_moving():
    """중심은 z 와 무관하다 — 좌우 이동은 힘 조절에 쓰지 않으므로 서로 간섭이 없다."""
    from rus_policy.rule_control import AXIS_INDEX
    c = RuleController(_q_gains(), find_axes=("thx",), q_target=0.90)
    c.phase = HOLD
    vel, note = c.update(0.0, True, -0.25, 0.06, q_view=0.80,
                         q_parts=_parts(centering=0.2), force_settled=False)
    assert vel[AXIS_INDEX["thy"]] != 0.0, note


def test_the_settled_check_is_the_runners_job():
    """러너가 F_n 과 f_bar 로 판단해 넘긴다. 설정값을 모르면 멈춰 세우지 않는다."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    fn = re.search(r"def _force_settled.*?\n    #", src, re.S).group(0)
    assert "self.f_bar is None" in fn and "return True" in fn, fn
    assert "force_settled=self._force_settled(fn)" in src


def test_the_rule_controller_can_never_command_the_beam_axis():
    """힘은 z 만으로 조절한다 — 제어기에 z 자리가 아예 없다 (조작자 지시 2026-09-14)."""
    from rus_policy.rule_control import AXIS_INDEX
    assert 2 not in AXIS_INDEX.values(), AXIS_INDEX


def test_the_lateral_gate_survives_without_a_rule_controller():
    """hold·placebo·policy·search 에는 규칙 제어기가 없다 — 거기서 부르면 죽었다.

    2026-09-14 본 시험 3 번째 에피소드(hold)가 여기서 AttributeError 로 멈췄다. 24 개
    중 3 개째였다. 그런데 **복귀는 조건과 무관하게 병진을 쓴다** — hold 도 기준 자세로
    돌아가야 하므로, 규칙 제어기가 없다고 통째로 건너뛰어도 안 된다.
    """
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    fn = re.search(r"def _enable_lateral_if_needed.*?\n    def ", src, re.S).group(0)
    assert "if self.rule is not None:" in fn, "규칙 제어기가 없는 경우를 안 가린다"
    assert "self.return_to is not None" in fn, "복귀에 병진이 필요한 것을 안 본다"
    guard = fn.index("if self.rule is not None:")
    use = fn.index("self.rule.align_axis")
    assert guard < use, "먼저 확인하지 않는다"
