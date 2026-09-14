"""법선 정렬 — Q_seg 가 합격한 여유에서만 결합을 개선한다."""

import numpy as np
import pytest

from rus_policy.normal_align import THX, THY, NormalAlign


class Surface:
    """기울어진 면. 기울기가 남아 있는 만큼 전단이 생긴다.

    ``true_sign`` 은 "어느 쪽으로 돌려야 전단이 주는가" 를 정하는 장비의 성질이다.
    제어기는 이것을 모르고 시작한다.
    """

    def __init__(self, tilt=(0.0, 0.0), gain=0.25, true_sign=1.0):
        self.tilt = np.array(tilt, float)      # [thx, thy] 잔여 기울기 [°]
        self.gain, self.true_sign = gain, true_sign

    def step(self, vel, dt):
        self.tilt = self.tilt - self.true_sign * np.array([vel[THX], vel[THY]]) * dt
        return self.shear()

    def shear(self):
        # 전단은 기울기에 비례한다. thy 기울기가 +x 전단을, thx 가 −y 전단을 만든다.
        return (self.gain * self.tilt[1], -self.gain * self.tilt[0])


def run(na, s, seconds=120.0, hz=5.0):
    t, dt = 0.0, 1.0 / hz
    sh = s.shear()
    for _ in range(int(seconds * hz)):
        vel, _ = na.step(t, sh)
        sh = s.step(vel, dt)
        t += dt
    return float(np.hypot(*sh))


def test_it_does_nothing_when_already_normal():
    na = NormalAlign(shear_ok_n=0.3)
    vel, note = na.step(0.0, (0.05, -0.05))
    assert not np.any(vel), note
    assert "충분히 법선" in note


def test_it_reduces_the_shear():
    na = NormalAlign(shear_ok_n=0.10, k_p=1.0, max_deg_s=0.5, max_excursion_deg=10.0)
    s = Surface(tilt=(2.0, -2.5), gain=0.25, true_sign=+1.0)
    before = float(np.hypot(*s.shear()))
    after = run(na, s)
    assert after < before * 0.5, f"{before:.2f} → {after:.2f} N"


def test_it_learns_the_sign_when_it_is_backwards():
    """부호를 틀리게 박아 두면 전단을 **키우는** 쪽으로 돈다 — 스스로 뒤집어야 한다."""
    na = NormalAlign(shear_ok_n=0.10, k_p=1.0, max_deg_s=0.5,
                     max_excursion_deg=10.0, check_s=3.0)
    s = Surface(tilt=(2.0, -2.5), gain=0.25, true_sign=-1.0)   # 규약이 반대인 장비
    before = float(np.hypot(*s.shear()))
    after = run(na, s, seconds=180.0)
    assert na.flips >= 1, "부호를 안 뒤집었다"
    assert after < before, f"{before:.2f} → {after:.2f} N — 여전히 키우고 있다"


def test_a_wrong_sign_costs_only_one_check_window():
    """한 확인 주기만 잃어야 한다 — 그 사이 누적이 상한을 넘으면 안 된다."""
    na = NormalAlign(shear_ok_n=0.10, max_deg_s=0.5, check_s=3.0, max_excursion_deg=4.0)
    s = Surface(tilt=(2.0, -2.5), true_sign=-1.0)
    run(na, s, seconds=20.0)
    assert na.excursion_deg <= 4.0 + 1e-6, na.pos


def test_the_excursion_is_capped():
    """이만큼 갔는데도 전단이 안 줄면 자세가 아니라 다른 것이 원인이다."""
    na = NormalAlign(shear_ok_n=0.05, max_deg_s=1.0, max_excursion_deg=3.0)
    s = Surface(tilt=(50.0, 50.0), gain=0.25, true_sign=0.0)   # 돌려도 안 변하는 면
    run(na, s, seconds=60.0)
    assert na.excursion_deg <= 3.0 + 1e-6, na.pos
    vel, note = na.step(999.0, s.shear())
    assert not np.any(vel) and "상한" in note, note


def test_it_never_touches_translation_or_the_beam_axis():
    na = NormalAlign(shear_ok_n=0.01)
    vel, _ = na.step(0.0, (1.0, -1.0))
    assert np.all(vel[:3] == 0.0), vel
    assert vel[5] == 0.0, "빔축 회전은 쓰지 않는다"


def test_unreadable_shear_is_not_a_command():
    na = NormalAlign()
    for bad in (None, (float("nan"), 0.0)):
        vel, note = na.step(0.0, bad)
        assert not np.any(vel), note


def test_reset_keeps_the_learned_sign():
    """자세를 새로 잡아도 부호는 장비의 성질이다 — 매번 다시 배우면 그때마다 손해다."""
    na = NormalAlign()
    na.sign, na.flips = -1.0, 1
    na.pos = np.array([2.0, 1.0])
    na.reset()
    assert na.sign == -1.0 and na.excursion_deg == 0.0


def test_speed_is_slow_by_default():
    """남는 여유에서 하는 일이다 — 빠르면 Q_seg 를 흔든다."""
    na = NormalAlign()
    assert na.max_deg_s <= 1.0
    vel, _ = na.step(0.0, (50.0, -50.0))
    assert np.all(np.abs(vel[3:5]) <= na.max_deg_s + 1e-9)


# --- 규칙 제어기와의 관계 (2026-09-14 조작자 지시: C 안) ------------------------

def _gains():
    from rus_policy.axis_gains import AxisGains
    return AxisGains(centroid_per_unit={"thy": 0.01913}, area_per_unit={"thx": -0.00189},
                     sign_agreement={"thy": 0.83})


def _parts(**kw):
    from rus_policy.view_quality import WEIGHTS
    p = {k: 1.0 for k in WEIGHTS}
    p.update(kw)
    return p


def test_it_only_runs_above_the_pass_line():
    """합격선 아래에서는 방광이 먼저다 — 같은 축을 두 목적이 다투면 안 된다."""
    from rus_policy.rule_control import HOLD, RuleController
    c = RuleController(_gains(), find_axes=("thx",), q_target=0.90)
    c.phase = HOLD
    # 합격 못 함 + 중심 부족 → 정렬이 이겨야 한다 (법선 정렬이 끼어들면 안 된다)
    vel, note = c.update(0.0, True, -0.25, 0.06, q_view=0.80,
                         q_parts=_parts(centering=0.2), shear_xy=(2.0, -2.0))
    assert "법선 정렬" not in note, note
    assert vel[THY] != 0.0, note


def test_it_uses_the_spare_capacity_when_passing():
    from rus_policy.rule_control import HOLD, RuleController
    c = RuleController(_gains(), find_axes=("thx",), q_target=0.90)
    c.phase = HOLD
    vel, note = c.update(0.0, True, 0.0, 0.06, q_view=0.94, q_parts=_parts(),
                         shear_xy=(1.0, -1.0))
    assert "합격" in note and "법선 정렬" in note, note
    assert np.any(vel[3:5]), "여유가 있는데 아무것도 안 한다"


def test_passing_with_low_shear_still_stands_still():
    """전단이 이미 작으면 건드릴 이유가 없다."""
    from rus_policy.rule_control import HOLD, RuleController
    c = RuleController(_gains(), find_axes=("thx",), q_target=0.90)
    c.phase = HOLD
    vel, note = c.update(0.0, True, 0.0, 0.06, q_view=0.94, q_parts=_parts(),
                         shear_xy=(0.05, 0.05))
    assert not np.any(vel), note
    assert "충분히 법선" in note, note


def test_it_can_be_turned_off():
    from rus_policy.rule_control import HOLD, RuleController
    c = RuleController(_gains(), find_axes=("thx",), q_target=0.90, normal_align=False)
    c.phase = HOLD
    vel, note = c.update(0.0, True, 0.0, 0.06, q_view=0.94, q_parts=_parts(),
                         shear_xy=(2.0, -2.0))
    assert not np.any(vel) and "가만히 둔다" in note, note


def test_the_runner_reads_shear_from_the_compensated_wrench():
    """보상 전 렌치에는 자중 10 N 이 실려 있다 — 그것을 전단으로 읽으면 안 된다."""
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    fn = re.search(r"def _shear_xy.*?\n    def ", src, re.S).group(0)
    assert "self.wrench" in fn and "return None" in fn, fn
    assert "shear_xy=self._shear_xy()" in src
    # 기록에도 남아야 한다 — 나중에 "결합이 왜 안 올랐나" 를 물을 수 있어야 한다
    for col in ('"shear_n"', '"na_deg"', '"na_sign"'):
        assert col in src, f"{col} 을 기록하지 않는다"
