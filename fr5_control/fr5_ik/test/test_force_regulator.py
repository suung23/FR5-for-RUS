"""force_regulator 회귀 테스트 — 무엇이 v_z 를 정하는가, 그리고 그 순서.

이 조절기는 프로브가 조직에 닿아 있는 동안 로봇이 스스로 내는 유일한 속도다.
틀리는 두 방향의 무게가 다르다.

  * **너무 밀면** 조직을 손상시킨다. 되돌릴 수 없다.
  * **너무 물러나면** 접촉을 잃는다. 성가시지만 다시 대면 된다.

그래서 한계 층이 admittance 항을 **덮는지**, 그 순서가 맞는지를 먼저 고정한다.
"""
from fr5_ik.force_regulator import ForceRegulator
import pytest

# probe.yaml contact_control 기본값
TARGET, DEADBAND, B_Z = 5.0, 0.5, 1000.0
MAX_SPEED, RETREAT = 0.010, 0.005
WARN, MAX = 9.0, 10.0


def make(**kw):
    """기본값으로 조절기 하나. 필요한 것만 덮어쓴다."""
    args = {
        "target_force_n": TARGET, "deadband_n": DEADBAND, "admittance_b_z": B_Z,
        "max_speed_m_s": MAX_SPEED, "warn_force_n": WARN, "max_force_n": MAX,
        "retreat_speed_m_s": RETREAT,
    }
    args.update(kw)
    return ForceRegulator(**args)


# -- 밴드 -------------------------------------------------------------------

@pytest.mark.parametrize("force", [4.6, 5.0, 5.4])
def test_inside_the_band_it_does_not_move(force):
    """밴드 안에서는 정지. 잡음 위에서 떨지 않기 위한 것이다."""
    assert make().update(force).v_z == 0.0


def test_band_edges_are_inclusive():
    """정확히 경계면 아직 밴드 안이다. 경계에서 깜빡이지 않는다."""
    assert make().update(TARGET - DEADBAND).v_z == 0.0
    assert make().update(TARGET + DEADBAND).v_z == 0.0


# -- admittance -------------------------------------------------------------

def test_too_light_advances():
    """힘이 부족하면 조직 쪽으로 간다."""
    out = make().update(1.0)
    assert out.v_z > 0


def test_too_firm_retreats():
    """밴드 위면 물러난다 — 경고에 닿기 전에 스스로 줄인다."""
    out = make().update(7.0)
    assert out.v_z < 0


def test_speed_follows_the_admittance_law():
    """v_z = (F* - F_n) / B_z. 상한에 안 걸리는 구간에서 정확히 성립한다."""
    force = 3.0
    assert make().update(force).v_z == pytest.approx((TARGET - force) / B_Z)


def test_speed_is_clamped():
    """오차가 아무리 커도 이 축 상한을 넘지 않는다."""
    assert make().update(-500.0).v_z == pytest.approx(MAX_SPEED)
    assert make(warn_force_n=1e6, max_force_n=1e7).update(500.0).v_z == pytest.approx(-MAX_SPEED)


def test_larger_damping_moves_slower():
    """B_z 는 감쇠다. 키우면 같은 오차에 천천히 간다."""
    fast = make(admittance_b_z=500.0).update(3.0).v_z
    slow = make(admittance_b_z=2000.0).update(3.0).v_z
    assert 0 < slow < fast


# -- 한계 층 ----------------------------------------------------------------

def test_never_advances_at_or_above_warn():
    """경고 이상에서 전진하는 경우가 **없다** — 힘 구간 전체에 대해 확인한다.

    이것은 분기로 막은 것이 아니라 ``target < warn`` 불변식에서 따라 나온다. 분기로
    막으면 그 분기는 절대 실행되지 않는 코드가 되고, 실행되지 않는 보호는 보호받고
    있다는 착각이다. 그래서 성질 자체를 시험한다.

    설정도 함께 흔든다 — 목표를 경고 바로 아래까지 올려도 성질이 유지되어야 한다.
    """
    for target in (1.0, 5.0, 8.99):
        reg = make(target_force_n=target)
        force = WARN
        while force <= MAX * 2:
            assert reg.update(force).v_z <= 0.0, f"target={target} force={force}"
            force += 0.05


def test_at_warn_it_may_still_retreat():
    """전진만 막는다. 물러나는 것은 언제나 허용된다."""
    assert make().update(9.2).v_z < 0


def test_at_max_it_retreats_regardless():
    """한계 이상이면 목표와 무관하게 후퇴한다."""
    out = make().update(10.5)
    assert out.v_z == pytest.approx(-RETREAT)
    assert "강제 후퇴" in out.reason


def test_max_overrides_the_band():
    """목표가 한계 위에 있을 수는 없지만, 밴드가 한계에 걸쳐도 후퇴가 이긴다."""
    out = make(deadband_n=8.0).update(10.2)
    assert out.v_z == pytest.approx(-RETREAT)


def test_forced_retreat_is_not_clamped_to_zero_by_the_warn_layer():
    """층 순서 확인 — 한계 층이 경고 층보다 먼저 결정한다."""
    assert make().update(12.0).v_z < 0


# -- 진입 시나리오 ----------------------------------------------------------

def test_settles_from_the_transition_force_to_the_band():
    """8 N 에서 접촉 프로빙에 들어와 목표 밴드로 수렴한다.

    전환 문턱(8 N)은 목표(5 N)보다 높다. 접촉을 확실히 잡은 뒤 작업 힘으로 내려오는
    것이 의도다 — 조절기가 그 방향으로 움직이는지 본다.
    """
    reg = make()
    force = 8.0
    for _ in range(400):
        v = reg.update(force).v_z
        # 조직을 1 N/mm 스프링으로 본 아주 거친 모형. 방향만 본다.
        force += v * 1000.0 * 1.0 * 0.01
        if abs(force - TARGET) <= DEADBAND:
            break
    assert abs(force - TARGET) <= DEADBAND


def test_contact_loss_drives_it_forward_again():
    """접촉을 잃으면(0 N) 다시 전진한다 — 밴드로 돌아가려 한다."""
    assert make().update(0.0).v_z > 0


# -- 설정 -------------------------------------------------------------------

def test_target_above_warn_is_rejected():
    """목표가 경고 위면 영원히 후퇴만 한다. 만들 때 막는다."""
    with pytest.raises(ValueError):
        make(target_force_n=9.5)


def test_warn_above_max_is_rejected():
    with pytest.raises(ValueError):
        make(warn_force_n=11.0)


@pytest.mark.parametrize("bad", [{"deadband_n": -0.1}, {"admittance_b_z": 0.0},
                                 {"max_speed_m_s": 0.0}, {"retreat_speed_m_s": -1.0}])
def test_nonsense_configuration_is_rejected(bad):
    with pytest.raises(ValueError):
        make(**bad)


def test_reason_is_always_populated():
    """조작자가 '왜 안 움직이나' 를 물을 때 답이 있어야 한다."""
    for force in (-1.0, 0.0, 5.0, 7.0, 9.5, 12.0):
        assert make().update(force).reason
