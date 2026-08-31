"""접촉 프로빙 힘 조절기 — 일정한 힘 구간을 유지한다 (DESIGN_NOTES §7, 2026-08-25).

접촉 프로빙 모드에서 **침투축(프로브 +z)은 로봇이 스스로 잡는다.** 조작자가 손으로
누르는 깊이를 맞추는 것이 아니라, 목표 힘 구간 안에 들어오도록 로봇이 z 를 움직인다.

**무엇을 잡는가.** 이 조절기는 스칼라 하나를 받아 목표에 맞춘다. 그 스칼라가 법선력
``F_n`` 인지 접촉력 크기 ``‖F‖`` 인지는 호출자가 정한다 (``us_diff_ik_node`` 의
``ft_sensor.contact_force_mode``, 기본 ``magnitude``). 2026-08-31 이전에는 법선력
전용이었다.

⚠️ 크기로 잡을 때의 성질 하나: ``‖F‖ ≥ |F_z|`` 이므로 횡력이 실리면 z 를 덜 눌러도
목표에 도달한다. 즉 프로브가 옆으로 끌리는 동안에는 **침투 깊이가 줄어드는 쪽**으로
동작한다. 조직에 과하게 파고드는 방향이 아니므로 안전한 쪽으로 틀리지만, "목표 힘을
유지 중" 이 곧 "법선으로 목표만큼 누르는 중" 은 아니다 — 화면이 총합·법선·횡력을
따로 보여 주는 이유다.

이것은 §7 admittance 의 첫 구현이다. 아직 policy 도 QP arbiter 도 없으므로 힘 축
하나만 닫는다. 나머지 축은 호출자가 정한다 (기본은 0 — 프로브가 제자리에서 힘만
유지한다).

제어 법칙
---------
목표 힘 ``F*`` 와 밴드 ``±d`` 를 둔다::

    e = F* - F_n
    |e| <= d        →  v_z = 0            밴드 안. 움직이지 않는다
    |e| >  d        →  v_z = e / B_z      admittance. 부족하면 전진, 과하면 후퇴

밴드를 두는 이유는 잡음 위에서 떨지 않기 위해서다. PX6D 잡음은 축당 ±0.05 N 이고
자중 미보상분이 자세에 따라 1.5 N 가량 움직이므로, 밴드 없이 순수 비례제어를 하면
프로브가 계속 미세하게 오르내린다.

**절대 넘지 않는다**
--------------------
두 가지가 이것을 보장한다. 하나는 검사이고, 하나는 구조다.

**구조 — 경고 이상에서는 전진할 수 없다.** 생성자가 ``target < warn`` 을 강제한다.
그러면 ``F_n >= warn`` 인 순간 ``e = target - F_n < 0`` 이므로 admittance 항이 반드시
음수다. 즉 "경고 이상에서 전진 금지" 를 위한 별도 분기가 필요 없다 — 그런 분기를
두면 **절대 실행되지 않는 코드**가 되고, 실행되지 않는 보호는 보호가 아니라 보호받고
있다는 착각이다. 이 성질은 시험이 힘 구간 전체에 대해 확인한다
(``test_never_advances_at_or_above_warn``).

**검사 — 한계 이상에서는 강제 후퇴.** ``F_n >= max_force_n`` 이면 목표와 무관하게
``v_z = -retreat_speed`` 를 낸다. 이것은 admittance 항과 밴드를 모두 덮는다. 힘이 이미
한계를 넘은 상황에서 "목표까지의 오차" 를 계산하는 것은 뜻이 없다.

⚠️ 이 조절기는 **속도만 낸다.** 실제로 얼마나 움직일지는 하류의 속도 클램프와
서보가 정한다. 여기서 낸 값이 그대로 실행된다고 가정하지 말 것.
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ForceRegulator", "RegulatorOutput"]


@dataclass(frozen=True)
class RegulatorOutput:
    """한 주기의 결정.

    Attributes:
        v_z: 프로브 +z 속도 [m/s]. 양수 = 조직으로 전진.
        reason: 무엇이 이 값을 정했는가. 로그·화면에 그대로 쓴다 — 조작자가
            "왜 안 움직이나" 를 물을 때 답이 되어야 한다.
    """

    v_z: float
    reason: str


class ForceRegulator:
    """접촉력을 목표 구간 안에 잡아 두는 1 축 admittance."""

    def __init__(
        self,
        target_force_n: float,
        deadband_n: float,
        admittance_b_z: float,
        max_speed_m_s: float,
        warn_force_n: float,
        max_force_n: float,
        retreat_speed_m_s: float,
    ) -> None:
        """조절기를 만든다.

        Args:
            target_force_n: 유지할 힘 [N]. 양수 = 압축.
            deadband_n: 목표 둘레 밴드 반폭 [N]. 이 안에서는 안 움직인다.
            admittance_b_z: 감쇠 [N·s/m]. 클수록 같은 오차에 천천히 움직인다.
            max_speed_m_s: 이 축 속도 상한 [m/s].
            warn_force_n: 이 힘 이상에서는 전진하지 않는다.
            max_force_n: 이 힘 이상에서는 강제로 후퇴한다.
            retreat_speed_m_s: 강제 후퇴 속도 [m/s]. 양수로 준다.

        Raises:
            ValueError: 값이 서로 모순된다.
        """
        if deadband_n < 0.0:
            raise ValueError(f"deadband_n 은 음수일 수 없다: {deadband_n}")
        if admittance_b_z <= 0.0:
            raise ValueError(f"admittance_b_z 는 양수여야 한다: {admittance_b_z}")
        if max_speed_m_s <= 0.0 or retreat_speed_m_s <= 0.0:
            raise ValueError("속도 상한과 후퇴 속도는 양수여야 한다")
        if not (target_force_n < warn_force_n <= max_force_n):
            # 목표가 경고 위에 있으면 조절기가 자기 목표를 향해 가는 것 자체가
            # 금지된다 — 영원히 후퇴만 한다. 만들 때 막는다.
            raise ValueError(
                f"target({target_force_n}) < warn({warn_force_n}) <= max({max_force_n}) "
                "이어야 한다"
            )

        self.target_force_n = float(target_force_n)
        self.deadband_n = float(deadband_n)
        self.admittance_b_z = float(admittance_b_z)
        self.max_speed_m_s = float(max_speed_m_s)
        self.warn_force_n = float(warn_force_n)
        self.max_force_n = float(max_force_n)
        self.retreat_speed_m_s = float(retreat_speed_m_s)

    def update(self, contact_force_n: float) -> RegulatorOutput:
        """접촉력 하나를 보고 이 주기의 z 속도를 정한다.

        Args:
            contact_force_n: 현재 접촉력 [N]. 양수 = 누름. 크기 ``‖F‖`` 인지 법선력
                ``F_n`` 인지는 호출자가 정한다.

        Returns:
            :class:`RegulatorOutput`.
        """
        force = float(contact_force_n)

        # 1) 한계 초과 — 목표와 무관하게 후퇴한다.
        if force >= self.max_force_n:
            return RegulatorOutput(
                v_z=-self.retreat_speed_m_s,
                reason=f"한계 초과 {force:.2f} ≥ {self.max_force_n:.1f} N — 강제 후퇴",
            )

        error = self.target_force_n - force

        # 2) 밴드 안 — 움직이지 않는다.
        if abs(error) <= self.deadband_n:
            return RegulatorOutput(
                v_z=0.0,
                reason=(
                    f"밴드 안 {force:.2f} N "
                    f"(목표 {self.target_force_n:.1f} ± {self.deadband_n:.1f})"
                ),
            )

        # 3) admittance. target < warn 불변식 덕분에, 여기서 나온 v_z 가 양수이면
        #    force < target < warn 이 이미 성립한다 — 경고 이상에서 전진하는 경우가
        #    구조적으로 없다. 별도 분기를 두지 않는 이유다.
        v_z = error / self.admittance_b_z
        v_z = max(-self.max_speed_m_s, min(self.max_speed_m_s, v_z))

        direction = "전진" if v_z > 0 else "후퇴"
        return RegulatorOutput(
            v_z=v_z,
            reason=f"{direction} {abs(v_z) * 1000:.1f} mm/s (오차 {error:+.2f} N)",
        )
