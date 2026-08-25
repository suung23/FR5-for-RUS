"""접근/접촉 판정기 — 법선력 하나로 단계를 가른다.

`teleop 종결 시점 = freespace 해제 시점 = 접촉 시작` 이라는 규칙을 한 곳에 담는다.
``joint_command_limiter`` 와 같은 이유로 rclpy 없이 돈다 — 이 판정의 버그는 잡음
한 번, 자세 한 번에 걸리는 종류라 실로봇 앞이 아니라 pytest 로 찾아야 한다.

**입력은 프로브 프레임의 F_z 다.** 부호 규약은 DESIGN_NOTES §4.4:

    F_n = normal_force_sign x F_z^probe          양수 = 압축

``normal_force_sign = -1.0`` (probe.yaml 현재값) 이면 압축이 F_z 음수로 들어오고,
``F_z <= -7 N`` 이 곧 ``F_n >= 7 N`` 이다. 부호가 반대로 밝혀지면 파라미터만 바꾸면
되고 이 모듈은 손대지 않는다.

세 가지를 넣었다. 셋 다 없으면 문턱 하나로는 못 쓴다.

**이력.** 진입과 이탈 문턱을 다르게 둔다. 하나로 두면 문턱 근처에서 상태가 떨린다.
그 떨림이 곧 freespace 클램프의 on/off 라 로봇이 덜컥거린다. 이탈 문턱은
``probe.yaml`` 의 ``watchdog.retreat_until_force_n`` (0.2 N, "분리 완료" 판정) 과
같은 자리에 두는 것이 자연스럽다 — 접촉이 끝났다는 판단은 한 곳에서만 내려야 한다.

**확인 시간.** 문턱을 넘은 것이 잡음이나 충격 스파이크 한 번일 수 있다. 연속으로
일정 시간 넘어야 접촉으로 친다. 1 kHz 에서 5 ms 면 5 표본이다. 이탈에도 같은 것을
둔다 — 누르는 중의 순간적인 힘 감소로 freespace 가 다시 켜지면 위험하다.

**걸쇠.** ``latched`` 는 접촉이 **한 번이라도** 있었는지를 기억한다. 상태가 접근으로
돌아와도 남는다. "teleop 종결" 처럼 되돌리면 안 되는 판단이 이것을 본다. 상태만
보면 프로브를 살짝 떼는 순간 teleop 이 되살아난다.

⚠️ **자중이 보상되지 않는다** (DESIGN_NOTES §2.3, ``payload.mass_kg`` 미식별).
센서는 프로브 자중과 접촉력을 함께 재고, 자중의 F_z 투영은 **자세에 따라 변한다**.
2026-08-24 무부하 측정에서 F_z 는 -1.2 ~ -1.6 N 이었다. 7 N 문턱에 대해 20 % 를
넘는 몫이다. ``bias_n`` 으로 상수 오프셋을 뺄 수 있게 열어 두었으나, 상수로는 자세
의존성을 지울 수 없다. 페이로드 식별 전까지 이 판정은 **보수적으로만** 쓸 것.
"""
from __future__ import annotations

__all__ = ["APPROACH", "CONTACT", "ContactDetector"]

#: 프로브가 아직 아무것도 만지지 않은 단계. freespace 스케일을 쓴다.
APPROACH = "approach"

#: 접촉 단계. freespace 를 내리고 접촉용 상한으로 간다.
CONTACT = "contact"


class ContactDetector:
    """법선력으로 접근/접촉을 가른다.

    시간은 밖에서 받는다 (``dt``). 벽시계를 안 읽으므로 기록을 되돌려 그대로 다시
    판정할 수 있고, 시험에서 초 단위 시나리오를 즉시 돌릴 수 있다.
    """

    def __init__(
        self,
        enter_n: float = 7.0,
        release_n: float = 0.2,
        normal_force_sign: float = -1.0,
        bias_n: float = 0.0,
        confirm_s: float = 0.005,
        release_confirm_s: float = 0.050,
    ) -> None:
        """판정기를 만든다.

        Args:
            enter_n: 접촉 진입 문턱 [N], ``F_n`` 기준. probe.yaml 의
                ``safety.max_normal_force_n`` 과 같은 값을 쓴다.
            release_n: 접촉 이탈 문턱 [N]. ``enter_n`` 보다 작아야 한다.
            normal_force_sign: ``F_n = sign x F_z``. probe.yaml 의
                ``ft_sensor.normal_force_sign``. §4.4 미검증 항목이다.
            bias_n: ``F_n`` 에서 뺄 상수 오프셋 [N]. 자중의 일부를 지우는 용도이며,
                자세 의존분은 지우지 못한다.
            confirm_s: 진입을 확정하기까지 연속으로 넘어야 하는 시간 [s].
            release_confirm_s: 이탈을 확정하기까지의 시간 [s]. 진입보다 길게 둔다 —
                누르는 중의 순간적인 힘 감소로 freespace 가 되살아나면 위험하다.

        Raises:
            ValueError: 문턱 순서가 뒤집혔거나 확인 시간이 음수다.
        """
        if release_n >= enter_n:
            raise ValueError(
                f"release_n({release_n}) 은 enter_n({enter_n}) 보다 작아야 한다 — "
                "같거나 크면 이력이 없어 문턱 근처에서 상태가 떤다"
            )
        if confirm_s < 0.0 or release_confirm_s < 0.0:
            raise ValueError("확인 시간은 음수일 수 없다")

        self.enter_n = float(enter_n)
        self.release_n = float(release_n)
        self.normal_force_sign = float(normal_force_sign)
        self.bias_n = float(bias_n)
        self.confirm_s = float(confirm_s)
        self.release_confirm_s = float(release_confirm_s)

        self.state = APPROACH
        self.latched = False
        self.normal_force_n = 0.0
        self._above_s = 0.0
        self._below_s = 0.0

    def normal_force(self, f_z: float) -> float:
        """프로브 프레임 F_z 를 법선력으로 바꾼다. 양수 = 압축."""
        return self.normal_force_sign * float(f_z) - self.bias_n

    def reset(self, *, unlatch: bool = False) -> None:
        """상태를 접근으로 되돌린다.

        Args:
            unlatch: 걸쇠까지 푼다. 새 시행을 시작할 때만 쓴다 — 접촉이 있었다는
                사실을 조용히 지우는 것이라 기본값은 유지다.
        """
        self.state = APPROACH
        self._above_s = 0.0
        self._below_s = 0.0
        if unlatch:
            self.latched = False

    def update(self, f_z: float, dt: float) -> str:
        """표본 하나를 넣고 현재 단계를 돌려준다.

        Args:
            f_z: 프로브 프레임의 z 방향 힘 [N]. 센서 프레임이 아니다 —
                ``ft_sensor.j6_to_sensor_rpy`` 로 옮긴 뒤의 값이어야 한다.
            dt: 직전 표본과의 간격 [s].

        Returns:
            :data:`APPROACH` 또는 :data:`CONTACT`.
        """
        f_n = self.normal_force(f_z)
        self.normal_force_n = f_n
        dt = max(0.0, float(dt))

        if self.state is APPROACH or self.state == APPROACH:
            # 문턱 아래로 한 번만 내려가도 확인 시간을 처음부터 다시 센다.
            # "대체로 넘었다" 는 접촉이 아니다.
            if f_n >= self.enter_n:
                self._above_s += dt
                if self._above_s >= self.confirm_s:
                    self.state = CONTACT
                    self.latched = True
                    self._below_s = 0.0
            else:
                self._above_s = 0.0
        else:
            if f_n <= self.release_n:
                self._below_s += dt
                if self._below_s >= self.release_confirm_s:
                    self.state = APPROACH
                    self._above_s = 0.0
            else:
                self._below_s = 0.0

        return self.state

    @property
    def freespace_allowed(self) -> bool:
        """지금 freespace 스케일을 써도 되는가.

        상태가 아니라 **걸쇠**를 본다. 접촉이 한 번이라도 있었으면 프로브를 떼도
        돌려주지 않는다 — 접촉 단계에 들어선 작업은 되돌아가는 것이 아니라 끝내고
        다시 시작하는 것이기 때문이다. 되돌리려면 :meth:`reset` 에 ``unlatch=True``.
        """
        return not self.latched

    def describe(self) -> str:
        """사람이 읽을 한 줄 요약. 감시 화면과 로그에 그대로 쓴다."""
        mark = "접촉" if self.state == CONTACT else "접근"
        latch = " [걸쇠]" if self.latched else ""
        return f"{mark}{latch}  F_n = {self.normal_force_n:+.3f} N"
