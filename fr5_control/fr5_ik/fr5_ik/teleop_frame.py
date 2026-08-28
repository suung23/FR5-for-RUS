"""teleop 병진 지령의 기준 프레임 (DESIGN_NOTES §10.5, 2026-08-21).

왜 필요한가
-----------
teleop 경로는 원래 **body → body** 였다::

    v_base = R_base_probe(t) · A · R_stylus(t)^T · v_world

``touch_twist_node`` 가 손 속도를 스타일러스 body 프레임으로 투영하고
(``vel_body = R^T · vel_world``), ``us_diff_ik_node`` 가 그 twist 를 프로브 body
프레임으로 되돌린다. "손을 오른쪽으로 밀면 로봇이 어디로 가는가" 를 **두 자세의 상대
관계**가 정한다는 뜻이다.

rate control 에서 그 관계는 유지되지 않는다. 스타일러스 자세는 장치가 주는 절대값이지만
프로브 자세는 **지령 각속도의 적분**이고, 그 각속도는 ``angular_scale``(1.2 배) ·
``angular_deadzone``(0.05 rad/s 이하 버림) · 클램프 · 필터 · DLS 감쇠를 통과한다.
어느 하나도 1:1 이 아니다.

특히 데드존이 **정류기**로 동작한다. 빠르게 비트는 동작은 통과하고 천천히 되돌리는
동작은 문턱 아래라 통째로 버려지므로, 한 방향으로만 적분된다.

2026-08-21 실측 (``touch_teleop/tools/teleop_frame_audit.py``): "스타일러스 +X 밀기"가
로봇 base 에서 가리키는 방향이 한 세션에 **10° → 104°** 로 단조 증가했고 (최대 135°),
데드맨을 놓았다 다시 잡아도 전혀 되돌아오지 않았다 (76.7→76.7, 86.3→87.3, 96.7→96.4).
한 파지 37 초 안에서만 10° → 76.7° 가 쌓였다 — **놓는 것이 원인을 만드는 게 아니라,
놓아도 리셋되지 않는 것이 문제였다.**

무엇을 하는가
-------------
병진에서 ``R_stylus(t)`` 와 ``R_base_probe(t)`` 를 **둘 다 걷어낸다**::

    v_world = R_stylus(t) · Aᵀ · v_cmd      지령을 장치 world 프레임으로 되돌리고
    v_base  = R_latch · v_world             고정 행렬 하나로 base 에 싣는다

``R_latch`` 는 상수다. 시각에 의존하는 항이 없으므로 **표류가 원리적으로 불가능하다.**

``R_latch`` 는 첫 파지 순간의 ``R_base_probe · A · R_stylusᵀ`` 로 잡는다. 그 순간에는
새 매핑이 기존 매핑과 **정확히 같아서**, 조작자가 파지를 시작할 때 느끼는 축은 지금과
동일하고 그 뒤로 어긋나지만 않는다.

회전에도 같은 것을 쓴다 (2026-08-25)
------------------------------------
처음에는 회전을 건드리지 않았다. 근거는 "손목을 비틀면 프로브가 비틀린다" 는 대응이
프로브를 보면서 조작자가 직접 확인·보정하는 축이라 병진처럼 화면 밖으로 어긋나지
않는다는 것이었다. **그 전제가 틀렸다** — 실기 조작에서 회전이 어긋난다는 보고가
나왔다.

회전 경로를 펼쳐 보면 병진과 같은 형태다::

    w_base = R_base_probe(t) · A · R_stylus(t)ᵀ · w_world

시간에 의존하는 회전 두 개가 그대로 있고, 그 사이를 지나는 것도 같다 —
``angular_scale`` · ``angular_deadzone`` · 클램프 · 필터 · DLS 감쇠. 데드존이
정류기로 동작하는 것까지 동일하다. 즉 표류 메커니즘이 병진과 다르지 않다.

**각속도에 같은 변환을 써도 되는 이유.** 각속도는 유사벡터(pseudovector)이므로 반사가
섞이면 벡터와 다르게 변환된다. 그러나 여기 등장하는 세 행렬은 모두 ``det = +1`` 인
진짜 회전이다 (``A = Rz · diag(+1,-1,-1)`` 는 ``det = +1``, 이것이 §10.4 에서 축 교환
대신 ``Rz`` 를 쓴 이유다). 진짜 회전에 대해서는 유사벡터도 벡터와 똑같이 변환되므로
``to_probe`` 를 병진·회전 양쪽에 그대로 쓸 수 있다.

병진과 회전은 **같은 ``R_latch``** 를 공유한다. 따로 잡으면 두 축이 서로 다른 순간에
고정되어, 손을 대각선으로 움직일 때 병진과 회전이 서로 다른 "오른쪽"을 갖는다.

조작자가 어디에 서 있는가 (2026-08-27)
--------------------------------------
위의 모든 것은 "손 좌표계 → base" 를 정할 뿐, **조작자가 로봇의 어느 쪽에 서 있는지**
는 모른다. 지금까지는 조작자와 로봇이 같은 방향을 보고 있다고 가정한 매핑이었다.
마주보고 서면 그 가정이 깨진다 — 조작자의 앞은 로봇의 뒤이고, 조작자의 오른쪽은
로봇의 왼쪽이다. 조작자에게는 "밀면 반대로 온다" 로 느껴진다.

고치는 방법은 base 수직축 둘레 회전 하나다::

    v_base = Rz_base(operator_yaw) · (지금까지의 v_base)

마주보기는 ``operator_yaw = 180°`` 다. 옆에 서면 ±90° 다.

**거울(반사)이 아니라 회전인 이유.** 사람이 "미러 모드" 라고 부르는 것은 보통 이
180° 회전이다. 진짜 거울상(``det = -1``)은 좌우만 뒤집고 앞뒤는 그대로 두는데, 로봇
반대편으로 걸어가면 **앞뒤도 같이 뒤집힌다.** 그것이 회전이다. 게다가 반사를 넣으면
그 twist 는 강체 운동이 아니게 되고, 각속도는 유사벡터라 회전 지령이 거울상으로
망가진다 (§10.4 가 축 교환 대신 ``Rz`` 를 쓴 이유와 같다). 그래서 반사는 제공하지
않는다 — 조작감으로 원하는 것은 회전 쪽이고, 그쪽만 물리적으로 성립한다.

**병진과 회전에 같은 ``Rz_base`` 를 쓴다.** 한쪽만 뒤집으면 미는 방향과 비트는
방향이 서로 다른 세계에 있게 된다.

값이 바뀌어도 **다음 파지부터** 적용된다. 조작 중에 바뀌면 같은 손동작에 로봇이
반대로 가고, 그 순간 조작자의 반사적인 교정은 상황을 악화시키는 방향이다.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "axis_mapping",
    "operator_rotation",
    "TeleopFrameMapper",
    "LinearFrameMapper",
]


def axis_mapping(tip_roll_deg: float) -> np.ndarray:
    """``touch_twist_node`` 의 ``A = Rz(tip_roll) · diag(+1, -1, -1)``.

    두 노드가 같은 ``A`` 를 써야 하므로 ``tip_roll_deg`` 는 양쪽이 같은
    ``probe.yaml`` 항목(``teleop.tip_roll_deg``)에서 읽는다. 값이 갈라지면 병진이
    조용히 90° 틀어진다.

    ``A`` 는 직교이고 ``det = +1`` 이라 역행렬이 전치와 같다 — 아래 되돌리기가
    수치오차 없이 정확한 이유다.
    """
    angle = math.radians(float(tip_roll_deg))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot_z = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ np.diag([1.0, -1.0, -1.0])


def operator_rotation(operator_yaw_deg: float) -> np.ndarray:
    """조작자가 선 자리를 나타내는 base 수직축 둘레 회전 ``Rz_base``.

    Args:
        operator_yaw_deg: 0 이면 조작자와 로봇이 같은 방향을 본다 (지금까지의 거동).
            180 이면 마주보고 선다 — 앞뒤와 좌우가 함께 뒤집힌다.

    Returns:
        3x3 회전. ``det = +1`` 이라 병진과 회전(유사벡터)에 똑같이 쓸 수 있다.
    """
    angle = math.radians(float(operator_yaw_deg))
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])


class TeleopFrameMapper:
    """teleop 지령(병진·회전)을 표류하지 않는 고정 프레임으로 옮긴다.

    Args:
        tip_roll_deg: ``teleop.tip_roll_deg``. teleop 노드와 **같은 값**이어야 한다.
        relatch_on_engage: 참이면 데드맨을 잡을 때마다 기준을 다시 잡는다. 거짓이면
            세션 첫 파지에 한 번만 잡고 유지한다 (기본).

            거짓이 기본인 이유: 파지마다 다시 잡으면 표류는 안 쌓이지만 **파지마다
            축이 달라진다.** 조작자에게는 "가끔 축이 바뀐다" 로 느껴지고, 그것은
            천천히 어긋나는 것보다 오히려 헷갈린다. 세션 내내 한 축으로 고정하는 편이
            예측 가능하다. 조작자가 자리를 옮겨 기준을 다시 잡고 싶을 때만 참으로 둔다.
    """

    def __init__(
        self,
        tip_roll_deg: float,
        relatch_on_engage: bool = False,
        operator_yaw_deg: float = 0.0,
    ) -> None:
        self.mapping = axis_mapping(tip_roll_deg)
        self.relatch_on_engage = bool(relatch_on_engage)
        #: 지금 **적용 중인** 조작자 각도. 파지 중에는 바뀌지 않는다.
        self.operator_yaw_deg = float(operator_yaw_deg)
        self.operator = operator_rotation(self.operator_yaw_deg)
        #: 다음 파지에 적용될 값. ``None`` 이면 대기 중인 변경이 없다.
        self.pending_yaw_deg = None
        #: world → base 고정 회전. ``None`` 이면 아직 기준을 안 잡았다.
        self.latched = None
        self.engage_count = 0

    # -- 조작자 위치 ------------------------------------------------------

    def request_operator_yaw(self, operator_yaw_deg: float) -> bool:
        """조작자 각도 변경을 **예약한다.** 즉시 반영하지 않는다.

        조작 중에 축이 뒤집히면 같은 손동작에 로봇이 반대로 가고, 그 순간 조작자가
        반사적으로 하는 교정은 상황을 악화시키는 방향이다. 그래서 손을 놓았다 다시
        잡을 때 바뀐다 — ``teleop.angular_frame`` 과 같은 규약이다.

        Args:
            operator_yaw_deg: 새 각도 [도]. ``(-180, 180]`` 으로 정규화해 다룬다.

        Returns:
            대기 중인 변경이 생겼거나 남아 있으면 ``True``.
        """
        wanted = _normalize_deg(operator_yaw_deg)
        if abs(wanted - self.operator_yaw_deg) < 1e-9:
            self.pending_yaw_deg = None
        else:
            self.pending_yaw_deg = wanted
        return self.pending_yaw_deg is not None

    @property
    def mirrored(self) -> bool:
        """조작자가 로봇을 마주보는 쪽인가 (|yaw| > 90°)."""
        return abs(self.operator_yaw_deg) > 90.0

    # -- 기준 잡기 --------------------------------------------------------

    def engage(self, rot_base_probe: np.ndarray, rot_stylus: np.ndarray) -> bool:
        """데드맨을 잡았다. 필요하면 기준을 (다시) 잡는다.

        Returns:
            이번 호출에서 기준을 새로 잡았으면 ``True``.
        """
        self.engage_count += 1
        # 파지 경계는 조작자 각도가 바뀌어도 안전한 유일한 순간이다. latch 를 다시
        # 잡지 않는 경우에도 이것만은 적용한다 — 안 그러면 예약이 영원히 안 걸린다.
        self._apply_pending_yaw()
        if self.latched is not None and not self.relatch_on_engage:
            return False
        # 이 순간에는 새 매핑이 기존 매핑과 정확히 같다 (조작자 각도가 그대로라면).
        # 파지 시작 축은 안 바뀌고, 그 뒤로 어긋나지만 않게 된다.
        self.latched = (
            self.operator
            @ np.asarray(rot_base_probe, dtype=float)
            @ self.mapping
            @ np.asarray(rot_stylus, dtype=float).T
        )
        return True

    def _apply_pending_yaw(self) -> bool:
        """예약된 조작자 각도를 적용한다. 적용했으면 ``True``.

        이미 잡아 둔 ``R_latch`` 가 있으면 **다시 잡지 않고 차이만 곱한다.**
        세션 기준을 새로 잡으면 조작자가 요청하지도 않은 축 이동이 함께 일어난다 —
        ``relatch_on_engage`` 가 거짓인 이유가 그것이다. 조작자 회전만 갈아 끼우면
        기준은 그대로 두고 방향만 뒤집힌다.
        """
        if self.pending_yaw_deg is None:
            return False
        previous = self.operator
        self.operator_yaw_deg = self.pending_yaw_deg
        self.operator = operator_rotation(self.operator_yaw_deg)
        if self.latched is not None:
            self.latched = self.operator @ previous.T @ self.latched
        self.pending_yaw_deg = None
        return True

    @property
    def ready(self) -> bool:
        """기준을 잡았는가. 아니면 호출자가 기존 거동으로 물러나야 한다."""
        return self.latched is not None

    # -- 변환 ------------------------------------------------------------

    def to_probe(
        self,
        cmd: np.ndarray,
        rot_base_probe: np.ndarray,
        rot_stylus: np.ndarray,
    ) -> np.ndarray:
        """프로브 프레임 지령을, 고정 프레임 해석과 같아지도록 다시 쓴다.

        병진과 회전 모두에 쓴다. 각속도가 유사벡터임에도 같은 식이 성립하는 이유는
        모듈 문서를 볼 것 — 관여하는 세 행렬이 전부 ``det = +1`` 이다.

        하류(``DlsSolver``)는 여전히 프로브 프레임 twist 를 받으므로, 결과를 프로브
        프레임으로 되돌려 준다. 클램프·추종오차 진단이 전부 프로브 축 기준이라
        그쪽 의미를 흐트러뜨리지 않기 위해서다.

        Args:
            cmd: teleop 이 보낸 3개 벡터 (프로브 프레임 의도). 병진 또는 회전.
            rot_base_probe: 현재 base → 프로브 회전.
            rot_stylus: 현재 스타일러스 자세 (장치 world 기준).

        Returns:
            프로브 프레임 3개 벡터. ``ready`` 가 거짓이면 입력을 그대로 돌려준다.
        """
        cmd = np.asarray(cmd, dtype=float).reshape(3)
        if self.latched is None:
            return cmd

        rot_base_probe = np.asarray(rot_base_probe, dtype=float)
        rot_stylus = np.asarray(rot_stylus, dtype=float)

        # 1) 지령을 장치 world 프레임으로 되돌린다 (A 가 직교라 전치가 곧 역행렬)
        world = rot_stylus @ self.mapping.T @ cmd
        # 2) 고정 행렬 하나로 base 에 싣는다 — 여기에 시간 의존 항이 없다
        base = self.latched @ world
        # 3) 하류가 기대하는 프로브 프레임으로
        return rot_base_probe.T @ base

    def to_probe_body(self, cmd: np.ndarray, rot_base_probe: np.ndarray) -> np.ndarray:
        """조작자 회전만 적용한다. 기준을 고정하지 **않는** 축에 쓴다.

        ``teleop.angular_frame: body`` 처럼 표류 대책을 끄고 쓰는 축에도 조작자
        위치는 적용되어야 한다. 병진만 뒤집히고 회전은 그대로면, 미는 방향과 비트는
        방향이 서로 다른 세계에 있게 된다.

        하류는 프로브 프레임 twist 를 받으므로 다시 프로브 프레임으로 돌려준다::

            cmd' = R_base_probeᵀ · Rz_base(operator_yaw) · R_base_probe · cmd

        Args:
            cmd: 프로브 프레임 3벡터 (병진 또는 회전).
            rot_base_probe: 현재 base → 프로브 회전.

        Returns:
            프로브 프레임 3벡터. 조작자 각도가 0 이면 입력 그대로다 (그때는 곱하지도
            않는다 — 지금까지의 거동이 수치적으로 **정확히** 보존된다).
        """
        cmd = np.asarray(cmd, dtype=float).reshape(3)
        if self.operator_yaw_deg == 0.0:
            return cmd
        rot_base_probe = np.asarray(rot_base_probe, dtype=float)
        return rot_base_probe.T @ self.operator @ rot_base_probe @ cmd

    def to_base(
        self,
        cmd: np.ndarray,
        rot_stylus: np.ndarray,
    ) -> np.ndarray:
        """같은 지령을 base 프레임으로. 진단·시험에서 "실제로 어디로 가는가" 용이다."""
        cmd = np.asarray(cmd, dtype=float).reshape(3)
        if self.latched is None:
            return cmd
        return self.latched @ np.asarray(rot_stylus, dtype=float) @ self.mapping.T @ cmd


def _normalize_deg(angle_deg: float) -> float:
    """각을 ``(-180, 180]`` 으로. 180 과 -180 이 서로 다른 설정처럼 보이지 않게 한다."""
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


#: 옛 이름. 이 매퍼는 이제 회전에도 쓰이므로 ``TeleopFrameMapper`` 가 맞는 이름이다.
LinearFrameMapper = TeleopFrameMapper
