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
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["axis_mapping", "TeleopFrameMapper", "LinearFrameMapper"]


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

    def __init__(self, tip_roll_deg: float, relatch_on_engage: bool = False) -> None:
        self.mapping = axis_mapping(tip_roll_deg)
        self.relatch_on_engage = bool(relatch_on_engage)
        #: world → base 고정 회전. ``None`` 이면 아직 기준을 안 잡았다.
        self.latched = None
        self.engage_count = 0

    # -- 기준 잡기 --------------------------------------------------------

    def engage(self, rot_base_probe: np.ndarray, rot_stylus: np.ndarray) -> bool:
        """데드맨을 잡았다. 필요하면 기준을 (다시) 잡는다.

        Returns:
            이번 호출에서 기준을 새로 잡았으면 ``True``.
        """
        self.engage_count += 1
        if self.latched is not None and not self.relatch_on_engage:
            return False
        # 이 순간에는 새 매핑이 기존 매핑과 정확히 같다. 파지 시작 축은 안 바뀌고,
        # 그 뒤로 어긋나지만 않게 된다.
        self.latched = np.asarray(rot_base_probe, dtype=float) @ self.mapping @ np.asarray(
            rot_stylus, dtype=float
        ).T
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


#: 옛 이름. 이 매퍼는 이제 회전에도 쓰이므로 ``TeleopFrameMapper`` 가 맞는 이름이다.
LinearFrameMapper = TeleopFrameMapper
