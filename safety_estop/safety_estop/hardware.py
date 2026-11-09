"""물리 비상정지 회로의 **관측**. 판정이 아니다.

회로는 ``README.md`` §1 이 정의한다. 소프트웨어는 그 회로에 끼어들지 않는다 —
버튼에서 컨트롤러 안전입력까지 어떤 ROS 노드도, 네트워크도, 이 파이썬도 지나지
않는다. 여기서 하는 일은 컨트롤러가 이미 내린 결론을 화면과 상태기계로 옮겨
적는 것뿐이다.

그래도 이 관측이 필요한 이유는 셋이다.

1. **콘솔이 알아야 한다.** 버튼이 눌린 줄 모르는 화면은 "로봇이 왜 안 움직이지" 를
   만든다.
2. **복귀를 막아야 한다.** 비상정지 상태에서 복귀 운동을 지령하면 안 된다. 실제로
   모터에 전원이 없어 움직이지도 않지만, 지령이 큐에 쌓였다가 리셋 순간 한꺼번에
   실행되는 것이 ISO 13850 이 금지하는 "리셋에 의한 재기동" 그 자체다.
3. **회로가 살아 있는지 봐야 한다.** 두 채널이 서로 다른 말을 하면 (한쪽만 열림)
   그것은 배선 고장이고, 그 상태의 회로는 다음번에 **안 멈출 수 있다**. 두 채널이
   일치할 때만 회로를 신뢰한다.

⚠️ 관측 자체는 TCP·XML-RPC 를 타고 온다. 즉 **이 신호가 끊기는 것** 과 **버튼이
눌리는 것** 은 다른 사건이고, 끊김을 "안 눌렸다" 로 읽으면 안 된다. ``stale`` 은
눌림과 똑같이 운동을 막되, 사유를 다르게 말한다.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HardwareLoop", "HardwareLoopState"]


@dataclass(frozen=True)
class HardwareLoop:
    """컨트롤러가 보고한 안전입력 한 장.

    Args:
        present: 이 설치에 물리 비상정지 회로가 **있는가**. 거짓이면 이 스택에는
            ISO 13850 의 비상정지가 없다 — 노드는 그것을 숨기지 않고 상태로 낸다.
        estop: 컨트롤러의 급정지 플래그 (``RobotStatePkg.EmergencyStop``).
        stop0: 안전정지 채널 0 (``safety_stop0_state``).
        stop1: 안전정지 채널 1 (``safety_stop1_state``).
        at: 이 값을 읽은 시각 [s].
    """

    present: bool
    estop: bool
    stop0: bool
    stop1: bool
    at: float

    def evaluate(self, now: float, stale_s: float = 0.5) -> "HardwareLoopState":
        stale = (now - self.at) > stale_s
        # 채널 불일치: 한쪽만 열렸다. 눌린 것으로 **취급** 하되 사유는 고장이다 —
        # 둘 중 하나는 거짓말을 하고 있고, 어느 쪽인지 소프트웨어는 모른다.
        disagree = self.stop0 != self.stop1
        asserted = bool(self.estop or self.stop0 or self.stop1 or disagree)
        if not self.present:
            reason = ("물리 비상정지 회로가 설정되지 않았다 — 이 스택에는 "
                      "ISO 13850 의 비상정지가 없다 (README §1)")
        elif stale:
            reason = (f"안전입력을 {now - self.at:.1f} s 째 못 읽었다 — "
                      "눌리지 않았다는 뜻이 아니다")
        elif disagree:
            reason = (f"안전입력 두 채널이 어긋난다 (ch0={int(self.stop0)}, "
                      f"ch1={int(self.stop1)}) — 배선 고장이다. 다음번에 안 멈출 수 있다")
        elif asserted:
            reason = "물리 비상정지가 눌려 있다"
        else:
            reason = ""
        return HardwareLoopState(
            present=self.present,
            asserted=asserted,
            stale=stale,
            channels_disagree=disagree,
            # 운동을 막는 조건. 회로가 없는 것은 막지 않는다 — 없는 설치에서
            # 아무것도 못 움직이게 하면 그 설치는 이 감시를 꺼 버린다. 대신
            # ``present=False`` 는 상태에 계속 떠 있고, 콘솔이 그것을 지우지 않는다.
            blocks_motion=bool(self.present and (asserted or stale)),
            reason=reason,
            age_s=now - self.at,
        )


@dataclass(frozen=True)
class HardwareLoopState:
    """위의 평가 결과. 콘솔과 상태기계가 함께 읽는다."""

    present: bool
    asserted: bool
    stale: bool
    channels_disagree: bool
    blocks_motion: bool
    reason: str
    age_s: float

    def as_dict(self) -> dict:
        return {
            "present": self.present,
            "asserted": self.asserted,
            "stale": self.stale,
            "channelsDisagree": self.channels_disagree,
            "blocksMotion": self.blocks_motion,
            "reason": self.reason,
            "ageS": self.age_s,
        }
