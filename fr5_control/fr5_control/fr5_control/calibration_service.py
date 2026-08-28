"""교정 세션 — 표본 수집과 절차 상태 (2026-08-26).

브리지 안에서 돌지만 rclpy 를 모른다. ROS 는 밖에서 wrench 와 자세를 넣어 주고, 이
모듈은 "지금 무엇을 모으는 중이고, 언제 끝났고, 결과가 쓸 만한가" 만 안다.

⚠️ **로봇을 움직이지 않는다.** 자세 이동은 조작자가 손으로 한다 (§7). 이 서비스가
할 수 있는 것은 "지금 자세에서 모아라" 뿐이고, 그것이 설계 의도다 — 교정 중에 로봇이
스스로 움직이면 조작자가 예상하지 못한 순간에 프로브가 움직인다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from fr5_control.wrench_calibration import (
    CalibrationPose,
    estimate_static_bias,
    fit_gravity_model,
    pose_coverage,
    MIN_POSES,
)

__all__ = ["CaptureState", "CalibrationSession"]


#: 자동 캡처 — 움직임으로 볼 관절 변화 [도]. 이보다 작으면 정지로 친다.
#: 컨트롤러가 내는 관절각의 정지 잡음보다 넉넉히 크게 잡았다.
AUTO_MOVE_THRESHOLD_DEG = 0.15

#: 자동 캡처 — 이만큼 조용해야 "멈췄다" 로 본다 [초].
#: 손을 뗀 직후에는 팔이 미세하게 흔들리고, 그때 잡으면 표본이 흔들린 값이 된다.
AUTO_SETTLE_S = 1.2

#: 자동 캡처 — 이미 잡은 자세와 이 각도 안쪽이면 건너뛴다 [도].
#: 비슷한 자세를 열두 번 잡아도 커버리지는 오르지 않는다. 그럴 바에는 안 잡고
#: "더 크게 돌려라" 를 보여 주는 편이 낫다.
AUTO_MIN_SEPARATION_DEG = 15.0


@dataclass
class StillnessDetector:
    """드래그가 멈춘 순간을 잡아낸다.

    조작자가 로봇을 손으로 옮기고 놓으면 자동으로 자세를 캡처하기 위한 것이다.
    자세마다 버튼을 누르려면 한 손이 묶이는데, 그 손은 로봇을 붙잡고 있어야 한다.

    **한 번 멈출 때 한 번만 잡는다.** 정지를 보고한 뒤에는 다시 움직이기 전까지
    보고하지 않는다 — 그러지 않으면 가만히 둔 자세를 초당 스무 번 잡는다.

    로봇을 움직이는 것은 조작자다. 이 검출기는 관절각을 볼 뿐 아무것도 명령하지
    않는다.
    """

    move_threshold_deg: float = AUTO_MOVE_THRESHOLD_DEG
    settle_s: float = AUTO_SETTLE_S
    _previous: np.ndarray | None = None
    _last_move_at: float = 0.0
    _armed: bool = False

    def reset(self) -> None:
        """처음 상태로. 자세를 버리거나 자동 모드를 껐다 켤 때 쓴다."""
        self._previous = None
        self._last_move_at = 0.0
        self._armed = False

    @property
    def armed(self) -> bool:
        """움직임을 봤고, 아직 그 정지를 보고하지 않았다."""
        return self._armed

    def update(self, joints_deg, now: float) -> str:
        """관절각 한 벌을 넣는다.

        Args:
            joints_deg: 관절각 6 개 [도].
            now: 지금 시각 [초].

        Returns:
            ``"settled"`` 면 방금 멈춘 것이다 (한 번만 나온다).
            ``"moving"`` 이면 움직이는 중. 그 외에는 빈 문자열.
        """
        current = np.asarray(joints_deg, dtype=float).reshape(-1)
        previous, self._previous = self._previous, current
        if previous is None or previous.shape != current.shape:
            self._last_move_at = now
            return ""

        if float(np.abs(current - previous).max()) > self.move_threshold_deg:
            self._last_move_at = now
            self._armed = True
            return "moving"

        if self._armed and (now - self._last_move_at) >= self.settle_s:
            self._armed = False
            return "settled"
        return ""


@dataclass
class CaptureState:
    """지금 수집 중인 것.

    Attributes:
        kind: ``"bias"`` 또는 ``"pose"``. 비어 있으면 수집 중이 아니다.
        label: 조작자에게 보여 준 이름.
        started_at: 시작 시각.
        target_s: 목표 수집 시간.
        samples: 모인 원시 wrench.
        gravity_sensor: 자세 수집일 때 그 순간의 ``{S}g``.
    """

    kind: str = ""
    label: str = ""
    started_at: float = 0.0
    target_s: float = 4.0
    samples: list = field(default_factory=list)
    gravity_sensor: np.ndarray | None = None
    gravity_flange: np.ndarray | None = None

    @property
    def active(self) -> bool:
        """수집 중인가."""
        return bool(self.kind)

    @property
    def elapsed_s(self) -> float:
        """시작한 지 얼마나 됐는가."""
        return time.time() - self.started_at if self.kind else 0.0

    @property
    def progress(self) -> float:
        """목표 시간의 몇 % 를 지났는가. 0..1."""
        if not self.kind or self.target_s <= 0:
            return 0.0
        return min(1.0, self.elapsed_s / self.target_s)


class CalibrationSession:
    """A·B 절차의 진행 상태를 들고 있는다."""

    def __init__(self) -> None:
        """빈 세션을 만든다."""
        self.capture = CaptureState()
        self.bias_result = None
        self.poses: list = []
        self.gravity_model = None
        self.last_error = ""

    # -- 수집 -------------------------------------------------------------

    def start_bias(self, seconds: float = 4.0) -> None:
        """전자 영점 수집을 시작한다."""
        self.capture = CaptureState(kind="bias", label="electronic zero",
                                    started_at=time.time(), target_s=seconds)

    def start_pose(
        self,
        label: str,
        gravity_sensor,
        seconds: float = 2.0,
        gravity_flange=None,
    ) -> None:
        """중력 식별용 한 자세를 수집한다.

        Args:
            label: 자세 이름. 기록에 남는다.
            gravity_sensor: 이 자세의 ``{S}g``. 수집 시작 시점에 고정한다 —
                로봇은 정지해 있어야 하므로 중간에 바뀌면 그 자체가 이상이다.
            seconds: 수집 시간.
            gravity_flange: 같은 순간의 ``{F}g``. 순기구학에서 바로 나오는,
                **가정이 섞이지 않은** 값이다. 플랜지→센서 회전을 적합에서 함께
                풀려면 이것이 있어야 한다.
        """
        self.capture = CaptureState(
            kind="pose", label=label, started_at=time.time(), target_s=seconds,
            gravity_sensor=np.asarray(gravity_sensor, dtype=float).reshape(3),
            gravity_flange=(
                None if gravity_flange is None
                else np.asarray(gravity_flange, dtype=float).reshape(3)
            ),
        )

    def coverage_hint(self) -> str:
        """어느 방향의 자세가 모자란지 한 문장으로.

        커버리지 숫자만 보여 주면 조작자는 그것을 어떻게 올리는지 알 수 없다.
        가장 덜 채워진 방향을 찾아, 프로브를 어느 쪽으로 기울이라는 말로 바꾼다.

        Returns:
            안내 문장. 자세가 적어 방향을 말할 수 없으면 빈 문자열.
        """
        if len(self.poses) < 3:
            return ""
        directions = []
        for pose in self.poses:
            vector = np.asarray(pose.gravity_sensor, dtype=float).reshape(3)
            norm = float(np.linalg.norm(vector))
            if norm > 1e-9:
                directions.append(vector / norm)
        if len(directions) < 3:
            return ""

        # 가장 작은 특이값의 방향이 곧 자세가 비어 있는 축이다.
        _, _, right = np.linalg.svd(np.array(directions))
        thin = right[-1]
        axis = int(np.argmax(np.abs(thin)))
        name = ("옆으로 (x)", "옆으로 (y)", "위아래로 (z)")[axis]
        return (
            f"중력이 {name} 실리는 자세가 모자라다 — 프로브를 그 축 쪽으로 "
            "더 크게 기울여서 몇 자세 더 잡아라"
        )

    def separation_deg(self, gravity_sensor) -> float:
        """이미 잡은 자세들과 얼마나 떨어진 자세인가 — 최소 각도 [도].

        중력이 센서 축에 실리는 **방향** 으로 잰다. 그것이 곧 이 자세가 모델에
        보태는 정보이고, 관절각이 달라도 중력 방향이 같으면 새 정보가 아니다.

        잡은 자세가 없으면 ``180.0`` 을 돌려준다 — 무엇을 잡아도 새 정보다.
        """
        candidate = np.asarray(gravity_sensor, dtype=float).reshape(3)
        norm = float(np.linalg.norm(candidate))
        if norm < 1e-9 or not self.poses:
            return 180.0
        candidate = candidate / norm

        worst = 180.0
        for pose in self.poses:
            other = np.asarray(pose.gravity_sensor, dtype=float).reshape(3)
            other_norm = float(np.linalg.norm(other))
            if other_norm < 1e-9:
                continue
            dot = float(np.clip(candidate @ (other / other_norm), -1.0, 1.0))
            worst = min(worst, math.degrees(math.acos(dot)))
        return worst

    def cancel(self) -> None:
        """수집을 버린다."""
        self.capture = CaptureState()

    def feed(self, raw_wrench) -> None:
        """표본 하나를 넣는다. 수집 중이 아니면 무시한다."""
        if self.capture.active:
            self.capture.samples.append(np.asarray(raw_wrench, dtype=float).reshape(6))

    def finish_if_due(self) -> str:
        """목표 시간을 채웠으면 마무리한다.

        Returns:
            무엇을 마쳤는지 (``"bias"`` / ``"pose"`` / ``""``).
        """
        if not self.capture.active or self.capture.elapsed_s < self.capture.target_s:
            return ""
        kind = self.capture.kind
        samples = np.array(self.capture.samples) if self.capture.samples else np.zeros((0, 6))

        if kind == "bias":
            try:
                self.bias_result = estimate_static_bias(samples, self.capture.elapsed_s)
                self.last_error = "" if self.bias_result.accepted else self.bias_result.reason
            except ValueError as exc:
                self.bias_result = None
                self.last_error = str(exc)
        else:
            if len(samples) < 20:
                self.last_error = f"자세 표본 부족 {len(samples)}"
            else:
                self.poses.append(CalibrationPose(
                    gravity_sensor=self.capture.gravity_sensor,
                    gravity_flange=self.capture.gravity_flange,
                    wrench=samples.mean(axis=0),
                    label=self.capture.label,
                ))
                self.last_error = ""
        self.capture = CaptureState()
        return kind

    # -- 적합 -------------------------------------------------------------

    def fit(self) -> bool:
        """모인 자세로 중력 모델을 푼다.

        Returns:
            모델이 유효한가. 거짓이어도 모델은 남는다 — 무엇이 왜 나빴는지 화면이
            보여 줘야 하기 때문이다.
        """
        if not self.poses:
            self.last_error = "자세가 없다"
            return False
        self.gravity_model = fit_gravity_model(self.poses)
        self.last_error = "" if self.gravity_model.valid else "; ".join(self.gravity_model.issues)
        return self.gravity_model.valid

    def drop_poses(self) -> None:
        """모은 자세를 버린다. 마운트를 바꾼 뒤에는 반드시 해야 한다."""
        self.poses = []
        self.gravity_model = None

    # -- 상태 -------------------------------------------------------------

    def status(self) -> dict:
        """화면이 그릴 진행 상태."""
        coverage = 0.0
        if self.poses:
            coverage, _ = pose_coverage([p.gravity_sensor for p in self.poses])
        return {
            "capturing": self.capture.kind,
            "captureLabel": self.capture.label,
            "captureProgress": round(self.capture.progress, 3),
            "captureSamples": len(self.capture.samples),
            "biasAccepted": None if self.bias_result is None else self.bias_result.accepted,
            "biasReason": "" if self.bias_result is None else self.bias_result.reason,
            "poseCount": len(self.poses),
            "poseLabels": [p.label for p in self.poses],
            "poseTarget": MIN_POSES,
            "coverage": round(coverage, 3),
            "lastError": self.last_error,
            "coverageHint": self.coverage_hint(),
            "gravity": self.gravity_model.to_dict() if self.gravity_model else None,
        }
