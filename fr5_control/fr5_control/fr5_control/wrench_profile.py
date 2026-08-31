"""교정 프로파일 — 저장, 되읽기, 런타임 보상 (2026-08-26).

한 프로파일은 "이 조립을, 이 장착으로, 이 시각에 교정했다" 를 통째로 담는다. 조각을
따로 저장하면 어느 영점이 어느 중력 모델과 짝인지 알 수 없게 되고, 그 조합이 틀리면
증상은 "힘이 조금 이상하다" 로만 나타난다.

런타임 파이프라인 (사양 §5)::

    {S}w_ext = {S}w_raw - {S}b - {S}w_g(q)
    {P}w_ext = blkdiag({P}R{S}, {P}R{S}) · {S}w_ext
    {P}τ_contact = {P}τ_ext - {P}r_{S→P} × {P}f_ext

순서가 중요하다. 중력 보상은 **센서 프레임에서** 해야 한다 — 모델의 질량·COM 이 그
프레임에서 식별됐기 때문이다. 프로브 프레임으로 옮긴 뒤에 빼면 회전이 섞여 틀린다.

⚠️ **오래된 교정을 조용히 쓰지 않는다.** 프로파일은 자기 나이와 장착 지문을 들고
다니며, 둘 중 하나라도 어긋나면 ``validity`` 가 그것을 말한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fr5_control.wrench_calibration import BiasResult, GravityModel
from fr5_control.wrench_frames import GRAVITY_B, FrameRegistration, wrench_rotate, wrench_translate

__all__ = ["CalibrationProfile", "CompensatedWrench", "compensate"]

#: 이보다 오래된 교정은 경고한다 [s]. 하루.
STALE_AFTER_S = 24 * 3600.0


@dataclass
class CompensatedWrench:
    """한 주기의 보상 결과. 각 단계를 모두 들고 있다.

    중간 단계를 버리지 않는 이유: 검증 화면이 "원값은 이런데 보상 뒤에 이렇게 된다"
    를 나란히 보여 줘야 하고, 어느 단계에서 이상해지는지가 곧 어느 교정이 틀렸는지다.
    """

    raw_sensor: np.ndarray
    bias_corrected_sensor: np.ndarray
    external_sensor: np.ndarray
    external_probe: np.ndarray
    contact_probe: np.ndarray

    @property
    def normal_force_n(self) -> float:
        """``F_{z_P}``. 양수 = 압축. 제어가 쓰는 유일한 스칼라다."""
        return float(self.contact_probe[2])

    def to_dict(self) -> dict:
        """화면·로그용."""
        return {
            "rawSensor": [float(v) for v in self.raw_sensor],
            "biasCorrectedSensor": [float(v) for v in self.bias_corrected_sensor],
            "externalSensor": [float(v) for v in self.external_sensor],
            "externalProbe": [float(v) for v in self.external_probe],
            "contactProbe": [float(v) for v in self.contact_probe],
            "normalForceN": self.normal_force_n,
        }


@dataclass
class CalibrationProfile:
    """저장되는 교정 한 벌.

    Attributes:
        registration: 프레임 등록 (장착각·뒤집힘·지렛대).
        bias: 전자 영점.
        gravity: 다자세 중력 모델. ``None`` 이면 A 단계만 끝난 상태다.
        created_at: 저장 시각 (epoch 초).
        robot_pose_deg: 영점을 잰 자세. 기록이자 재현용이다.
        sensor_temperature_c: 있으면 기록한다. PX6D 는 안 준다.
        mounting_note: 조립 구성 메모. 마운트를 바꾸면 여기가 달라져야 한다.
        working_tare: 작업 자세에서 잰 잔여 렌치 (프로브 프레임). 보상 뒤에도
            0.2 N 남짓이 남는데, 접근을 시작하는 자세에서 그것을 0 으로 만든다.
        tare_flange_z: 그 영점을 잰 자세의 플랜지 z 성분. **이 영점은 그 자세에서만
            정확하다** — 다른 자세로 가면 뺀 만큼이 그대로 오차가 된다. 얼마나
            멀어졌는지 말할 수 있도록 함께 남긴다.
        tare_pose_deg: 그 영점을 잰 관절각.
    """

    registration: FrameRegistration
    bias: BiasResult
    gravity: GravityModel | None = None
    created_at: float = field(default_factory=time.time)
    robot_pose_deg: list = field(default_factory=list)
    sensor_temperature_c: float | None = None
    mounting_note: str = ""
    working_tare: np.ndarray | None = None
    tare_flange_z: float | None = None
    tare_pose_deg: list = field(default_factory=list)

    # -- 유효성 -----------------------------------------------------------

    def validity(self, now: float | None = None,
                 stale_after_s: float | None = None) -> tuple:
        """제어를 열어도 되는지 판정한다. ``(valid, issues)`` 를 돌려준다.

        A 단계만 끝난 프로파일은 **유효하지 않다.** 그 상태로 접촉 제어를 열면 자세가
        바뀌는 순간 힘이 수 N 씩 어긋난다 — 영점을 잰 자세에서만 맞는 값이다.

        Args:
            now: 기준 시각. 시험용.
            stale_after_s: 만료 시간 [s]. ``None`` 이면 :data:`STALE_AFTER_S`.
                0 이하이면 **나이를 보지 않는다.**

                기본 24 시간은 "센서 영점이 하루면 흐른다" 는 보수적 가정이다.
                그 가정을 측정으로 대체했다면(같은 프로파일로 여러 날에 걸쳐
                무접촉 잔차가 유지되는 것을 확인했다면) 늘리는 것이 맞다. 다만
                **늘린다고 흐름이 멈추지는 않는다** — 만료를 끄는 것은 "흐르지
                않는다" 는 주장이며, 그 주장의 근거는 이 코드가 아니라 사용자가
                들고 있어야 한다.
        """
        now = time.time() if now is None else now
        limit = STALE_AFTER_S if stale_after_s is None else float(stale_after_s)
        issues = []
        if not self.bias.accepted:
            issues.append(f"전자 영점 거부됨: {self.bias.reason}")
        if self.gravity is None:
            issues.append("중력 보상 미완료 — 자세가 바뀌면 값이 어긋난다")
        elif not self.gravity.valid:
            issues.extend(f"중력 모델: {m}" for m in self.gravity.issues)
        age = now - self.created_at
        if limit > 0.0 and age > limit:
            issues.append(f"교정이 오래됐다 ({age / 3600:.1f} 시간 전)")
        return (not issues), issues

    def age_s(self, now: float | None = None) -> float:
        """교정한 지 얼마나 됐는가 [s]."""
        return (time.time() if now is None else now) - self.created_at

    # -- 저장 -------------------------------------------------------------

    def to_dict(self) -> dict:
        """저장용."""
        return {
            "version": 1,
            "createdAt": float(self.created_at),
            "registration": self.registration.to_dict(),
            "bias": self.bias.to_dict(),
            "gravity": self.gravity.to_dict() if self.gravity else None,
            "robotPoseDeg": [float(v) for v in self.robot_pose_deg],
            "sensorTemperatureC": self.sensor_temperature_c,
            "mountingNote": self.mounting_note,
            "workingTare": (
                None if self.working_tare is None
                else [float(v) for v in np.asarray(self.working_tare).reshape(6)]
            ),
            "tareFlangeZ": self.tare_flange_z,
            "tarePoseDeg": list(self.tare_pose_deg),
        }

    def save(self, path) -> None:
        """파일로 저장한다. 원자적으로 쓴다 — 반쯤 쓰인 프로파일을 다음 기동이 읽으면 안 된다."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1))
        tmp.replace(path)

    @staticmethod
    def load(path) -> "CalibrationProfile":
        """저장본을 되읽는다."""
        data = json.loads(Path(path).read_text())
        b = data["bias"]
        return CalibrationProfile(
            registration=FrameRegistration.from_dict(data["registration"]),
            bias=BiasResult(
                bias=np.asarray(b["bias"], dtype=float).reshape(6),
                std=np.asarray(b["std"], dtype=float).reshape(6),
                samples=int(b["samples"]),
                duration_s=float(b["duration_s"]),
                accepted=bool(b["accepted"]),
                reason=str(b["reason"]),
            ),
            gravity=GravityModel.from_dict(data["gravity"]) if data.get("gravity") else None,
            created_at=float(data.get("createdAt", 0.0)),
            robot_pose_deg=list(data.get("robotPoseDeg", [])),
            sensor_temperature_c=data.get("sensorTemperatureC"),
            mounting_note=str(data.get("mountingNote", "")),
            working_tare=(
                None if data.get("workingTare") is None
                else np.asarray(data["workingTare"], dtype=float).reshape(6)
            ),
            tare_flange_z=data.get("tareFlangeZ"),
            tare_pose_deg=list(data.get("tarePoseDeg", [])),
        )


def compensate(profile: CalibrationProfile, raw_wrench,
               rot_base_flange=None) -> CompensatedWrench:
    """원시 wrench 를 접촉점 wrench 로 바꾼다 (사양 §5).

    Args:
        profile: 쓸 교정.
        raw_wrench: 센서가 낸 ``[Fx, Fy, Fz, Mx, My, Mz]``.
        rot_base_flange: ``{B}R{F}``. ``None`` 이면 중력 보상을 건너뛴다 —
            자세를 모르는 채 중력을 빼는 것은 추측이지 보상이 아니다.

    Returns:
        :class:`CompensatedWrench`. 모든 중간 단계를 포함한다.
    """
    raw = np.asarray(raw_wrench, dtype=float).reshape(6)

    bias_corrected = raw - profile.bias.bias

    if profile.gravity is not None and rot_base_flange is not None:
        # ``{S}g`` 는 **모델이 식별될 때 쓴 회전** 으로 만들어야 한다.
        #
        # 적합은 플랜지→센서 회전을 데이터에서 함께 푼다. 그렇게 나온 질량과 COM 은
        # 그 회전과 한 짝이다. 여기서 등록의 **가정된** 회전으로 중력을 만들면 모델이
        # 본 적 없는 방향의 중력을 빼게 되고, 두 회전이 89° 어긋나 있던 실기에서
        # 잔여 힘이 0.17 N 이어야 할 자세에 4.09 N 이 남았다.
        #
        # 적합이 회전을 풀지 않았으면(옛 프로파일) 단위행렬이고, 그때는 등록을 쓴
        # 예전 거동과 같아진다.
        rotation = np.asarray(profile.gravity.rotation_sensor_from_flange, dtype=float)
        if np.allclose(rotation, np.eye(3)):
            g_s = profile.registration.gravity_in_sensor(rot_base_flange)
        else:
            g_s = rotation @ (np.asarray(rot_base_flange, dtype=float).T @ GRAVITY_B)
        # 중력 모델은 원값 기준으로 식별됐으므로, 전자 영점을 빼기 전 값에서 뺀 뒤
        # 다시 영점을 적용하는 것과 같아지도록 잔여 바이어스만 여기서 처리한다.
        external_sensor = bias_corrected - (
            profile.gravity.predict(g_s) - profile.bias.bias
        )
    else:
        external_sensor = bias_corrected

    rot = profile.registration.rotation_probe_from_sensor()
    external_probe = wrench_rotate(rot, external_sensor)
    contact_probe = wrench_translate(
        external_probe, profile.registration.r_sensor_to_probe_m
    )

    # 작업 자세 영점.
    #
    # 다자세 보상을 다 하고도 0.2 N 남짓이 남는다 — 자중 200 g 에 센서 오프셋
    # 8.5 N 이라는 신호비의 한계다. 접근을 시작하는 자세에서 그것을 0 으로 만들면
    # 무접촉이 정확히 0 으로 읽히고, 5 N 대역이 온전히 접촉에만 쓰인다.
    #
    # **그 자세에서만 정확하다.** 상수를 빼는 것이므로 다른 자세로 가면 뺀 만큼이
    # 오차로 돌아온다. 그래서 언제 어느 자세에서 쟀는지를 프로파일이 들고 다니고,
    # 브리지가 지금 자세와 얼마나 떨어졌는지 화면에 낸다.
    if profile.working_tare is not None:
        contact_probe = contact_probe - np.asarray(
            profile.working_tare, dtype=float
        ).reshape(6)

    return CompensatedWrench(
        raw_sensor=raw,
        bias_corrected_sensor=bias_corrected,
        external_sensor=external_sensor,
        external_probe=external_probe,
        contact_probe=contact_probe,
    )
