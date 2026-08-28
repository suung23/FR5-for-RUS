"""프로브 TCP 를 손으로 잡아 재는 절차 도구.

    ros2 run fr5_control probe_tcp_id --ip 192.168.58.3
    python3 -m fr5_control.probe_tcp_id --replay poses.json     # 로봇 없이 재계산

**이 도구는 로봇을 움직이지 않는다.** 컨트롤러에 읽기로만 붙어 관절각을 가져올
뿐이다. 자세를 옮기는 것은 조작자이며, 드래그(프리드라이브) 모드는 티치펜던트에서
켠다. 교정 절차가 로봇을 자동으로 움직이지 않는다는 규칙은 렌치 교정과 같다.

**1단계 — 피벗.** 프로브 면의 한 점을 공간의 고정점에 대고, **그 점을 유지한 채**
자세만 바꿔 가며 여러 번 잡는다. 고정점은 뾰족할수록 좋다 (원뿔 팁, 센터 펀치).
프로브 면은 뾰족하지 않으므로 **면에 기준점을 표시하고** 매번 같은 자국을 대야
한다 — 이 절차의 정확도는 수학이 아니라 여기서 결정된다.

**2단계 — 평면.** 프로브 면을 평평한 기준면에 밀착시킨 자세를 2 회 이상 잡는다.
``+z_P`` 는 조직으로 파고드는 방향이므로 면을 향한다. 기준면의 법선은 ``--normal``
로 주거나 (기본값은 베이스 +z, 즉 수평한 테이블), 1 단계에서 얻은 팁으로 테이블
위 점 3 개 이상을 짚어 맞춘다 (``plane`` 단계).

**3단계 — 배열 방향.** 남는 자유도는 프로브 축 둘레 회전 하나다. 배열 장축을
베이스에서 방향을 아는 곧은 모서리에 맞춘 자세를 잡으면 정해진다. 이 단계를
건너뛰면 ``--array-hint`` 의 기본값(베이스 +x)이 쓰이고, 결과에 그 사실이 남는다.

결과는 표로 찍고 ``probe.yaml`` 에 붙여 넣을 수 있는 형태로 낸다. **잔차와 자세
다양성을 항상 같이 낸다** — 값만 보고 믿을 수 있는 절차가 아니기 때문이다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np

from fr5_control.probe_tcp import (
    PivotPose,
    axis_in_flange,
    fit_plane,
    pose_from_joints,
    rotation_from_axes,
    rpy_from_rotation,
    solve_pivot,
)


def _connect(ip: str):
    """컨트롤러에 **읽기로만** 붙는다."""
    from fr5_control.robot_backend import FairinoBackend

    backend = FairinoBackend(ip)
    backend.connect()
    return backend


def _prompt(message: str) -> bool:
    """엔터면 참, ``s`` 면 건너뛰기(거짓), ``q`` 면 종료."""
    try:
        answer = input(message).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    if answer == "q":
        sys.exit(0)
    return answer != "s"


def _capture(backend, label: str) -> PivotPose:
    joints = backend.joint_positions_deg()
    pose = pose_from_joints(joints, label)
    print(
        f"    관절 [{', '.join(f'{v:7.2f}' for v in joints)}]  "
        f"플랜지 ({pose.position[0] * 1000:7.1f}, {pose.position[1] * 1000:7.1f}, "
        f"{pose.position[2] * 1000:7.1f}) mm"
    )
    return pose


def _report_pivot(result) -> None:
    print()
    print("  피벗 결과")
    print("  " + "-" * 62)
    offset_mm = result.offset_flange_m * 1000.0
    print(
        f"    오프셋 (플랜지 기준)   x {offset_mm[0]:+8.2f}   "
        f"y {offset_mm[1]:+8.2f}   z {offset_mm[2]:+8.2f}  mm"
    )
    print(f"    잔차 RMS               {result.rms_mm:8.2f} mm   (최대 {result.max_mm:.2f})")
    print(f"    자세 다양성            {result.spread_deg:8.1f}°")
    print(f"    최소 특이값            {result.singular_values[-1]:8.4f}")
    print()
    print("    자세별 잔차")
    for pose, residual in zip(result_poses(result), result.residuals_mm):
        flag = "  ← 크다" if residual > 2.0 else ""
        print(f"      {pose:<12s} {residual:6.2f} mm{flag}")
    if result.issues:
        print()
        for issue in result.issues:
            print(f"    ⚠️  {issue}")
    else:
        print()
        print("    ✅ 유효")


def result_poses(result):
    """보고용 자세 이름. 잔차와 같은 순서다."""
    return getattr(result, "_labels", [f"자세 {i + 1}" for i in range(len(result.residuals_mm))])


def run_pivot(backend, count: int) -> tuple:
    print()
    print("=" * 66)
    print("  1단계 — 피벗 (병진)")
    print("=" * 66)
    print("  고정점에 프로브의 **같은 지점**을 댄 채 자세만 바꿔 가며 잡는다.")
    print("  자세는 크게 벌릴수록 좋다 — 30° 이상 기울여야 풀린다.")
    print("  두 자세로는 원리적으로 풀리지 않으므로 최소 3, 권장 5 이상이다.")
    print()

    poses = []
    while True:
        index = len(poses) + 1
        label = f"자세 {index}"
        if not _prompt(f"  [{index}/{count}] 고정점에 대고 엔터 (s 건너뛰기, q 종료): "):
            if len(poses) >= 3:
                break
            print("    3 자세 미만이라 풀 수 없다. 계속 잡아라.")
            continue
        poses.append(_capture(backend, label))
        if len(poses) >= count:
            interim = solve_pivot(poses)
            setattr(interim, "_labels", [p.label for p in poses])
            _report_pivot(interim)
            if interim.valid:
                return interim, poses
            if not _prompt("\n  자세를 더 잡겠는가? 엔터 계속 / s 이대로 진행: "):
                return interim, poses

    result = solve_pivot(poses)
    setattr(result, "_labels", [p.label for p in poses])
    _report_pivot(result)
    return result, poses


def run_plane(backend, tip_offset) -> np.ndarray | None:
    """1 단계에서 얻은 팁으로 기준면을 맞춘다. 수평 가정을 없앤다."""
    print()
    print("=" * 66)
    print("  2단계 (선택) — 기준면 맞추기")
    print("=" * 66)
    print("  방금 얻은 팁으로 기준면 위의 서로 떨어진 점 3 개 이상을 짚는다.")
    print("  건너뛰면 기준면이 수평(베이스 +z)이라고 가정한다.")
    print()
    if not _prompt("  맞추겠는가? 엔터 진행 / s 건너뛰기: "):
        return None

    points = []
    while True:
        index = len(points) + 1
        if not _prompt(f"  [{index}] 면 위의 점에 대고 엔터 (s 그만): "):
            if len(points) >= 3:
                break
            print("    점이 3 개는 있어야 평면이 된다.")
            continue
        pose = _capture(backend, f"평면 {index}")
        points.append(pose.rotation @ tip_offset + pose.position)

    normal, _, residual = fit_plane(points)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(normal @ np.array([0, 0, 1.0]))))))
    print()
    print(f"    법선  ({normal[0]:+.4f}, {normal[1]:+.4f}, {normal[2]:+.4f})")
    print(f"    수평에서 {tilt:.2f}° 기울어져 있다.  평면 잔차 {residual:.2f} mm")
    return normal


def run_axis(backend, normal, array_hint) -> tuple:
    print()
    print("=" * 66)
    print("  3단계 — 프로브 축 (회전)")
    print("=" * 66)
    print("  프로브 면을 기준면에 **밀착**시킨 자세를 2 회 이상 잡는다.")
    print("  축 둘레로만 돌려 가며 잡으면 밀착 여부가 서로 검증된다.")
    print()

    rotations = []
    while True:
        index = len(rotations) + 1
        if not _prompt(f"  [{index}] 밀착시키고 엔터 (s 그만): "):
            if len(rotations) >= 2:
                break
            print("    2 자세는 있어야 일치도를 볼 수 있다.")
            continue
        rotations.append(_capture(backend, f"밀착 {index}").rotation)

    result = axis_in_flange(rotations, -normal)
    print()
    print(
        f"    프로브 축 (플랜지 기준)  ({result.axis_flange[0]:+.4f}, "
        f"{result.axis_flange[1]:+.4f}, {result.axis_flange[2]:+.4f})"
    )
    print(f"    자세 간 불일치            {result.spread_deg:.2f}°")
    for issue in result.issues:
        print(f"    ⚠️  {issue}")

    rotation = rotation_from_axes(result.axis_flange, array_hint)
    return result, rotation


def emit(result, rotation, array_measured: bool, path: str | None) -> None:
    xyz = [round(float(v), 6) for v in result.offset_flange_m]
    rpy = [round(float(v), 6) for v in rpy_from_rotation(rotation)]

    print()
    print("=" * 66)
    print("  probe.yaml 에 넣을 값")
    print("=" * 66)
    print("    tool:")
    print(f"      j6_to_probe_xyz: [{xyz[0]}, {xyz[1]}, {xyz[2]}]")
    print(f"      j6_to_probe_rpy: [{rpy[0]}, {rpy[1]}, {rpy[2]}]")
    print(
        f"      # rpy [도]: ({math.degrees(rpy[0]):.2f}, "
        f"{math.degrees(rpy[1]):.2f}, {math.degrees(rpy[2]):.2f})"
    )
    print("      allow_missing_tool: false")
    if not array_measured:
        print()
        print("    ⚠️  배열 방향을 재지 않았다. 축 둘레 회전은 기본 힌트로 채운 값이며,")
        print("        영상면 안팎(x/y)이 뒤바뀌어 있을 수 있다. 병진과 z 축은 유효하다.")
    if result.issues:
        print()
        print("    ⚠️  피벗이 유효하지 않다. 위 값을 넣기 전에 다시 잡아라.")

    if path:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "j6_to_probe_xyz": xyz,
                    "j6_to_probe_rpy": rpy,
                    "pivot_rms_mm": result.rms_mm,
                    "pivot_max_mm": result.max_mm,
                    "orientation_spread_deg": result.spread_deg,
                    "array_direction_measured": array_measured,
                    "issues": result.issues,
                    "poses_deg": [list(p) for p in getattr(result, "_joints", [])],
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        print(f"\n  기록: {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ip", default="192.168.58.3", help="FR5 컨트롤러 주소")
    parser.add_argument("--poses", type=int, default=6, help="피벗 목표 자세 수")
    parser.add_argument("--out", help="결과 JSON 경로")
    parser.add_argument(
        "--normal",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="기준면 법선 (베이스 프레임). 주면 2단계를 건너뛴다",
    )
    parser.add_argument(
        "--array-hint",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.0],
        metavar=("X", "Y", "Z"),
        help="배열 장축이 향하는 베이스 방향. 재지 않으면 기본값이 쓰인다",
    )
    parser.add_argument("--replay", help="관절각 JSON 으로 로봇 없이 재계산")
    args = parser.parse_args(argv)

    if args.replay:
        with open(args.replay, encoding="utf-8") as handle:
            data = json.load(handle)
        poses = [pose_from_joints(j, f"자세 {i + 1}") for i, j in enumerate(data["pivot_deg"])]
        result = solve_pivot(poses)
        setattr(result, "_labels", [p.label for p in poses])
        setattr(result, "_joints", [p.joint_deg for p in poses])
        _report_pivot(result)
        normal = np.array(data.get("normal", [0.0, 0.0, 1.0]), dtype=float)
        flush = [pose_from_joints(j).rotation for j in data.get("flush_deg", [])]
        if flush:
            axis = axis_in_flange(flush, -normal)
            rotation = rotation_from_axes(axis.axis_flange, args.array_hint)
        else:
            rotation = np.eye(3)
        emit(result, rotation, bool(data.get("array_measured")), args.out)
        return 0

    print(f"\n  FR5 {args.ip} 에 **읽기로만** 붙는다. 이 도구는 로봇을 움직이지 않는다.")
    print("  티치펜던트에서 드래그 모드를 켜고 손으로 옮겨라.\n")
    backend = _connect(args.ip)

    result, poses = run_pivot(backend, args.poses)
    setattr(result, "_joints", [p.joint_deg for p in poses])

    normal = np.array(args.normal, dtype=float) if args.normal else None
    if normal is None:
        normal = run_plane(backend, result.offset_flange_m)
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    normal = normal / np.linalg.norm(normal)

    _, rotation = run_axis(backend, normal, np.array(args.array_hint, dtype=float))
    emit(result, rotation, array_measured=False, path=args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
