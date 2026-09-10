#!/usr/bin/env python3
"""수집된 자세 로그에서 교정 프로파일을 다시 적합해 저장한다 — **정렬 문턱을 명시적으로 지정해서.**

    python3 phantom_stiffness/refit_calibration.py --dry-run
    python3 phantom_stiffness/refit_calibration.py --max-alignment-spread 0.17

왜 이것이 따로 있는가
--------------------
브리지의 `calib.save` 는 중력 모델이 **유효할 때만** 저장한다. 그 판정에 쓰이는
`max_alignment_spread` 기본값은 0.15 이고 브리지는 그것을 파라미터로 내보내지 않는다.
문턱을 넘지 못하면 저장 자체가 불가능하고, 브리지를 재시작하면 메모리의 자세가 사라진다.

이 스크립트는 **자세 로그(`~/.ros/fr5_px6d_calibration_poses.jsonl`)** 를 읽어 같은 적합을
다시 돌린다. 적합·판정·직렬화는 전부 `fr5_control` 의 코드를 그대로 쓴다 — 여기서 다시
구현하지 않는다. 다른 것은 **문턱을 인자로 준다는 것 하나뿐**이며, 그 값과 이유가
프로파일의 `mountingNote` 에 남는다.

⚠️ **문턱을 올리는 것은 안전 게이트를 완화하는 일이다.** 반드시 근거를 갖고 하고,
저장한 뒤 `px6d_verify` 로 **적합에 안 쓴 자세**의 잔차를 실측해 판단하라. 문턱은
대리 지표이고 verify 가 직접 측정이다.

정렬 산포의 뜻: 풀어낸 3x3 "질량 행렬" 이 등방(`m·I`) 에서 얼마나 벗어났는가를
`|m|·√3` 으로 나눈 값이다. 페이로드가 가벼우면(이 조립은 ≈0.2 kg) 분모가 작아 같은
채널 이득 불일치가 크게 보인다. 자세를 더 모아도 줄지 않는 성질이다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
for _p in (os.path.join(_ROOT, "fr5_control", "fr5_control"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fr5_control.wrench_calibration import (            # noqa: E402
    BiasResult, CalibrationPose, fit_gravity_model,
)
from fr5_control.wrench_frames import FrameRegistration  # noqa: E402
from fr5_control.wrench_profile import CalibrationProfile  # noqa: E402

DEFAULT_POSES = os.path.expanduser("~/.ros/fr5_px6d_calibration_poses.jsonl")
DEFAULT_OUT = os.path.expanduser("~/.ros/fr5_px6d_calibration.json")


def select_cone(poses: list[CalibrationPose], cone_deg: float) -> list[CalibrationPose]:
    """프로브가 아래를 보는 원뿔 안의 자세만 남긴다.

    왜 좁히는가. 이 센서는 채널 이득이 서로 달라 (2026-09-11 실측: y −20 %, z +20 %)
    하나의 질량·회전으로 **모든** 방향을 설명하지 못한다. 그러면 적합은 전 방향에
    걸쳐 타협하고, 정작 쓰는 자세에서도 최선이 아니게 된다.

    실제 프로빙에서 프로브는 늘 아래를 본다. 쓰는 영역으로 좁히면 그 안에서 훨씬
    잘 맞는다 — 12 자세 전체 0.297 N 대 ±60° 6 자세 0.190 N (2026-09-11).

    ⚠️ **대가는 바깥이다.** 원뿔 밖에서는 보상이 더 나빠진다. 이 프로파일로 팔을
    크게 눕히지 말 것. `working_tare` 를 작업 자세에서 얹으면 그 자리는 0 이 된다.

    프로브가 아래를 보면 중력은 플랜지 프레임의 **+z** 로 온다 (툴 z 와 중력이 같은 방향).
    """
    kept = []
    for pose in poses:
        g = pose.gravity_flange
        if g is None:
            continue
        u = np.asarray(g, float)
        norm = float(np.linalg.norm(u))
        if norm < 1e-6:
            continue
        angle = math.degrees(math.acos(max(-1.0, min(1.0, float(u[2]) / norm))))
        if angle <= cone_deg:
            kept.append(pose)
    return kept


def load_poses(path: str, count: int) -> list[CalibrationPose]:
    """자세 로그의 **마지막 `count` 개**를 읽는다.

    브리지는 적합할 때마다 현재 자세 목록을 통째로 덧붙인다. 그래서 파일에는 여러
    세션이 쌓여 있고, 최신 세션은 항상 꼬리에 있다.
    """
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if len(rows) < count:
        raise SystemExit(f"자세가 {len(rows)} 개뿐이다 (요청 {count})")
    return [
        CalibrationPose(
            wrench=np.asarray(r["wrench"], dtype=float),
            gravity_sensor=np.asarray(r["gravity_sensor"], dtype=float),
            gravity_flange=(np.asarray(r["gravity_flange"], dtype=float)
                            if r.get("gravity_flange") is not None else None),
            label=str(r.get("label", "")),
        )
        for r in rows[-count:]
    ]


def registration_from_yaml(path: str) -> FrameRegistration:
    """probe.yaml 의 적층 값에서 등록을 만든다 — 브리지와 **같은 경로**로."""
    with open(path) as fh:
        doc = yaml.safe_load(fh)

    found: dict = {}

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k in ("j6_to_sensor_xyz", "j6_to_sensor_rpy",
                         "j6_to_probe_xyz", "j6_to_probe_rpy") and k not in found:
                    found[k] = v
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(doc)
    missing = {"j6_to_sensor_xyz", "j6_to_sensor_rpy",
               "j6_to_probe_xyz", "j6_to_probe_rpy"} - set(found)
    if missing:
        raise SystemExit(f"probe.yaml 에서 적층 값을 못 찾았다: {sorted(missing)}")
    return FrameRegistration.from_stack(
        found["j6_to_sensor_xyz"], found["j6_to_sensor_rpy"],
        found["j6_to_probe_xyz"], found["j6_to_probe_rpy"],
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses-log", default=DEFAULT_POSES)
    ap.add_argument("--count", type=int, default=20, help="꼬리에서 읽을 자세 수")
    ap.add_argument("--probe-yaml",
                    default=os.path.join(_ROOT, "install", "fr5_control", "share",
                                         "fr5_control", "config", "probe.yaml"))
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--min-poses", type=int, default=None,
                   help="자세 수 하한. 기본은 fit_gravity_model 의 12. 원뿔로 좁히면 "
                        "그만큼 줄므로 함께 내려야 한다 — 내리는 것은 판단이므로 명시한다")
    ap.add_argument("--down-cone-deg", type=float, default=None, metavar="DEG",
                   help="프로브가 아래에서 이 각 이내인 자세만 쓴다. 채널 이득이 어긋난 "
                        "센서를 **실제 쓰는 자세 영역**에 맞추는 길이다 (권장 60)")
    ap.add_argument("--max-alignment-spread", type=float, default=0.15,
                    help="정렬 산포 상한. 기본 0.15 는 fit_gravity_model 의 값과 같다. "
                         "올리면 안전 게이트를 완화하는 것이다")
    ap.add_argument("--note", default="", help="mountingNote 에 남길 메모")
    ap.add_argument("--dry-run", action="store_true", help="적합만 하고 저장하지 않는다")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    poses = load_poses(args.poses_log, args.count)
    print(f"자세 {len(poses)} 개  ({poses[0].label} … {poses[-1].label})")
    if args.down_cone_deg is not None:
        before = len(poses)
        poses = select_cone(poses, args.down_cone_deg)
        print(f"아래 원뿔 ±{args.down_cone_deg:.0f}° 로 제한: {before} → {len(poses)} 자세")
        if len(poses) < 4:
            raise SystemExit("원뿔 안의 자세가 4 개 미만이다 — 원뿔을 넓히거나 더 수집하라")

    reg = registration_from_yaml(args.probe_yaml)
    print(f"등록: 장착각 {reg.mounting_angle_deg:.4f}°  뒤집힘 {reg.axial_flip}  "
          f"레버암 {np.round(reg.r_sensor_to_probe_m, 4).tolist()} m")

    kw = {"max_alignment_spread": args.max_alignment_spread}
    if args.min_poses is not None:
        kw["min_poses"] = args.min_poses
    model = fit_gravity_model(poses, **kw)
    print()
    print("=== 중력 모델 ===")
    print(f"  질량        {model.mass_kg:.4f} kg")
    print(f"  무게중심    {np.round(model.com_sensor_m, 4).tolist()} m")
    print(f"  정렬 산포   {model.alignment_spread:.4f}   (상한 {args.max_alignment_spread})")
    print(f"  커버리지    {model.coverage:.4f}")
    print(f"  힘 잔차     rms {model.rms_force_n:.4f} N   축별 "
          f"{np.round(model.per_axis_force_n, 4).tolist()}")
    print(f"  모멘트 잔차 rms {model.rms_torque_nm:.5f} N·m")
    print(f"  valid       {model.valid}   issues {model.issues}")
    if not model.valid:
        print("\n중력 모델이 유효하지 않다 — 저장하지 않는다.", file=sys.stderr)
        return 1

    # 전자 영점: 자세 로그에는 없다. 다만 **자세를 아는 정상 경로에서는 상쇄된다** —
    # compensate() 가 `raw - bias - (predict - bias)` = `raw - predict` 로 접히기 때문이다.
    # 자세를 모르는 경로(rot_base_flange=None)에서만 쓰이므로, 중력이 0 일 때
    # predict 가 내는 값(= residual_bias)과 같게 두는 것이 두 경로를 일치시킨다.
    bias = BiasResult(
        bias=np.asarray(model.residual_bias, dtype=float),
        std=np.zeros(6),
        samples=0,
        duration_s=0.0,
        accepted=True,
        reason=("자세 로그에서 재적합 — 전자 영점은 중력 모델의 residual_bias 로 둔다 "
                "(자세를 아는 경로에서는 compensate 안에서 상쇄된다)"),
    )

    note = args.note or (
        f"refit_calibration.py · max_alignment_spread={args.max_alignment_spread} "
        f"(기본 0.15 완화) · 자세 {len(poses)}"
        + (f" · 아래 원뿔 ±{args.down_cone_deg:.0f}° 제한" if args.down_cone_deg else "")
        + (f" · min_poses={args.min_poses}" if args.min_poses is not None else "")
        + f" · {time.strftime('%Y-%m-%d %H:%M')}"
    )
    profile = CalibrationProfile(registration=reg, bias=bias, gravity=model,
                                 mounting_note=note)
    valid, issues = profile.validity()
    print()
    print(f"프로파일 유효성: {valid}   issues {issues}")

    if args.dry_run:
        print("\n--dry-run: 저장하지 않았다.")
        return 0

    if os.path.exists(args.out):
        os.replace(args.out, args.out + ".prev")
        print(f"직전 파일을 {args.out}.prev 로 옮겼다")
    profile.save(args.out)
    print(f"저장 → {args.out}")

    # 쓴 것을 그대로 다시 읽어 확인한다. 형식이 어긋나면 브리지가 조용히 원값을 낸다.
    back = CalibrationProfile.load(args.out)
    v2, i2 = back.validity()
    print(f"되읽기 확인: valid {v2}   issues {i2}")
    print(f"  질량 {back.gravity.mass_kg:.4f} kg · 정렬 산포 {back.gravity.alignment_spread:.4f}")
    print()
    print("브리지에 반영: 콘솔에서 calib.load, 또는 브리지 재시작")
    return 0 if v2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
