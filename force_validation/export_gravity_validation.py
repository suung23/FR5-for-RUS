#!/usr/bin/env python3
r"""교정 때 잡은 자세들에서 `gravity_compensation_validation.csv` 를 만든다.

    python3 export_gravity_validation.py \\
      --poses ~/.ros/fr5_px6d_calibration_poses.jsonl \\
      --profile ~/.ros/fr5_px6d_calibration.json \\
      --out captures/gravity_compensation_validation.csv

무부하 다자세 교정에서 이미 잰 자세들이 곧 사양이 말하는 validation pose 다.
각 자세에서 **보상을 적용한 뒤 남는** probe-frame 잔차를 계산해 낸다.

새로 재지 않는다. 이미 있는 기록을 사양이 요구하는 표 형식으로 옮길 뿐이며,
보상 계산은 런타임과 같은 함수(:func:`fr5_control.wrench_profile.compensate`)를
쓴다 — 여기서 따로 구현하면 두 값이 갈라진다.

**한계 하나를 미리 적어 둔다.** 이 기록에는 자세의 회전이 중력 방향으로만 남아
있어 ``tcp_r*_rad`` 세 각을 복원할 수 없다. 그래서 그 열은 비워 두고, 분석은
자세별이 아니라 전체 불확도를 쓰게 된다. 자세별로 주고 싶으면 capture 때 회전을
함께 남겨야 한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np


def main(argv=None) -> int:
    """자세 기록과 프로파일을 읽어 검증 CSV 를 쓴다."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--poses", default=os.path.expanduser("~/.ros/fr5_px6d_calibration_poses.jsonl"))
    parser.add_argument(
        "--profile", default=os.path.expanduser("~/.ros/fr5_px6d_calibration.json"))
    parser.add_argument(
        "--out", default="captures/gravity_compensation_validation.csv")
    parser.add_argument(
        "--include-working-tare", action="store_true",
        help="작업 자세 영점까지 적용한 잔차를 낸다. 기본은 적용하지 않는다 — "
             "이 자세들은 영점보다 먼저 잡혔고, 여기서 묻는 것은 **중력보상이 "
             "얼마나 맞는가** 이지 그 위에 얹은 상수가 아니다",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    for path in (args.poses, args.profile):
        if not os.path.exists(path):
            print(f"파일이 없다: {path}", file=sys.stderr)
            return 2

    try:
        from fr5_control.wrench_profile import CalibrationProfile, compensate
    except ImportError as exc:
        print(f"fr5_control 을 불러올 수 없다 ({exc}) — install/setup.bash 를 source 하라",
              file=sys.stderr)
        return 2

    profile = CalibrationProfile.load(args.profile)
    if profile.gravity is None:
        print("프로파일에 중력 모델이 없다 — 교정을 먼저 끝내라", file=sys.stderr)
        return 2

    if not args.include_working_tare and profile.working_tare is not None:
        # 이 자세들은 작업 영점보다 먼저 잡혔다. 그 상수를 얹으면 중력보상의
        # 잔차가 아니라 "지금 시스템이 내는 값" 이 되고, 두 가지가 섞인다.
        print("  작업 자세 영점은 빼고 계산한다 (--include-working-tare 로 포함 가능)")
        profile.working_tare = None

    rows = []
    skipped = 0
    with open(args.poses, encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            flange_gravity = record.get("gravity_flange")
            if flange_gravity is None:
                skipped += 1
                continue

            # {S}g 를 만든 회전을 거꾸로 써서 {B}R{F} 대신 쓸 수 있는 형태로 넘긴다.
            # compensate() 는 {B}R{F} 를 받아 g_F = R^T g_B 를 만드는데, 여기서는
            # g_F 를 이미 알고 있으므로 같은 결과가 나오는 행렬을 세운다.
            g_flange = np.asarray(flange_gravity, dtype=float)
            base_gravity = np.array([0.0, 0.0, -9.80665])
            rot_base_flange = _rotation_taking(base_gravity, g_flange)

            result = compensate(profile, np.asarray(record["wrench"], dtype=float),
                                rot_base_flange)
            residual = np.asarray(result.contact_probe, dtype=float)
            rows.append({
                "pose_id": record.get("label") or str(index),
                "timestamp_utc": "",
                "tcp_rx_rad": "",
                "tcp_ry_rad": "",
                "tcp_rz_rad": "",
                "gravity_residual_Fx_P_N": f"{residual[0]:.6f}",
                "gravity_residual_Fy_P_N": f"{residual[1]:.6f}",
                "gravity_residual_Fz_P_N": f"{residual[2]:.6f}",
                "valid_pose": "TRUE",
            })

    if not rows:
        print("쓸 수 있는 자세가 없다 (gravity_flange 가 없는 옛 기록뿐이다)",
              file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    columns = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    normals = np.array([float(row["gravity_residual_Fz_P_N"]) for row in rows])
    print(f"  자세 {len(rows)} 개" + (f" (건너뜀 {skipped})" if skipped else ""))
    print(f"  normal 잔차: 평균 {normals.mean():+.4f} N · SD {normals.std(ddof=1):.4f} N · "
          f"최대 |e| {np.abs(normals).max():.4f} N")
    print(f"  {args.out}")
    print("  ⚠️ tcp_r*_rad 는 비어 있다 — 이 기록에 회전이 남아 있지 않아서다.")
    print("     분석은 자세별이 아니라 전체(global) 중력 불확도를 쓴다.")
    return 0


def _rotation_taking(source, target):
    """``R^T · source = target`` 을 만족하는 회전 하나.

    자세 기록에는 ``{F}g`` 만 있고 ``{B}R{F}`` 가 없다. 보상 계산은 그 행렬을
    받지만 실제로 쓰는 것은 ``R^T g_B`` 하나이므로, 그 값을 내는 아무 회전이나
    있으면 결과가 같다.
    """
    def frame(vector):
        first = np.asarray(vector, dtype=float)
        first = first / np.linalg.norm(first)
        helper = np.array([1.0, 0.0, 0.0])
        if abs(float(first @ helper)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0])
        second = np.cross(first, helper)
        second /= np.linalg.norm(second)
        return np.column_stack([first, second, np.cross(first, second)])

    return frame(source) @ frame(target).T


if __name__ == "__main__":
    sys.exit(main())
