#!/usr/bin/env python3
"""지령 프레임과 실제 프레임의 어긋남을 **기록에서 잰다**.

    python3 scripts/measure_frame_map.py runs/main_rule --save runs/frame_map.json

무엇을 하는가
-------------
에피소드의 ``decisions.csv`` 에서 **낸 회전 지령**(cmd_wx·wy·wz, 프로브 프레임)과
**실제로 일어난 회전**(ee_q* 의 차분을 프로브 좌표로 되돌린 것)을 모아, 둘을 잇는 직교
사상을 맞춘다.

왜 재는가
---------
러너는 자세를 ``ee_wrt_base``(로봇 컨트롤러의 TCP)에서 읽고, 지령은 솔버가 자기 FK 의
프로브 프레임으로 읽는다. 두 프레임이 같다는 보장이 없고, 실제로 2026-09-14 실측에서
**z 축 둘레 90.2°** 어긋나 있었다. 그것을 모른 채 기준 자세로 돌아가려 하면 90° 돌아간
방향으로 나가므로 단조 발산한다 (24 → 70 mm).

규약을 문서에서 유도하지 않고 재는 이유는 이 저장소가 이미 두 번 데였기 때문이다:
probe2 의 실행되지 않은 병진 이득, thz 의 "좋아 보이는" 면적 이득. 재면 틀릴 수 없다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

import _common  # noqa: F401  — sys.path 부트스트랩

from rus_policy.frame_map import MAX_RESIDUAL, fit

#: 이 각속도 아래의 표본은 버린다 [°/s]. 잡음이 지령보다 크면 사상이 흐려진다.
MIN_CMD_DEG_S = 0.5
#: 이 밖의 간격은 버린다 [s]. 기록이 끊긴 자리를 미분하면 엉뚱한 값이 나온다.
DT_RANGE = (0.05, 1.0)


def _rot(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], float)


def pairs(path: Path) -> tuple[list, list]:
    """한 에피소드에서 (지령, 실제) 짝을 뽑는다. 둘 다 프로브 좌표 [°/s]."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    def col(n):
        return np.array([float(r[n]) if r.get(n) not in (None, "", "nan") else np.nan
                         for r in rows], float)
    try:
        t = col("t")
        W = np.stack([col("cmd_wx"), col("cmd_wy"), col("cmd_wz")])
        Q = np.stack([col("ee_qw"), col("ee_qx"), col("ee_qy"), col("ee_qz")])
    except KeyError:
        return [], []
    ok = np.all(np.isfinite(Q), 0) & np.all(np.isfinite(W), 0)
    t, W, Q = t[ok], W[:, ok], Q[:, ok]
    C, V = [], []
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        if not (DT_RANGE[0] < dt < DT_RANGE[1]):
            continue
        if np.abs(W[:, i]).sum() < MIN_CMD_DEG_S:
            continue
        A, B = _rot(Q[:, i]), _rot(Q[:, i + 1])
        dR = B @ A.T
        # 작은 회전의 벡터부. dt 가 작아 1 차 근사가 성립한다.
        v = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2.0
        C.append(W[:, i])
        V.append(A.T @ (np.degrees(v) / dt))        # 기저 → 프로브
    return C, V


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="세션 폴더 또는 에피소드 폴더")
    p.add_argument("--save", default="", help="여기에 frame_map.json 을 쓴다")
    a = p.parse_args()

    root = Path(a.run)
    files = sorted(root.glob("**/decisions.csv"))
    if not files:
        print(f"decisions.csv 를 못 찾았다: {root}", file=sys.stderr)
        return 1
    C, V = [], []
    for f in files:
        c, v = pairs(f)
        if c:
            print(f"  {f.parent.name:>30}  표본 {len(c)}")
        C += c
        V += v
    if len(C) < 20:
        print(f"표본이 {len(C)} 개뿐이다 — 회전 지령이 실린 에피소드가 필요하다",
              file=sys.stderr)
        return 1

    fm = fit(np.array(C).T, np.array(V).T)
    fm.source = str(root)
    print(f"\n표본 {fm.n_samples}\n최적 직교 사상 (지령 → 실제):")
    for row in fm.R:
        print("   " + "  ".join(f"{x:+.3f}" for x in row))
    print(f"\n→ {fm.describe()}")
    if fm.residual > MAX_RESIDUAL:
        print(f"⚠️  잔차 {fm.residual:.2f} 가 크다 (> {MAX_RESIDUAL}) — 순수 회전이 아니다.\n"
              "    축별 추종 열화나 특이점 근처일 수 있다. 쓰기 전에 살펴보십시오.")
    if fm.is_identity:
        print("   두 프레임이 같다 — 보정할 것이 없다.")
    if a.save:
        fm.save(a.save)
        print(f"\n기록: {a.save}")
        print("  러너에 넘기려면:  --frame-map " + a.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
