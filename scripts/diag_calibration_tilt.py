#!/usr/bin/env python3
"""적층이 동축인지 — 수집한 자세에서 플랜지→센서 회전을 **제약 없이** 풀어 본다.

    python3 scripts/diag_calibration_tilt.py            # 최근 자세 전부
    python3 scripts/diag_calibration_tilt.py --count 6  # 마지막 6 자세만 (빠른 확인)

왜 필요한가
----------
`fit_gravity_model` 은 정렬을 **z 둘레 회전 하나**로 제약해 푼다 (`solve_axial_alignment`).
그것은 가정이 아니라 측정이었다 — 축방향 압축이 Fz 채널로 깨끗하게 갔다는 사실. 자중이
200 g 대에 센서 오프셋이 수백 N 이라 미지수를 하나로 줄여야 풀리기도 한다.

문제는 **그 전제가 깨졌을 때 조용하다는 것**이다. 적층이 몇 도 기울어 앉으면 z-회전
모델이 그것을 담지 못하고, 잔차가 한 축(대개 z)으로 몰리면서 "정렬 특이값 산포가 크다"
로만 보고된다. 그 메시지는 "자세 하나가 흔들렸다" 를 가리켜 엉뚱한 곳을 찾게 만든다.

이 도구는 같은 자세로 **제약 없는 3x3** 을 풀어 둘을 가른다:

  * 자유 회전으로 잔차가 크게 떨어지고 회전축이 z 에서 벗어나 있다 → **기울어 앉았다.**
    볼트·어댑터 면을 본다. 소프트웨어로 풀 일이 아니다.
  * 자유 회전으로도 안 맞는다 → 회전으로 설명 안 되는 오차. 센서나 강체성을 의심한다.

2026-09-11 실측 예: z-회전 제약 잔차 0.347 N / 자유 3x3 0.112 N, 회전축이 z 에서 4.85°.
플랜지 볼트가 고르게 조여지지 않았던 것이었다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

DEFAULT_LOG = os.path.expanduser("~/.ros/fr5_px6d_calibration_poses.jsonl")


def load(path: str, count: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """자세 로그 꼬리에서 `count` 개. 브리지는 적합할 때마다 현재 목록을 통째로 덧붙인다."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    if not rows:
        raise SystemExit(f"{path} 가 비어 있다")
    rows = rows[-count:] if count > 0 else rows
    missing = [r.get("label", "?") for r in rows if r.get("gravity_flange") is None]
    if missing:
        raise SystemExit(f"gravity_flange 가 없는 자세가 있다: {missing[:3]}")
    G = np.array([r["gravity_flange"] for r in rows], float)
    F = np.array([r["wrench"][:3] for r in rows], float)
    return G, F, [str(r.get("label", "")) for r in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses-log", default=DEFAULT_LOG)
    ap.add_argument("--count", type=int, default=0, help="꼬리에서 볼 자세 수 (0 = 전부)")
    ap.add_argument("--min-span", type=float, default=0.15,
                   help="중력 방향이 3 축을 얼마나 채워야 판정하는가 (최소/최대 특이값 비)")
    ap.add_argument("--tilt-warn-deg", type=float, default=1.5,
                    help="이 각을 넘으면 기울어 앉은 것으로 본다")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])

    G, F, labels = load(a.poses_log, a.count)
    n = len(G)
    if n < 4:
        raise SystemExit(f"자세가 {n} 개다 — 3x3 을 풀려면 4 개 이상 필요하다")

    # 자세가 한 평면에 몰려 있으면 3x3 은 **풀리지 않는다.** 그래도 최소제곱은
    # 답을 내놓는다 — 몇 점을 정확히 지나는 엉뚱한 해를. 2026-09-11 에 평면에 놓인
    # 자세 6 개로 "질량 69 kg, 기울기 36°" 가 나왔고, 잔차는 0.009 N 으로 오히려
    # 좋아 보였다. 그 답을 믿으면 멀쩡한 조립을 뜯게 된다. 먼저 막는다.
    directions = G / np.linalg.norm(G, axis=1, keepdims=True)
    spread = np.linalg.svd(directions, compute_uv=False)
    span = float(spread.min() / spread.max())
    if span < a.min_span:
        print(f"자세 {n} 개  ({labels[0]} … {labels[-1]})")
        print(f"  중력 방향의 3 축 span  {span:.4f}  (필요 {a.min_span})")
        print()
        print("⛔ 자세가 한 평면에 몰려 있어 3x3 을 풀 수 없다 — 판정하지 않는다.")
        print("   프로브를 수평으로 두고 **J6(프로브 자기 축)를 45°씩** 돌려 몇 자세를 더 잡아라.")
        print("   그래야 중력이 센서 y 축에도 실린다. 교정 화면의 coverage 와 같은 양이다.")
        return 2

    design = np.hstack([G, np.ones((n, 1))])
    sol, *_ = np.linalg.lstsq(design, F, rcond=None)
    K = sol[:3, :].T
    residual = float(np.sqrt(((F - design @ sol) ** 2).mean()))

    # K ≈ m·R 인가. 극분해로 회전을 떼어내고 특이값으로 등방성을 본다.
    U, S, Vt = np.linalg.svd(K)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
        S = S.copy()
        S[-1] *= -1
    R = U @ Vt
    angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(axis[2]))))))

    print(f"자세 {n} 개  ({labels[0]} … {labels[-1]})")
    print(f"  자유 3x3 잔차      {residual:.4f} N")
    print(f"  특이값             {np.round(S, 4)}   비등방 {(S.max()-S.min())/S.mean()*100:.1f} %")
    print(f"  질량(특이값 평균)  {S.mean():.4f} kg")
    print(f"  회전               {angle:.2f}°  축 {np.round(axis, 3)}")
    print(f"  z 축에서의 기울기  {tilt:.2f}°   (동축이면 0)")
    print()
    if tilt > a.tilt_warn_deg:
        print(f"⚠️ 기울어 앉았다 ({tilt:.2f}° > {a.tilt_warn_deg}°).")
        print("   z-회전만 허용하는 교정 적합은 이것을 담지 못하고 잔차를 한 축으로 민다.")
        print("   플랜지 볼트를 고르게 다시 조이고, 어댑터 면이 평행한지 본다. 재수집 필요.")
        return 1
    print(f"✅ 동축으로 볼 만하다 ({tilt:.2f}° ≤ {a.tilt_warn_deg}°). 교정을 계속 진행해도 된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
