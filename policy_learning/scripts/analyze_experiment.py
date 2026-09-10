#!/usr/bin/env python3
"""파일럿·본 시험 분석 — 임계, p_B, 폐기율, 그리고 본 시험 규모.

    python3 scripts/analyze_experiment.py runs/pilot
    python3 scripts/analyze_experiment.py runs/pilot --sweep       # 임계 민감도

파일럿의 산출물은 결과가 아니라 **본 시험을 설계할 수 있게 하는 네 가지** 다.

1. 판정 임계   ``--sweep`` 이 면적비·연결성분을 훑어 성공률이 어떻게 움직이는지 보인다.
               조건 사이가 가장 잘 갈리는 자리가 아니라, **평탄한 자리**를 골라야 한다 —
               성공률이 임계에 민감한 지점을 고르면 그 선택이 결과를 만든다.
2. p_B         위약 성공률. 본 시험 규모를 좌우한다.
3. 도달 불가   에피소드가 시작조차 못 한 비율.
4. 폐기율      시작 조건(면적비 < 2 % · Q_raw ≥ 0.6) 을 못 맞춘 비율.

비율에는 **Wilson 구간**을 쓴다. n=6 에서 정규근사는 음수 하한을 내놓는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import _common  # noqa: F401  — sys.path 부트스트랩

from rus_policy.episode import Thresholds, judge


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """(비율, 하한, 상한). n 이 작을 때 정규근사가 내는 음수 하한을 피한다."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def sample_size_two_proportions(p_b: float, p_c: float, alpha: float = 0.05,
                                power: float = 0.80) -> int:
    """두 비율을 가르는 데 필요한 **팔당** 에피소드 수 (정규근사, 양측).

    p_B 를 모르면 규모를 못 정한다는 것이 파일럿의 요점이다. 여기 나오는 수는 하한으로
    본다 — 자세마다 난이도가 다르므로 실제로는 자세를 블록으로 보는 설계가 필요하다.
    """
    if not (0 < p_b < 1 and 0 < p_c < 1) or abs(p_c - p_b) < 1e-9:
        return -1
    z_a, z_b = 1.959963985, {0.80: 0.8416, 0.90: 1.2816}.get(round(power, 2), 0.8416)
    p_bar = (p_b + p_c) / 2
    num = (z_a * math.sqrt(2 * p_bar * (1 - p_bar)) +
           z_b * math.sqrt(p_b * (1 - p_b) + p_c * (1 - p_c))) ** 2
    return int(math.ceil(num / (p_c - p_b) ** 2))


def load_episodes(root: Path) -> list[dict]:
    rows = []
    path = root / "summary.csv"
    if path.is_file():
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    return rows


def rejudge(root: Path, thr: Thresholds) -> dict[str, list[bool]]:
    """저장된 상태 시계열로 임계를 바꿔 다시 판정한다."""
    out: dict[str, list[bool]] = {}
    for d in sorted(root.glob("ep*/")):
        npz, meta = d / "states.npz", d / "meta.json"
        if not (npz.is_file() and meta.is_file()):
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        if not m.get("episode_started"):
            continue
        z = np.load(npz)
        t, st, t0 = z["t"], z["state"], float(z["t_start"])
        keep = t >= t0
        if not keep.any():
            continue
        out.setdefault(str(m.get("condition", "?")), []).append(
            bool(judge(t[keep], st[keep], thr)["success"]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", help="세션 폴더 (run_experiment.py 의 --out)")
    p.add_argument("--sweep", action="store_true", help="임계 민감도를 훑는다")
    p.add_argument("--hold-s", type=float, default=3.0)
    p.add_argument("--target-lift", type=float, default=0.30,
                   help="본 시험에서 검출하려는 p_C − p_B")
    args = p.parse_args()
    root = Path(args.session)
    rows = load_episodes(root)
    if not rows:
        raise SystemExit(f"{root/'summary.csv'} 가 없습니다 — 아직 에피소드를 돌리지 않았습니다.")

    n = len(rows)
    started = [r for r in rows if r["started"] == "1"]
    print(f"\n에피소드 {n} · 시작함 {len(started)} · 폐기 {n - len(started)}")
    p_, lo, hi = wilson(n - len(started), n)
    print(f"  ④ 시작조건 폐기율  {p_:.1%}  [{lo:.1%}, {hi:.1%}]   ← 본 시험 자세 수를 정할 때 나눗셈")

    print("\n조건별 성공률 (Wilson 95 %)")
    rate = {}
    for cond in sorted({r["condition"] for r in started}):
        sub = [r for r in started if r["condition"] == cond]
        k = sum(r["success"] == "1" for r in sub)
        pr, a, b = wilson(k, len(sub))
        rate[cond] = pr
        print(f"  {cond:9s} {k}/{len(sub)}  {pr:.1%}  [{a:.1%}, {b:.1%}]")

    if "placebo" in rate:
        p_b = rate["placebo"]
        print(f"\n  ② p_B(위약) = {p_b:.1%}")
        p_c = min(0.99, max(0.01, p_b + args.target_lift))
        need = sample_size_two_proportions(max(p_b, 0.01), p_c)
        print(f"  본 시험 규모: p_C − p_B = {args.target_lift:.0%} 를 검출하려면 "
              + (f"팔당 약 {need} 에피소드" if need > 0 else "계산 불가 (p_B 가 0 또는 1)"))
        if len(started) < n:
            need_poses = math.ceil(need / max(1e-9, len(started) / n))
            print(f"  폐기율을 감안하면 자세 약 {need_poses} 개를 잡아야 팔당 {need} 개가 남는다")

    if args.sweep:
        print("\n① 임계 민감도 — 성공률이 **평탄한** 자리를 고른다 (민감한 자리를 고르면 그 선택이 결과다)")
        print(f"  {'면적비':>7} {'연결성분':>8} | " + " ".join(f"{c:>9}" for c in sorted(rate)))
        for area in (0.04, 0.06, 0.08, 0.10, 0.12):
            for comp in (0.70, 0.80, 0.90):
                res = rejudge(root, Thresholds(area_min=area, component_min=comp,
                                               hold_s=args.hold_s))
                if not res:
                    print("  (states.npz 가 없습니다 — 이 세션은 임계를 다시 훑을 수 없습니다)")
                    return 0
                cells = " ".join(
                    f"{np.mean(res.get(c, [])):>8.0%}" if res.get(c) else f"{'-':>9}"
                    for c in sorted(rate))
                print(f"  {area:>7.2f} {comp:>8.2f} | {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
