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


#: 탐색 설정이 다르면 **다른 조건**이다. 2026-09-12 에 걸음 2°→5° · min_gain 0.01→0.02 로
#: 바꿨고(dQ/dθ≈0.005/° 라 2° 는 정지 잡음 0.007 에 묻혔다), 그 전후를 한 팔로 묶으면 안 된다.
#: 그날 이전 에피소드는 meta 에 설정이 없다 — 그때 값은 2.0° · 0.01 이었다.
UNRECORDED = "설정미기록"


def search_variant(meta: dict) -> str:
    s = meta.get("search")
    if not isinstance(s, dict):
        return UNRECORDED
    step, gain = s.get("step_deg"), s.get("min_gain")
    return f"{step:g}°/{gain:g}" if None not in (step, gain) else UNRECORDED


def variants_by_dir(root: Path) -> dict[str, str]:
    """에피소드 폴더 이름 → 탐색 설정 꼬리표. search 가 아닌 조건은 담지 않는다."""
    out = {}
    for meta in sorted(root.glob("ep*/meta.json")):
        m = json.loads(meta.read_text(encoding="utf-8"))
        if str(m.get("condition")) == "search":
            out[meta.parent.name] = search_variant(m)
    return out


def split_search(cond: str, key: str, variants: dict[str, str]) -> str:
    """설정이 여러 가지면 ``search`` 를 설정별로 가른다. 하나뿐이면 그대로 둔다."""
    if cond != "search" or len(set(variants.values())) < 2:
        return cond
    return f"search[{variants.get(key, UNRECORDED)}]"


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
    variants = variants_by_dir(root)
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
        cond = split_search(str(m.get("condition", "?")), d.name, variants)
        out.setdefault(cond, []).append(
            bool(judge(t[keep], st[keep], thr)["success"]))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", help="세션 폴더 (run_experiment.py 의 --out)")
    p.add_argument("--sweep", action="store_true", help="임계 민감도를 훑는다 (조건별)")
    p.add_argument("--blind-thresholds", action="store_true",
                   help="임계를 **조건을 가린 채** 고른다. 조건별 성공률을 보고 고르면 그 선택이 "
                        "결과를 만든다 — 본 시험을 이어서 돌릴 때는 이쪽을 쓴다")
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

    variants = variants_by_dir(root)
    for r in started:
        r["condition"] = split_search(r["condition"], r.get("dir", ""), variants)
    if len(set(variants.values())) > 1:
        print(f"\n⚠️ 탐색 설정이 {len(set(variants.values()))} 가지다 — 팔을 나눠 센다 "
              f"({', '.join(sorted(set(variants.values())))}). "
              f"{UNRECORDED} 은 2026-09-12 이전(걸음 2° · min_gain 0.01)이다.")

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

    if args.blind_thresholds:
        print("\n① 임계 선택 — **조건을 가린다.** 전체를 한 덩어리로 보고 평탄한 자리를 고른다.")
        print("   기울기가 작을수록 그 임계에서 성공률이 임계 자체에 덜 휘둘린다는 뜻이다.")
        print(f"  {'면적비':>7} {'연결성분':>8} {'성공률':>8} {'면적비 기울기':>14}")
        grid = [(a, c) for a in (0.04, 0.06, 0.08, 0.10, 0.12) for c in (0.70, 0.80, 0.90)]
        pooled = {}
        for area, comp in grid:
            res = rejudge(root, Thresholds(area_min=area, component_min=comp, hold_s=args.hold_s))
            if not res:
                print("  (states.npz 가 없습니다 — 이 세션은 임계를 다시 훑을 수 없습니다)")
                return 0
            allv = [v for lst in res.values() for v in lst]      # 조건을 섞는다
            pooled[(area, comp)] = float(np.mean(allv)) if allv else float("nan")
        for area, comp in grid:
            here = pooled[(area, comp)]
            nb = [pooled.get((a, comp)) for a in (round(area - 0.02, 2), round(area + 0.02, 2))]
            nb = [v for v in nb if v is not None and np.isfinite(v)]
            slope = (max(nb) - min(nb)) if len(nb) == 2 else float("nan")
            print(f"  {area:>7.2f} {comp:>8.2f} {here:>8.0%} {slope:>13.1%}")
        print("\n  기울기가 가장 작은 줄을 고르고, 그 값을 본 시험에 --success-* 로 고정한다.")
        print("  고른 뒤에는 바꾸지 않는다 — 바꾸는 순간 사전 등록이 아니다.")

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
