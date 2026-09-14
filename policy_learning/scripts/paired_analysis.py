#!/usr/bin/env python3
"""자세로 짝지은 조건 비교 — 본 시험의 1 차 분석.

    python3 scripts/paired_analysis.py results/main_rule/episodes.csv

왜 짝짓기인가
-------------
같은 자세에서 조건을 모두 돌았으므로 자세가 짝이 된다. 자세마다 난이도가 크게 다르므로
(방광이 원뿔 안에 있는가, 접촉이 좋은가) 독립 표본으로 보면 그 분산이 조건 차이를 덮는다.

McNemar 정확검정은 **불일치 쌍만** 쓴다. 8 쌍 중 6 쌍이 불일치면 양측 p = 2/2⁶ = 0.031 이고,
5 쌍이면 0.0625 다 — 자세 수가 유의선을 직접 정한다. 5 자세로는 p < 0.05 가 구조적으로
불가능하다 (전승해도 0.0625).
"""
from __future__ import annotations

import argparse
import collections
import csv
from math import comb
from pathlib import Path

CONDS = ("rule", "search", "hold")


def mcnemar(pairs: list[tuple[bool, bool]]) -> tuple[int, int, int, float]:
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = b + c
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
    return len(pairs), b, c, p


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episodes", help="results/<세션>/episodes.csv")
    ap.add_argument("--max-episode", type=int, default=None, help="이 번호 이후는 뺀다")
    args = ap.parse_args()

    g: dict[int, dict[str, bool]] = collections.defaultdict(dict)
    for r in csv.DictReader(Path(args.episodes).open(encoding="utf-8")):
        n = int(r["episode"][2:6])
        if args.max_episode and n > args.max_episode:
            continue
        if r["condition"] in CONDS:
            g[int(r["episode"].split("_pose")[1][:2])][r["condition"]] = (r["success"] == "True")

    print(f"{'자세':>4s} " + " ".join(f"{c:>7s}" for c in CONDS))
    for p in sorted(g):
        print(f"{p:4d} " + " ".join(f"{str(g[p].get(c, '—')):>7s}" for c in CONDS))
    print()
    for c in CONDS:
        v = [g[p][c] for p in g if c in g[p]]
        lo, hi = wilson(sum(v), len(v))
        print(f"  {c:7s} {sum(v)}/{len(v)}  {sum(v)/len(v)*100:5.1f} %  Wilson 95 % [{lo*100:.1f}, {hi*100:.1f}]")
    print()
    for a, b in [("rule", "hold"), ("search", "hold"), ("rule", "search")]:
        pr = [(g[p][a], g[p][b]) for p in sorted(g) if a in g[p] and b in g[p]]
        n, B, C, pv = mcnemar(pr)
        star = " ✓" if pv < 0.05 else ""
        print(f"  McNemar {a:6s} 대 {b:6s}: {n} 쌍 · 불일치 {B + C} ({a}만 {B}, {b}만 {C})"
              f" → 양측 p = {pv:.4f}{star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
