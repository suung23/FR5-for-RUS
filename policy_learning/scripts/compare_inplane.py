#!/usr/bin/env python3
"""학습 런들을 **면내 방향**으로 줄 세운다.

    python3 scripts/compare_inplane.py runs/rsp_fixed runs/beta_*

왜 이 지표인가
--------------
시험이 재는 것이 면내 방향(`centroid_dx` 와 정렬축 지령의 상관)이므로, 모델도 그것으로
고른다. ``select_*`` 계열은 **사전분포 z + Q̂ 선택** 경로, 즉 실행시와 같은 경로에서 나온
값이라 누설에 면역이다 (``mae_*`` 는 사후분포라 낙관적이다 — 런북 §3).

``select_corr_dx_thy`` 는 부호까지 의미가 있다: 2026-09-14 확정한 규약에서 **음수가 옳다**
(dx > 0 이면 θy < 0 으로 되돌려야 한다). 양수면 모델이 중심을 **밀어내고** 있다는 뜻이다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(run: Path) -> list[dict]:
    f = run / "metrics.jsonl"
    if not f.is_file():
        return []
    out = []
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" in r and "val/select_corr_dx_thy" in r:
            out.append(r)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+")
    p.add_argument("--min-epoch", type=int, default=3,
                   help="이 에폭 이전은 무시 — 1~2 에폭의 상관은 아직 β 워밍업 중이라 과장된다")
    args = p.parse_args()

    print(f"{'런':22s} {'에폭':>5s} {'corr(dx,θy)':>12s} {'dir_thy':>8s} {'sel_nmae':>9s} "
          f"{'θy mae':>7s} {'vy_std':>7s} {'beta':>6s}")
    print("-" * 82)
    best = []
    for r in args.runs:
        run = Path(r)
        R = [x for x in rows(run) if x["epoch"] >= args.min_epoch]
        if not R:
            print(f"{run.name:22s}  (metrics 없음)")
            continue
        # 면내 방향이 가장 강하고 **부호가 옳은**(음수) epoch
        pick = min(R, key=lambda x: x["val/select_corr_dx_thy"])
        best.append((pick["val/select_corr_dx_thy"], run.name, pick))
        print(f"{run.name:22s} {pick['epoch']:5d} {pick['val/select_corr_dx_thy']:+12.3f} "
              f"{pick.get('val/select_dir_thy', float('nan')):8.3f} "
              f"{pick.get('val/select_nmae', float('nan')):9.2f} "
              f"{pick.get('val/mae_thy_deg', float('nan')):7.2f} "
              f"{pick.get('val/mode_collapse_vy_std', float('nan')):7.2f} "
              f"{pick.get('val/beta', float('nan')):6.2f}")
        last = R[-1]
        print(f"{'  └ 마지막 에폭':22s} {last['epoch']:5d} {last['val/select_corr_dx_thy']:+12.3f} "
              f"{last.get('val/select_dir_thy', float('nan')):8.3f} "
              f"{last.get('val/select_nmae', float('nan')):9.2f}")
    if best:
        best.sort()
        c, name, pick = best[0]
        print(f"\n면내 방향이 가장 강한 런: **{name}** (epoch {pick['epoch']}, corr {c:+.3f})")
        if c > -0.3:
            print("⚠ 상관의 크기가 작다 — 이 모델로는 면내 방향을 주장하기 어렵다.")
        if c > 0:
            print("⚠ 부호가 **양수**다 — 중심을 밀어내는 방향이다. 프레임 규약을 다시 확인할 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
