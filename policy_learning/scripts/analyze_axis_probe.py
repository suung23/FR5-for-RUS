#!/usr/bin/env python3
"""축 흔들기(condition=axis_probe) 기록에서 **기하 이득**을 낸다.

    python3 scripts/analyze_axis_probe.py runs/probe/ep0001_pose01_axis_probe

무엇을 내는가
-------------
축마다 "이 축으로 1° (또는 1 mm) 움직이면 영상 속 방광이 얼마나 옮겨지는가":

    d(centroid_dx)/d(지령)   면내(in-plane) 축일수록 크다 — 좌우 정렬에 쓸 축
    d(area_ratio)/d(지령)    면외(out-of-plane) 축일수록 크다 — 단면이 바뀐다

부호가 곧 제어기의 부호다. ``centroid_dx`` 가 양수(방광이 오른쪽)일 때 그것을 0 으로
되돌리려면 ``-centroid_dx / 이득`` 만큼 지령하면 된다.

왜 이걸 따로 재는가: 프리핸드 데이터의 기울기(43 °/단위)는 **행동** 기울기라 못 쓴다.
기존 실기 로그도 여러 축이 섞여 돌아 축별로 분리되지 않는다 (rus_policy.axis_probe 머리말).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _common import setup_logging  # noqa: F401

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rus_policy.axis_probe import AXIS_INDEX          # noqa: E402
from rus_policy.perception import STATE_FEATURE_NAMES  # noqa: E402

CMD = {"x": "cmd_vx", "y": "cmd_vy", "thx": "cmd_wx", "thy": "cmd_wy", "thz": "cmd_wz"}
UNIT = {"x": "mm", "y": "mm", "thx": "°", "thy": "°", "thz": "°"}


def segments(rows: list[dict]) -> list[tuple[str, list[int]]]:
    """``probe_label`` 이 같은 연속 구간으로 나눈다 (done 은 뺀다)."""
    out, cur, lab = [], [], None
    for i, r in enumerate(rows):
        v = r.get("probe_label", "")
        if v != lab:
            if cur and lab and lab != "done":
                out.append((lab, cur))
            cur, lab = [], v
        cur.append(i)
    if cur and lab and lab != "done":
        out.append((lab, cur))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode", help="axis_probe 에피소드 폴더")
    p.add_argument("--min-visible", type=float, default=0.5,
                   help="구간에서 마스크가 이 비율 이상 있어야 쓴다")
    args = p.parse_args()

    ep = Path(args.episode)
    rows = list(csv.DictReader(open(ep / "decisions.csv", encoding="utf-8")))
    if not rows or "probe_label" not in rows[0]:
        print("probe_label 이 없다 — axis_probe 로 찍은 에피소드가 아니다.")
        return 1
    z = np.load(ep / "states.npz")
    I = {n: i for i, n in enumerate(STATE_FEATURE_NAMES)}
    ts, st = z["t"] - z["t"][0], z["state"]
    dx, area, has = st[:, I["centroid_dx"]], st[:, I["area_ratio"]], st[:, I["has_mask"]] > 0.5
    td = np.array([float(r["t_episode"]) for r in rows])

    by_axis: dict[str, list[tuple[float, float, float]]] = {}
    for lab, idx in segments(rows):
        name, sign = lab[:-1], (1.0 if lab[-1] == "+" else -1.0)
        if name not in AXIS_INDEX:
            continue
        t0, t1 = td[idx[0]], td[idx[-1]]
        m = (ts >= t0) & (ts <= t1)
        if m.sum() < 5 or has[m].mean() < args.min_visible:
            continue
        cmd = np.mean([abs(float(r[CMD[name]] or 0.0)) for r in (rows[i] for i in idx)])
        # 회전 지령은 rad/s 로 기록된다 — °/s 로 되돌린다
        if name.startswith("th"):
            cmd = np.degrees(cmd)
        else:
            cmd = cmd * 1000.0                     # m/s → mm/s
        k = np.where(m)[0]
        span = ts[k[-1]] - ts[k[0]]
        if span <= 0 or cmd <= 0:
            continue
        move = sign * cmd * span                   # 그 구간의 총 지령량 [° 또는 mm]
        by_axis.setdefault(name, []).append(
            (move, dx[k[-1]] - dx[k[0]], area[k[-1]] - area[k[0]]))

    if not by_axis:
        print("쓸 구간이 없다 — 방광이 계속 안 보였을 수 있다 (--min-visible 확인).")
        return 1

    print(f"\n=== 축별 기하 이득 ({ep.name}) ===")
    print(f"{'축':>5} {'구간':>5} {'d(centroid_dx)/지령':>20} {'d(area)/지령':>15}")
    best = None
    for name in ("thx", "thy", "thz", "x", "y"):
        v = by_axis.get(name)
        if not v or len(v) < 2:
            print(f"{name:>5} {len(v or []):>5} {'표본 부족':>20}")
            continue
        M = np.array(v)
        g_dx = float(np.polyfit(M[:, 0], M[:, 1], 1)[0])
        g_ar = float(np.polyfit(M[:, 0], M[:, 2], 1)[0])
        u = UNIT[name]
        print(f"{name:>5} {len(v):>5} {g_dx:>+16.5f} /{u:<2} {g_ar:>+11.5f} /{u:<2}")
        if best is None or abs(g_dx) > abs(best[1]):
            best = (name, g_dx)

    if best and abs(best[1]) > 0:
        name, g = best
        print(f"\n좌우 정렬에 쓸 축: **{name}** (이득 {g:+.5f} /{UNIT[name]})")
        print(f"  centroid_dx 를 0 으로 되돌리는 지령 = -centroid_dx / {g:+.5f} "
              f"[{UNIT[name]}]")
        print(f"  예: centroid_dx = +0.10 이면 {-0.10 / g:+.1f} {UNIT[name]}")
    print("\n면내 축은 centroid_dx 를 크게 움직이고, 면외 축은 area 를 바꾸되 centroid_dx 는 덜 건드린다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
