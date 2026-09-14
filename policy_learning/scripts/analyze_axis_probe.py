#!/usr/bin/env python3
"""축 흔들기(condition=axis_probe) 기록에서 **기하 이득**을 낸다.

    python3 scripts/analyze_axis_probe.py runs/probe          # 가장 최근 axis_probe 를 고른다
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
from rus_policy.axis_gains import AxisGains            # noqa: E402
from rus_policy.axis_probe import AXIS_INDEX          # noqa: E402
from rus_policy.perception import STATE_FEATURE_NAMES  # noqa: E402

CMD = {"x": "cmd_vx", "y": "cmd_vy", "thx": "cmd_wx", "thy": "cmd_wy", "thz": "cmd_wz"}
#: 이만큼은 있어야 기울기를 믿는다. 양·음 각 2 회면 4 — 기본 설정이 그것을 만든다.
MIN_BURSTS = 3
#: 지령한 양의 이 비율만큼은 **실제로 움직여야** 그 축을 쟀다고 본다.
#:
#: 2026-09-14: 병진 지령이 제어 스택의 gate_axes 에서 지워지고 있었는데(정책 모드는 회전만
#: 통과) 분석기는 그걸 모른 채 "x 병진 이득 0.00708 /mm, r=1.00" 을 냈다. 로봇은 0.07 mm 밖에
#: 안 움직였다. 지령만 보고 이득을 내면 **실행되지 않은 축에도 그럴듯한 숫자가 붙는다.**
MIN_REALIZED = 0.3
UNIT = {"x": "mm", "y": "mm", "thx": "°", "thy": "°", "thz": "°"}


def bursts(rows: list[dict]) -> list[tuple[str, float, float, float]]:
    """지령이 살아 있는 구간마다 (축, 시작 t, 끝 t, 부호 있는 지령량).

    ``probe_label`` 로만 나누면 안 된다 — 반복 2 회가 같은 이름이라 한 덩어리로 뭉친다
    (2026-09-12: 축마다 점이 1~2 개뿐이라 회귀가 안 됐다). 지령이 0 이 아닌 구간으로 나눈다.

    단위는 기록된 그대로다: 회전 °/s, 병진 mm/s (``_publish`` 가 변환 **전** 값을 돌려준다).
    """
    out, cur = [], None
    for r in rows:
        name = str(r.get("probe_label", ""))[:-1]
        if name not in AXIS_INDEX:
            name = ""
        v = float(r.get(CMD.get(name, ""), 0.0) or 0.0) if name else 0.0
        t = float(r["t_episode"])
        if name and abs(v) > 1e-9:
            if cur and cur[0] == name and np.sign(cur[3]) == np.sign(v):
                cur[2], cur[4] = t, cur[4] + [v]
            else:
                if cur:
                    out.append(cur)
                cur = [name, t, t, v, [v]]
        elif cur:
            out.append(cur)
            cur = None
    if cur:
        out.append(cur)
    got = []
    for name, t0, t1, _, vs in out:
        if t1 > t0:
            got.append((name, t0, t1, float(np.mean(vs)) * (t1 - t0)))
    return got


def resolve_episode(path: Path) -> Path | None:
    """에피소드 폴더면 그대로, run 폴더면 그 안의 **가장 최근 axis_probe** 를 고른다.

    자리표시자를 손으로 채우게 하면 셸에서 깨진다 (2026-09-12 `<에피소드폴더>` 를 그대로
    붙여 넣어 bash 가 리다이렉션으로 읽었다).
    """
    if (path / "decisions.csv").is_file():
        return path
    got = []
    # 에피소드 폴더 이름이 모드마다 다르다 — run_experiment 는 ep*, --loop 은 attempt_*.
    for meta in path.glob("*/meta.json"):
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(m.get("condition")) == "axis_probe":
            got.append(meta.parent)
    return max(got, key=lambda d: d.stat().st_mtime) if got else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episode",
                   help="axis_probe 에피소드 폴더, 또는 run 폴더 (그 안의 가장 최근 것을 고른다)")
    p.add_argument("--save", nargs="?", const="", default=None,
                   help="이득을 파일로 남긴다 (기본: <에피소드>/axis_gains.json). "
                        "규칙 제어기(--condition rule)가 이 파일을 읽는다")
    p.add_argument("--min-visible", type=float, default=0.5,
                   help="구간에서 마스크가 이 비율 이상 있어야 쓴다")
    args = p.parse_args()

    ep = resolve_episode(Path(args.episode))
    if ep is None:
        print(f"{args.episode} 안에 axis_probe 에피소드가 없다.")
        return 1
    rows = list(csv.DictReader(open(ep / "decisions.csv", encoding="utf-8")))
    if not rows or "probe_label" not in rows[0]:
        print("probe_label 이 없다 — axis_probe 로 찍은 에피소드가 아니다.")
        return 1
    z = np.load(ep / "states.npz")
    I = {n: i for i, n in enumerate(STATE_FEATURE_NAMES)}
    ts, st = z["t"] - z["t"][0], z["state"]
    td = np.array([float(r["t_episode"]) for r in rows])
    have_pose = all(k in rows[0] for k in ("ee_x", "ee_qw"))
    if have_pose:
        ee_p = np.array([[float(r["ee_x"]), float(r["ee_y"]), float(r["ee_z"])] for r in rows])
        ee_q = np.array([[float(r["ee_qw"]), float(r["ee_qx"]),
                          float(r["ee_qy"]), float(r["ee_qz"])] for r in rows])

    def realized(t0: float, t1: float, rot: bool) -> float:
        """그 구간에 로봇이 **실제로** 움직인 양 [° 또는 mm]."""
        if not have_pose:
            return float("nan")
        i, j = int(np.searchsorted(td, t0)), min(int(np.searchsorted(td, t1)), len(td) - 1)
        if j <= i:
            return float("nan")
        if rot:
            d = min(1.0, abs(float(np.dot(ee_q[i], ee_q[j]))))
            return float(np.degrees(2 * np.arccos(d)))
        return float(np.linalg.norm(ee_p[j] - ee_p[i])) * 1000.0
    dx, area, has = st[:, I["centroid_dx"]], st[:, I["area_ratio"]], st[:, I["has_mask"]] > 0.5

    def window(a: float, b: float):
        """[a,b] 구간의 (centroid_dx, area, 마스크 있는 표본 수, 전체 표본 수).

        면적은 **마스크가 없을 때 0 이 정답**이다 — 방광이 화면 밖으로 나간 것이 곧 그 축의
        효과다. 중심 위치는 마스크가 있어야만 정의된다. 그래서 둘을 따로 센다
        (2026-09-12: 면외 축에서 방광이 사라져 구간을 통째로 버렸다).
        """
        w = (ts >= a) & (ts <= b)
        if not w.any():
            return np.nan, np.nan, 0, 0
        wm = w & has
        return (float(dx[wm].mean()) if wm.any() else np.nan,
                float(area[w].mean()), int(wm.sum()), int(w.sum()))

    #: 초음파 지연 0.2 s 를 넘겨 영상이 따라온 뒤를 본다. 쉬는 구간(1.5 s) 안에 들어가야 한다.
    LAG, WIN = 0.4, 0.8

    by_axis: dict[str, list[tuple[float, float, float]]] = {}
    moved: dict[str, list[tuple[float, float]]] = {}
    for name, t0, t1, amount in bursts(rows):
        moved.setdefault(name, []).append(
            (abs(amount), realized(t0, t1, name.startswith("th"))))
        pre = window(t0 - LAG - WIN, t0 - LAG)        # 흔들기 직전의 쉬는 구간
        post = window(t1 + LAG, t1 + LAG + WIN)       # 흔든 뒤의 쉬는 구간
        if pre[3] < 3 or post[3] < 3:
            continue                                   # 구간 자체가 비었다
        d_dx = (post[0] - pre[0]) if (pre[2] >= 3 and post[2] >= 3) else np.nan
        by_axis.setdefault(name, []).append((amount, d_dx, post[1] - pre[1]))

    if not by_axis:
        print("쓸 구간이 없다 — 방광이 계속 안 보였을 수 있다 (--min-visible 확인).")
        return 1

    gains = AxisGains(source=str(ep))
    print(f"\n=== 축별 기하 이득 ({ep.name}) ===")
    print(f"{'축':>5} {'구간':>5} {'d(centroid_dx)/지령':>18} {'부호일치':>7} {'r':>6} "
          f"{'d(area)/지령':>15}")
    best = None
    for name in ("thx", "thy", "thz", "x", "y"):
        # 실행되지 않은 축은 아예 세지 않는다 — 지령만으로는 이득이 아니다.
        mv = np.array(moved.get(name, []), dtype=float).reshape(-1, 2)
        if len(mv) and np.isfinite(mv[:, 1]).any():
            want, got = mv[:, 0].mean(), np.nanmean(mv[:, 1])
            if want > 0 and got / want < MIN_REALIZED:
                print(f"{name:>5} {len(by_axis.get(name) or []):>5} "
                      f"{'실행 안 됨 — 지령 %.1f, 실제 %.2f' % (want, got):>26}")
                continue
        v = by_axis.get(name)
        if not v or len(v) < MIN_BURSTS:
            print(f"{name:>5} {len(v or []):>5} {'구간 부족 (≥%d 필요)' % MIN_BURSTS:>22}")
            continue
        M = np.array(v)
        # 지령량이 서로 비슷하면 기울기가 발산한다 — 2 점으로 −273/° 같은 값이 나왔다.
        if np.ptp(M[:, 0]) < 0.5 * np.abs(M[:, 0]).mean():
            print(f"{name:>5} {len(v):>5} {'지령량이 안 퍼져 기울기 불안정':>22}")
            continue
        ok = np.isfinite(M[:, 1])
        u = UNIT[name]
        g_ar = float(np.polyfit(M[:, 0], M[:, 2], 1)[0])
        gains.area_per_unit[name] = g_ar
        if ok.sum() >= MIN_BURSTS and np.ptp(M[ok, 0]) > 0:
            g_dx = float(np.polyfit(M[ok, 0], M[ok, 1], 1)[0])
            r = float(np.corrcoef(M[ok, 0], M[ok, 1])[0, 1]) if ok.sum() > 2 else float("nan")
            agree = float(np.mean(np.sign(M[ok, 0]) == np.sign(M[ok, 1])))
            print(f"{name:>5} {len(v):>5} {g_dx:>+14.5f} /{u:<2} {agree:>7.0%} "
                  f"{r:>+6.2f} {g_ar:>+11.5f} /{u:<2}")
            gains.centroid_per_unit[name] = g_dx
            gains.r_centroid[name] = r
            gains.sign_agreement[name] = float(
                np.mean(np.sign(M[ok, 0]) == np.sign(M[ok, 1])))
            if best is None or abs(g_dx) > abs(best[1]):
                best = (name, g_dx)
        else:
            print(f"{name:>5} {len(v):>5} {'중심 측정 불가 (%d/%d)' % (ok.sum(), len(M)):>18} "
                  f"{'':>7} {g_ar:>+11.5f} /{u:<2}")

    if best and abs(best[1]) > 0:
        name, g = best
        print(f"\n좌우 정렬에 쓸 축: **{name}** (이득 {g:+.5f} /{UNIT[name]})")
        print(f"  centroid_dx 를 0 으로 되돌리는 지령 = -centroid_dx / {g:+.5f} "
              f"[{UNIT[name]}]")
        print(f"  예: centroid_dx = +0.10 이면 {-0.10 / g:+.1f} {UNIT[name]}")
    print("\n면내 축은 centroid_dx 를 크게 움직이고, 면외 축은 area 를 바꾸되 centroid_dx 는 덜 건드린다.")

    if args.save is not None:
        out = Path(args.save) if args.save else ep / "axis_gains.json"
        gains.save(out)
        ca, sa = gains.centering_axis(), gains.search_axis()
        print(f"\n저장 → {out}")
        print(f"  정렬 축 {ca or '없음 (이득이 문턱 아래)'} · 찾기 축 {sa or '없음'}")
        print(f"  규칙 제어기: --condition rule --gains {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
