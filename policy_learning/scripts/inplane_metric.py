#!/usr/bin/env python3
"""에피소드의 **면내 방향** 지표 — 중심 편차와 정렬축 지령이 같은 이야기를 하는가.

    python3 scripts/inplane_metric.py runs/main_rule
    python3 scripts/inplane_metric.py runs/main_sono --axis thy

무엇을 재는가
-------------
`centroid_dx` (영상에서 방광이 좌우로 얼마나 치우쳤나) 와 **정렬축 지령**의 상관이다.
되돌리는 방향으로 지령하면 둘은 반대 부호여야 하므로, 읽기 쉽게 **부호를 뒤집어**
"올바름 = +1" 로 보고한다 (규칙 기반 제어기가 정의상 +1.00 에 가깝게 나온다).

성공률만으로는 무엇이 좋아졌는지 알 수 없다 — 면외 훑기가 우연히 방광을 지나가도
성공은 오른다. 이 지표는 **학습이 겨냥한 바로 그 축**을 직접 본다.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

AXIS_CMD = {"thx": "cmd_wx", "thy": "cmd_wy", "thz": "cmd_wz", "x": "cmd_vx", "y": "cmd_vy"}
CENTROID_DX = 2


def episode(dirp: Path, axis: str | None) -> dict | None:
    st = dirp / "states.npz"
    dc = dirp / "decisions.csv"
    if not (st.is_file() and dc.is_file()):
        return None
    meta = json.loads((dirp / "meta.json").read_text(encoding="utf-8")) if (dirp / "meta.json").is_file() else {}
    ax = axis or meta.get("rule_align_axis") or "thy"
    col = AXIS_CMD.get(ax, "cmd_wy")
    z = np.load(st)
    t_s, dx = z["t"], np.asarray(z["state"])[:, CENTROID_DX]
    has = np.asarray(z["state"])[:, 0] > 0.5
    rows = list(csv.DictReader(dc.open(encoding="utf-8")))
    if not rows:
        return None
    t_c = np.array([float(r["t"]) for r in rows])
    cmd = np.array([float(r[col] or 0.0) for r in rows])
    on = np.array([str(r.get("enabled", "")).strip() in ("1", "True", "true") for r in rows])
    # 정렬 국면만 본다. 규칙 제어기는 '찾기' 동안 **면외 축**을 훑으므로, 에피소드 전체를
    # 뭉쳐 재면 정렬과 무관한 지령이 상관을 희석한다 (2026-09-14: 전체 +0.21 대 정렬만 +0.93).
    # rule_phase 가 없는 조건(policy)은 지령이 나간 구간 전체를 정렬로 본다.
    phase = np.array([1.0 if (r.get("rule_phase", "") or "유지") == "유지" else 0.0 for r in rows])
    t0 = float(z["t_start"]) if "t_start" in z.files and np.isfinite(z["t_start"]) else t_s[0]
    # 지령을 상태 시각으로 옮긴다 (지령은 tick, 상태는 프레임 — 레이트가 다르다)
    sel = (t_s >= t0) & has
    if sel.sum() < 30:
        return {"dir": dirp.name, "condition": meta.get("condition", "?"), "axis": ax, "n": int(sel.sum())}
    c_on = np.interp(t_s[sel], t_c, on.astype(float))
    c_ph = np.interp(t_s[sel], t_c, phase)
    c = np.interp(t_s[sel], t_c, cmd)
    d = dx[sel]
    m = (c_on > 0.5) & (c_ph > 0.5) & (np.abs(c) > 1e-6)   # 지령이 나갔고, 정렬 중인 구간만
    out = {"dir": dirp.name, "condition": meta.get("condition", "?"), "axis": ax,
           "n": int(m.sum()), "cmd_frac": round(float(m.mean()), 3)}
    if m.sum() >= 30 and np.std(d[m]) > 1e-9 and np.std(c[m]) > 1e-9:
        r = float(np.corrcoef(d[m], c[m])[0, 1])
        out["corr_raw"] = round(r, 3)
        out["inplane"] = round(-r, 3)             # 올바름 = +1
        out["sign_agree"] = round(float(np.mean(np.sign(d[m]) != np.sign(c[m]))), 3)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session")
    p.add_argument("--axis", default=None, help="정렬축 (기본: meta.json 의 rule_align_axis, 없으면 thy)")
    args = p.parse_args()
    eps = sorted(Path(args.session).glob("ep*"))
    rows = [r for r in (episode(e, args.axis) for e in eps) if r]
    print(f"{'에피소드':28s} {'조건':9s} {'축':5s} {'n':>6s} {'지령비':>6s} {'면내(+1=옳음)':>13s} {'부호일치':>8s}")
    print("-" * 82)
    for r in rows:
        print(f"{r['dir']:28s} {r['condition']:9s} {r['axis']:5s} {r['n']:6d} "
              f"{r.get('cmd_frac', float('nan')):6.2f} "
              f"{r.get('inplane', float('nan')):13.3f} {r.get('sign_agree', float('nan')):8.3f}")
    print()
    for cond in sorted({r["condition"] for r in rows}):
        v = [r["inplane"] for r in rows if r["condition"] == cond and "inplane" in r]
        if v:
            print(f"  {cond:9s} 면내 중앙 {np.median(v):+.3f}  (n={len(v)} 에피소드, 범위 {min(v):+.2f}…{max(v):+.2f})")
        else:
            print(f"  {cond:9s} 지령이 없어 계산 불가 (hold 는 정상)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
