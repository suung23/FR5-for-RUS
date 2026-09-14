#!/usr/bin/env python3
"""기록된 에피소드를 **지금의 판정 문턱으로 다시 판정**하고 요약을 덮어쓴다.

판정 문턱은 사전 등록 계획이 "예비 촬영이 정한다" 고 남겨 둔 자리였고, 2026-09-14 에
면적비 0.08 → 0.04 로 정해졌다 (rus_policy.episode.Thresholds 주석에 근거). 그 전에 돈
에피소드의 summary.csv·meta.json·session.json 에는 옛 판정이 박혀 있다. 섞이면 어느 줄이
어느 문턱의 결과인지 알 수 없으므로 한 번에 다시 계산한다.

원본은 ``*.bak_<문턱>`` 으로 남긴다 — 덮어쓰기 전의 값을 되짚을 수 있어야 한다.

    python3 policy_learning/scripts/rejudge_runs.py --dry-run
    python3 policy_learning/scripts/rejudge_runs.py
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rus_policy.episode import Thresholds, judge     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "policy_learning" / "runs"


def verdict_of(states: Path, thr: Thresholds) -> dict:
    z = np.load(states)
    return judge(list(z["t"]), z["state"], thr)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=str(RUNS), help="runs 디렉터리")
    p.add_argument("--dry-run", action="store_true", help="무엇이 바뀌는지만 보인다")
    args = p.parse_args()

    thr = Thresholds()
    tag = f"bak_area{thr.area_min:g}".replace(".", "p")
    root = Path(args.runs)
    changed = 0

    for meta in sorted(root.glob("**/meta.json")):
        st = meta.parent / "states.npz"
        if not st.is_file():
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        old = bool((m.get("verdict") or {}).get("success"))
        v = verdict_of(st, thr)
        new = bool(v.get("success"))
        if old != new or (m.get("thresholds") or {}).get("area_min") != thr.area_min:
            changed += 1
            print(f"  {meta.parent.relative_to(root)}  성공 {old} → {new}"
                  f"  (최장 연속 {v.get('best_run_s', 0):.1f} s)")
            if not args.dry_run:
                shutil.copy(meta, meta.with_suffix(f".json.{tag}"))
                m["verdict"] = v
                m["thresholds"] = {"area_min": thr.area_min, "component_min": thr.component_min,
                                   "centroid_max": thr.centroid_max, "hold_s": thr.hold_s}
                m["rejudged"] = "2026-09-14 판정 문턱 개정 (면적비 0.08 → 0.04)"
                meta.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    # summary.csv 는 세션 폴더에 있고 에피소드 폴더 이름(dir 열)으로 이어진다.
    for summary in sorted(root.glob("*/summary.csv")):
        rows = list(csv.DictReader(open(summary, encoding="utf-8")))
        if not rows:
            continue
        touched = False
        for r in rows:
            st = summary.parent / r.get("dir", "") / "states.npz"
            if not st.is_file():
                continue
            v = verdict_of(st, thr)
            for key, val in (("success", int(bool(v.get("success")))),
                             ("best_run_s", v.get("best_run_s", "")),
                             ("good_fraction", v.get("good_fraction", "")),
                             ("area_max", v.get("area_max", "")),
                             ("q_mean", v.get("q_mean", "")),
                             ("q_final_10s", v.get("q_final_10s", ""))):
                if key in r and str(r[key]) != str(val):
                    r[key], touched = val, True
        if touched:
            print(f"  {summary.relative_to(root)} 갱신")
            if not args.dry_run:
                shutil.copy(summary, summary.with_suffix(f".csv.{tag}"))
                with open(summary, "w", newline="", encoding="utf-8") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                    w.writeheader(); w.writerows(rows)

    # session.json 의 에피소드별 verdict 도 같은 값을 들고 있다.
    for sess in sorted(root.glob("*/session.json")):
        s = json.loads(sess.read_text(encoding="utf-8"))
        touched = False
        for ep in s.get("plan", []):
            d = sess.parent / f"ep{ep['index']:04d}_pose{ep['pose']:02d}_{ep['condition']}"
            st = d / "states.npz"
            if not st.is_file():
                continue
            v = verdict_of(st, thr)
            if bool((ep.get("verdict") or {}).get("success")) != bool(v.get("success")):
                touched = True
            ep["verdict"] = v
        if touched:
            print(f"  {sess.relative_to(root)} 갱신")
            if not args.dry_run:
                shutil.copy(sess, sess.with_suffix(f".json.{tag}"))
                sess.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'(모의) ' if args.dry_run else ''}판정이 바뀐 에피소드 {changed} 개 "
          f"· 문턱 면적비 {thr.area_min} · 백업 접미사 .{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
