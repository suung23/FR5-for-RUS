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
from rus_policy.episode import Thresholds, judge, judge_two_phase     # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "policy_learning" / "runs"


def _force_series(episode: Path) -> tuple[list[float], list[float]]:
    """decisions.csv 의 (t, F_n). 없으면 빈 것 — 힘 항은 NaN 이 된다."""
    path = episode / "decisions.csv"
    if not path.is_file():
        return [], []
    t: list[float] = []
    f: list[float] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                t.append(float(row["t"]))
                f.append(float(row["F_n"]))
            except (KeyError, TypeError, ValueError):
                continue
    return t, f


def verdict_of(states: Path, thr: Thresholds) -> dict:
    """실행 중과 **같은 판정**을 낸다.

    🔁 2026-09-14 저녁: 여기가 ``judge`` 만 불렀다. 실행 중에는 run_policy 가
    ``judge`` 뒤에 ``judge_two_phase`` 를 겹쳐 기록한다 (찾기 / 유지 / 그때의 힘).
    그래서 재판정한 meta.json 에서는 ``hold_q_pass_fraction`` 같은 유지 지표가 통째로
    사라졌고, **``q_view_min`` 이 기록의 어디에도 반영되지 않았다** — 문턱을 옮겨도
    바뀌는 것이 없으니 옮긴 값이 맞는지 확인할 방법도 없었다.
    """
    z = np.load(states)
    t, state = list(z["t"]), z["state"]
    out = judge(t, state, thr)
    ft, fn = _force_series(states.parent)
    out.update(judge_two_phase(t, state, thr,
                               force_t=ft or None, force_n=fn or None))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default=str(RUNS), help="runs 디렉터리")
    p.add_argument("--dry-run", action="store_true", help="무엇이 바뀌는지만 보인다")
    # 판정 코드 자체가 바뀌었을 때 쓴다 (문턱은 그대로인데 나오는 항목이 늘어난 경우).
    # 문턱만 비교하는 아래 조건은 그것을 알아채지 못한다.
    p.add_argument("--force", action="store_true", help="문턱이 같아도 전부 다시 쓴다")
    args = p.parse_args()

    thr = Thresholds()
    # 🔁 2026-09-14 저녁: 태그가 area_min 만으로 만들어져 있었다. 그러면 문턱을 두 번째로
    # 고칠 때(Q_seg 눈금 개편) 같은 이름이 나오고, **0.08 → 0.04 때 남긴 원본 백업을
    # 덮어쓴다.** 백업의 값어치는 되짚을 수 있다는 것뿐이므로 그것이 없어지면 백업이 아니다.
    # 판정에 들어가는 문턱 전부를 태그에 넣고, 아래에서 기존 백업은 절대 건드리지 않는다.
    tag = (f"bak_area{thr.area_min:g}_q{thr.q_view_min:g}").replace(".", "p")
    root = Path(args.runs)
    changed = 0

    def backup(src: Path, suffix: str) -> None:
        """원본을 남긴다. **이미 있으면 그대로 둔다** — 백업은 덮어쓰는 것이 아니다."""
        dst = src.with_suffix(f"{suffix}.{tag}")
        if not dst.exists():
            shutil.copy(src, dst)

    for meta in sorted(root.glob("**/meta.json")):
        st = meta.parent / "states.npz"
        if not st.is_file():
            continue
        m = json.loads(meta.read_text(encoding="utf-8"))
        old = bool((m.get("verdict") or {}).get("success"))
        v = verdict_of(st, thr)
        new = bool(v.get("success"))
        stored = m.get("thresholds") or {}
        if (args.force or old != new or stored.get("area_min") != thr.area_min
                or stored.get("q_view_min") != thr.q_view_min
                or "hold_q_pass_fraction" not in v.keys() - m.keys()
                and "hold_q_pass_fraction" not in (m.get("verdict") or {})):
            changed += 1
            print(f"  {meta.parent.relative_to(root)}  성공 {old} → {new}"
                  f"  (최장 연속 {v.get('best_run_s', 0):.1f} s)")
            if not args.dry_run:
                backup(meta, ".json")
                m["verdict"] = v
                m["thresholds"] = {"area_min": thr.area_min, "component_min": thr.component_min,
                                   "centroid_max": thr.centroid_max, "hold_s": thr.hold_s,
                                   "q_view_min": thr.q_view_min}
                m["rejudged"] = ("2026-09-14 판정 문턱 개정 (면적비 0.08 → 0.04) · "
                                 "저녁: Q_seg 눈금 개편에 맞춰 q_view_min 0.97 → 0.936")
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
            # 🔁 2026-09-14 밤: 목록에 q_seg·유지 지표가 없었다. 그래서 **Q_seg 눈금을
            # 두 번 바꾸는 동안 요약표의 q_seg 열은 옛 값 그대로였다** — 재판정했다는
            # 말과 달리 표에 남는 숫자는 옛 자였다. 그리고 없는 열은 만들어 준다:
            # 유지 지표는 나중에 생긴 열이라 옛 요약표에는 칸 자체가 없다.
            for key, val in (("success", int(bool(v.get("success")))),
                             ("best_run_s", v.get("best_run_s", "")),
                             ("good_fraction", v.get("good_fraction", "")),
                             ("area_max", v.get("area_max", "")),
                             ("q_seg", v.get("q_view_mean", "")),
                             ("q_seg_final_10s", v.get("q_view_final_10s", "")),
                             ("q_mean", v.get("q_mean", "")),
                             ("q_final_10s", v.get("q_final_10s", "")),
                             ("hold_q_mean", v.get("hold_q_mean", "")),
                             ("hold_q_pass_fraction", v.get("hold_q_pass_fraction", "")),
                             ("hold_good_fraction", v.get("hold_good_fraction", ""))):
                if key not in r or str(r[key]) != str(val):
                    r[key], touched = val, True
        if touched:
            print(f"  {summary.relative_to(root)} 갱신")
            if not args.dry_run:
                backup(summary, ".csv")
                with open(summary, "w", newline="", encoding="utf-8") as fh:
                    fields = list(rows[0])
                    for extra in rows[1:]:
                        fields += [k for k in extra if k not in fields]
                    w = csv.DictWriter(fh, fieldnames=fields, restval="")
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
                backup(sess, ".json")
                sess.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'(모의) ' if args.dry_run else ''}판정이 바뀐 에피소드 {changed} 개 "
          f"· 문턱 면적비 {thr.area_min} · 백업 접미사 .{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
