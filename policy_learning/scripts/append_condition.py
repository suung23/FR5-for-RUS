#!/usr/bin/env python3
"""이미 도는 세션의 계획에 **조건을 하나 덧붙인다.**

    python3 scripts/append_condition.py runs/main_rule policy            # 모든 자세에
    python3 scripts/append_condition.py runs/main_rule policy --poses 6,7,8

왜 새 세션을 만들지 않는가
--------------------------
자세 자동 복귀(`--return-to-pose`)는 **같은 세션 안에서 `order_in_pose ≥ 2`** 일 때만 걸린다
(`run_experiment.pose_arg`). 새 세션을 만들면 모든 자세가 "새 자세" 로 취급돼 조작자가 8 개를
손으로 다시 잡아야 하고, 그 순간 조건 차이와 자세 차이가 섞인다 — 2026-09-14 rule_v2 에서
프로브가 8 회에 59 mm 밀려 유지 성적이 81 % → 0 % 로 떨어진 그 함정이다.

같은 세션에 붙이면 `pose_start` 가 이미 있으므로 로봇이 스스로 기준 자리로 돌아간다.

⚠️ **덧붙인 조건은 그 자세에서 가장 늦게 돈다.** 이미 끝난 자세(여기서는 1–5)에서는
조건 순서 무작위화가 깨지고 시간 순서가 교란 요인으로 남는다 — 젤이 마르고 팬텀이 샌다.
피할 수 없다면 **결과와 함께 보고**해야 한다. 아직 안 돈 자세(6–8)에는 다른 조건과 함께
섞이므로 이 문제가 없다.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session")
    p.add_argument("condition")
    p.add_argument("--poses", default=None, help="쉼표 목록 (기본: 계획에 있는 모든 자세)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    path = Path(args.session) / "session.json"
    s = json.loads(path.read_text(encoding="utf-8"))
    plan = s["plan"]
    poses = ([int(x) for x in args.poses.split(",")] if args.poses
             else sorted({e["pose"] for e in plan}))
    have = {(e["pose"], e["condition"]) for e in plan}
    nxt = max(e["index"] for e in plan) + 1
    added = []
    for pose in poses:
        if (pose, args.condition) in have:
            print(f"  자세 {pose}: 이미 있음 — 건너뜀")
            continue
        order = max([e["order_in_pose"] for e in plan if e["pose"] == pose] or [0]) + 1
        e = {"index": nxt, "pose": pose, "order_in_pose": order,
             "condition": args.condition, "status": "pending",
             "appended": True}
        plan.append(e)
        added.append(e)
        nxt += 1
    plan.sort(key=lambda e: (e["pose"], e["order_in_pose"]))

    pend = [e for e in plan if e["status"] == "pending"]
    print(f"\n덧붙임 {len(added)} · 계획 전체 {len(plan)} · 남은 에피소드 {len(pend)}")
    for e in added:
        ret = "자동 복귀 ○" if e["order_in_pose"] > 1 and str(e["pose"]) in s.get("pose_start", {}) else "⚠ 손으로 잡아야 함"
        print(f"  자세 {e['pose']} · 순서 {e['order_in_pose']} · {e['condition']}   {ret}")
    if args.dry_run:
        print("\n(--dry-run: 저장하지 않았습니다)")
        return 0
    if args.condition not in s.get("conditions", []):
        s.setdefault("conditions", []).append(args.condition)
    shutil.copy(path, path.with_suffix(".json.bak"))
    path.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {path}  (원본 {path.with_suffix('.json.bak').name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
