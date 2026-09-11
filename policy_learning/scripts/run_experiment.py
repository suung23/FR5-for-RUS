#!/usr/bin/env python3
"""실험 세션 드라이버 — 에피소드를 계획하고, 순서를 섞고, 하나씩 돌린다.

    # ① 파일럿: 시작 자세 6 개 × 조건 3 개 = 18 에피소드
    python3 scripts/run_experiment.py runs/qres2_ep25.pt --out runs/pilot \\
            --poses 6 --conditions hold,placebo,policy --duration 90

    # ② 본 시험: 자세 25 개 × 조건 4 개 = 100 에피소드
    python3 scripts/run_experiment.py runs/qres2_ep25.pt --out runs/main \\
            --poses 25 --conditions hold,placebo,policy,expert --duration 90

계획을 먼저 파일로 쓰고(``session.json``) 그 순서대로 돈다. 중간에 끊겨도 같은 ``--out`` 으로
다시 부르면 **남은 것만** 이어서 한다 — 5 시간짜리 세션을 한 번에 끝낼 수 있다고 가정하지 않는다.

왜 순서를 섞는가
---------------
자세 안에서 조건 순서를 섞는다. 안 섞으면 조건 효과와 **시간 효과**가 붙는다 — 풍선은 계속
변형되고 젤은 마르고 조작자는 지친다. 같은 자세에서 hold 를 늘 먼저 하면 hold 가 늘 가장 좋은
젤 상태를 받는다.

자세 순서는 섞지 않는다. 조작자가 자세를 물리적으로 다시 잡아야 하므로 자세를 오가면 그 시간이
에피소드보다 길어진다.

가림 (--blind)
-------------
콘솔에 조건을 인쇄하지 않는다. 조작자는 접근만 하고 손을 놓으므로 조건을 알 필요가 없고,
알면 hold 에서 무의식적으로 더 좋은 자세를 잡는다. ``expert`` 는 사람이 지령해야 하므로 그
에피소드만 조건을 알린다. 진짜 조건은 ``session.json`` 에 남는다.

산출물
------
``<out>/session.json`` 계획과 진행 · ``<out>/ep0001_*/`` 에피소드별 기록 ·
``<out>/summary.csv`` 판정 한 줄씩. 파일럿의 목적은 이 요약으로 임계와 p_B 를 정하는 것이다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import _common  # noqa: F401  — sys.path 부트스트랩 (설치 없이 실행)

from rus_policy.episode import CONDITIONS


def build_plan(poses: int, conditions: list[str], repeats: int, seed: int) -> list[dict]:
    """자세별로 조건 순서를 섞어 에피소드 목록을 만든다."""
    rng = random.Random(seed)
    plan = []
    for pose in range(1, poses + 1):
        block = [c for c in conditions for _ in range(repeats)]
        rng.shuffle(block)
        for order, cond in enumerate(block, 1):
            plan.append({"index": len(plan) + 1, "pose": pose, "order_in_pose": order,
                         "condition": cond, "status": "pending"})
    return plan


def load_or_create(out: Path, args) -> dict:
    path = out / "session.json"
    if path.is_file():
        s = json.loads(path.read_text(encoding="utf-8"))
        print(f"이어서 진행: {path}  ({sum(e['status'] != 'pending' for e in s['plan'])}"
              f"/{len(s['plan'])} 완료)")
        return s
    out.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = set(conditions) - set(CONDITIONS)
    if bad:
        raise SystemExit(f"모르는 조건: {sorted(bad)}  (가능: {list(CONDITIONS)})")
    s = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.checkpoint, "duration_s": args.duration, "seed": args.seed,
        "conditions": conditions, "poses": args.poses, "repeats": args.repeats,
        "plan": build_plan(args.poses, conditions, args.repeats, args.seed),
    }
    path.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    return s


def save(out: Path, session: dict) -> None:
    (out / "session.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")


def append_summary(out: Path, row: dict) -> None:
    path = out / "summary.csv"
    new = not path.is_file()
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def run_episode(args, ep: dict, ep_dir: Path) -> dict:
    cmd = [sys.executable, str(Path(__file__).with_name("run_policy.py")), args.checkpoint,
           "--condition", ep["condition"], "--duration", str(args.duration),
           "--out", str(ep_dir), "--axes", args.axes,
           # 게이트는 **막는 것이 아니라 재는 것**이다. run_policy 는 on/off 와 무관하게
           # gate_met_at_start 를 기록하므로(§542), 기본을 off 로 두어도 "시작 시점에
           # 조건이 맞았는가" 는 그대로 남는다 — 계획서 §3 이 요구하는 것은 그 기록이다.
           #
           # on 으로 두었더니 팬텀 위에서 한 에피소드도 열리지 않았다 (2026-09-11:
           # 면적비 0.037 인데 문턱이 0.02). 문턱이 이 팬텀에 안 맞으면 **전량 폐기**가
           # 되고 데이터가 한 줄도 안 남는다. 파일럿의 목적은 문턱을 고르는 것인데,
           # 그 문턱이 파일럿을 막으면 고를 근거를 만들 수 없다.
           "--start-gate", args.start_gate,
           "--gate-area-max", str(args.gate_area_max),
           "--gate-quality-min", str(args.gate_quality_min),
           "--success-area-min", str(args.success_area_min),
           "--success-component-min", str(args.success_component_min),
           "--success-centroid-max", str(args.success_centroid_max),
           "--placebo-mode", args.placebo_mode,
           "--placebo-seed", str(args.seed * 1000 + ep["index"]),
           "--success-hold-s", str(args.success_hold_s),
           "--placebo-delay-s", str(args.placebo_delay_s),
           "--max-deg-s", str(args.max_deg_s), "--max-mm-s", str(args.max_mm_s),
           "--start-force", str(args.start_force)]
    if args.gamma is not None:
        cmd += ["--gamma", str(args.gamma)]
    if args.execute and ep["condition"] in ("policy", "placebo"):
        cmd.append("--execute")          # hold·expert 는 애초에 지령하지 않는다
    cmd += args.extra
    print("  $ " + " ".join(cmd))
    subprocess.run(cmd, check=False)
    meta = ep_dir / "meta.json"
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True, help="세션 폴더")
    p.add_argument("--poses", type=int, default=6)
    p.add_argument("--conditions", default="hold,placebo,policy")
    p.add_argument("--repeats", type=int, default=1, help="자세×조건 반복 횟수")
    p.add_argument("--duration", type=float, default=90.0)
    p.add_argument("--seed", type=int, default=0, help="조건 순서 섞기 — 세션을 재현한다")
    p.add_argument("--blind", action="store_true", help="조건을 콘솔에 인쇄하지 않는다 (expert 제외)")
    p.add_argument("--execute", action="store_true", help="실제로 지령한다 (없으면 전 조건 DRY-RUN)")
    p.add_argument("--axes", default="rot")
    # 안전 상한은 러너가 아니라 여기서 정한다 — 에피소드마다 다르면 조건 비교가 깨진다.
    p.add_argument("--max-deg-s", type=float, default=3.0, help="회전 지령 상한 [°/s]")
    p.add_argument("--max-mm-s", type=float, default=3.0, help="병진 지령 상한 [mm/s]")
    p.add_argument("--start-force", type=float, default=1.0, help="이 접촉력[N] 미만이면 지령하지 않는다")
    p.add_argument("--start-gate", default="off", choices=["off", "on"],
                   help="on 이면 시작 조건을 못 맞춘 에피소드를 폐기한다. off 여도 충족 "
                        "여부는 그대로 기록된다 — 분석에서 사후에 걸러낼 수 있다")
    p.add_argument("--gate-area-max", type=float, default=0.02)
    p.add_argument("--gate-quality-min", type=float, default=0.6)
    p.add_argument("--success-area-min", type=float, default=0.08)
    p.add_argument("--success-component-min", type=float, default=0.80)
    p.add_argument("--success-centroid-max", type=float, default=0.30)
    p.add_argument("--placebo-mode", default="random-direction",
                   choices=["random-direction", "stale-obs"])
    p.add_argument("--success-hold-s", type=float, default=3.0)
    p.add_argument("--placebo-delay-s", type=float, default=30.0)
    p.add_argument("--gamma", type=float, default=None,
                   help="방향 관성 — run_policy 로 넘긴다. 없으면 체크포인트 값")
    p.add_argument("extra", nargs="*", help="run_policy.py 에 그대로 넘길 인자")
    # 선언 안 된 옵션은 run_policy 로 흘려보낸다. 예전에는 parse_args 가 그것을 거부했고,
    # `--` 를 앞에 붙여도 위치 인자 extra 가 앞에서 이미 빈 값으로 소비돼 역시 거부됐다 —
    # 러너 전용 옵션을 드라이버 너머로 넘기는 통로가 사실상 없었다.
    args, unknown = p.parse_known_args()
    args.extra = list(args.extra) + [a for a in unknown if a != "--"]

    out = Path(args.out)
    session = load_or_create(out, args)
    plan = session["plan"]
    todo = [e for e in plan if e["status"] == "pending"]
    if not todo:
        print("계획한 에피소드를 모두 마쳤습니다.")
        return 0
    print(f"\n남은 에피소드 {len(todo)} / {len(plan)}   "
          f"예상 {len(todo) * (args.duration + 120) / 3600:.1f} 시간 "
          f"(에피소드 {args.duration:.0f} s + 복귀·판정 2 분)")
    if not args.execute:
        print("⚠️  DRY-RUN — 지령이 나가지 않습니다. 실제 실험은 --execute 를 붙이십시오.")

    for ep in todo:
        label = ep["condition"] if (not args.blind or ep["condition"] == "expert") else "····"
        print(f"\n{'=' * 68}\n에피소드 {ep['index']}/{len(plan)}   "
              f"자세 {ep['pose']}   조건 {label}\n{'=' * 68}")
        print("  텔레옵으로 시작 자세를 잡고, 접촉이 생기면 손을 놓으십시오.")
        if ep["condition"] == "expert":
            print("  ⚠️ 이 에피소드는 **조작자가 계속 지령**합니다 (상한선 조건).")
        try:
            ans = input("  준비되면 Enter · s=건너뜀 · q=중단 > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n중단합니다.")
            break
        if ans == "q":
            break
        if ans == "s":
            ep["status"] = "skipped"
            save(out, session)
            continue

        ep_dir = out / f"ep{ep['index']:04d}_pose{ep['pose']:02d}_{ep['condition']}"
        meta = run_episode(args, ep, ep_dir)
        verdict = meta.get("verdict") or {}
        started = bool(meta.get("episode_started"))
        gate_met = meta.get("gate_met_at_start")
        ep["status"] = "done" if started else "gate_reject"
        ep["gate_met_at_start"] = gate_met
        ep["verdict"] = verdict
        save(out, session)
        append_summary(out, {
            "index": ep["index"], "pose": ep["pose"], "condition": ep["condition"],
            "started": int(started), "success": int(bool(verdict.get("success"))),
            "gate_met": "" if gate_met is None else int(bool(gate_met)),
            "best_run_s": verdict.get("best_run_s", ""), "q_mean": verdict.get("q_mean", ""),
            "q_final_10s": verdict.get("q_final_10s", ""), "area_max": verdict.get("area_max", ""),
            "good_fraction": verdict.get("good_fraction", ""), "dir": ep_dir.name,
        })
        if not started:
            print("  시작 조건 미충족 — 폐기로 기록합니다 (--start-gate on).")
        else:
            gate_note = "" if gate_met is None else (
                "  시작조건 충족" if gate_met else "  시작조건 미충족(기록만)")
            print(f"  성공={bool(verdict.get('success'))}  "
                  f"최장 연속={verdict.get('best_run_s', float('nan')):.1f} s{gate_note}")

    done = sum(e["status"] == "done" for e in plan)
    rej = sum(e["status"] == "gate_reject" for e in plan)
    met = sum(1 for e in plan if e.get("gate_met_at_start"))
    print(f"\n완료 {done} · 폐기 {rej} · 남음 {sum(e['status'] == 'pending' for e in plan)}")
    if args.start_gate == "off":
        print(f"시작 조건을 충족한 채 시작한 에피소드 {met}/{done} — "
              f"분석에서 사후에 걸러낼 수 있습니다 (summary.csv 의 gate_met).")
    print(f"요약: {out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
