#!/usr/bin/env python3
"""세션 시작 전 **눈금이 그대로인가**를 확인한다.

    python3 scripts/preflight_freeze.py                                  # 정적 확인만
    python3 scripts/preflight_freeze.py --reference-episode runs/ref/ep0001_pose01_hold
    python3 scripts/preflight_freeze.py --attest --out runs/main_sono/preflight.json

왜 필요한가
-----------
두 실행(오늘의 규칙 기반, 나중의 Sonologger 기반)을 **같은 자로** 비교하려면 자가 그대로여야
한다. 그런데 자를 이루는 것들이 코드·설정·물리 세계에 흩어져 있다:

* 지각 모델이 바뀌면 면적·중심이 바뀐다           → U-Net 체크포인트 id
* ``AREA_FULL`` 이 바뀌면 q_view 가 통째로 다른 값 → view_quality.py
* 판정 문턱이 바뀌면 성공률이 바뀐다               → thresholds
* **팬텀 충전이 바뀌면 AREA_FULL 이 무효가 된다**  → 면적 분포로 잡는다

마지막 것이 가장 위험하다. 코드 어디에도 안 적히고, 조작자가 물풍선을 다시 채우면 조용히
바뀌며, 그 순간 면적비가 AREA_FULL 을 넘어 q_view 의 면적 항이 포화한다 (2026-09-14 실기에서
실제로 밟았다: 기준 실행의 면적 p90 이 0.1174 로 AREA_FULL 0.12 바로 아래였다).

그래서 충전 상태를 **숫자로** 묶는다 — 기준 실행에서 방광이 보인 프레임의 면적비 분포를
지문으로 저장하고, 새 세션의 참조 에피소드와 대조한다. 말로 하는 다짐은 그 아래에 둔다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# venv 로 스스로 갈아탄다. 시스템 python 으로 부르면 torch 가 없어 U-Net 확인에서 죽는데,
# **세션 직전에 부르는 도구가 죽으면 조작자는 확인을 건너뛴다** (2026-09-14 실기).
_VENV = ROOT.parent / "Unet_seg" / ".venv" / "bin" / "python3"
if "--no-unet" not in sys.argv and _VENV.is_file():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        import os
        os.execv(str(_VENV), [str(_VENV), str(Path(__file__).resolve())] + sys.argv[1:])

OK, BAD = "  ✔", "  ✘"


def check(results: list, name: str, expect, got, ok: bool | None = None) -> None:
    ok = (expect == got) if ok is None else ok
    results.append((ok, name, expect, got))
    print(f"{OK if ok else BAD} {name}: {got}" + ("" if ok else f"   ← 기대 {expect}"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default=str(ROOT / "configs" / "experiment_freeze.json"))
    p.add_argument("--reference-episode", default=None,
                   help="에피소드 디렉터리 하나 (states.npz 를 읽는다)")
    p.add_argument("--recent", type=int, default=6,
                   help="세션에서 **가장 최근** 이 개수의 에피소드만 본다 (기본 6). "
                        "전체를 보면 변화 이전 에피소드가 통과시킨다")
    p.add_argument("--reference-session", default=None,
                   help="세션 폴더 — 그 안의 **모든** 에피소드에서 도달한 최대 면적을 쓴다. "
                        "에피소드를 손으로 고르면 방광을 못 찾은 것을 집기 쉽다 (2026-09-14 실기)")
    p.add_argument("--attest", action="store_true", help="조작자 확인 문항을 묻는다")
    p.add_argument("--out", default=None, help="확인 결과를 남길 JSON")
    p.add_argument("--no-unet", action="store_true", help="U-Net 적재를 건너뛴다 (torch 없는 환경)")
    args = p.parse_args()

    man = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    res: list = []
    print(f"고정 목록: {args.manifest}  (기준 실행 {man.get('reference_run')})\n")

    # --- 코드·설정 (정적) -------------------------------------------------
    from rus_policy.config import load_config
    from rus_policy.view_quality import AREA_FULL
    from _common import DEFAULT_CONFIG

    cfg = load_config(DEFAULT_CONFIG)
    check(res, "unet_checkpoint", man["unet"]["checkpoint"], str(cfg.paths.unet_checkpoint))
    check(res, "unet_config", man["unet"]["config"], str(cfg.paths.unet_config))
    check(res, "AREA_FULL", man["view_quality"]["AREA_FULL"], AREA_FULL)
    sha = hashlib.sha1((ROOT / "rus_policy" / "view_quality.py").read_bytes()).hexdigest()[:12]
    check(res, "view_quality.py sha1", man["view_quality"]["source_sha1"], sha)

    if not args.no_unet:
        from rus_policy.perception import build_backend
        cid = build_backend(cfg).checkpoint_id
        check(res, "unet checkpoint_id", man["unet"]["checkpoint_id"], cid)

    # 판정 문턱은 run_experiment 의 기본값으로 산다 — 그 기본값이 곧 프로토콜이다.
    import run_experiment as RX
    d = {a.dest: a.default for a in RX.build_parser()._actions} if hasattr(RX, "build_parser") else None
    if d is None:                       # build_parser 가 없으면 소스에서 읽는다
        import re
        src = (ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
        d = {m.group(1).replace("-", "_"): float(m.group(2))
             for m in re.finditer(r'"--([a-z\-]+)",\s*type=float,\s*default=([0-9.]+)', src)}
    for key, dest in [("q_view_min", "success_q_view_min"), ("area_min", "success_area_min"),
                      ("component_min", "success_component_min"), ("centroid_max", "success_centroid_max"),
                      ("hold_s", "success_hold_s")]:
        if dest in d:
            check(res, f"thresholds.{key}", man["thresholds"][key], d[dest])
    import re as _re
    _rp = (ROOT / "scripts" / "run_policy.py").read_text(encoding="utf-8")
    _m = _re.search(r'"--rule-good-enough",\s*type=float,\s*default=([0-9.]+)', _rp)
    if _m:
        # 위치 B 부터는 기본값이 아니라 **명령줄로 준 값**이 기준이다 (--rule-good-enough).
        # 여기서는 고정 목록의 값이 실행 인자 목록에 들어 있는지를 본다.
        want = str(man["rule"]["q_good_enough"])
        args_ = man.get("runner", {}).get("rule_args", [])
        ok = ("--rule-good-enough" in args_
              and args_[args_.index("--rule-good-enough") + 1].rstrip("0").rstrip(".") == want.rstrip("0").rstrip("."))
        check(res, "rule.q_good_enough (실행 인자)", want,
              (args_[args_.index("--rule-good-enough") + 1] if "--rule-good-enough" in args_ else f"기본값 {_m.group(1)}"),
              ok=ok)

    # --- 규칙 제어기의 입력 파일 -----------------------------------------
    # rule 조건은 이 두 파일이 정의한다. 바뀌면 "같은 규칙 기반" 이 아니다.
    for key, ent in man.get("rule_inputs", {}).items():
        if key.startswith("_"):
            continue
        f = ROOT / ent["path"]
        got = hashlib.sha1(f.read_bytes()).hexdigest()[:12] if f.is_file() else "(파일 없음)"
        check(res, f"rule_inputs.{key}", ent["sha1"], got)

    # --- 팬텀 충전 (물리) -------------------------------------------------
    fill = None
    src = None
    if args.reference_session or args.reference_episode:
        if args.reference_session:
            # 도달 **가능한** 최대 면적이므로 세션 전체에서 가장 큰 값이 맞다. 한 에피소드만
            # 보면 그 에피소드가 방광을 찾았는지에 좌우된다 — 못 찾은 것을 집으면 충전이
            # 정상인데도 탈락한다.
            # **최근 것만 본다.** 세션 전체에서 최대를 고르면 변화 이전의 에피소드가
            # 통과시켜 버린다 — 2026-09-14 에 실제로 그랬다: 19 시에 팬텀이 절반으로
            # 줄었는데 15 시의 ep0001(0.1415)을 집어 "정상" 이라고 답했다.
            eps = sorted(Path(args.reference_session).glob("ep*/states.npz"),
                         key=lambda f: f.parent.stat().st_mtime)[-args.recent:]
            vs, src = [], "(없음)"
            best = -1.0
            for f in eps:
                a = np.asarray(np.load(f)["state"])[:, 4]
                a = a[a > 0.005]
                if a.size >= 50:
                    vs.append(a)
                    if float(a.max()) > best:
                        best, src = float(a.max()), f.parent.name
            v = np.concatenate(vs) if vs else np.array([])
            print(f"  ‥ 팬텀 충전: 세션 {Path(args.reference_session).name} 의 **최근** "
                  f"{len(vs)} 개 에피소드, 최대는 {src}")
        else:
            v = np.asarray(np.load(Path(args.reference_episode) / "states.npz")["state"])[:, 4]
            v = v[v > 0.005]
        if v.size < 50:
            check(res, "팬텀 충전 지문", "방광이 보인 프레임 ≥ 50", f"{v.size} 장", ok=False)
        else:
            mx = float(v.max())
            lo, hi = man["phantom_fill"]["accept_range"]
            fill = {"max_area": round(mx, 4), "accept_range": [lo, hi], "n": int(v.size),
                    "reference": man["phantom_fill"]["episode_max_area"]["top5_median"]}
            check(res, "팬텀 충전 — 도달 가능한 최대 면적", f"{lo:.4f}–{hi:.4f}", f"{mx:.4f}",
                  ok=lo <= mx <= hi)
    else:
        print("  ‥ 팬텀 충전: 참조가 없어 확인하지 않았다 (--reference-session 권장)")

    # --- 조작자 확인 ------------------------------------------------------
    attest = {}
    if args.attest:
        print()
        for q in man["operator_attest"]:
            a = input(f"  {q}?  [y/N] ").strip().lower()
            attest[q] = (a == "y")
            if a != "y":
                res.append((False, f"확인: {q}", "y", a or "(없음)"))

    bad = [r for r in res if not r[0]]
    print(f"\n{'=' * 60}")
    if bad:
        print(f"✘ {len(bad)} 항목이 기준과 다릅니다 — 이 세션은 기준 실행과 **같은 자로 볼 수 없습니다**:")
        for _, name, expect, got in bad:
            print(f"    {name}: {got}  (기준 {expect})")
    else:
        print("✔ 모든 고정 항목이 기준과 같습니다.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"manifest": args.manifest, "passed": not bad,
                                   "checks": [{"ok": o, "name": n, "expect": e, "got": g} for o, n, e, g in res],
                                   "phantom_fill": fill, "attest": attest},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"기록: {out}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
