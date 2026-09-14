#!/usr/bin/env python3
"""논문 작업에 필요한 것을 **저장소 안으로** 내보낸다.

    python3 Paper/icra_bladder/export_results.py --run policy_learning/runs/main_rule

왜 필요한가
-----------
``policy_learning/runs/`` 는 통째로 gitignore 다 (3.2 GB). 이 저장소는 100 MB 하드
리밋에 두 번 걸린 이력이 있어서 (`.gitignore` 주석) 원시 세션을 넣을 수 없다. 그런데
논문은 그 안의 숫자와 프레임 없이는 못 쓴다. **이 스크립트가 그 사이를 잇는다** —
실험 PC 에서 한 번 돌리면, 그 뒤로는 저장소만 있으면 어디서든 논문을 끝까지 쓸 수 있다.

무엇이 나가는가 (``Paper/icra_bladder/data/``)
----------------------------------------------
    results.md                 조건별·자세별 표 (그대로 논문에 옮길 수 있는 형태)
    summary/<run>.csv          원본 요약표 사본
    verdicts.json              에피소드별 판정 전체 (힘·유지·찾기 지표 포함)
    frames/<ep>/index.csv      프레임마다 t · Q_seg · 면적 · 중심 · 잘림 · 힘
    frames/<ep>/f####.png      B-mode 원본 (무손실, 256², 1 Hz)
    frames/<ep>/m####.png      같은 프레임의 U-Net 마스크 (이진, 수 KB)
    frames/roi_mask.png        부채꼴 ROI (그림에 경계를 그릴 때 쓴다)
    contact/<ep>.jpg           한 장짜리 컨택트 시트 — 눈으로 고르는 용도

마스크를 따로 내보내는 이유는 **오프라인에서 torch 없이 겹쳐 그리기 위해서**다.
``make_figure.py`` 는 numpy 와 cv2 만 있으면 돈다.

프레임을 1 Hz 로 줄인 이유
--------------------------
전량(10 Hz)은 PNG 로 430 MB 다. 1 Hz 는 43 MB 이고, 120 s 에피소드마다 120 장이라
논문 그림 하나 고르기에는 넘친다. 더 촘촘한 것이 필요하면 ``--hz`` 로 올리되,
2 Hz 가 86 MB 라는 것을 알고 올릴 것.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "policy_learning"))


def _fmt(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if v != v else f"{v:.{nd}f}"


def load_rows(run: Path) -> list[dict]:
    rows = list(csv.DictReader(open(run / "summary.csv", encoding="utf-8")))
    for r in rows:
        meta = run / r["dir"] / "meta.json"
        v = (json.loads(meta.read_text(encoding="utf-8")).get("verdict") or {}) if meta.is_file() else {}
        r["_verdict"] = v
    return rows


def results_markdown(run: Path, rows: list[dict]) -> str:
    import numpy as np

    def g(r, k):
        try:
            return float(r["_verdict"].get(k))
        except (TypeError, ValueError):
            return float("nan")

    def med(v):
        v = [x for x in v if x == x]
        return np.median(v) if v else float("nan")

    conds = sorted({r["condition"] for r in rows})
    poses = sorted({r["pose"] for r in rows}, key=int)
    out = [f"# {run.name} — 결과", "",
           f"에피소드 {len(rows)} · 자세 {len(poses)} · 조건 {', '.join(conds)}", "",
           "## 조건별", "",
           "| 조건 | 성공 | 찾음 | 찾기 [s] | 최장 연속 [s] | hold_q_mean | hold_good_frac | F_max [N] | >4.5 N [s] | >5 N [s] |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for c in conds:
        gr = [r for r in rows if r["condition"] == c]
        found = [r for r in gr if r["_verdict"].get("found")]
        out.append(
            f"| {c} | {sum(int(r['success']) for r in gr)}/{len(gr)} | {len(found)}/{len(gr)} "
            f"| {_fmt(med([g(r,'find_time_s') for r in found]), 1)} "
            f"| {_fmt(med([g(r,'best_run_s') for r in gr]), 1)} "
            f"| {_fmt(med([g(r,'hold_q_mean') for r in gr]))} "
            f"| {_fmt(med([g(r,'hold_good_fraction') for r in gr]))} "
            f"| {_fmt(max([g(r,'force_max_n') for r in gr] or [float('nan')]), 2)} "
            f"| {_fmt(max([g(r,'time_above_warn_s') for r in gr] or [float('nan')]), 1)} "
            f"| {_fmt(max([g(r,'time_above_limit_s') for r in gr] or [float('nan')]), 1)} |")

    out += ["", "## 자세별 (짝지은 비교)", "",
            "| 자세 | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for p in poses:
        cells = []
        for c in conds:
            gr = [r for r in rows if r["pose"] == p and r["condition"] == c]
            if not gr:
                cells.append("—")
                continue
            r = gr[0]
            cells.append(f"{'성공' if int(r['success']) else '실패'} {_fmt(g(r,'best_run_s'),1)} s "
                         f"· q {_fmt(g(r,'hold_q_mean'))} · {r['dir']}")
        out.append(f"| {p} | " + " | ".join(cells) + " |")

    # 짝지은 정확 이항검정 — 대조군이 있을 때만.
    if "hold" in conds:
        from math import comb
        for c in [x for x in conds if x != "hold"]:
            b = c_ = n = 0
            for p in poses:
                a = [r for r in rows if r["pose"] == p and r["condition"] == c]
                h = [r for r in rows if r["pose"] == p and r["condition"] == "hold"]
                if not (a and h):
                    continue
                n += 1
                b += int(a[0]["success"]) and not int(h[0]["success"])
                c_ += int(h[0]["success"]) and not int(a[0]["success"])
            if b + c_:
                pv = sum(comb(b + c_, k) for k in range(b, b + c_ + 1)) / 2 ** (b + c_)
            else:
                pv = 1.0
            floor = 1 / 2 ** n if n else float("nan")
            out += ["", f"**{c} vs hold** — 짝 {n} · {c} 만 성공 {b} · hold 만 성공 {c_} · "
                        f"정확 이항검정 단측 p = {pv:.4f}",
                    "", f"> ⚠️ 짝 {n} 개로 도달 가능한 최소 p 는 {floor:.4f} 다. "
                        f"{'p < 0.05 가 구조적으로 불가능하다.' if floor > 0.05 else ''}"]
    return "\n".join(out) + "\n"


def export_frames(run: Path, rows: list[dict], out: Path, hz: float, quality: int) -> None:
    import cv2
    from rus_policy.config import load_config
    from rus_policy.perception import build_backend
    from rus_policy.view_quality import view_quality

    be = build_backend(load_config(str(ROOT / "policy_learning/configs/policy_default.yaml")))
    cv2.imwrite(str(out / "frames" / "roi_mask.png"), (be.roi.astype(np.uint8) * 255))

    for r in rows:
        ep = run / r["dir"]
        fz = ep / "frames.npz"
        if not fz.is_file():
            print(f"  {r['dir']}: frames.npz 없음 — 건너뛴다")
            continue
        z = np.load(fz)
        F, ft = z["frames"], z["t"]
        S = np.load(ep / "states.npz")
        st, stt = S["state"], S["t"]
        force = {}
        dpath = ep / "decisions.csv"
        if dpath.is_file():
            for row in csv.DictReader(open(dpath, encoding="utf-8")):
                try:
                    force[round(float(row["t"]), 1)] = float(row["F_n"])
                except (KeyError, TypeError, ValueError):
                    pass
        d = out / "frames" / r["dir"]
        d.mkdir(parents=True, exist_ok=True)
        step = max(1, int(round(len(ft) / max(ft[-1] - ft[0], 1e-6) / hz)))
        idx = list(range(0, len(F), step))
        recs, tiles = [], []
        for n, j in enumerate(idx):
            cs = be.step_detailed(F[j])[4]
            m = np.asarray(cs.binary_mask).astype(np.uint8)
            i = int(np.argmin(np.abs(stt - ft[j])))
            q, _ = view_quality(st[i])
            cv2.imwrite(str(d / f"f{n:04d}.png"), F[j], [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(str(d / f"m{n:04d}.png"), m * 255, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            recs.append({"n": n, "raw_index": j, "t": round(float(ft[j] - ft[0]), 2),
                         "q_seg": round(q, 4), "area": round(float(st[i, 4]), 5),
                         "centroid_dx": round(float(st[i, 2]), 4),
                         "centroid_dy": round(float(st[i, 3]), 4),
                         "border": round(float(st[i, 12]), 4),
                         "has_mask": int(st[i, 0] > 0.5),
                         "force_n": force.get(round(float(ft[j]), 1), "")})
            v = cv2.cvtColor(F[j], cv2.COLOR_GRAY2BGR)
            c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(v, c, -1, (0, 0, 255), 1)
            cv2.putText(v, f"{n} {recs[-1]['t']:.0f}s q{q:.2f}", (3, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 255, 255), 1)
            tiles.append(v)
        with open(d / "index.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recs[0]))
            w.writeheader()
            w.writerows(recs)
        cols = 12
        pad = (-len(tiles)) % cols
        tiles += [np.zeros_like(tiles[0])] * pad
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        cv2.imwrite(str(out / "contact" / f"{r['dir']}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, quality])
        print(f"  {r['dir']}: {len(idx)} 장")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=str(ROOT / "policy_learning/runs/main_rule"))
    p.add_argument("--out", default=str(Path(__file__).with_name("data")))
    p.add_argument("--hz", type=float, default=1.0, help="프레임 내보내기 밀도 [Hz]")
    p.add_argument("--quality", type=int, default=85, help="컨택트 시트 JPEG 품질")
    p.add_argument("--no-frames", action="store_true", help="표만 다시 만든다 (빠르다)")
    a = p.parse_args()

    run, out = Path(a.run), Path(a.out)
    for sub in ("summary", "frames", "contact"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = load_rows(run)
    print(f"{run.name}: 에피소드 {len(rows)}")

    (out / "summary" / f"{run.name}.csv").write_bytes((run / "summary.csv").read_bytes())
    (out / "verdicts.json").write_text(json.dumps(
        {r["dir"]: {"pose": r["pose"], "condition": r["condition"],
                    "success": int(r["success"]), **r["_verdict"]} for r in rows},
        ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "results.md").write_text(results_markdown(run, rows), encoding="utf-8")
    print(f"  표 → {out/'results.md'}")

    if not a.no_frames:
        export_frames(run, rows, out, a.hz, a.quality)
    print("완료")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
