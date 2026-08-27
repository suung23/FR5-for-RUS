#!/usr/bin/env python3
"""수정 라벨 재학습의 효과를 환자 단위로 비교한다.

세 개의 평가 디렉터리를 읽는다 (scripts/evaluate.py 가 만든 frame_metrics.csv):

    eval_base_origGT   기존 체크포인트 x 원본 GT     -- 과거 보고서의 재현
    eval_base_editGT   기존 체크포인트 x 수정 GT     -- 라벨만 바뀐 효과
    eval_edit_editGT   재학습 체크포인트 x 수정 GT   -- 재학습까지 반영

부트스트랩은 환자 단위(프레임 단위가 아니라)로 뽑는다. 같은 클립의 프레임은
독립이 아니므로 프레임 부트스트랩은 신뢰구간을 몇 배로 좁게 만든다.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

import numpy as np

METRICS = ("dice", "iou", "hd95")


def per_patient(path: str) -> dict[str, dict[str, float]]:
    acc: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            for m in METRICS:
                v = r.get(m, "")
                if v not in ("", "nan", "None"):
                    acc[r["patient_id"]][m].append(float(v))
    return {p: {m: float(np.mean(v)) for m, v in d.items() if v} for p, d in acc.items()}


def boot(values: list[float], n: int = 4000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(values, float)
    draws = a[rng.integers(0, len(a), size=(n, len(a)))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/labeledit")
    ap.add_argument("--output", default="runs/labeledit/comparison.json")
    a = ap.parse_args()

    # 정해진 순서를 먼저, 나중에 추가된 런은 뒤에 붙인다.
    order = ["eval_base_origGT", "eval_base_editGT", "eval_seed43_editGT",
             "eval_edit_editGT", "eval_editnoP033_editGT",
             "eval_edit_man_editGT", "eval_edit_man_hydro_editGT"]
    found = sorted(d for d in os.listdir(a.runs)
                   if d.startswith("eval_") and os.path.isdir(os.path.join(a.runs, d)))
    names = [n for n in order if n in found] + [n for n in found if n not in order]
    tables = {}
    for n in names:
        p = os.path.join(a.runs, n, "frame_metrics.csv")
        if os.path.exists(p):
            tables[n] = per_patient(p)
        else:
            print("missing (skipped): " + p)


    pats = sorted(set().union(*(set(t) for t in tables.values())))
    short = {n: n.replace("eval_", "").replace("_editGT", "").replace("_origGT", ":origGT")
             for n in tables}
    hdr = f"{'patient':9}" + "".join(f"{short[n]:>20}" for n in tables)
    print("\n=== Dice, per patient (frame mean) ===")
    print(hdr)
    for p in pats:
        row = f"{p:9}"
        for n in tables:
            v = tables[n].get(p, {}).get("dice")
            row += f"{v:20.4f}" if v is not None else f"{'-':>20}"
        print(row)

    summary = {}
    print("\n=== Summary (patient-mean, patient-level bootstrap 95% CI, 4000 draws) ===")
    for n, t in tables.items():
        summary[n] = {}
        line = f"{n:20}"
        for m in METRICS:
            vals = [d[m] for d in t.values() if m in d]
            if not vals:
                continue
            lo, hi = boot(vals)
            summary[n][m] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                             "ci95": [lo, hi], "n_patients": len(vals)}
            if m == "dice":
                line += f"Dice {np.mean(vals):.4f} +/- {np.std(vals, ddof=1):.3f} [{lo:.3f}, {hi:.3f}]   "
            elif m == "iou":
                line += f"IoU {np.mean(vals):.4f}   "
            else:
                line += f"HD95 {np.mean(vals):.2f} px"
        print(line)

    # 배제된 test 환자를 뺀 코호트도 같은 frame_metrics 에서 뽑는다 -- 테스트셋을
    # 바꾸지 않고 두 숫자를 다 보여주기 위해서다.
    flagged_test = ["P000", "P020", "P021", "P043"]
    print("\n=== Retained test cohort (VERDICTS 불량 4명 제외, 12 patients) ===")
    for n, t in tables.items():
        vals = [d["dice"] for p, d in t.items() if p not in flagged_test and "dice" in d]
        lo, hi = boot(vals)
        summary.setdefault(n, {})["dice_retained_cohort"] = {
            "mean": float(np.mean(vals)), "ci95": [lo, hi], "n_patients": len(vals)}
        print(f"  {short[n]:24} Dice {np.mean(vals):.4f} [{lo:.3f}, {hi:.3f}]  n={len(vals)}")

    # 짝지은 비교: 기준을 바꿔가며 여러 쌍을 본다.
    pairs = [("eval_base_editGT", "eval_seed43_editGT", "재현성 바닥 (라벨 동일, seed만 42->43)"),
             ("eval_base_editGT", "eval_edit_editGT", "수정 라벨 재학습"),
             ("eval_edit_editGT", "eval_editnoP033_editGT", "P033 편집 되돌림"),
             ("eval_edit_editGT", "eval_edit_man_editGT", "+ manual 배제"),
             ("eval_edit_man_editGT", "eval_edit_man_hydro_editGT", "+ hydro 필터")]
    print("\n=== Paired deltas on the fixed 16-patient test set ===")
    for base, other, label in pairs:
        if base not in tables or other not in tables:
            continue
        b, e = tables[base], tables[other]
        common = sorted(set(b) & set(e))
        d = np.array([e[p]["dice"] - b[p]["dice"] for p in common])
        lo, hi = boot(d.tolist())
        summary[f"paired::{other}_minus_{base}"] = {
            "label": label, "mean_delta_dice": float(d.mean()), "ci95": [lo, hi],
            "improved": int((d > 0).sum()), "n_patients": len(common)}
        print(f"  {label:36} {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"improved {int((d > 0).sum())}/{len(common)}")

    if False:
        # 같은 GT, 같은 환자 -- 짝지어 비교해야 환자 난이도 분산이 상쇄된다.
        b, e = tables["eval_base_editGT"], tables["eval_edit_editGT"]
        common = sorted(set(b) & set(e))
        d = np.array([e[p]["dice"] - b[p]["dice"] for p in common])
        lo, hi = boot(d.tolist())
        summary["paired_edit_minus_base_on_editGT"] = {
            "mean_delta_dice": float(d.mean()), "ci95": [lo, hi],
            "n_patients": len(common),
            "improved": int((d > 0).sum()), "worsened": int((d < 0).sum()),
        }
        print(f"\nPaired delta (retrained - baseline, same corrected GT): "
              f"{d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  "
              f"improved {int((d > 0).sum())}/{len(common)}")

    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w") as fh:
        json.dump({"per_patient": tables, "summary": summary}, fh, indent=2)
    print("\nwrote " + a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
