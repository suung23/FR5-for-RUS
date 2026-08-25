#!/usr/bin/env python3
"""라벨 결함 스크리닝 — 사람 검토(2026-08-25)가 찾아낸 결함 유형을 모델 없이 재현한다.

기존 감사(`dataset_audit/label_appearance.py`, `label_integrity.py`)는 P021 을 통과시켰다.
사람이 직접 보고 "라벨이 잘못됐다" 고 판정했으므로, 그 감사들은 이 결함에 둔감하다.
train/val 94명은 사람 검토를 거치지 않았으므로 같은 결함이 얼마나 있는지 알아야 한다.

검토에서 나온 결함 두 가지를 각각 지표로 만든다.

  A. 경계가 영상 구조를 따르지 않는다  (P021)
     진짜 경계는 밝기 gradient 위에 앉는다. 라벨 윤곽의 |∇I| 를, 같은 마스크를
     몇 픽셀 밀어 만든 귀무 윤곽의 |∇I| 와 비교한다.
        edge_ratio = mean|∇I|(윤곽) / mean|∇I|(이동 윤곽)
     1.0 근처면 "아무 데나 그은 선"과 구별되지 않는다.

  B. 라벨 안에 방광이 아닌 것이 섞여 있다  (P043)
     내강은 무에코라 어둡고 균질하다. 라벨 내부에서 섹터 중앙값보다 밝은 화소의
     비율을 센다.
        bright_frac = |{내부 화소 > 섹터 median}| / |내부|
     높을수록 비내강 조직을 함께 삼킨 것이다.

둘 다 **모델을 쓰지 않는다.** 예측을 근거로 라벨을 의심하면 순환논증이 된다.

    python3 screen_labels.py --frames 15
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

import numpy as np
from PIL import Image

ROOT = "/home/rosotauser/datasets/pfus/"


def contour(m: np.ndarray) -> np.ndarray:
    inner = m.copy()
    for s in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        inner &= np.roll(m, s, axis=(0, 1))
    return m & ~inner


def grad_mag(img: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(img)
    return np.hypot(gx, gy)


def frame_scores(image_path: str, mask_path: str, shift: int = 7):
    img = np.asarray(Image.open(ROOT + image_path).convert("L"), np.float32)
    m = np.asarray(Image.open(ROOT + mask_path)) > 0
    if m.sum() < 50:
        return None
    g = grad_mag(img)
    c = contour(m)
    on = float(g[c].mean())

    # 귀무: 같은 모양을 네 방향으로 민 윤곽. 국소 텍스처를 통제한다.
    nulls = []
    for dy, dx in ((shift, 0), (-shift, 0), (0, shift), (0, -shift)):
        cs = np.roll(c, (dy, dx), axis=(0, 1))
        nulls.append(float(g[cs].mean()))
    edge_ratio = on / (float(np.mean(nulls)) + 1e-6)

    # 섹터 = 완전 검은 레터박스를 제외한 영역
    sector = img > 5
    med = float(np.median(img[sector])) if sector.any() else 0.0
    inside = img[m]
    bright_frac = float((inside > med).mean())
    return edge_ratio, bright_frac, float(inside.mean()), med, int(m.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    a = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT + "manifest.csv")))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["patient_id"]].append(r)
    for v in by.values():
        v.sort(key=lambda r: int(r["frame_index"]))

    out = []
    for pid, rs in sorted(by.items()):
        pick = np.linspace(0, len(rs) - 1, min(a.frames, len(rs))).astype(int)
        vals = [frame_scores(rs[i]["image_path"], rs[i]["mask_path"]) for i in pick]
        vals = [v for v in vals if v]
        if not vals:
            continue
        er = float(np.median([v[0] for v in vals]))
        bf = float(np.median([v[1] for v in vals]))
        out.append({"patient_id": pid, "split": rs[0]["split"], "n_sampled": len(vals),
                    "edge_ratio": round(er, 3), "bright_frac": round(bf, 3),
                    "inside_mean": round(float(np.median([v[2] for v in vals])), 1),
                    "sector_median": round(float(np.median([v[3] for v in vals])), 1),
                    "area_px": int(np.median([v[4] for v in vals]))})

    p = os.path.join(a.out, "label_screen.csv")
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wr.writeheader(); wr.writerows(out)

    known_bad = {"P021", "P043"}
    E = np.array([r["edge_ratio"] for r in out]); B = np.array([r["bright_frac"] for r in out])
    print("환자 %d명.  중앙값  edge_ratio %.2f   bright_frac %.2f" % (len(out), np.median(E), np.median(B)))
    print("\n[A] 경계가 구조를 안 따름 — edge_ratio 낮은 순 12명")
    for r in sorted(out, key=lambda r: r["edge_ratio"])[:12]:
        print("   %-6s %-6s edge %.2f  bright %.2f  area %5d  %s"
              % (r["patient_id"], r["split"], r["edge_ratio"], r["bright_frac"], r["area_px"],
                 "◄ 사람 판정: 불량" if r["patient_id"] in known_bad else ""))
    print("\n[B] 비내강 혼입 — bright_frac 높은 순 12명")
    for r in sorted(out, key=lambda r: -r["bright_frac"])[:12]:
        print("   %-6s %-6s bright %.2f  edge %.2f  area %5d  %s"
              % (r["patient_id"], r["split"], r["bright_frac"], r["edge_ratio"], r["area_px"],
                 "◄ 사람 판정: 불량" if r["patient_id"] in known_bad else ""))
    for pid in ("P021", "P043", "P000"):
        r = next((x for x in out if x["patient_id"] == pid), None)
        if r:
            print("\n%s  edge %.2f (백분위 %.0f)   bright %.2f (백분위 %.0f)"
                  % (pid, r["edge_ratio"], 100 * (E < r["edge_ratio"]).mean(),
                     r["bright_frac"], 100 * (B < r["bright_frac"]).mean()))
    print("\n→", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
