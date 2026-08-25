#!/usr/bin/env python3
"""Dice 가 낮은 환자의 **GT 라벨을 사람이 직접 판정**하기 위한 검토 시트.

의도적으로 모델 예측을 그리지 않는다. 판정 대상은 라벨이지 모델이 아니고,
예측을 겹쳐 두면 "모델이 저렇게 봤으니 라벨이 틀렸겠지" 쪽으로 눈이 끌린다.
대신 프레임별 Dice 를 숫자로만 적어, 어느 프레임을 특히 볼지 고르게 한다.

산출 (out/):
    <PID>_context.png   전체 프레임 + GT 윤곽      — 위치·해부 맥락
    <PID>_zoom.png      GT 주변 확대 + 윤곽        — 경계 정확도
    verdict_template.csv  판정 기록용 빈 표

    python3 make_review_sheets.py --patients P021 P043 P000
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/home/rosotauser/datasets/pfus/"
EVAL = "runs/pfus_bladder/eval_test/evaluation.json"


def contour(mask: np.ndarray) -> np.ndarray:
    """이진 마스크의 1픽셀 경계. 침식 후 차집합 — scipy 없이."""
    m = mask.astype(bool)
    inner = m.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        inner &= np.roll(m, (dy, dx), axis=(0, 1))
    return m & ~inner


def load(pid: str):
    rows = [r for r in csv.DictReader(open(ROOT + "manifest.csv")) if r["patient_id"] == pid]
    rows.sort(key=lambda r: int(r["frame_index"]))
    return rows


def dice_by_frame(pid: str) -> dict[int, float]:
    """frame_index 로 키를 잡는다. frame_id 는 'P021/seq0/000000' 형식이라 파일명과 다르다."""
    d = json.load(open(EVAL))
    return {int(f["frame_index"]): f["dice"] for f in d["spatial"]["per_frame"]
            if f["patient_id"] == pid and f["dice"] is not None}


def sheet(pid, rows, dices, out, n=20, zoom=False):
    pick = np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)
    cols, r_ = 5, int(np.ceil(len(pick) / 5))
    fig, axes = plt.subplots(r_, cols, figsize=(cols * 3.0, r_ * 3.0))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, pick):
        r = rows[i]
        img = np.asarray(Image.open(ROOT + r["image_path"]).convert("L"), np.float32)
        gt = np.asarray(Image.open(ROOT + r["mask_path"])) > 0
        rgb = np.stack([img] * 3, -1) / 255.0
        if gt.any():
            c = contour(gt)
            rgb[c] = [0.1, 1.0, 0.3]                      # GT = 초록
        if zoom and gt.any():
            ys, xs = np.nonzero(gt)
            cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
            h = int(max(np.ptp(ys), np.ptp(xs)) * 0.72) + 14
            y0, y1 = max(cy - h, 0), min(cy + h, rgb.shape[0])
            x0, x1 = max(cx - h, 0), min(cx + h, rgb.shape[1])
            rgb = rgb[y0:y1, x0:x1]
        ax.imshow(np.clip(rgb, 0, 1))
        fid = os.path.splitext(os.path.basename(r["image_path"]))[0]
        dc = dices.get(int(r["frame_index"]))
        ax.set_title("%s   Dice %s" % (fid, "n/a" if dc is None else "%.2f" % dc),
                     fontsize=8, color=("#b0452b" if (dc is not None and dc < 0.5) else "#333"))
    fig.suptitle("%s  —  GT label review, %s   (green = GT contour; prediction NOT drawn)"
                 % (pid, "zoom" if zoom else "full frame"), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("  ", out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="+", default=["P021", "P043", "P000"])
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    verdicts = []
    for pid in a.patients:
        rows = load(pid)
        dices = dice_by_frame(pid)
        vals = [v for v in dices.values()]
        print("%s  프레임 %d  Dice 평균 %.3f" % (pid, len(rows), float(np.mean(vals)) if vals else float("nan")))
        sheet(pid, rows, dices, os.path.join(a.out, pid + "_context.png"), a.frames, zoom=False)
        sheet(pid, rows, dices, os.path.join(a.out, pid + "_zoom.png"), a.frames, zoom=True)
        verdicts.append({"patient_id": pid, "n_frames": len(rows),
                         "dice_mean": round(float(np.mean(vals)), 3) if vals else "",
                         "verdict": "", "note": ""})

    p = os.path.join(a.out, "verdict_template.csv")
    with open(p, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["patient_id", "n_frames", "dice_mean", "verdict", "note"])
        wr.writeheader(); wr.writerows(verdicts)
    print("\n판정 기록:", p)
    print("verdict 열에  label_ok / label_wrong / partial  중 하나를 적어주십시오.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
