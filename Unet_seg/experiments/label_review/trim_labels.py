#!/usr/bin/env python3
"""라벨 다듬기 — 비내강까지 삼킨 GT 에서 **빼기만** 한다.

검토(2026-08-25)에서 나온 결함: 라벨이 방광 내강 밖의 밝은 조직까지 포함한다.
그 상태로는 Dice 가 모델이 아니라 라벨을 재고, 학습에도 잘못된 경계를 가르친다.

원칙 — **더하지 않는다.** 원 라벨 밖으로는 절대 나가지 않고 안에서만 깎는다.
라벨이 놓친 내강을 추측해 넣는 것은 사람의 판단이 필요한 일이고, 자동으로 하면
새 오류를 만든다.

방법
    1. 라벨 내부 화소의 밝기에 Otsu 를 걸어 어두운 쪽(무에코 = 내강)을 남긴다.
    2. 형태학적 닫기로 내부 잡음(debris·혼탁)을 메운다.
    3. 최대 연결성분만 남기고 구멍을 채운다.
    4. 남은 넓이가 원본의 `--min-keep` 미만이면 **적용하지 않고 표시만** 한다.
       그런 프레임은 자동으로 다룰 문제가 아니다.

원본 마스크는 건드리지 않는다. 결과는 별도 디렉터리에 쓰고, 승인 후 매니페스트가
그쪽을 가리키게 한다.

    python3 trim_labels.py --patients P043 P060 --preview      # 검토 시트만
    python3 trim_labels.py --patients P043 --apply             # 마스크 기록
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os

import cv2
import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = "/home/rosotauser/datasets/pfus/"
OUTMASK = ROOT + "masks_trimmed/"


def trim(img: np.ndarray, m: np.ndarray, close_px: int, min_keep: float):
    """원 라벨 안에서만 깎는다. (다듬은 마스크, 유지비율, 적용여부)"""
    inside = img[m]
    if inside.size < 50:
        return m, 1.0, False
    thr, _ = cv2.threshold(inside.astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    keep = m & (img <= thr)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
        keep &= m                                   # 닫기가 라벨 밖으로 새지 않게
    n, lab, stats, _ = cv2.connectedComponentsWithStats(keep.astype(np.uint8), 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = lab == big
    ff = keep.astype(np.uint8).copy()
    h, w = ff.shape
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    keep = keep | (ff == 0)                          # 구멍 채우기
    ratio = float(keep.sum()) / float(m.sum())
    return (keep, ratio, True) if ratio >= min_keep else (m, ratio, False)


def load(pid):
    rs = [r for r in csv.DictReader(open(ROOT + "manifest.csv")) if r["patient_id"] == pid]
    rs.sort(key=lambda r: int(r["frame_index"]))
    return rs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", nargs="+", required=True)
    ap.add_argument("--close-px", type=int, default=9)
    ap.add_argument("--min-keep", type=float, default=0.35,
                    help="이 비율 미만으로 깎이면 적용하지 않고 표시만 한다")
    ap.add_argument("--preview", action="store_true", help="검토 시트만 만든다")
    ap.add_argument("--apply", action="store_true", help="다듬은 마스크를 기록한다")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)) + "/trim")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    summary = []
    for pid in a.patients:
        rs = load(pid)
        ratios, applied = [], 0
        for r in rs:
            img = np.asarray(Image.open(ROOT + r["image_path"]).convert("L"), np.float32)
            m = np.asarray(Image.open(ROOT + r["mask_path"])) > 0
            if m.sum() < 50:
                continue
            new, ratio, ok = trim(img, m, a.close_px, a.min_keep)
            ratios.append(ratio); applied += ok
            if a.apply:
                p = OUTMASK + r["mask_path"].split("masks/", 1)[1]
                os.makedirs(os.path.dirname(p), exist_ok=True)
                Image.fromarray((new * 255).astype(np.uint8)).save(p)

        med = float(np.median(ratios)) if ratios else float("nan")
        summary.append({"patient_id": pid, "n": len(ratios), "keep_median": round(med, 3),
                        "applied": applied, "skipped": len(ratios) - applied})
        print("  %-6s 프레임 %3d  유지 중앙 %.2f  적용 %3d  보류 %2d"
              % (pid, len(ratios), med, applied, len(ratios) - applied))

        if a.preview:
            pick = np.linspace(0, len(rs) - 1, min(a.frames, len(rs))).astype(int)
            fig, axes = plt.subplots(3, 4, figsize=(4 * 3.1, 3 * 3.2))
            for ax in axes.ravel():
                ax.axis("off")
            for ax, i in zip(axes.ravel(), pick):
                r = rs[i]
                img = np.asarray(Image.open(ROOT + r["image_path"]).convert("L"), np.float32)
                m = np.asarray(Image.open(ROOT + r["mask_path"])) > 0
                new, ratio, ok = trim(img, m, a.close_px, a.min_keep)
                rgb = np.stack([img] * 3, -1) / 255.0
                for msk, col in ((m, [1.0, 0.25, 0.15]), (new, [0.1, 1.0, 0.3])):
                    c = msk & ~np.all([np.roll(msk, s, (0, 1)) for s in
                                       ((1, 0), (-1, 0), (0, 1), (0, -1))], axis=0)
                    rgb[c] = col
                if m.any():
                    ys, xs = np.nonzero(m)
                    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
                    h = int(max(np.ptp(ys), np.ptp(xs)) * 0.72) + 14
                    rgb = rgb[max(cy - h, 0):cy + h, max(cx - h, 0):cx + h]
                ax.imshow(np.clip(rgb, 0, 1))
                ax.set_title("f%03d  keep %.2f%s" % (i, ratio, "" if ok else "  SKIP"),
                             fontsize=8, color=("#333" if ok else "#b0452b"))
            fig.suptitle("%s  —  red = original GT,  green = trimmed  (subtract only)" % pid, fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(a.out, pid + "_trim.png"), dpi=110)
            plt.close(fig)

    json.dump(summary, open(os.path.join(a.out, "trim_summary.json"), "w"), indent=1)
    if a.apply:
        print("\n마스크 기록:", OUTMASK)
    print("→", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
