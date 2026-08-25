#!/usr/bin/env python3
"""assets/ 사진에서 배경을 지우고 물체만 남긴다 (GrabCut).

    python3 make_cutouts.py

결과는 assets/cut_*.png (RGBA, 배경 투명). 그림 스크립트는 이 파일을 읽는다.
사각형 초기값은 사람이 눈으로 잡은 값이다 — 사진을 새로 찍으면 다시 잡아야 한다.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "assets")

# name -> (전경을 감싸는 사각형 비율, 사전 크롭 비율 or None, GrabCut 반복)
# 배경과 물체가 둘 다 밝은 무채색이면 색 모형만으로는 못 가른다 (흰 장비 위의 흰
# 프로브, 나뭇결 위의 회색 출력물). 그런 사진은 실루엣을 손으로 따고, GrabCut 은
# 그 선 ±12 px 띠 안에서만 진짜 경계로 붙게 쓴다. 좌표는 사전 크롭 후 비율이다.
POLYGONS = {
    "photo_probe_survey.png": [
        (0.240, 0.128), (0.400, 0.065), (0.580, 0.032), (0.720, 0.038), (0.805, 0.078),
        (0.815, 0.150), (0.816, 0.285), (0.790, 0.368), (0.690, 0.418), (0.630, 0.470),
        (0.620, 0.600), (0.600, 0.720), (0.560, 0.820), (0.550, 1.000), (0.450, 1.000),
        (0.440, 0.820), (0.400, 0.700), (0.380, 0.550), (0.370, 0.470), (0.300, 0.430),
        (0.212, 0.395), (0.202, 0.320), (0.205, 0.190),
    ],
    "photo_probe_shell.png": [
        (0.335, 0.138), (0.695, 0.138), (0.712, 0.180), (0.700, 0.248), (0.635, 0.283),
        (0.595, 0.330), (0.582, 0.420), (0.575, 0.500), (0.630, 0.520), (0.635, 0.620),
        (0.600, 0.650), (0.615, 0.700), (0.615, 0.860), (0.585, 0.870), (0.560, 0.880),
        (0.500, 0.880), (0.485, 0.870), (0.440, 0.860), (0.440, 0.700), (0.420, 0.660),
        (0.395, 0.620), (0.390, 0.520), (0.435, 0.500), (0.428, 0.420), (0.415, 0.330),
        (0.372, 0.283), (0.312, 0.248), (0.308, 0.180),
    ],
}

# 배경과 물체가 둘 다 밝은 무채색이면 색만으로는 못 가른다. 그런 사진은 사람이 찍은
# 전경/배경 상자를 같이 준다 (GrabCut 의 붓질에 해당). 좌표는 사전 크롭 후 비율이다.
SCRIBBLES = {
    "photo_probe_survey.png": {
        "fg": [(0.25, 0.06, 0.78, 0.38), (0.42, 0.45, 0.58, 0.85)],
        "bg": [(0.00, 0.00, 0.14, 1.00), (0.88, 0.00, 1.00, 1.00),
               (0.00, 0.00, 1.00, 0.02), (0.00, 0.46, 0.32, 1.00),
               (0.72, 0.46, 1.00, 1.00)],
    },
    "photo_probe_shell.png": {
        "fg": [(0.36, 0.15, 0.66, 0.26), (0.45, 0.32, 0.57, 0.48),
               (0.46, 0.55, 0.58, 0.68), (0.47, 0.73, 0.58, 0.85)],
        "bg": [(0.00, 0.00, 0.20, 1.00), (0.80, 0.00, 1.00, 1.00),
               (0.00, 0.00, 1.00, 0.08), (0.00, 0.92, 1.00, 1.00),
               (0.00, 0.30, 0.28, 0.90), (0.72, 0.30, 1.00, 0.90)],
    },
}

# name -> (rect, pre-crop, iters, 배경으로 확정할 채도 문턱 or None)
JOBS = {
    "photo_probe_survey.png": ((0.06, 0.02, 0.94, 0.97), (0.30, 0.02, 0.72, 0.78), 6, None),
    "photo_probe_cad.png":    ((0.05, 0.12, 0.97, 0.92), (0.26, 0.12, 0.98, 0.81), 6, None),
    "photo_probe_shell.png":  ((0.16, 0.03, 0.82, 0.70), None, 6, 22),
    "photo_imu_mount.png":    ((0.09, 0.10, 0.92, 0.90), None, 6, 45),
}


def cutout(name, rect, pre, iters, sat_bg=None):
    im = Image.open(os.path.join(A, name)).convert("RGB")
    if pre:
        w, h = im.size
        im = im.crop((int(pre[0] * w), int(pre[1] * h),
                      int(pre[2] * w), int(pre[3] * h)))
    # GrabCut 은 화소 수에 비례해 느려진다. 긴 변 1400 으로 줄여 풀고 마스크만 되돌린다.
    big = np.asarray(im)
    scale = 1400.0 / max(big.shape[:2])
    small = cv2.resize(big, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1 else big
    h, w = small.shape[:2]
    r = (int(rect[0] * w), int(rect[1] * h),
         int((rect[2] - rect[0]) * w), int((rect[3] - rect[1]) * h))
    poly = POLYGONS.get(name)
    if poly:
        fg = np.zeros((h, w), np.uint8)
        pts = np.array([[int(x * w), int(y * h)] for x, y in poly], np.int32)
        cv2.fillPoly(fg, [pts], 255)
    else:
        mask = np.zeros((h, w), np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(small, mask, r, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255,
                      0).astype(np.uint8)

    # 2 차 통과 — 1 차 마스크를 씨앗으로 다시 푼다. 나뭇결 같은 배경 조각이 붙어 있으면
    # 채도로 확실한 배경을 못박아 준다 (물체는 전부 무채색이다).
    band = 13 if poly else 25
    m2 = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    m2[cv2.dilate(fg, np.ones((band, band), np.uint8)) > 0] = cv2.GC_PR_FGD
    m2[cv2.erode(fg, np.ones((band, band), np.uint8)) > 0] = cv2.GC_FGD
    m2[cv2.dilate(fg, np.ones((band * 2 + 1, band * 2 + 1), np.uint8)) == 0] = cv2.GC_BGD
    sc = None if poly else SCRIBBLES.get(name)
    if sc:
        for x0, y0, x1, y1 in sc.get("bg", []):
            m2[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = cv2.GC_BGD
        for x0, y0, x1, y1 in sc.get("fg", []):
            m2[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = cv2.GC_FGD
    if sat_bg is not None:
        sat = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[:, :, 1]
        m2[(sat > sat_bg) & (m2 != cv2.GC_FGD)] = cv2.GC_BGD
    bgd2, fgd2 = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(small, m2, None, bgd2, fgd2, 4, cv2.GC_INIT_WITH_MASK)
    fg = np.where((m2 == cv2.GC_FGD) | (m2 == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # 가장 큰 덩어리만 남기고 구멍을 메운다 — 손가락·그림자 조각이 붙는 것을 막는다
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    if n > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fg = np.where(lab == k, 255, 0).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    inset = max(3, int(0.010 * min(h, w)))
    fg = cv2.erode(fg, np.ones((inset, inset), np.uint8))
    fg = cv2.GaussianBlur(fg, (5, 5), 0)

    alpha = cv2.resize(fg, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_LINEAR)
    out = np.dstack([big, alpha])
    ys, xs = np.where(alpha > 40)
    if ys.size:                                   # 여백을 잘라 낸다
        pad = int(0.02 * max(big.shape[:2]))
        y0, y1 = max(0, ys.min() - pad), min(alpha.shape[0], ys.max() + pad)
        x0, x1 = max(0, xs.min() - pad), min(alpha.shape[1], xs.max() + pad)
        out = out[y0:y1, x0:x1]
    dst = os.path.join(A, "cut_" + name)
    Image.fromarray(out).save(dst)
    print(f"  -> {dst}   {out.shape[1]}x{out.shape[0]}   fg {alpha.mean() / 255:.2f}")


if __name__ == "__main__":
    for n, (rect, pre, it, sat) in JOBS.items():
        cutout(n, rect, pre, it, sat)
