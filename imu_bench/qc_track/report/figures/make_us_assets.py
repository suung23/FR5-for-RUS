#!/usr/bin/env python3
"""그림에 넣을 초음파 프레임 두 장을 만든다 — 원본과 세그멘테이션 겹침.

    python3 make_us_assets.py

프레임은 PFUS 라벨 감사 세트에서 **방광 내강이 가장 크게 보이는 실측 프레임**을
고른 것이고, 겹침은 저장된 Slim U-Net 예측(채움)과 정답 마스크(청록 외곽선)를
그대로 얹은 것이다. 해부를 그려 넣지 않는다.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
AUDIT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "Unet_seg",
                                     "experiments", "pfus_label_audit"))
SAMPLE = "S0040_P041_frame_000"          # 마스크 면적 최대 (0.152) = 넓은 방광

FILL, LINE = (224, 111, 122), (221, 221, 34)   # BGR 기준: 보라 채움 · 청록 외곽선


def main() -> int:
    img = cv2.imread(os.path.join(AUDIT, "annotation_audit", "images",
                                  SAMPLE + ".png"), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(AUDIT, "annotation_audit", "original_masks",
                                 SAMPLE + ".png"), cv2.IMREAD_GRAYSCALE)
    pred = np.load(os.path.join(AUDIT, "predictions",
                                "predictions_slim.npz"))[SAMPLE].astype(np.uint8)
    if pred.shape != img.shape:
        pred = cv2.resize(pred, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(ASSETS, "us_frame_wide.png"), rgb)

    over = rgb.copy()
    over[pred > 0] = FILL
    out = cv2.addWeighted(over, 0.45, rgb, 0.55, 0)
    cnt, _ = cv2.findContours((gt > 127).astype(np.uint8), cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, LINE, 3)
    cv2.imwrite(os.path.join(ASSETS, "us_frame_wide_seg.png"), out)

    for n in ("us_frame_wide.png", "us_frame_wide_seg.png"):
        p = os.path.join(ASSETS, n)
        print(f"  -> {p}   {Image.open(p).size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
