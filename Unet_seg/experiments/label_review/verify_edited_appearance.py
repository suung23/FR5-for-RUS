#!/usr/bin/env python3
"""수동 편집 라벨이 실제로 '내강 위에' 놓였는지 모델 없이 검증한다.

dataset_audit/label_appearance.py 의 frame_stats 를 그대로 재사용한다.
같은 프레임, 같은 지표, 라벨만 원본 vs 편집본으로 바꿔 짝지어 비교하므로
차이는 라벨에서만 온다.

    contrast  = mean(라벨 주변 링) - mean(라벨 내부).  내강이면 클수록 옳다.
    texture   = std(내부) / std(링).                   내강이면 작을수록 옳다.
    darkness  = 내부 평균의 섹터 내 백분위.            내강이면 작을수록 옳다.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pfus_label_audit" / "dataset_audit"))
from label_appearance import frame_stats  # noqa: E402

SIZE = 256
ROOT = Path("/home/rosotauser/datasets/pfus")


def load(path: Path, is_mask: bool) -> np.ndarray:
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    a = cv2.resize(a, (SIZE, SIZE), interpolation=interp)
    return a > 127 if is_mask else a


def main() -> int:
    rows = list(csv.DictReader(open(ROOT / "manifest.csv")))
    edited = sorted(p.name for p in (ROOT / "masks_edited").iterdir() if p.is_dir())
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pid = r["patient_id"]
        if pid not in edited:
            continue
        ep = ROOT / ("masks_edited/" + r["mask_path"].split("masks/", 1)[1])
        if not ep.exists():
            continue
        img = load(ROOT / r["image_path"], False)
        for tag, mp in (("orig", ROOT / r["mask_path"]), ("edit", ep)):
            s = frame_stats(img, load(mp, True))
            if s:
                for k, v in s.items():
                    acc[pid][tag + "_" + k].append(v)

    keys = ("contrast", "texture", "darkness_pct")
    print(f"{'patient':9}{'n':>5}" + "".join(f"{k:>26}" for k in keys))
    print(f"{'':9}{'':>5}" + "".join(f"{'orig -> edit  (delta)':>26}" for _ in keys))
    for pid in edited:
        d = acc[pid]
        n = len(d["orig_contrast"])
        line = f"{pid:9}{n:5d}"
        for k in keys:
            o, e = np.mean(d["orig_" + k]), np.mean(d["edit_" + k])
            line += f"{o:9.3f} ->{e:8.3f} ({e - o:+6.3f})"
        print(line)
    print("\ncontrast 는 클수록, texture 와 darkness 는 작을수록 '내강 라벨'에 가깝다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
