#!/usr/bin/env python3
"""수동 편집 라벨(masks_edited/)을 반영한 매니페스트를 만든다.

편집기(mask_editor/server.py)는 사람이 손댄 프레임만 masks_edited/ 에 쓴다.
따라서 한 환자 안에서도 '고친 프레임'과 '원본 그대로인 프레임'이 섞인다.
그 상태로 학습하면 같은 환자의 같은 해부구조에 서로 다른 경계 정의를 가르치게
되므로, 기본값은 **편집된 환자의 미편집 프레임을 버리는 것**이다
(--keep-unedited 로 원본을 그대로 쓰게 할 수 있다).

원본 매니페스트/마스크는 건드리지 않는다.

    python3 build_edited_manifest.py                    # manifest_edited.csv
    python3 build_edited_manifest.py --keep-unedited
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
from PIL import Image

ROOT = "/home/rosotauser/datasets/pfus/"
EDIT = ROOT + "masks_edited/"


def edited_rel(mask_path: str) -> str:
    return "masks_edited/" + mask_path.split("masks/", 1)[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=ROOT + "manifest.csv")
    ap.add_argument("--output", default=ROOT + "manifest_edited.csv")
    ap.add_argument("--report", default=ROOT + "manifest_edited_report.json")
    ap.add_argument("--keep-unedited", action="store_true",
                    help="편집된 환자의 미편집 프레임을 원본 라벨로 남긴다")
    ap.add_argument("--ignore-edits", nargs="*", default=[], metavar="PID",
                    help="이 환자의 편집본은 무시하고 원본 라벨을 그대로 쓴다 "
                         "(편집이 오히려 나빠진 환자를 되돌릴 때)")
    a = ap.parse_args()

    with open(a.manifest, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)

    edited_patients = sorted(p for p in os.listdir(EDIT)
                             if os.path.isdir(EDIT + p) and p not in a.ignore_edits)

    stats = {p: {"frames": 0, "edited": 0, "dropped": 0, "iou": [], "keep_ratio": [],
                 "components": [], "empty": 0} for p in edited_patients}
    out = []
    for r in rows:
        pid = r["patient_id"]
        if pid not in edited_patients:
            out.append(r)
            continue
        s = stats[pid]
        s["frames"] += 1
        rel = edited_rel(r["mask_path"])
        if not os.path.exists(ROOT + rel):
            if a.keep_unedited:
                out.append(r)
            else:
                s["dropped"] += 1
            continue
        o = np.asarray(Image.open(ROOT + r["mask_path"]).convert("L")) > 127
        e = np.asarray(Image.open(ROOT + rel).convert("L")) > 127
        if o.shape != e.shape:
            raise SystemExit("shape mismatch: %s" % rel)
        union = (o | e).sum()
        s["iou"].append(float((o & e).sum() / union) if union else 1.0)
        s["keep_ratio"].append(float(e.sum() / max(o.sum(), 1)))
        if e.sum() == 0:
            s["empty"] += 1
        # 편집기는 브러시라서 조각이 남을 수 있다 -- 세어만 두고 고치지는 않는다.
        lab = _components(e)
        s["components"].append(lab)
        s["edited"] += 1
        r = dict(r)
        r["mask_path"] = rel
        out.append(r)

    with open(a.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    report = {
        "source_manifest": a.manifest,
        "output_manifest": a.output,
        "keep_unedited": a.keep_unedited,
        "edited_patients": edited_patients,
        "ignored_edits": sorted(a.ignore_edits),
        "rows_in": len(rows),
        "rows_out": len(out),
        "per_patient": {
            p: {
                "frames_in_manifest": s["frames"],
                "frames_edited": s["edited"],
                "frames_dropped": s["dropped"],
                "mean_iou_orig_vs_edit": round(float(np.mean(s["iou"])), 4) if s["iou"] else None,
                "mean_area_ratio_edit_over_orig": round(float(np.mean(s["keep_ratio"])), 4) if s["keep_ratio"] else None,
                "frames_with_multiple_components": int(sum(c > 1 for c in s["components"])),
                "max_components": int(max(s["components"])) if s["components"] else 0,
                "empty_edited_masks": s["empty"],
            }
            for p, s in stats.items()
        },
    }
    with open(a.report, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0


def _components(mask: np.ndarray) -> int:
    """8-이웃 연결성분 개수 (cv2 없이도 돌게 scipy 우선, 없으면 cv2)."""
    try:
        from scipy import ndimage
        return int(ndimage.label(mask, structure=np.ones((3, 3)))[1])
    except ImportError:
        import cv2
        return int(cv2.connectedComponents(mask.astype(np.uint8), 8)[0]) - 1


if __name__ == "__main__":
    raise SystemExit(main())
