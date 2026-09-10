#!/usr/bin/env python3
"""C10UR 팬텀: A-line 접촉 실측 → 학습 세트(수동 + 의사라벨 + 음성) 생성.

세션 사이에 매니페스트를 여러 개 만들어 두지 않기 위한 스크립트다. 학습에 쓰는 매니페스트는
언제나 이 명령 하나로 다시 만든다 — 입력은 `manifest.csv`(사람이 그린 masks/) 와 직전 추론 결과뿐이다.

    python scripts/phantom_training_sets.py contact                     # A-line 접촉 → contact_alines.npz
    python scripts/phantom_training_sets.py sets --run runs/<이전 추론>  # → manifest_train.csv
    python scripts/phantom_training_sets.py export --run runs/<최종 추론>  # → masks_final/ + manifest_masked.csv

**접촉 판정.** 부채꼴 이미지의 밝기로는 안 된다: 근거리(깊이 표본 0-48)는 트랜스듀서 링다운이라 공기 중에도
밝다 (실측 공기 89 / 접촉 89). 갈리는 곳은 중간 깊이 48-160 이다 (공기 6 / 접촉 55). 그래서 극좌표 원본에서
A-line 160 개마다 중간 깊이 평균을 재고, 임계값을 넘는 비율을 프레임의 접촉 비율로 쓴다.
2026-09-10 실측: 완전 접촉(>=0.9) 70.0 % / 부분 27.1 % / 비접촉(<0.15) 2.9 %.

**의사라벨 선택.** 사람 라벨 117 장 기준으로 Dice 는 접촉 비율에 따라 0.933 → 0.833 → 0.627 로 떨어지고,
사람이 "방광 없음"으로 표시한 프레임은 전부 부분 접촉 구간에 있다. 그래서 자가학습 양성 표본은 **완전 접촉
프레임만** 쓴다 (부분 접촉의 판단은 사람 라벨로만 배운다). 음성 표본은 공기 프레임의 빈 마스크다 — 이것을
빼면 모델이 "없다"고 말하는 능력을 잃는다 (실측: 비접촉 666 장 중 오탐 35 → 631).

홀드아웃 세션에는 어떤 의사라벨도 넣지 않는다.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "phantom_c10ur"
LOGS = HERE.parent.parent / "imu_bench" / "logs"

#: A-line 이 조직과 결합했다고 보는 중간 깊이(표본 48-160) 평균 임계값. 공기 ~6, 접촉 ~55.
COUPLING_THRESHOLD = 20.0
MIDFIELD = slice(48, 160)
#: 세션 단위 홀드아웃 (policy_learning/data/sessions.csv 의 배정과 같다).
VAL = {"us_imu_20260909_161540", "us_imu_20260909_181532"}
TEST = {"us_imu_20260909_182305"}


def measure_contact(out: Path) -> None:
    """세션마다 (프레임, A-line) 결합 여부 불리언 배열을 저장한다."""
    sessions = sorted({r["patient_id"] for r in csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8"))})
    res = {}
    for pid in sessions:
        meta = json.loads((LOGS / pid / "session.meta.json").read_text(encoding="utf-8"))
        shape = tuple(meta["us"]["frame_shape"])
        frames = np.memmap(str(LOGS / pid / "us_frames.bin"), np.uint8, "r").reshape(-1, *shape)
        cov = np.zeros((len(frames), shape[0]), bool)
        for a in range(0, len(frames), 200):
            blk = np.asarray(frames[a:a + 200], np.float32)
            cov[a:a + 200] = blk[:, :, MIDFIELD].mean(axis=2) > COUPLING_THRESHOLD
        res[pid] = cov
        r = cov.mean(axis=1)
        print("%-22s n=%4d  접촉비율 median %.2f  full %5.1f%%  partial %5.1f%%  none %5.1f%%"
              % (pid[-6:], len(frames), np.median(r), 100 * (r >= 0.9).mean(),
                 100 * ((r > 0.15) & (r < 0.9)).mean(), 100 * (r <= 0.15).mean()))
    np.savez_compressed(out, **res)
    allr = np.concatenate([v.mean(axis=1) for v in res.values()])
    print("전체 %d 프레임: full %.1f%%  partial %.1f%%  none %.1f%%  →  %s"
          % (len(allr), 100 * (allr >= 0.9).mean(), 100 * ((allr > 0.15) & (allr < 0.9)).mean(),
             100 * (allr <= 0.15).mean(), out))


def parse_frame_id(frame_id: str) -> tuple[str, int]:
    """추론 결과의 frame_id → (세션, 프레임 번호).

    매니페스트로 돌린 추론은 "<세션>/S00/000123", --input 폴더로 돌린 추론은 파일 이름
    "<세션>_00123" 이다. 두 경우 모두 받는다.
    """
    if "/" in frame_id:
        pid, _, fi = frame_id.split("/")
        return pid, int(fi)
    pid, fi = frame_id.rsplit("_", 1)
    return pid, int(fi)


def neighbour_iou(run: Path, pid: str, indices: list[int]) -> dict[int, float]:
    """이웃 프레임 마스크와의 IoU — 사람 라벨 87 장에서 Dice 와 r=0.69 로 검증된 오류 지표."""
    import cv2

    masks = [cv2.imread(str(run / "masks" / f"{pid}_S00_{i:06d}_mask.png"), cv2.IMREAD_GRAYSCALE) > 127
             for i in indices]
    out = {}
    for k, i in enumerate(indices):
        nb = []
        for j in (k - 1, k + 1):
            if 0 <= j < len(masks):
                union = (masks[k] | masks[j]).sum()
                nb.append(1.0 if union == 0 else (masks[k] & masks[j]).sum() / union)
        out[i] = float(np.mean(nb)) if nb else 0.0
    return out


def build_sets(run: Path, out: Path, pos_per_session: int, neg_per_session: int) -> None:
    import cv2
    from PIL import Image

    cov = dict(np.load(DATA / "contact_alines.npz"))
    rows = list(csv.DictReader(open(DATA / "manifest.csv", encoding="utf-8")))
    manual = {(r["patient_id"], int(r["frame_index"])) for r in rows if r["mask_path"]}
    labelled_sessions = {p for p, _ in manual}

    states = {}
    for r in csv.DictReader(open(run / "control_states.csv", encoding="utf-8")):
        states[parse_frame_id(r["frame_id"])] = r
    if len(states) != len(rows):
        raise SystemExit(f"{run}/control_states.csv 가 {len(states)} 행 — 매니페스트 {len(rows)} 행과 다르다. "
                         "다른 프로세스가 같은 폴더에 쓰고 있지 않은지 확인할 것.")

    by_session = collections.defaultdict(list)
    for pid, fi in states:
        by_session[pid].append(fi)
    tiou = {}
    for pid, idx in by_session.items():
        idx.sort()
        tiou.update({(pid, i): v for i, v in neighbour_iou(run, pid, idx).items()})

    entropy = np.array([float(r["mean_boundary_entropy"]) for r in states.values()])
    entropy_max = float(np.median(entropy))

    positives, negatives = [], []
    for (pid, fi), st in states.items():
        if pid in VAL | TEST or (pid, fi) in manual:
            continue
        contact = float(cov[pid][fi].mean())
        if contact < 0.10:
            negatives.append((pid, fi))
            continue
        if contact < 0.90 or st["valid_for_control"] not in ("True", "true", "1"):
            continue
        area = float(st["mask_area_ratio"])
        if not (0.010 <= area <= 0.060):
            continue
        if float(st["mean_boundary_entropy"]) > entropy_max or tiou[(pid, fi)] < 0.95:
            continue
        positives.append((pid, fi))

    def spread(items, per_session):
        grouped = collections.defaultdict(list)
        for pid, fi in items:
            grouped[pid].append(fi)
        out_items = []
        for pid, idx in sorted(grouped.items()):
            idx.sort()
            out_items += [(pid, i) for i in idx[:: max(1, len(idx) // per_session)][:per_session]]
        return out_items

    pos, neg = spread(positives, pos_per_session), spread(negatives, neg_per_session)
    print("양성 후보 %d → %d 장 / 음성 후보 %d → %d 장" % (len(positives), len(pos), len(negatives), len(neg)))

    pos_dir, neg_dir = DATA / "pseudo_pos", DATA / "pseudo_neg"
    for d in (pos_dir, neg_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    for pid, fi in pos:
        shutil.copyfile(run / "masks" / f"{pid}_S00_{fi:06d}_mask.png", pos_dir / f"{pid}_{fi:05d}.png")
    for pid, fi in neg:
        Image.fromarray(np.zeros((256, 256), np.uint8)).save(neg_dir / f"{pid}_{fi:05d}.png")

    pos_set, neg_set = set(pos), set(neg)
    counts = collections.Counter()
    written = []
    for r in rows:
        r = dict(r)
        key = (r["patient_id"], int(r["frame_index"]))
        r["split"] = ("val" if r["patient_id"] in VAL else "test" if r["patient_id"] in TEST
                      else "train" if r["patient_id"] in labelled_sessions else r["split"])
        if key in manual:
            source = "manual"
        elif key in neg_set:
            r["mask_path"] = "pseudo_neg/%s_%05d.png" % key
            source = "neg"
        elif key in pos_set:
            r["mask_path"] = "pseudo_pos/%s_%05d.png" % key
            source = "pos"
        else:
            r["mask_path"] = ""
            source = "none"
        if source != "none":
            counts[(r["split"], source)] += 1
        written.append(r)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(written)
    for k in sorted(counts, key=str):
        print("  %-5s %-6s %d" % (k[0], k[1], counts[k]))
    print("→ %s" % out)


def export_masks(run: Path, out_dir: Path, manifest_out: Path, area_floor: float,
                 data: Path = DATA, mask_name: str = "%s_S00_%06d_mask.png") -> None:
    """최종 마스크 내보내기: 사람 라벨 우선, 나머지는 의사라벨, 접촉 물리로 걸러낸다.

    거르는 것 세 가지 — 모두 사람 라벨과 모순되지 않는 선에서만:
      * 비접촉 프레임(접촉 비율 < 0.10) → 빈 마스크. 측정 자체가 없다.
      * 결합된 A-line 밖에 절반 넘게 놓인 마스크 → 빈 마스크. 그 각도에는 측정이 없다.
      * 면적이 사람이 그린 최소치(0.0101)보다 작은 조각 → 빈 마스크.
    """
    import cv2
    import sys as _sys

    _sys.path.insert(0, str(HERE.parent.parent / "policy_learning"))
    from rus_policy.bmode import BmodeConverter

    cov = dict(np.load(data / "contact_alines.npz"))
    meta = json.loads((LOGS / "us_imu_20260909_161540" / "session.meta.json").read_text(encoding="utf-8"))
    conv = BmodeConverter(meta, (160, 512), out_size=256)
    idx_map = conv(np.repeat(np.arange(160, dtype=np.uint8)[:, None], 512, axis=1)).astype(np.int16)
    fan = conv(np.full((160, 512), 255, np.uint8)) > 127

    rows = list(csv.DictReader(open(data / "manifest.csv", encoding="utf-8")))
    manual = {(r["patient_id"], int(r["frame_index"])): r["mask_path"] for r in rows if r["mask_path"]}
    states = {}
    for r in csv.DictReader(open(run / "control_states.csv", encoding="utf-8")):
        states[parse_frame_id(r["frame_id"])] = r

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    written, tally = [], collections.Counter()
    for r in rows:
        key = (r["patient_id"], int(r["frame_index"]))
        name = "%s_%05d" % key
        contact = float(cov[key[0]][key[1]].mean())
        st = states[key]
        if key in manual:
            mask = cv2.imread(str(data / manual[key]), cv2.IMREAD_GRAYSCALE)
            source, reason = "manual", ""
        else:
            mask = cv2.imread(str(run / "masks" / (mask_name % key)), cv2.IMREAD_GRAYSCALE)
            source, reason = "pseudo", ""
            binary = mask > 127
            if contact < 0.10:
                reason = "no_contact"
            elif binary.sum() and binary.mean() < area_floor:
                reason = "speck"
            elif contact < 0.90 and binary.sum():
                coupled = cov[key[0]][key[1]][np.clip(idx_map, 0, 159)] & fan
                if (binary & coupled).sum() / binary.sum() < 0.5:
                    reason = "outside_coupled_sector"
            if reason:
                mask = np.zeros((256, 256), np.uint8)
        tally[(source, reason or "kept")] += 1
        cv2.imwrite(str(out_dir / ("%s.png" % name)), mask)
        written.append({**r, "mask_path": "%s/%s.png" % (out_dir.name, name), "label_source": source,
                        "zeroed_reason": reason, "contact_ratio": "%.3f" % contact,
                        "area_ratio": "%.4f" % float((mask > 127).mean()),
                        "neighbour_iou": "", "valid_for_control": st["valid_for_control"]})
    with open(manifest_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(written[0].keys()))
        w.writeheader()
        w.writerows(written)
    empty = sum(1 for r in written if float(r["area_ratio"]) == 0)
    print("%s: %d 장 (사람 %d, 의사라벨 %d)" % (out_dir, len(written), len(manual), len(written) - len(manual)))
    for k in sorted(tally, key=str):
        print("   %-7s %-22s %d" % (k[0], k[1], tally[k]))
    print("   빈 마스크 합계 %d (%.1f %%)  →  %s" % (empty, 100 * empty / len(written), manifest_out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contact", help="A-line 접촉 실측")
    c.add_argument("--out", type=Path, default=DATA / "contact_alines.npz")
    s = sub.add_parser("sets", help="학습 매니페스트 + 의사라벨/음성 표본")
    s.add_argument("--run", type=Path, required=True, help="직전 추론 폴더 (masks/ + control_states.csv)")
    s.add_argument("--out", type=Path, default=DATA / "manifest_train.csv")
    s.add_argument("--pos-per-session", type=int, default=25)
    s.add_argument("--neg-per-session", type=int, default=15)
    e = sub.add_parser("export", help="최종 마스크 + manifest_masked.csv")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--out-dir", type=Path, default=DATA / "masks_final")
    e.add_argument("--manifest-out", type=Path, default=DATA / "manifest_masked.csv")
    e.add_argument("--area-floor", type=float, default=0.008, help="사람이 그린 최소 면적 0.0101 아래")
    e.add_argument("--data", type=Path, default=DATA, help="데이터셋 폴더 (manifest.csv + contact_alines.npz)")
    e.add_argument("--mask-name", default="%s_S00_%06d_mask.png",
                   help="추론 폴더의 마스크 파일명 규칙. --input 으로 돌린 추론은 '%s_%05d_mask.png'")
    a = ap.parse_args()
    if a.cmd == "contact":
        measure_contact(a.out)
    elif a.cmd == "sets":
        build_sets(a.run, a.out, a.pos_per_session, a.neg_per_session)
    else:
        export_masks(a.run, a.out_dir, a.manifest_out, a.area_floor, a.data, a.mask_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
