#!/usr/bin/env python3
"""C10UR 팬텀 수집 세션 (imu_bench/logs/us_imu_*) → 이 저장소의 manifest 데이터셋 + 수동 라벨링 큐.

    python scripts/prepare_phantom_sessions.py \
        --sessions ../policy_learning/data/sessions.csv \
        --output-dir data/phantom_c10ur --per-session 15

만드는 것
  data/phantom_c10ur/images/<session>_<frame:05d>.png   부채꼴 B-mode 256×256 — policy_learning 의
                                                        BmodeConverter 와 **같은 변환** (학습·실행 입력 일치)
  data/phantom_c10ur/manifest.csv                       patient_id = 세션 (split 단위), sequence_id = S00,
                                                        frame_index, timestamp, mask_path (라벨 있으면 채움)
  data/phantom_c10ur/label_queue.csv                    사람이 그릴 프레임 목록: 세션마다 --per-session 장,
                                                        시간상 고르게, 정지 구간(sync.npz still) 우선
  data/phantom_c10ur/to_label/<session>/<...>.png       큐 프레임 복사본 (편집기·autolabel 입력)

라벨은 masks/<same name>.png (0/255) 로 두면 --refresh 로 manifest 의 mask_path 가 채워진다:
    python scripts/prepare_phantom_sessions.py --output-dir data/phantom_c10ur --refresh

세션이 늘면 같은 명령을 다시 돌린다 — 이미 있는 PNG 는 건너뛴다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (str(REPO / "policy_learning"), str(REPO / "imu_bench" / "host")):
    if p not in sys.path:
        sys.path.insert(0, p)


def read_sessions(manifest: Path) -> list[dict]:
    rows = []
    with open(manifest, encoding="utf-8") as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            d = Path(r["session_dir"])
            if not d.is_absolute():
                d = (manifest.parent / d).resolve()
            rows.append({"dir": d, "split": (r.get("split") or "").strip(), "subject": r.get("subject", "")})
    return rows


def load_session_frames(session_dir: Path):
    meta = json.loads((session_dir / "session.meta.json").read_text(encoding="utf-8"))
    shape = tuple(meta["us"].get("frame_shape") or (256, 256))
    frames = np.memmap(str(session_dir / "us_frames.bin"), np.uint8, "r").reshape(-1, *shape)
    t = np.loadtxt(session_dir / "us_index.csv", delimiter=",", skiprows=1, usecols=0, ndmin=1)
    still = None
    sync = session_dir / "sync.npz"
    if sync.is_file():
        z = np.load(sync)
        if "still" in z:
            still = np.asarray(z["still"], bool)
    return meta, frames, t, still


def pick_label_frames(t: np.ndarray, still, n: int, min_gap_s: float = 3.0) -> list[int]:
    """시간상 고르게 n 장. 각 슬롯 안에서는 정지 프레임을 우선 (경계 흔들림 없는 깨끗한 라벨)."""
    if len(t) == 0 or n <= 0:
        return []
    edges = np.linspace(t[0], t[-1], n + 1)
    picks = []
    for k in range(n):
        lo, hi = edges[k], edges[k + 1]
        idx = np.where((t >= lo) & (t < hi))[0] if k < n - 1 else np.where((t >= lo) & (t <= hi))[0]
        if idx.size == 0:
            continue
        if still is not None and still[idx].any():
            cand = idx[still[idx]]
            # 정지 구간의 한가운데 (경계에서 멀리)
            i = cand[len(cand) // 2]
        else:
            i = idx[len(idx) // 2]
        if picks and t[i] - t[picks[-1]] < min_gap_s:
            continue
        picks.append(int(i))
    return picks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", default=str(REPO / "policy_learning" / "data" / "sessions.csv"))
    ap.add_argument("--output-dir", default=str(HERE.parent / "data" / "phantom_c10ur"))
    ap.add_argument("--per-session", type=int, default=15, help="세션당 수동 라벨 프레임 수")
    ap.add_argument("--every", type=int, default=1, help="이미지로 내보낼 프레임 간격 (1 = 전부)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--refresh", action="store_true", help="이미지 내보내기 없이 masks/ 를 보고 manifest 만 갱신")
    args = ap.parse_args()

    from PIL import Image

    out = Path(args.output_dir)
    img_dir, mask_dir, queue_dir = out / "images", out / "masks", out / "to_label"
    for d in (img_dir, mask_dir, queue_dir):
        d.mkdir(parents=True, exist_ok=True)

    sessions = read_sessions(Path(args.sessions))
    rows: list[dict] = []
    queue: list[dict] = []
    n_new = 0
    for s in sessions:
        sd: Path = s["dir"]
        if not (sd / "us_frames.bin").is_file():
            print("건너뜀 (프레임 없음):", sd.name)
            continue
        meta, frames, t, still = load_session_frames(sd)
        if len(frames) < 100:
            print("건너뜀 (프레임 %d):" % len(frames), sd.name)
            continue
        conv = None
        if not args.refresh:
            from rus_policy.bmode import BmodeConverter
            conv = BmodeConverter(meta, frames.shape[1:], out_size=args.size)
        picks = set(pick_label_frames(t, still, args.per_session))
        for i in range(0, len(frames), args.every):
            name = "%s_%05d.png" % (sd.name, i)
            ip = img_dir / name
            if conv is not None and not ip.is_file():
                Image.fromarray(conv(frames[i])).save(ip)
                n_new += 1
            mp = mask_dir / name
            rows.append({
                "patient_id": sd.name, "sequence_id": "S00", "frame_index": i,
                "image_path": "images/" + name,
                "mask_path": ("masks/" + name) if mp.is_file() else "",
                "timestamp": "%.3f" % (t[i] - t[0]), "split": s["split"] or "train",
                "still": int(bool(still[i])) if still is not None else "",
            })
            if i in picks:
                queue.append({"patient_id": sd.name, "frame_index": i, "image_path": "images/" + name,
                              "mask_path": "masks/" + name, "labeled": int(mp.is_file()),
                              "timestamp": "%.1f" % (t[i] - t[0])})
                qd = queue_dir / sd.name
                qd.mkdir(exist_ok=True)
                if ip.is_file() and not (qd / name).is_file():
                    (qd / name).write_bytes(ip.read_bytes())
        print("%s: 프레임 %d, 라벨 큐 %d, 정지 %s" % (
            sd.name, len(frames), len(picks), "%.0f %%" % (100 * still.mean()) if still is not None else "-"))

    fields = ["patient_id", "sequence_id", "frame_index", "image_path", "mask_path", "timestamp", "split", "still"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    with open(out / "label_queue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "frame_index", "image_path", "mask_path", "labeled", "timestamp"])
        w.writeheader(); w.writerows(queue)
    n_lab = sum(1 for r in rows if r["mask_path"])
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], [0, 0])
        by_split[r["split"]][0] += 1
        by_split[r["split"]][1] += bool(r["mask_path"])
    print("\nmanifest: %s  프레임 %d (새 PNG %d)  라벨 있음 %d" % (out / "manifest.csv", len(rows), n_new, n_lab))
    for k, (n, l) in sorted(by_split.items()):
        print("  %-5s 프레임 %5d  라벨 %d" % (k, n, l))
    print("라벨 큐: %s  %d 장 (그린 것 %d)" % (out / "label_queue.csv", len(queue), sum(q["labeled"] for q in queue)))
    print("편집기:  python scripts/mask_editor/server.py --data %s --queue" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
