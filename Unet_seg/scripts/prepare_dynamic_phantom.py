#!/usr/bin/env python3
"""다이나믹 팬텀 프레임 → 라벨링 큐 (최소 장수).

r6 체크포인트는 2026-09-09/10 의 팬텀 A 로만 학습됐다. 09-11 부터 쓰는 다이나믹
팬텀에서는 방광이 또렷한 프레임에도 신경망이 **확률 0.000** 을 낸다 (문턱 문제가
아니라 도메인 차이다). 그 구멍을 메우는 데 필요한 최소 라벨을 고른다.

고르는 규칙 셋:
  1. **실패한 프레임만.** 이미 맞히는 프레임을 더 라벨해 봐야 배울 것이 없다.
     실패 여부는 실행 중에 그 모델이 남긴 ``has_mask`` 로 판정한다 (같은 모델·같은
     후처리라 다시 추론할 필요가 없다).
  2. **모양이 겹치면 하나만.** 32x32 로 줄여 k-means 로 묶고 군집마다 중심에 가장
     가까운 한 장만 뽑는다. 프리핸드 영상은 이웃 프레임이 거의 같아서, 이 단계가
     없으면 같은 그림을 40번 라벨하게 된다.
  3. **음성도 섞는다.** 양성만 추가하면 모델이 "없다" 를 못 하게 된다 — 팬텀 A
     라운드에서 비접촉 오탐이 315 → 631 로 뛴 그 실패다 (METHODS_NOTE §5.3).
     검출된 프레임에서도 일부를 넣어 유무 판단의 기준을 유지한다.

    python3 Unet_seg/scripts/prepare_dynamic_phantom.py --n 40
    python3 Unet_seg/scripts/mask_editor/server.py --data data/phantom_dyn --queue
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "policy_learning" / "runs"
OUT = ROOT / "Unet_seg" / "data" / "phantom_dyn"   # --out-dir 로 덮는다


def coupling_mask(size):
    """부채꼴 ROI 의 **깊은 영역**. 위쪽 35 % 는 트랜스듀서 링다운이라 공기 중에도 밝다."""
    import sys
    sys.path.insert(0, str(ROOT / "Unet_seg"))
    from rus_perception.control.roi import RoiConfig, build_roi_mask
    from rus_perception.utils.config import load_config as load_unet_config
    cfg = load_unet_config(str(ROOT / "Unet_seg/configs/phantom_c10ur_finetune.yaml"))
    rd = dict(cfg.section("control").get("roi") or {})
    if rd.get("path") and not Path(str(rd["path"])).is_absolute():
        rd["path"] = str(ROOT / "Unet_seg" / str(rd["path"]))
    roi = np.asarray(build_roi_mask(size, RoiConfig.from_dict(rd))) > 0
    roi[: int(0.35 * size[0])] = False
    return roi


def collect(since: datetime.date) -> list[dict]:
    """frames.npz 가 있는 실행에서 (프레임, 그때의 has_mask) 를 모은다."""
    items = []
    for fp in sorted(glob.glob(str(RUNS / "**" / "frames.npz"), recursive=True)):
        d = Path(fp).parent
        if datetime.date.fromtimestamp(os.path.getmtime(fp)) < since:
            continue
        sp = d / "states.npz"
        if not sp.is_file():
            continue
        fz, sz = np.load(fp), np.load(sp)
        frames, ft = fz["frames"], fz["t"]
        state, st = sz["state"], sz["t"]
        run = str(d.relative_to(RUNS)).replace("/", "_")
        for i in range(len(frames)):
            j = int(np.argmin(np.abs(st - ft[i])))
            items.append({"run": run, "index": i, "t": float(ft[i] - ft[0]),
                          "has_mask": bool(state[j, 0] > 0.5), "frame": frames[i]})
    return items


def _kmeans(x: np.ndarray, k: int, seed: int, iters: int = 25):
    """k-means++ 초기화의 작은 구현. sklearn 을 이 저장소에 들이지 않으려고 직접 둔다."""
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(k - 1):
        d = np.min(((x[:, None, :] - np.stack(centers)[None]) ** 2).sum(-1), axis=1)
        total = d.sum()
        centers.append(x[rng.choice(len(x), p=d / total) if total > 0 else rng.integers(len(x))])
    c = np.stack(centers)
    labels = np.zeros(len(x), int)
    for _ in range(iters):
        labels = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        moved = False
        for j in range(k):
            m = labels == j
            if m.any():
                new = x[m].mean(0)
                moved |= not np.allclose(new, c[j])
                c[j] = new
        if not moved:
            break
    return labels, c


def pick(items: list[dict], n: int, min_gap_s: float, seed: int) -> list[dict]:
    """군집당 한 장. 같은 실행 안에서는 min_gap_s 보다 붙은 것을 다시 뽑지 않는다."""
    x = np.stack([np.asarray(Image.fromarray(it["frame"]).resize((32, 32)), np.float32).ravel()
                  for it in items])
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    k = min(n, len(items))
    labels, centers = _kmeans(x, k, seed)
    chosen: list[dict] = []
    for c in range(k):
        idx = np.flatnonzero(labels == c)
        if not len(idx):
            continue
        order = idx[np.argsort(np.linalg.norm(x[idx] - centers[c], axis=1))]
        for i in order:                       # 중심에 가까운 순서로, 시간이 겹치지 않는 첫 장
            it = items[int(i)]
            if all(not (o["run"] == it["run"] and abs(o["t"] - it["t"]) < min_gap_s)
                   for o in chosen):
                chosen.append(it)
                break
    return chosen


def pick_coupled(items: list[dict], args) -> list[dict]:
    """닿아 있는 프레임을, **지금 배포된 모델이 뭐라고 하는지**로 층화해 고른다.

    고치려는 것은 "닿았는데 방광이 없는" 프레임의 오탐이다. 그 오탐은 정의상 모델이
    마스크를 내는 프레임에 있으므로 그쪽을 절반 넘게 담는다. 나머지는 모델이 아무것도
    못 낸 결합 프레임 — 그중 진짜로 방광이 있는 것이 있으면 검출 쪽도 같이 고쳐진다.

    어느 쪽이 참인지는 이 함수가 모른다. 그것을 사람이 정하는 것이 라벨링이다.
    """
    import sys
    sys.path.insert(0, str(ROOT / "Unet_seg"))
    from rus_perception.inference.predictor import Predictor, PredictorConfig
    from rus_perception.utils.config import load_config as load_unet_config
    from rus_perception.data.io import resize_image

    cfg = load_unet_config(str(ROOT / "Unet_seg/configs/phantom_c10ur_finetune.yaml"))
    size = tuple(int(v) for v in cfg.section("data")["image_size"])
    deep = coupling_mask(size)
    pred = Predictor.from_checkpoint(
        str(ROOT / args.checkpoint), model_config=cfg.section("model"),
        predictor_config=PredictorConfig(
            input_size=size,
            intensity_normalization=str(cfg.get("data.intensity_normalization", "zero_one")),
            normalization_stats=cfg.get("data.normalization_stats"),
            device="cuda", restore_original_size=False))

    with_mask, without = [], []
    for it in items:
        img = np.asarray(it["frame"], np.float32)
        if img[deep].mean() < args.coupling_min:
            continue                       # 안 닿았다 — 그 구간은 이미 0/33 로 깨끗하다
        prob, _ = pred.predict_probability(resize_image(img / 255.0, size))
        area = (prob > 0.5).sum() / prob.size
        it["model_area"] = float(area)
        (with_mask if area > 0.001 else without).append(it)
    print(f"결합 프레임 — 모델이 마스크를 냄 {len(with_mask)}, 안 냄 {len(without)}")
    # 검출된 쪽의 몫. 오탐을 겨냥할 때는 크게(0.65), 놓치는 경계를 겨냥할 때는 작게 준다.
    n_pos = int(round(args.n * args.detected_share))
    out = pick(with_mask, min(n_pos, len(with_mask)), args.min_gap_s, args.seed) if n_pos else []
    out += pick(without, args.n - len(out), args.min_gap_s, args.seed + 1)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=40, help="라벨할 총 장수")
    p.add_argument("--detected-share", type=float, default=0.3,
                   help="이미 검출되는 프레임의 몫 — 유무 판단을 유지하기 위한 대조")
    p.add_argument("--since", default="2026-09-11", help="이 날짜 이후의 실행만")
    p.add_argument("--min-gap-s", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", default="missed", choices=["missed", "coupled", "random"],
                   help="missed=모델이 놓친 프레임 (검출을 고친다) · "
                        "coupled=닿아 있는 프레임 (닿았는데 방광이 없을 때의 오탐을 고친다) · "
                        "random=**균등 무작위** (성능을 재는 감사용 — 학습에 쓰지 말 것)")
    p.add_argument("--split", default="train", choices=["train", "val", "test"],
                   help="manifest 에 박을 split. 감사 세트는 test 로 박아 둔다 — "
                        "표시가 없으면 다음 라운드에서 조용히 학습에 들어간다")
    p.add_argument("--checkpoint", default="Unet_seg/checkpoints/phantom_dyn_r1/last.pt",
                   help="coupled 모드에서 층화에 쓸 모델 — 지금 배포된 것을 쓴다")
    p.add_argument("--coupling-min", type=float, default=78.0,
                   help="이 밝기(0-255) 이상이면 닿아 있다고 본다. 라벨 양성의 최소값이 78.4 였다")
    p.add_argument("--out-dir", default="phantom_dyn",
                   help="Unet_seg/data 아래 폴더 이름")
    args = p.parse_args()

    global OUT
    OUT = ROOT / "Unet_seg" / "data" / args.out_dir
    since = datetime.date.fromisoformat(args.since)
    items = collect(since)
    missed = [it for it in items if not it["has_mask"]]
    found = [it for it in items if it["has_mask"]]
    print(f"후보 {len(items)} 장 — 놓친 것 {len(missed)}, 검출된 것 {len(found)}")

    if args.mode == "random":
        # 고르지 않는 것이 요점이다. 어떤 기준으로든 추리면 그 순간 감사가 아니게 된다.
        rng = np.random.default_rng(args.seed)
        picks = [items[i] for i in rng.choice(len(items), size=min(args.n, len(items)),
                                              replace=False)]
    elif args.mode == "coupled":
        picks = pick_coupled(items, args)
    else:
        n_found = int(round(args.n * args.detected_share))
        picks = pick(missed, args.n - n_found, args.min_gap_s, args.seed)
        if found and n_found:
            picks += pick(found, n_found, args.min_gap_s, args.seed)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    man, queue = [], []
    for it in sorted(picks, key=lambda r: (r["run"], r["index"])):
        name = f"{it['run']}_{it['index']:05d}.png"
        Image.fromarray(it["frame"]).save(OUT / "images" / name)
        man.append({"patient_id": it["run"], "sequence_id": "S00", "frame_index": it["index"],
                    # 경로는 **이 폴더 기준**이다. 편집기(server.py)도 학습(load_manifest)도
                    # manifest 가 있는 디렉터리를 root 로 잡는다.
                    "image_path": f"images/{name}", "mask_path": f"masks/{name}",
                    "timestamp": round(it["t"], 3), "split": args.split})
        queue.append({"patient_id": it["run"], "frame_index": it["index"],
                      "image_path": f"images/{name}", "mask_path": f"masks/{name}",
                      "labeled": 0, "timestamp": round(it["t"], 3),
                      "model_has_mask": int(it.get("model_area", 0.0) > 0.001
                                            if "model_area" in it else it["has_mask"]),
                      "model_area": round(float(it.get("model_area", float("nan"))), 4)})

    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(man[0]))
        w.writeheader(); w.writerows(man)
    with open(OUT / "label_queue.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(queue[0]))
        w.writeheader(); w.writerows(queue)

    runs = sorted({r["patient_id"] for r in man})
    print(f"큐 {len(man)} 장 (놓친 것 {sum(1 for q in queue if not q['model_has_mask'])}, "
          f"검출된 것 {sum(q['model_has_mask'] for q in queue)}) · 실행 {len(runs)} 개")
    print(f"→ {OUT}/label_queue.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
