#!/usr/bin/env python3
"""학습된 policy 를 split 하나에서 평가 — 실행시 선택 경로(select_action)로.

    python scripts/eval_policy.py runs/exp1/best.pt --dataset data/policy_dataset.h5 --split test --csv eval.csv

출력: 6 축 순변위 MAE·부호 정확도, σ 정규화 오차(nmae), 모드 마진, (csv) 샘플별 예측·라벨.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from _common import setup_logging

from rus_policy.dataset import PolicyH5Dataset
from rus_policy.model import AXIS_NAMES, AXIS_UNITS, Y_AXIS
from rus_policy.train import load_policy, to_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset", default=None, help="HDF5 (기본: 체크포인트 설정의 paths.dataset)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--z-samples", type=int, default=None, help="CVAE z 후보 수 (기본: train.z_samples_eval)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--csv", default=None, help="샘플별 결과 CSV")
    args = parser.parse_args()
    setup_logging()

    model, cfg = load_policy(args.checkpoint, device=args.device)
    device = next(model.parameters()).device
    ds_path = Path(args.dataset) if args.dataset else cfg.resolve(cfg.paths.dataset)
    ds = PolicyH5Dataset(ds_path, split=args.split, augment=False)
    if len(ds) == 0:
        raise SystemExit(f"{ds_path}: split={args.split} 샘플이 없습니다")
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    M = args.z_samples or cfg.train.z_samples_eval

    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            sel = model.select_action(batch, n_samples=M, gamma=cfg.train.gamma_mode_consistency,
                                      prev_dy=batch["vec"][:, 1] * batch["vec"][:, 3])
            net_hat = sel["net"].cpu().numpy()
            net_lab = batch["P"][:, -1].cpu().numpy()
            sig = batch["sigma_net"].cpu().numpy()
            margin = sel["mode_margin"].cpu().numpy()
            if margin.ndim > 1:
                margin = margin[:, Y_AXIS]
            Qh = sel["Q_hat"].mean(1).cpu().numpy()
            for b in range(net_hat.shape[0]):
                row: dict[str, float] = {"index": int(batch["index"][b])}
                for i, nm in enumerate(AXIS_NAMES):
                    row[f"d{nm}_hat"] = float(net_hat[b, i])
                    row[f"d{nm}"] = float(net_lab[b, i])
                    row[f"sigma_{nm}"] = float(sig[b, i])
                row["mode_margin"] = float(margin[b])
                row["Q_hat_mean"] = float(Qh[b])
                rows.append(row)
    hat = np.array([[r[f"d{nm}_hat"] for nm in AXIS_NAMES] for r in rows])
    lab = np.array([[r[f"d{nm}"] for nm in AXIS_NAMES] for r in rows])
    sig = np.array([[r[f"sigma_{nm}"] for nm in AXIS_NAMES] for r in rows])
    err = np.abs(hat - lab)
    summary = {
        "checkpoint": args.checkpoint, "split": args.split, "n": len(rows), "head": cfg.model.head,
        # σ 정규화 오차 — 학습 로그의 select_nmae 와 같은 정의. 축 단위가 달라도 하나로 비교된다.
        "nmae": float((err / np.maximum(sig, 1e-6)).mean()),
        "mode_margin_mean": float(np.nanmean([r["mode_margin"] for r in rows])),
        "Q_hat_mean": float(np.mean([r["Q_hat_mean"] for r in rows])),
    }
    for i, (nm, un) in enumerate(zip(AXIS_NAMES, AXIS_UNITS)):
        big = np.abs(lab[:, i]) > sig[:, i]
        summary[f"mae_{nm}_{un}"] = float(err[:, i].mean())
        summary[f"median_err_{nm}_{un}"] = float(np.median(err[:, i]))
        summary[f"sign_acc_{nm}"] = (float((np.sign(hat[big, i]) == np.sign(lab[big, i])).mean())
                                     if big.any() else float("nan"))
        summary[f"within_2sigma_{nm}"] = float((err[:, i] < 2 * sig[:, i]).mean())
        summary[f"sigma_{nm}_median"] = float(np.median(sig[:, i]))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"샘플별 결과: {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
