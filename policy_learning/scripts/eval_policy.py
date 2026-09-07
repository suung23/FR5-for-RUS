#!/usr/bin/env python3
"""학습된 policy 를 split 하나에서 평가 — 실행시 선택 경로(select_action)로.

    python scripts/eval_policy.py runs/exp1/best.pt --dataset data/policy_dataset.h5 --split test --csv eval.csv

출력: 축별 순변위 MAE, Δy 부호 정확도, 모드 마진, (csv) 샘플별 예측·라벨.
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
                margin = margin[:, 1]
            Qh = sel["Q_hat"].mean(1).cpu().numpy()
            for b in range(net_hat.shape[0]):
                rows.append({"index": int(batch["index"][b]), "dx_hat": net_hat[b, 0], "dy_hat": net_hat[b, 1],
                             "dth_hat": net_hat[b, 2], "dx": net_lab[b, 0], "dy": net_lab[b, 1], "dth": net_lab[b, 2],
                             "sigma_xy": sig[b, 0], "sigma_th": sig[b, 2], "mode_margin": float(margin[b]),
                             "Q_hat_mean": float(Qh[b])})
    hat = np.array([[r["dx_hat"], r["dy_hat"], r["dth_hat"]] for r in rows])
    lab = np.array([[r["dx"], r["dy"], r["dth"]] for r in rows])
    sig = np.array([[r["sigma_xy"], r["sigma_xy"], r["sigma_th"]] for r in rows])
    err = np.abs(hat - lab)
    big_y = np.abs(lab[:, 1]) > sig[:, 1]
    summary = {
        "checkpoint": args.checkpoint, "split": args.split, "n": len(rows), "head": cfg.model.head,
        "mae_x_mm": float(err[:, 0].mean()), "mae_y_mm": float(err[:, 1].mean()), "mae_th_deg": float(err[:, 2].mean()),
        "median_err_x_mm": float(np.median(err[:, 0])), "median_err_y_mm": float(np.median(err[:, 1])),
        "vy_sign_acc": float((np.sign(hat[big_y, 1]) == np.sign(lab[big_y, 1])).mean()) if big_y.any() else float("nan"),
        "vx_sign_acc": float((np.sign(hat[:, 0]) == np.sign(lab[:, 0]))[np.abs(lab[:, 0]) > sig[:, 0]].mean()),
        "within_2sigma_x": float((err[:, 0] < 2 * sig[:, 0]).mean()),
        "mode_margin_mean": float(np.nanmean([r["mode_margin"] for r in rows])),
        "label_sigma_xy_median": float(np.median(sig[:, 0])),
    }
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
