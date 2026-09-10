#!/usr/bin/env python3
"""Q̂ 진단 — 품질 예측기가 행동을 보는가, 그 방향이 맞는가.

    python3 scripts/diag_quality.py runs/qres2/last.pt
    python3 scripts/diag_quality.py runs/*/last.pt --dataset data/policy_dataset.h5

이 정책의 목적은 시연 궤적 모방이 아니라 "어떤 움직임이 어떤 영상 변화를 만드는가" 이므로,
행동 재구성 오차(mae_*) 가 아니라 **Q̂ 이 행동에 반응하는가** 가 성패를 가른다. 두 가지를 잰다.

민감도
    |Q̂(o, A) − Q̂(o, A′)| / std(Q).  행동 입력을 흔들었을 때 출력이 얼마나 움직이나.
    ⚠️ **단독으로 믿으면 안 된다.** 학습이 덜 된 모델은 무작위 가중치가 모든 입력에 반응하므로
    높게 나온다 (2026-09-10: 초기 epoch 에서 11 % → ep25 에 3 % 로 하락). epoch 을 함께 본다.

순위상관 ρ
    모델의 행동 기여분(잔차 헤드) vs 관측이 설명하지 못한 실제 잔차의 스피어만 상관.
    무작위 가중치는 ρ ≈ 0 이라 위 함정에 걸리지 않는다. **1 순위 지표.**
    귀무가설 하 표준오차 ≈ 1/√(n−1) 이므로 n=575 에서 |ρ| > 0.084 면 p < 0.05.

2026-09-10 기준값 (25 epoch 완료, λ_quality 스윕):
    λ=1e2  ρ=+0.150 (p≈3e-4)   λ=10  +0.135   λ=1e3  +0.106   λ=1  +0.050 (유의하지 않음)
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from _common import setup_logging

from rus_policy.dataset import PolicyH5Dataset
from rus_policy.train import load_policy, to_device


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--dataset", default=None, help="HDF5 (기본: 체크포인트 설정의 paths.dataset)")
    p.add_argument("--split", default="val", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)

    for path in args.checkpoints:
        model, cfg = load_policy(path, device=args.device)
        dev = next(model.parameters()).device
        ds = PolicyH5Dataset(args.dataset or cfg.resolve(cfg.paths.dataset), split=args.split)
        ep = int(torch.load(path, map_location="cpu", weights_only=False)["epoch"])

        d_shuf, d_zero, d_rot, qs, rp, rt = [], [], [], [], [], []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=args.batch_size):
                batch = to_device(batch, dev)
                out = model(batch, use_posterior=True)
                P, pooled = batch["P"], out.memory_pooled
                q = model.predict_quality(pooled, P)
                perm = torch.randperm(P.shape[0], device=dev)
                P_rot = P.clone()
                P_rot[..., 3:] = 0                                  # 회전만 지운다
                d_shuf.append((q - model.predict_quality(pooled, P[perm])).abs().mean(1).cpu().numpy())
                d_zero.append((q - model.predict_quality(pooled, torch.zeros_like(P))).abs().mean(1).cpu().numpy())
                d_rot.append((q - model.predict_quality(pooled, P_rot)).abs().mean(1).cpu().numpy())

                base, resid = model.quality_parts(pooled, P)
                m = batch["Q_valid"].float()
                keep = m.sum(1) > 0
                avg = lambda t: ((t * m).sum(1) / m.sum(1).clamp_min(1))[keep]
                rp.append(avg(resid).cpu().numpy())
                rt.append((avg(batch["Q"]) - avg(base)).cpu().numpy())
                qs.append(batch["Q"][batch["Q_valid"]].cpu().numpy())

        cat = lambda xs: np.concatenate(xs)
        s = cat(qs).std()
        rp_, rt_ = cat(rp), cat(rt)
        rho = spearman(rp_, rt_)
        se = 1.0 / np.sqrt(max(len(rp_) - 1, 1))
        pct = lambda v: f"{cat(v).mean() / s * 100:5.1f}%"
        print(f"{path}")
        print(f"  epoch {ep}  n={len(rp_)}  잔차분해={cfg.model.quality_residual}  Q std={s:.4f}")
        print(f"  민감도   뒤섞기 {pct(d_shuf)}   행동0 {pct(d_zero)}   회전만0 {pct(d_rot)}")
        print(f"  순위상관 ρ = {rho:+.4f}   (z = {rho/se:+.1f}, {'유의' if abs(rho) > 1.96*se else '유의하지 않음'})")
        print(f"  행동기여 std {rp_.std():.5f}  vs  실제잔차 std {rt_.std():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
