#!/usr/bin/env python3
"""ACT policy 학습.

    python scripts/train_policy.py                                  # configs/policy_default.yaml
    python scripts/train_policy.py --dataset data/policy_dataset.h5 --out runs/exp1
    python scripts/train_policy.py --set model.head=discrete --set train.epochs=100
    python scripts/train_policy.py --resume runs/exp1/last.pt

출력: <out>/config.yaml, metrics.jsonl, last.pt, best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import add_config_args, config_from_args

from rus_policy.train import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--dataset", default=None, help="HDF5 (기본: paths.dataset)")
    parser.add_argument("--out", default=None, help="출력 디렉터리 (기본: paths.output_dir)")
    parser.add_argument("--resume", default=None, help="이어서 학습할 체크포인트")
    args = parser.parse_args()
    cfg = config_from_args(args)

    trainer = Trainer(cfg, dataset_path=Path(args.dataset) if args.dataset else None,
                      output_dir=Path(args.out) if args.out else None)
    if args.resume:
        trainer.resume(Path(args.resume))
    history = trainer.fit()
    last = history[-1] if history else {}
    print(f"\n완료: {trainer.out}  best val {trainer.best:.4f}")
    for key in ("val/mae_x_mm", "val/mae_y_mm", "val/mae_th_deg", "val/vy_sign_acc",
                "val/mode_collapse_vy_std", "val/mode_margin", "val/select_vy_sign_acc"):
        if key in last:
            print(f"  {key:28s} {last[key]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
