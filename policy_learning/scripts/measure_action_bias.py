#!/usr/bin/env python3
"""Q̂ 의 관측과 무관한 행동 기울어짐을 재서 체크포인트 옆에 저장한다.

    python3 scripts/measure_action_bias.py runs/qres2_ep25.pt
    python3 scripts/measure_action_bias.py runs/qres2_ep25.pt --device cpu   # 로봇 실험 중

왜: 선택은 후보 M 개 중 Σ_k Q̂ 가 가장 큰 것을 고르는데, Q̂ 의 행동 기울기는 부호가 관측과
무관하게 거의 같다 (2026-09-12: θx 98 % · θz 95 %). 그 상수가 만드는 점수 차(0.012)가 후보
사이의 실제 차(중앙 0.021)와 맞먹어, 64 개 중 최댓값은 거의 언제나 같은 방향이 된다 —
디코더가 낸 후보는 θx 양수 51 % 로 고른데 고른 뒤에는 32 % 다.

재서 빼면 선택이 한쪽으로 치우치지 않는다. **방향을 알게 되는 것은 아니다** — 남은 신호
자체가 약하다 (A 대 −A 판별 52~56 %). 그것은 개입 데이터로 다시 학습해야 한다.

저장 위치는 ``<체크포인트>.action_bias.json`` 이고, run_policy 가 그대로 읽는다.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from _common import setup_logging

from rus_policy.dataset import PolicyH5Dataset
from rus_policy.debias import (MIN_SIGN_AGREEMENT, bias_report, default_bias_path,
                               gate_by_agreement, save_action_bias)
from rus_policy.model import AXIS_NAMES
from rus_policy.train import load_policy, to_device


def _selected_balance(model, loader, bias, gamma, M, max_samples):
    """고른 후보의 θx 부호 분포 — 쏠림이 실제로 사라졌는지 보는 직접 증거."""
    dev = next(model.parameters()).device
    out, n = [], 0
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, dev)
            torch.manual_seed(0)
            sel = model.select_action(batch, n_samples=M, gamma=gamma,
                                      prev_dy=batch["vec"][:, 1] * batch["vec"][:, 3],
                                      action_bias=bias)
            out.append(sel["net"][:, 3:6].cpu().numpy())
            n += sel["net"].shape[0]
            if max_samples and n >= max_samples:
                break
    return np.concatenate(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--dataset", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="auto",
                   help="⚠️ 로봇 실험 중인 기계에서는 cpu — 지각이 같은 GPU 를 쓴다")
    p.add_argument("--samples", type=int, default=320, help="기울기를 잴 관측 수")
    p.add_argument("--z-samples", type=int, default=64)
    p.add_argument("--out", default=None, help="기본: <체크포인트>.action_bias.json")
    p.add_argument("--min-agreement", type=float, default=MIN_SIGN_AGREEMENT,
                   help="이 비율 이상 부호가 같은 축만 뺀다. 낮추면 신호까지 지운다")
    p.add_argument("--dry-run", action="store_true", help="재기만 하고 저장하지 않는다")
    args = p.parse_args()
    setup_logging()

    model, cfg = load_policy(args.checkpoint, device=args.device)
    ds_path = args.dataset or cfg.resolve(cfg.paths.dataset)
    ds = PolicyH5Dataset(ds_path, split=args.split, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)

    rep = bias_report(model, loader, max_samples=args.samples)
    # 부호가 관측마다 뒤집히는 축은 빼지 않는다 — 그 평균은 상수가 아니라 신호의 평균이다.
    bias = gate_by_agreement(rep["mean"], rep["sign_agreement"], args.min_agreement)
    print(f"\n=== Q̂ 의 행동 기울기 ({rep['n_obs']} 관측, {args.split}) ===")
    print(f"{'축':>5} {'평균':>12} {'표준편차':>12} {'부호 일치':>10} {'처리':>8}")
    for i, nm in enumerate(AXIS_NAMES):
        used = "뺀다" if abs(bias[i]) > 0 else "그대로"
        print(f"{nm:>5} {rep['mean'][i]:>12.5f} {rep['std'][i]:>12.5f} "
              f"{rep['sign_agreement'][i]:>9.0%} {used:>8}")
    print("\n부호 일치가 높을수록 관측과 무관한 상수다 — 그만큼이 순위를 정하고 있었다.")

    gamma = cfg.train.gamma_mode_consistency
    before = _selected_balance(model, loader, None, gamma, args.z_samples, args.samples)
    after = _selected_balance(model, loader, bias, gamma, args.z_samples, args.samples)
    print(f"\n=== 고른 후보의 방향 분포 (γ={gamma:g}) ===")
    print(f"{'축':>5} {'빼기 전 양수':>12} {'뺀 뒤 양수':>12}")
    for i, nm in enumerate(AXIS_NAMES[3:]):
        print(f"{nm:>5} {np.mean(before[:, i] > 0):>11.0%} {np.mean(after[:, i] > 0):>11.0%}")
    print("  (디코더가 낸 후보 자체는 θx 양수 51 % 로 고르다 — 50 % 에 가까워야 맞다)")

    if args.dry_run:
        print("\n--dry-run: 저장하지 않았다.")
        return 0
    out = args.out or default_bias_path(args.checkpoint)
    save_action_bias(out, bias, rep)
    print(f"\n저장 → {out}")
    print("run_policy 가 --action-bias auto (기본) 로 이 파일을 읽는다.")
    print(json.dumps({"action_bias": [round(v, 6) for v in bias]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
