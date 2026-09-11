#!/usr/bin/env python3
"""품질 헤드만 다른 라벨로 다시 학습하고, 선택을 바꿀 만큼 달라졌는지 잰다.

    python3 scripts/retrain_quality.py runs/qres2_ep25.pt --out runs/qarea_head
    python3 scripts/retrain_quality.py runs/qres2_ep25.pt --out runs/qarea_head --eval-only

**체크포인트 자신의 설정에서 출발한다.** 기본 yaml 로 시작하면 모델 구조(6 자유도·잔차 Q̂)와
손실 가중치(λ_Q)가 어긋나 가중치 로드가 깨지거나 다른 학습이 된다.

학습하는 것은 quality_head · quality_base (전체의 3 %) 뿐이다. 나머지는 얼리고 eval 모드로
묶으므로 **행동 후보는 그대로이고 줄 세우는 기준만 바뀐다** — 재학습 전후를 그대로 비교할 수
있는 이유다 (tests/test_quality_retrain.py).

끝나면 옛 헤드와 새 헤드를 같은 test 관측에서 잰다 (rus_policy.quality_eval):
라벨 적합 · 행동 효과 상관 · 방향별 Q̂ 차 · z 재추첨 일치율.

⚠️ 2026-09-11 사전 판정: 이 데이터(IMU 프리핸드 스윕)에서 미래 면적 변화에 행동이 보태는
설명력은 ΔR² ≈ 0 이었다 — 움직임의 **크기**는 면적 변화의 크기와 ρ 0.55 로 묶이지만 **방향**은
축별 |ρ| < 0.13. 조작자가 보면서 고른 움직임이라 방향의 효과가 선택과 뒤섞여 있다. 이 스크립트가
"방향을 가르지 못한다" 를 내면 그것이 예상된 결과다. 파일럿의 위약(무작위 방향) 에피소드가
그 뒤섞임을 끊는 데이터다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import setup_logging

from rus_policy.dataset import PolicyH5Dataset
from rus_policy.quality_eval import evaluate_quality_head, verdict
from rus_policy.train import Trainer, load_policy


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", help="출발 체크포인트 (가중치와 설정을 모두 여기서 가져온다)")
    p.add_argument("--out", required=True)
    p.add_argument("--target", default="Q_area", choices=["Q_area", "Q"])
    p.add_argument("--dataset", default=None, help="HDF5 (기본: 체크포인트 설정의 paths.dataset)")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4, help="헤드만 학습하므로 본 학습보다 크게")
    p.add_argument("--device", default="auto",
                   help="⚠️ 로봇 실험 중인 기계에서는 cpu — 지각이 같은 GPU 를 쓴다")
    p.add_argument("--eval-samples", type=int, default=500, help="평가에 쓸 test 관측 수 상한")
    p.add_argument("--eval-only", action="store_true", help="학습 없이 --out/best.pt 를 평가만")
    p.add_argument("--cached", action="store_true",
                   help="얼린 인코더의 특징을 한 번만 뽑고 헤드만 학습한다 — CPU 에서 수 분. "
                        "Trainer 경로와 손실식이 같다 (tests: test_cached_loss_matches_the_trainer_loss). "
                        "증강은 쓰지 않는다")
    args = p.parse_args()
    setup_logging()

    old_model, cfg = load_policy(args.checkpoint, device=args.device)
    ds_path = Path(args.dataset) if args.dataset else cfg.resolve(cfg.paths.dataset)
    out = Path(args.out)

    if args.cached and not args.eval_only:
        from rus_policy.quality_eval import cache_features, train_head_cached
        out.mkdir(parents=True, exist_ok=True)
        work, _ = load_policy(args.checkpoint, device=args.device)
        mk = lambda split: torch.utils.data.DataLoader(
            PolicyH5Dataset(ds_path, split=split, augment=False), batch_size=16, shuffle=False)
        print("특징 캐시 — 인코더를 train·val 에 한 번씩만 돌린다")
        tr_f, va_f = cache_features(work, mk("train"), args.target), cache_features(work, mk("val"), args.target)
        print(f"  train {tr_f['pooled'].shape[0]} · val {va_f['pooled'].shape[0]} · "
              f"라벨 유효 {float(tr_f['valid'].float().mean()):.0%}")
        res = train_head_cached(work, tr_f, va_f, epochs=args.epochs, lr=args.lr)
        cfg.loss.quality_target = args.target
        cfg.train.init_from = str(Path(args.checkpoint).resolve())
        cfg.train.train_only = "quality"
        # load_policy 가 그대로 읽는 형식. run_policy 에 이 파일을 그대로 넘기면 된다.
        torch.save({"model_state": work.state_dict(), "config": cfg.to_dict(),
                    "epoch": res["best_epoch"], "best": res["best_val"], "best_metric": "qual",
                    "trained_by": "retrain_quality --cached"}, out / "best.pt")
        print(f"\n학습 완료 (캐시): {out / 'best.pt'}  best val {res['best_val']:.5f} @ ep {res['best_epoch']}")
    elif not args.eval_only:
        cfg.train.init_from = str(Path(args.checkpoint).resolve())
        cfg.train.train_only = "quality"
        cfg.loss.quality_target = args.target
        cfg.train.epochs = args.epochs
        cfg.train.lr = args.lr
        cfg.train.device = args.device
        cfg.train.warmup_epochs = 1
        trainer = Trainer(cfg, dataset_path=ds_path, output_dir=out)
        trainer.fit()
        print(f"\n학습 완료: {out}  best val/qual {trainer.best:.4f}")

    new_model, _ = load_policy(out / "best.pt", device=args.device)
    test = PolicyH5Dataset(ds_path, split="test", augment=False)
    loader = torch.utils.data.DataLoader(test, batch_size=16, shuffle=False)
    kw = dict(target=args.target, max_samples=args.eval_samples,
              gamma_ref=cfg.train.gamma_mode_consistency)
    old = evaluate_quality_head(old_model, loader, **kw)
    new = evaluate_quality_head(new_model, loader, **kw)

    print(f"\n=== 품질 헤드 비교 — 목표 {args.target}, test 관측 {new['n_obs']} ===")
    rows = [("라벨 적합 R²", "fit_r2", "{:+.3f}"),
            ("방향만: A 대 −A 부호 일치 ★", "mirror_sign_agree", "{:.1%}"),
            ("크기만: 크게 움직이는 쪽 선호", "prefers_larger", "{:.1%}"),
            ("행동 효과 상관 ρ (크기·방향 섞임)", "effect_spearman", "{:+.3f}"),
            ("방향별 Q̂ 차 (중앙)", "gap_median", "{:.4f}"),
            ("z 재추첨 방향 일치율", "consistency", "{:.1%}"),
            (f"Q̂ 가 방향을 정하는 몫 (γ={kw['gamma_ref']:g})", "decided_frac", "{:.1%}")]
    for label, key, fmt in rows:
        print(f"  {label:32s} 옛 {fmt.format(old[key]):>8}   새 {fmt.format(new[key]):>8}")
    v = verdict(old, new)
    print("  ★ = 방향 판단의 주 지표. 나머지는 크기와 방향이 섞여 있다 (quality_eval.verdict)")
    print(f"\n  → {v}")
    (out / "quality_eval.json").write_text(
        json.dumps({"target": args.target, "old": old, "new": new, "verdict": v},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n기록: {out / 'quality_eval.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
