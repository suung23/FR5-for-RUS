#!/usr/bin/env python3
"""§8-7 (A7 / L12) — temporal ensembling 이 이봉 모드를 소멸시키는지 직접 확인한다.

    python scripts/exp_ensembling.py [--ticks 400] [--trials 200] [--json out.json]

**학습이 전혀 필요 없는 순수 수치 실험이다** (POLICY_LEARNING_MATH §8 "7번"). 학습된
policy 대신 (3.1) 의 이봉 행동분포를 그대로 내는 합성 policy 를 쓴다 — chunk 마다 `±d` 를
확률 ½ 로 내는 것으로 충분하다.

측정하는 것

  유지율   mean |a_y 방출| / |a_y chunk|.  **표준 ensembling 에서 0 으로 가면 L12 가 맞다** —
           로봇이 제자리에 선다. 처방을 얹었을 때 1 로 회복되는지가 §3.5 의 판정이다.
  전환율   윈도우 내 `a_y` 부호 전환 [Hz]. 🟡 1 Hz 초과면 모드 진동 (§3.5 의 필수 진단).
  커밋률   시나리오 B 에서 다수 모드에 머문 tick 비율. 처방이 **정당한 증거까지** 뭉개지
           않는지 보는 대조.

시나리오

  A  p(+) = 0.5   완전 대칭. 두 모드가 똑같이 좋다 — 어느 쪽이든 **커밋해서 움직이는 것**이
                  정답이고, 평균해서 0 을 내는 것이 실패다.
  B  p(+) = 0.8   약한 증거. 다수 모드로 수렴해야 하고, 처방이 그것을 막으면 안 된다.

처방 (§3.5)

  1  모드 일관성 보너스 — 혼합이 아니라 **선택** 단계다. (3.3) 을 그대로 시뮬레이션한다:
     z 후보 M 개의 Q̂ ~ N(0, σ_Q) 에 γ·sign_match 를 더해 argmax. 대칭 근방에서 Q̂ 가 두
     모드에 대등하다는 §3.4 의 전제를 σ_Q 로 표현한다.
  2  모드 내 평균 (ensemble.mode=cluster)
  3  경계 히스테리시스 (ensemble.mode=hysteresis)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np

from _common import setup_logging  # noqa: F401  (경로 부트스트랩)

from rus_policy.config import PolicyConfig
from rus_policy.ensemble import SIGN_AXIS_Y, EnsembleConfig, TemporalEnsembler


def select_sign(rng: np.random.RandomState, committed: int, gamma: float, sigma_q: float,
                p_plus: float, n_samples: int) -> float:
    """(3.3) 의 모드 선택. gamma=0 이면 처방 1 없음 = Q̂ 잡음만으로 고른다.

    z 후보 M 개를 뽑고 각각의 부호를 p_plus 로 정한다 (사전분포가 이봉이라는 뜻). 후보의
    Q̂ 는 대칭 근방에서 두 모드에 대등하므로 N(0, σ_Q) 로 둔다 (§3.4 의 "우연 수준").
    """
    signs = np.where(rng.rand(n_samples) < p_plus, 1.0, -1.0)
    score = rng.normal(0.0, sigma_q, n_samples)
    if gamma > 0.0 and committed != 0:
        score = score + gamma * signs * committed
    return float(signs[int(np.argmax(score))])


def run_trial(rng: np.random.RandomState, ens_cfg: EnsembleConfig, k: int, dt: float, d_mm: float,
              ticks: int, p_plus: float, gamma: float, sigma_q: float, n_samples: int) -> dict:
    ens = TemporalEnsembler(k, dt, ens_cfg)
    a_chunk = d_mm / (k * dt)                 # chunk 한 장이 의도한 |a_y|
    emitted, committed_plus = [], 0
    for _ in range(ticks):
        sign = select_sign(rng, ens.committed, gamma, sigma_q, p_plus, n_samples)
        chunk = np.zeros((k, 3))
        chunk[:, SIGN_AXIS_Y] = sign * a_chunk
        ens.push(chunk, score=0.0)
        a, _ = ens.step()
        emitted.append(a[SIGN_AXIS_Y])
        committed_plus += int(ens.committed > 0)
    e = np.asarray(emitted)
    return {"retention": float(np.abs(e).mean() / a_chunk),
            "net_progress": float(abs(e.sum()) / (len(e) * a_chunk)),
            "flip_rate_hz": float(ens.flip_rate_hz),
            "commit_plus_frac": committed_plus / ticks}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticks", type=int, default=400, help="시행당 policy tick 수")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--d-mm", type=float, default=4.0, help="chunk 당 면외 순변위 |d|")
    p.add_argument("--gamma", type=float, default=None, help="처방 1 계수 (기본: train.gamma_mode_consistency)")
    p.add_argument("--sigma-q", type=float, default=1.0, help="대칭 근방 Q̂ 의 표준편차 (§3.4)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None)
    args = p.parse_args()

    cfg = PolicyConfig()
    k, dt = cfg.timing.chunk_steps, cfg.timing.chunk_dt
    gamma = cfg.train.gamma_mode_consistency if args.gamma is None else args.gamma
    M = cfg.train.z_samples_eval
    base = EnsembleConfig(deadband_mm=0.2 * args.d_mm)

    variants = [
        ("혼합 없음 (대조)",           replace(base, mode="none"),                              0.0),
        ("표준 ACT ensembling",        replace(base, mode="mean"),                              0.0),
        ("  + 처방1 모드일관성",       replace(base, mode="mean"),                            gamma),
        ("처방2 모드내 평균",          replace(base, mode="cluster"),                           0.0),
        ("  + 처방1",                  replace(base, mode="cluster"),                         gamma),
        ("처방3 경계 히스테리시스",    replace(base, mode="hysteresis", boundary_only=True,
                                               switch_margin=0.5),                             0.0),
        ("  + 처방1",                  replace(base, mode="hysteresis", boundary_only=True,
                                               switch_margin=0.5),                           gamma),
    ]
    scenarios = [("A  p(+)=0.5  완전 대칭", 0.5), ("B  p(+)=0.8  약한 증거", 0.8)]

    out: dict = {"config": {"k": k, "dt": dt, "d_mm": args.d_mm, "gamma": gamma, "sigma_q": args.sigma_q,
                            "z_samples": M, "ticks": args.ticks, "trials": args.trials},
                 "scenarios": {}}
    for title, p_plus in scenarios:
        print(f"\n{title}   (chunk k={k}, f_p={1/dt:.0f} Hz, |d|={args.d_mm} mm, γ={gamma}, σ_Q={args.sigma_q})")
        print(f"  {'처방':28s} {'유지율':>8s} {'순진행':>8s} {'전환율Hz':>9s} {'커밋+':>7s}")
        rows = {}
        for name, ecfg, g in variants:
            res = [run_trial(np.random.RandomState(args.seed + 1000 * i), ecfg, k, dt, args.d_mm,
                             args.ticks, p_plus, g, args.sigma_q, M) for i in range(args.trials)]
            agg = {kk: float(np.mean([r[kk] for r in res])) for kk in res[0]}
            warn = "  ← 모드 진동" if agg["flip_rate_hz"] > base.flip_rate_warn_hz else ""
            if agg["retention"] < 0.25:
                warn += "  ← 모드 소멸"
            print(f"  {name:28s} {agg['retention']:8.3f} {agg['net_progress']:8.3f} "
                  f"{agg['flip_rate_hz']:9.2f} {agg['commit_plus_frac']:7.2f}{warn}")
            rows[name.strip()] = agg
        out["scenarios"][title] = rows

    print("\n판정 기준: 유지율 ≈ 0 이면 L12 (평균이 모드를 지움) 확인. 처방이 1 로 회복시키면 §3.5 유효.")
    print("          전환율 > 1 Hz 는 §3.5 의 모드 진동 경보선.")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        print(f"결과: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
