"""§5.3 loss.

    L = L_act + λ_F L_force + λ_Q L_qual + λ_s L_smooth + λ_c L_feas + λ_r L_risk + β L_KL

(a) L_act  이종분산 Huber — chunk **순변위** (가중 1/σ_net²) + 궤적 **형상** R_i (가중 w_shape/σ_shape²).
           per-step 속도는 감독하지 않는다 (§5.2: 라벨은 적분해서, 예측은 미분해서).
           σ 는 샘플·축·소스마다 다르다 → 프리핸드/텔레오퍼레이션이 자동으로 올바르게 섞인다.
(b) 힘축 (v_z, ω_x, ω_y) 는 라벨에서 제외 — 데이터셋 단계에서 이미 policy 3축만 P 에 들어 있다.
(c) L_force  F̂_n vs F̃_n  (F_valid 마스크: 프리핸드는 0)
(d) L_qual   Q̂(o, P̃) vs Q̃  (Q_valid 마스크) — 모드 선택이 여기 의존하므로 λ_Q 를 낮추지 말 것
(e) L_feas   |v_z,req| = |F* − F̂|/B_z 가 v_z,max 를 넘는 만큼  (힘 채널 있을 때만)
    L_risk   비대칭 (5.6): 접촉 상실 쪽 w₋ ≫ 과압 쪽 w₊
(f) L_smooth Δ² Â
(g) L_KL     β 는 붕괴와 **누설** 사이에서 고른다 (§5.3g, 2026-09-08 개정). β 가 너무 작으면 인코더가
             라벨을 z 에 실어 보내고 디코더가 o 를 무시한다 — 사전분포 z 로는 무작위 chunk 가 나온다.
             `beta_scale` 로 워밍업(0 → 1) 을 받는다. 누설 진단은 train.py 의 val/leak_gap_mm.

이산 헤드: 순변위 bin 에 대한 CE. 타깃은 σ_net 폭의 가우시안 소프트 라벨이라 라벨 잡음이
자연스럽게 들어간다 (FK 라벨은 뾰족, IMU 라벨은 퍼짐).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .config import LossConfig
from .model import AXIS_NAMES, AXIS_UNITS, Y_AXIS, ActPolicy, PolicyOutput


def huber(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """요소별 Huber ρ_δ(x). delta 는 브로드캐스트 가능."""
    ax = x.abs()
    quad = 0.5 * x * x
    lin = delta * (ax - 0.5 * delta)
    return torch.where(ax <= delta, quad, lin)


def trajectory_shape(P: torch.Tensor) -> torch.Tensor:
    """R_i = P_i − (i/k) P_k, i = 1..k−1  (선형 성분을 뺀 형상). (B,k+1,3) → (B,k−1,3)."""
    k = P.shape[1] - 1
    frac = torch.arange(1, k, device=P.device, dtype=P.dtype)[None, :, None] / k
    return P[:, 1:k] - frac * P[:, k:k + 1]


def soft_bin_targets(net: torch.Tensor, sigma: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """(B,3) 순변위, (B,3) σ, (3,n) bin 중심 → (B,3,n) 가우시안 소프트 라벨 (범위 밖은 끝 bin 으로)."""
    width = (centers[:, 1] - centers[:, 0])[None, :, None]                   # (1,3,1)
    clipped = torch.maximum(torch.minimum(net, centers[:, -1][None]), centers[:, 0][None])
    s = torch.maximum(sigma, width[:, :, 0] * 0.5)                           # 최소 반 bin
    logits = -0.5 * ((centers[None] - clipped[..., None]) / s[..., None]) ** 2
    return logits.softmax(-1)


def compute_loss(model: ActPolicy, out: PolicyOutput, batch: dict[str, torch.Tensor], cfg: LossConfig,
                 beta_scale: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """beta_scale: KL 워밍업 계수 (0..1). 학습 루프가 epoch 에 따라 올린다."""
    P_lab = batch["P"]                                  # (B,k+1,3)
    P_hat = out.P_hat
    k = P_lab.shape[1] - 1
    sig_net = batch["sigma_net"]                        # (B,3)
    sig_shape = batch["sigma_shape"]
    logs: dict[str, float] = {}

    # (a) 순변위 + 형상
    w_net = (1.0 / sig_net ** 2).clamp(max=cfg.net_weight_cap)
    # δ 는 "도달 가능한 잔차" 규모여야 한다. 라벨 정밀도(σ_net≈1mm)에 묶어 두면 δ≈2mm 가 되는데
    # 실제 잔차는 10mm 대라 모든 샘플이 L1 영역에 들어간다. 그러면 그래디언트 크기가 오차와 무관해져
    # (3mm 틀리나 30mm 틀리나 같은 힘) 최적해가 가중 중앙값 = 상수가 된다 — 2026-09-10 exp1~3 이
    # 전부 상수 예측기로 붕괴한 원인. 실측 라벨 산포 수준으로 바닥을 깐다.
    mm_, dg_ = cfg.huber_delta_min_mm, cfg.huber_delta_min_deg
    floor = torch.tensor([mm_, mm_, mm_, dg_, dg_, dg_], device=sig_net.device, dtype=sig_net.dtype)
    delta_net = torch.maximum(cfg.huber_delta_sigma * sig_net, floor)
    net_err = P_hat[:, -1] - P_lab[:, -1]
    if model.head_type == "cvae":
        l_net = (w_net * huber(net_err, delta_net)).sum(1).mean()
    else:
        centers = model.bin_centers(P_lab.device)
        target = soft_bin_targets(P_lab[:, -1], sig_net, centers)
        logp = out.logits.log_softmax(-1)
        l_net = -(target * logp).sum(-1).sum(1).mean()
        # 이산 헤드에서도 연속 헤드의 순변위를 약하게 맞춘다 (형상·Q̂ 입력의 스케일 유지)
        l_net = l_net + 0.1 * (w_net * huber(net_err, delta_net)).sum(1).mean()
    w_shape = cfg.w_shape / sig_shape ** 2
    # 형상 항도 같은 함정에 빠진다. 바닥을 σ_shape/σ_net 비율만큼 줄여 같은 관계를 유지한다.
    delta_shape = torch.maximum(cfg.huber_delta_sigma * sig_shape,
                                floor * (sig_shape / sig_net.clamp_min(1e-6)))
    shape_err = trajectory_shape(P_hat) - trajectory_shape(P_lab)              # (B,k−1,3)
    l_shape = (w_shape[:, None] * huber(shape_err, delta_shape[:, None])).sum(2).mean(1).mean() \
        if k > 1 else torch.zeros((), device=P_lab.device)
    l_act = l_net + l_shape
    logs["act_net"], logs["act_shape"] = l_net.item(), l_shape.item()

    # (d) 품질 헤드 — 시연된 chunk (라벨) 에 대한 Q̃ 로 감독
    qkey = cfg.quality_target
    if qkey not in ("Q", "Q_area"):
        raise ValueError(f"loss.quality_target 는 'Q' 또는 'Q_area' 여야 한다: {qkey!r}")
    qmask = batch[f"{qkey}_valid"].float()
    qlab = batch[qkey]
    denom = qmask.sum().clamp_min(1.0)

    def _mse(pred: torch.Tensor) -> torch.Tensor:
        return ((pred - qlab) ** 2 * qmask).sum() / denom

    base, resid = model.quality_parts(out.memory_pooled, P_lab)
    if model.quality_residual:
        # b 는 관측만으로 Q 를 맞히고, 잔차는 b 가 남긴 것만 맡는다. b.detach() 로 그래디언트를
        # 끊어야 잔차 경로가 설명한 몫을 b 가 도로 흡수하지 않는다.
        l_base = _mse(base)
        l_resid = _mse(base.detach() + resid)
        l_qual = l_base + l_resid
        logs["qual_base"], logs["qual_resid"] = l_base.item(), l_resid.item()
        # 진단: 잔차가 실제로 행동을 쓰고 있는가 (0 이면 Q̂ 이 행동에 눈이 멀었다는 뜻)
        logs["qual_resid_rms"] = float(resid.detach().pow(2).mean().sqrt())
    else:
        l_qual = _mse(base + resid)
    logs["qual"] = l_qual.item()
    logs["qual_frac"] = float(qmask.mean())

    # (c) 힘 헤드 + (e) 실행가능성·위험 — 힘 라벨이 있을 때만
    fmask = batch["F_valid"].float()
    if fmask.sum() > 0:
        l_force = ((out.F_hat - batch["F"]) ** 2 * fmask).sum() / fmask.sum()
        Fn_star = batch["Fn_star"][:, None]
        v_req = (Fn_star - out.F_hat) / cfg.B_z * 1000.0                        # mm/s
        l_feas = (F.relu(v_req.abs() - cfg.v_z_max_mm_s) ** 2 * fmask).sum() / fmask.sum()
        w_minus = cfg.w_minus_over_plus
        l_risk = ((w_minus * F.relu(Fn_star - out.F_hat) ** 2 + F.relu(out.F_hat - Fn_star) ** 2)
                  * fmask).sum() / fmask.sum()
    else:
        l_force = l_feas = l_risk = torch.zeros((), device=P_lab.device)
    logs["force"], logs["feas"], logs["risk"] = l_force.item(), l_feas.item(), l_risk.item()

    # (f) 매끄러움 (정규화된 속도의 2차 차분)
    a_n = out.a_hat / (model.action_scale / (k * model.dt))
    l_smooth = (a_n[:, 2:] - 2 * a_n[:, 1:-1] + a_n[:, :-2]).pow(2).mean() if k > 2 \
        else torch.zeros((), device=P_lab.device)
    logs["smooth"] = l_smooth.item()

    # (g) KL
    if out.mu is not None:
        l_kl = (-0.5 * (1 + out.logvar - out.mu ** 2 - out.logvar.exp())).sum(1).mean()
    else:
        l_kl = torch.zeros((), device=P_lab.device)
    logs["kl"] = l_kl.item()

    beta = cfg.beta_kl * float(beta_scale)
    total = (l_act + cfg.lambda_force * l_force + cfg.lambda_quality * l_qual + cfg.lambda_smooth * l_smooth
             + cfg.lambda_feas * l_feas + cfg.lambda_risk * l_risk + beta * l_kl)
    logs["total"] = total.item()
    logs["beta"] = beta
    # 진단: 축별 순변위 절대오차 (mm, mm, deg)
    with torch.no_grad():
        ae = net_err.abs().mean(0)
        for i, (nm, un) in enumerate(zip(AXIS_NAMES, AXIS_UNITS)):
            logs[f"mae_{nm}_{un}"] = float(ae[i])
        yi = Y_AXIS
        sign_ok = (torch.sign(P_hat[:, -1, yi]) == torch.sign(P_lab[:, -1, yi])).float()
        big = P_lab[:, -1, yi].abs() > sig_net[:, yi]
        logs["vy_sign_acc"] = float(sign_ok[big].mean()) if big.any() else float("nan")
    return total, logs
