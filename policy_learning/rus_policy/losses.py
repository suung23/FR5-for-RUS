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
(g) L_KL     β 작게 — posterior collapse 는 L6 로 회귀 (§5.3g)

이산 헤드: 순변위 bin 에 대한 CE. 타깃은 σ_net 폭의 가우시안 소프트 라벨이라 라벨 잡음이
자연스럽게 들어간다 (FK 라벨은 뾰족, IMU 라벨은 퍼짐).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .config import LossConfig
from .model import ActPolicy, PolicyOutput


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
                 ) -> tuple[torch.Tensor, dict[str, float]]:
    P_lab = batch["P"]                                  # (B,k+1,3)
    P_hat = out.P_hat
    k = P_lab.shape[1] - 1
    sig_net = batch["sigma_net"]                        # (B,3)
    sig_shape = batch["sigma_shape"]
    logs: dict[str, float] = {}

    # (a) 순변위 + 형상
    w_net = (1.0 / sig_net ** 2).clamp(max=cfg.net_weight_cap)
    delta_net = cfg.huber_delta_sigma * sig_net
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
    delta_shape = cfg.huber_delta_sigma * sig_shape
    shape_err = trajectory_shape(P_hat) - trajectory_shape(P_lab)              # (B,k−1,3)
    l_shape = (w_shape[:, None] * huber(shape_err, delta_shape[:, None])).sum(2).mean(1).mean() \
        if k > 1 else torch.zeros((), device=P_lab.device)
    l_act = l_net + l_shape
    logs["act_net"], logs["act_shape"] = l_net.item(), l_shape.item()

    # (d) 품질 헤드 — 시연된 chunk (라벨) 에 대한 Q̃ 로 감독
    Q_hat = model.predict_quality(out.memory_pooled, P_lab)
    qmask = batch["Q_valid"].float()
    l_qual = ((Q_hat - batch["Q"]) ** 2 * qmask).sum() / qmask.sum().clamp_min(1.0)
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

    total = (l_act + cfg.lambda_force * l_force + cfg.lambda_quality * l_qual + cfg.lambda_smooth * l_smooth
             + cfg.lambda_feas * l_feas + cfg.lambda_risk * l_risk + cfg.beta_kl * l_kl)
    logs["total"] = total.item()
    # 진단: 축별 순변위 절대오차 (mm, mm, deg)
    with torch.no_grad():
        ae = net_err.abs().mean(0)
        logs["mae_x_mm"], logs["mae_y_mm"], logs["mae_th_deg"] = float(ae[0]), float(ae[1]), float(ae[2])
        sign_ok = (torch.sign(P_hat[:, -1, 1]) == torch.sign(P_lab[:, -1, 1])).float()
        big = P_lab[:, -1, 1].abs() > sig_net[:, 1]
        logs["vy_sign_acc"] = float(sign_ok[big].mean()) if big.any() else float("nan")
    return total, logs
