"""품질 헤드가 **행동을 줄 세울 수 있는가** — 재학습 전후 비교용.

선택은 같은 관측에서 후보 행동 M 개를 ``Σ Q̂`` (+ 방향 관성 보너스) 로 고른다. 그러니
Q̂ 가 쓸모 있으려면 라벨을 잘 맞히는 것만으로는 부족하고, **같은 관측에서 행동에 따라
값이 달라져야** 한다. 2026-09-11 에 옛 Q̂ 로 잰 값:

  방향별 최선 후보의 Q̂ 합 차   중앙 0.021  (방향 관성 보너스 차 1.60 의 1/75)
  z 를 다시 뽑아도 같은 방향     66.7 %      (순수 잡음이면 50 %)

이 모듈은 그 두 수와 라벨 적합을 함께 잰다. 무겁지 않게 관측 표본 수를 제한할 수 있다.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch

from .model import Y_AXIS


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _candidates(model, batch, M: int, seed: int):
    """후보 M 개와 각 후보의 Q̂ 합. select_action 과 같은 경로로 만든다."""
    torch.manual_seed(seed)
    B = batch["frames"].shape[0]
    mem, pad, pooled = model.encode_observation(batch)
    mr, pr, por = (x.repeat_interleave(M, 0) for x in (mem, pad, pooled))
    z = torch.randn(B * M, model.z_dim, device=mem.device)
    _, P, _, _ = model.decode(mr, pr, B * M, z)
    q = model.predict_quality(por, P).reshape(B, M, model.k).sum(-1)
    P = P.reshape(B, M, model.k + 1, -1)
    return q.cpu().numpy(), torch.sign(P[:, :, -1, Y_AXIS]).cpu().numpy(), pooled


@torch.no_grad()
def evaluate_quality_head(model, loader, target: str = "Q_area", M: int = 64,
                          max_samples: Optional[int] = None, gamma_ref: float = 0.2) -> dict:
    """품질 헤드 하나를 잰다.

    Returns:
        fit_r2            Q̂(o, 시연 행동) 대 라벨의 R² (유효 칸만)
        effect_spearman   예측한 행동 효과 [Q̂(o,A) − Q̂(o,0)] 대 실제 변화의 순위상관
        gap_median        방향(y 부호)별 최선 후보의 Q̂ 합 차 |Δ| 중앙값
        consistency       z 를 다시 뽑아도 Q̂ 가 같은 방향을 고르는 비율 (0.5 = 잡음)
        decided_frac      gamma_ref 의 보너스보다 Q̂ 차가 큰 비율 — Q̂ 가 방향을 정하는 몫
    """
    model.eval()
    k = model.k
    fit_p, fit_y, eff_p, eff_y, gaps, agree, n = [], [], [], [], [], [], 0
    mir_p, mir_y, mag = [], [], []
    for batch in loader:
        dev = next(model.parameters()).device
        batch = {key: (v.to(dev) if torch.is_tensor(v) else v) for key, v in batch.items()}
        lab, valid = batch[target], batch[f"{target}_valid"]
        mem, pad, pooled = model.encode_observation(batch)
        qa = model.predict_quality(pooled, batch["P"])                          # (B,k)
        q0 = model.predict_quality(pooled, torch.zeros_like(batch["P"]))
        v = valid.cpu().numpy()
        fit_p.append(qa.cpu().numpy()[v])
        fit_y.append(lab.cpu().numpy()[v])
        ends = v[:, 0] & v[:, -1]
        if ends.any():
            eff_p.append((qa - q0)[:, -1].cpu().numpy()[ends])
            eff_y.append((lab[:, -1] - lab[:, 0]).cpu().numpy()[ends])
            # 거울 검사 — 크기는 같고 방향만 반대. 방향을 아는 헤드만 이 차로 실제 변화의 부호를
            # 맞힌다. 위의 행동 효과 상관은 크기와 방향이 섞여 있어, "크게 움직일수록 면적이 더
            # 변한다" 만 배워도 오른다 (2026-09-11 재학습: 효과 상관 +0.20 인데 방향만은 +0.07).
            qm = model.predict_quality(pooled, -batch["P"])
            q2 = model.predict_quality(pooled, 2 * batch["P"])
            mir_p.append((qa - qm)[:, -1].cpu().numpy()[ends])
            mir_y.append((lab[:, -1] - lab[:, 0]).cpu().numpy()[ends])
            mag.append(((q2 - qa)[:, -1] > 0).cpu().numpy()[ends])

        qa_, sa, _ = _candidates(model, batch, M, seed=11)
        qb_, sb, _ = _candidates(model, batch, M, seed=22)
        for b in range(qa_.shape[0]):
            pos, neg = sa[b] > 0, sa[b] < 0
            if pos.any() and neg.any():
                gaps.append(abs(qa_[b][pos].max() - qa_[b][neg].max()))
                if (sb[b] > 0).any() and (sb[b] < 0).any():
                    agree.append(np.sign(sa[b][qa_[b].argmax()]) == np.sign(sb[b][qb_[b].argmax()]))
        n += qa_.shape[0]
        if max_samples and n >= max_samples:
            break

    fp, fy = np.concatenate(fit_p), np.concatenate(fit_y)
    r2 = 1.0 - np.sum((fy - fp) ** 2) / max(np.sum((fy - fy.mean()) ** 2), 1e-12)
    ep = np.concatenate(eff_p) if eff_p else np.array([])
    ey = np.concatenate(eff_y) if eff_y else np.array([])
    gaps = np.asarray(gaps)
    bonus_gap = gamma_ref * k
    mp = np.concatenate(mir_p) if mir_p else np.array([])
    my = np.concatenate(mir_y) if mir_y else np.array([])
    nz = my != 0
    return {
        "n_obs": n,
        "fit_r2": float(r2),
        "effect_spearman": _spearman(ep, ey),
        "gap_median": float(np.median(gaps)) if gaps.size else float("nan"),
        "consistency": float(np.mean(agree)) if agree else float("nan"),
        "decided_frac": float(np.mean(gaps > bonus_gap)) if gaps.size else float("nan"),
        "gamma_ref": gamma_ref,
        # 방향만: A 대 −A 중 실제로 면적을 키운 쪽을 맞힌 비율 (0.5 = 모름). 방향 판단의 주 지표.
        "mirror_sign_agree": float(np.mean(np.sign(mp[nz]) == np.sign(my[nz]))) if nz.any() else float("nan"),
        "mirror_spearman": _spearman(mp, my),
        # 크기만: 같은 방향으로 두 배 움직이는 쪽을 더 좋게 보는 비율.
        "prefers_larger": float(np.mean(np.concatenate(mag))) if mag else float("nan"),
    }


def verdict(old: dict, new: dict) -> str:
    """재학습이 **방향** 판단을 바꿨는가 — 사람이 읽을 한 줄.

    거울 검사(A 대 −A)로 판정한다. 일치율·행동 효과 상관은 크기와 방향이 섞여 있어 주 지표로
    쓰면 안 된다: 2026-09-11 Q_area 재학습에서 일치율 65 → 79 %, 효과 상관 −0.03 → +0.20 으로
    크게 좋아 보였지만 방향만 떼어 보면 52 → 56 % 였다. 앞의 두 수가 오른 것은 대부분 "크게
    움직일수록 면적이 더 변한다" 를 배운 몫이다. 이 함수의 첫 판본은 그 두 수로 판정해 "방향
    판단이 나아졌다" 를 냈다.
    """
    a_new, a_old = new.get("mirror_sign_agree", float("nan")), old.get("mirror_sign_agree", float("nan"))
    if not np.isfinite(a_new):
        return "판정 불가 — 양 끝 라벨이 유효한 관측이 없다"
    size = ""
    if np.isfinite(new.get("prefers_larger", float("nan"))) and new["prefers_larger"] > 0.6:
        size = f" · 크게 움직이는 쪽을 더 좋게 본다 ({new['prefers_larger']:.0%})"
    if a_new < 0.55:
        return (f"⚠️ 방향을 가르지 못한다 — A 대 −A 부호 일치 {a_new:.0%} (옛 {a_old:.0%}, 50 % = 모름){size}. "
                "이 데이터에는 '어느 쪽으로 가야 커지는가' 가 거의 없다 — 개입 데이터가 필요하다")
    if a_new < 0.65:
        return (f"방향 판단이 약하다 — A 대 −A 부호 일치 {a_new:.0%} (옛 {a_old:.0%}, 50 % = 모름){size}. "
                "선택에 맡기기에는 부족하다")
    return f"방향 판단이 나아졌다 — A 대 −A 부호 일치 {a_old:.0%} → {a_new:.0%}{size}"


# ---- 얼린 인코더 위에서 헤드만 빠르게 ---------------------------------------------

def quality_loss(model, pooled: torch.Tensor, P: torch.Tensor, lab: torch.Tensor,
                 valid: torch.Tensor) -> torch.Tensor:
    """losses.compute_loss 의 품질 항과 **같은 식**. 두 경로가 어긋나지 않게 여기 한 번만 쓴다.

    잔차 모드면 b 는 관측만으로 맞히고, 잔차는 b.detach() 위에서 남은 것만 맡는다.
    """
    m = valid.float()
    denom = m.sum().clamp_min(1.0)

    def mse(pred):
        return ((pred - lab) ** 2 * m).sum() / denom

    base, resid = model.quality_parts(pooled, P)
    if model.quality_residual:
        return mse(base) + mse(base.detach() + resid)
    return mse(base + resid)


@torch.no_grad()
def cache_features(model, loader, target: str) -> dict:
    """얼린 인코더의 pooled 특징을 한 번만 뽑는다. 헤드만 학습하면 인코더를 매 epoch 다시
    돌릴 이유가 없다 — CPU 에서 epoch 당 수 분이 수 초가 된다. 증강은 쓰지 않는다."""
    model.eval()
    dev = next(model.parameters()).device
    pooled, P, lab, valid = [], [], [], []
    for batch in loader:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        _, _, p = model.encode_observation(batch)
        pooled.append(p.cpu()); P.append(batch["P"].cpu())
        lab.append(batch[target].cpu()); valid.append(batch[f"{target}_valid"].cpu())
    return {"pooled": torch.cat(pooled), "P": torch.cat(P),
            "lab": torch.cat(lab), "valid": torch.cat(valid)}


def train_head_cached(model, train: dict, val: dict, epochs: int = 60, lr: float = 3e-4,
                      weight_decay: float = 1e-4, batch_size: int = 64, seed: int = 0,
                      log=print) -> dict:
    """quality_head · quality_base 만 학습한다. val 손실이 가장 낮은 epoch 의 가중치로 끝낸다."""
    torch.manual_seed(seed)
    heads = [model.quality_head, model.quality_base]
    for prm in model.parameters():
        prm.requires_grad = False
    params = [prm for h in heads for prm in h.parameters()]
    for prm in params:
        prm.requires_grad = True
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    n = train["pooled"].shape[0]
    best, best_ep, best_state, hist = float("inf"), 0, None, []
    for ep in range(epochs):
        model.eval()
        for h in heads:
            h.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            j = perm[i:i + batch_size]
            loss = quality_loss(model, train["pooled"][j], train["P"][j],
                                train["lab"][j], train["valid"][j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(quality_loss(model, val["pooled"], val["P"], val["lab"], val["valid"]))
        hist.append(v)
        if np.isfinite(v) and v < best:
            best, best_ep = v, ep + 1
            best_state = [copy.deepcopy(h.state_dict()) for h in heads]
        if (ep + 1) % 10 == 0:
            log(f"  ep {ep + 1:3d}  val {v:.5f}  (best {best:.5f} @ {best_ep})")
    if best_state is None:
        raise RuntimeError("val 손실이 한 번도 유한하지 않았다 — 라벨 유효 칸이 있는지 확인하라")
    for h, st in zip(heads, best_state):
        h.load_state_dict(st)
    model.eval()
    return {"best_val": best, "best_epoch": best_ep, "val_history": hist}
