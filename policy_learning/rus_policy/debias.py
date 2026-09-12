"""Q̂ 의 **관측과 무관한** 행동 기울어짐을 재서 빼낸다.

    bias = measure_action_bias(model, loader)      # (6,) — Σ_k Q̂ 의 축별 기울기
    save_action_bias(path, bias);  bias = load_action_bias(path)

왜 필요한가
-----------
선택은 후보 M 개 중 ``Σ_k Q̂`` 가 가장 큰 것을 고른다. 그런데 Q̂ 를 행동에 대해 미분해 보면
기울기의 **부호가 관측과 무관하게 거의 같다** (2026-09-12, test 320 개):

    θx  부호 일치 98 %   평균 −0.00307 /°
    θz  부호 일치 95 %   평균 +0.00287 /°
    θy  부호 일치 65 %   평균 +0.00156 /°

즉 Q̂ 는 영상이 무엇이든 −θx 를 좋게 본다. 후보들이 ±2° 로 퍼져 있으면 이 상수가 만드는
점수 차는 0.012 인데, 후보 사이의 실제 Q̂ 차는 중앙 0.021 이다 — 순위의 절반 이상을 관측과
무관한 상수가 정한다. 그래서 64 개 중 최댓값을 고르면 거의 언제나 같은 방향이 나온다
(디코더가 낸 후보는 θx 양수 51 % 로 고른데 고른 뒤에는 32 %, Q_area 재학습 헤드는 9 %).

``Q̂(o,A) ≈ f(o) + ĝ·A + (관측에 따라 달라지는 몫)`` 에서 ``ĝ·A`` 를 빼면 남는 것이 관측에
따라 달라지는 몫이다. 그것이 원래 선택에 쓰여야 할 값이다.

**이것은 쏠림을 없앨 뿐, 방향을 알게 하지는 않는다.** 빼고 나면 선택은 한쪽으로 치우치지
않지만, 남은 신호 자체가 약하다 (크기 같고 방향만 반대인 행동을 가르는 정확도 52~56 %).
방향을 제대로 고르려면 개입 데이터로 다시 학습해야 한다 — 파일럿의 위약 에피소드가 그것이다.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from .model import ACTION_DIM

#: 유한차분에 쓸 크기. 회전은 °, 병진은 mm — 후보들이 실제로 퍼져 있는 범위와 같은 자릿수.
STEP = (2.0, 2.0, 2.0, 2.0, 2.0, 2.0)


def _ramp(model, B: int, axis: int, size: float, device) -> torch.Tensor:
    """축 하나로만 일정 속도로 가는 chunk 궤적 (B, k+1, 6)."""
    P = torch.zeros(B, model.k + 1, ACTION_DIM, device=device)
    P[:, :, axis] = torch.linspace(0, 1, model.k + 1, device=device) * size
    return P


#: 이 비율 이상으로 부호가 같은 축만 뺀다.
#:
#: 평균을 빼는 것은 "관측과 무관한 상수" 를 지우는 일이다. 부호가 관측마다 뒤집히는 축은
#: 그 평균이 상수가 아니라 **관측에 따라 달라지는 몫의 평균**이라, 빼면 신호를 지운다.
#: 2026-09-12 실측: θy 는 부호 일치가 65 % 뿐인데 평균을 빼자 고른 방향이 49 % → 34 % 로
#: 오히려 치우쳤다 (θx 는 98 % 로 36 % → 58 %, θz 는 95 % 로 45 % → 62 %).
MIN_SIGN_AGREEMENT = 0.90


def gate_by_agreement(mean: np.ndarray, agreement: Sequence[float],
                      threshold: float = MIN_SIGN_AGREEMENT) -> np.ndarray:
    """부호 일치가 문턱에 못 미치는 축은 0 으로 — 그 축은 빼지 않는다."""
    out = np.asarray(mean, float).copy()
    out[np.asarray(agreement, float) < threshold] = 0.0
    return out


@torch.no_grad()
def measure_action_bias(model, loader, max_samples: Optional[int] = 320) -> np.ndarray:
    """관측 여러 개에서 잰 ``Σ_k Q̂`` 의 축별 평균 기울기 (6,).

    평균을 쓰는 이유: 빼려는 것은 **관측과 무관한 몫**이고, 관측에 따라 달라지는 몫은
    평균에서 상쇄된다. 중앙값을 쓰면 분포가 한쪽으로 몰릴 때 그 몫까지 지운다.
    """
    model.eval()
    dev = next(model.parameters()).device
    got, n = [], 0
    for batch in loader:
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        B = batch["frames"].shape[0]
        _, _, pooled = model.encode_observation(batch)
        g = []
        for ax in range(ACTION_DIM):
            eps = STEP[ax]
            qp = model.predict_quality(pooled, _ramp(model, B, ax, +eps, dev)).sum(1)
            qm = model.predict_quality(pooled, _ramp(model, B, ax, -eps, dev)).sum(1)
            g.append(((qp - qm) / (2 * eps)).cpu().numpy())
        got.append(np.stack(g, 1))
        n += B
        if max_samples and n >= max_samples:
            break
    return np.concatenate(got).mean(0).astype(np.float64)


def bias_report(model, loader, max_samples: Optional[int] = 320) -> dict:
    """기울기의 평균과 **부호 일치율**. 일치율이 높아야 '관측과 무관' 이라 부를 수 있다."""
    model.eval()
    dev = next(model.parameters()).device
    got, n = [], 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            B = batch["frames"].shape[0]
            _, _, pooled = model.encode_observation(batch)
            g = []
            for ax in range(ACTION_DIM):
                eps = STEP[ax]
                qp = model.predict_quality(pooled, _ramp(model, B, ax, +eps, dev)).sum(1)
                qm = model.predict_quality(pooled, _ramp(model, B, ax, -eps, dev)).sum(1)
                g.append(((qp - qm) / (2 * eps)).cpu().numpy())
            got.append(np.stack(g, 1))
            n += B
            if max_samples and n >= max_samples:
                break
    G = np.concatenate(got)
    return {
        "n_obs": int(len(G)),
        "mean": G.mean(0).tolist(),
        "std": G.std(0).tolist(),
        "sign_agreement": [float(max((G[:, i] > 0).mean(), (G[:, i] < 0).mean()))
                           for i in range(ACTION_DIM)],
    }


def save_action_bias(path: Path, bias: np.ndarray, report: Optional[dict] = None) -> None:
    payload = {"action_bias": list(map(float, np.asarray(bias).reshape(ACTION_DIM)))}
    if report:
        payload["report"] = report
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_action_bias(path: Path) -> Optional[np.ndarray]:
    """없으면 ``None`` — 파일이 없다고 러너가 서지는 않는다."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["action_bias"], dtype=np.float64).reshape(ACTION_DIM)


def default_bias_path(checkpoint: str | Path) -> Path:
    """체크포인트 옆에 둔다 — 모델이 바뀌면 기울어짐도 달라지므로 한 짝이어야 한다."""
    return Path(str(checkpoint) + ".action_bias.json")


# ---- 실행 중에 재는 기울어짐 -------------------------------------------------

@dataclass
class OnlineActionBias:
    """로봇이 **지금 보고 있는 영상**에서 기울어짐을 재서 뺀다.

    오프라인 보정은 상수를 학습 데이터(손으로 스캔한 프리핸드)에서 한 번 재어 고정한다.
    로봇이 보는 것은 팬텀 영상이라 분포가 다르고, 그래서 상수도 다르다 — 2026-09-12 실기에서
    θx 는 맞았지만(예측 55 % · 실제 49 %) θz 는 빗나갔다(예측 60 % · 실제 23 %).

    모델은 매 결정마다 후보 M 개와 각각의 Σ_k Q̂ 를 이미 계산한다. 그 M 쌍에 직선을 맞추면
    **이 관측에서의 기울기** 가 나온다 (상수 성분 + 이 관측 고유의 성분). 여러 결정에 걸쳐
    평균하면 관측 고유의 성분은 부호가 엎치락뒤치락하며 상쇄되고 상수만 남는다.

    ⚠️ **평균 구간이 짧거나 프로브가 한자리에 오래 있으면 신호까지 지운다.** 관측이 서로
    닮으면 관측 고유의 성분이 상쇄되지 않고, 그것을 상수로 착각해 뺀다. 오프라인에서 θy 에
    그 일이 있었다 (부호 일치 65 % 인데 뺐더니 고른 방향이 49 % → 34 %). 그래서 여기서도
    **구간 안에서 부호가 일관된 축만** 뺀다.

    학습이 아니다 — 가중치는 하나도 바뀌지 않고, 저장하지도 않는다. 모델이 이미 낸 값의
    상수항을 실행 중에 정규화할 뿐이다.
    """

    window: int = 150                       # 5 Hz 에서 30 s
    min_count: int = 50                     # 이만큼 쌓이기 전에는 빼지 않는다
    min_agreement: float = MIN_SIGN_AGREEMENT
    _buf: deque = field(default_factory=lambda: deque(maxlen=150), init=False)

    def __post_init__(self) -> None:
        self._buf = deque(maxlen=int(self.window))

    def update(self, net: np.ndarray, q_sum: np.ndarray) -> None:
        """후보들의 순변위 (M,6) 과 각각의 Σ_k Q̂ (M,) 로 이 결정의 기울기를 잰다."""
        N = np.asarray(net, float).reshape(-1, ACTION_DIM)
        y = np.asarray(q_sum, float).reshape(-1)
        if len(N) < ACTION_DIM + 2 or not np.isfinite(y).all():
            return
        A = np.concatenate([N, np.ones((len(N), 1))], 1)     # 절편을 함께 푼다
        try:
            sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        except np.linalg.LinAlgError:
            return
        if np.isfinite(sol).all():
            self._buf.append(sol[:ACTION_DIM])

    @property
    def n(self) -> int:
        return len(self._buf)

    def bias(self) -> Optional[np.ndarray]:
        """구간 평균에서 **부호가 일관된 축만** 남긴 보정값. 아직 모자라면 ``None``."""
        if len(self._buf) < self.min_count:
            return None
        G = np.stack(self._buf)
        agree = [float(max((G[:, i] > 0).mean(), (G[:, i] < 0).mean()))
                 for i in range(ACTION_DIM)]
        return gate_by_agreement(G.mean(0), agree, self.min_agreement)

    def describe(self) -> str:
        b = self.bias()
        if b is None:
            return f"온라인 기울어짐 — {self.n}/{self.min_count} 결정 모으는 중"
        used = [n for n, v in zip(("x", "y", "z", "θx", "θy", "θz"), b) if v != 0.0]
        return (f"온라인 기울어짐 {self.n} 결정 · 빼는 축 {', '.join(used) or '없음'} "
                f"(θx {b[3]:+.5f} θy {b[4]:+.5f} θz {b[5]:+.5f})")
