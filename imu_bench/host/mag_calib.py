"""자력계 하드아이언/소프트아이언(대각) 보정.

측정된 자기장은 m_meas = S * (R * m_earth) + b 형태로 왜곡된다.
  b : 하드아이언 — 센서와 함께 회전하는 자석/철심이 만드는 고정 오프셋
  S : 소프트아이언 — 축별 감도 왜곡 (여기서는 대각 성분만 잡는다)
보정 후에는 어떤 자세로 돌려도 |m| 이 일정한 구(sphere)가 되어야 한다.
"""

import json

import numpy as np


def fit_sphere(samples):
    """최소자승 구 피팅. |m - c|^2 = r^2 를 선형화해 중심 c 와 반지름 r 을 얻는다.

        m·m = 2c·m + (r^2 - c·c)
        A = [2mx, 2my, 2mz, 1],  b = |m|^2  ->  x = [cx, cy, cz, r^2-|c|^2]
    """
    m = np.asarray(samples, dtype=float)
    A = np.hstack([2.0 * m, np.ones((len(m), 1))])
    b = np.sum(m * m, axis=1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:3]
    r = float(np.sqrt(max(0.0, x[3] + c @ c)))
    return c, r


def fit(samples):
    """하드아이언 오프셋 + 축별 스케일을 구한다. dict 로 반환."""
    m = np.asarray(samples, dtype=float)
    center, radius = fit_sphere(m)
    centered = m - center

    # 축별 반경 편차를 스케일로 흡수 (대각 소프트아이언 근사)
    span = np.array([
        (centered[:, i].max() - centered[:, i].min()) / 2.0 for i in range(3)
    ])
    span[span < 1e-6] = 1.0
    scale = span.mean() / span

    corrected = centered * scale
    norms = np.linalg.norm(corrected, axis=1)
    return {
        "hard_iron": center.tolist(),
        "scale": scale.tolist(),
        "radius_uT": radius,
        "n_samples": int(len(m)),
        # 품질 지표: 보정 후 |m| 이 얼마나 일정한가 (작을수록 좋다)
        "residual_pct": float(100.0 * norms.std() / norms.mean()) if norms.mean() else 0.0,
        "coverage_deg": _coverage(centered),
    }


def _coverage(centered):
    """방향 커버리지 추정 — 단위벡터들의 평균 길이가 0 에 가까울수록 고르게 돌린 것."""
    n = np.linalg.norm(centered, axis=1, keepdims=True)
    n[n < 1e-9] = 1.0
    u = centered / n
    return float(100.0 * (1.0 - np.linalg.norm(u.mean(axis=0))))


def apply(cal, m):
    if cal is None:
        return m
    return (np.asarray(m, dtype=float) - np.array(cal["hard_iron"])) * np.array(cal["scale"])


def save(cal, path):
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
