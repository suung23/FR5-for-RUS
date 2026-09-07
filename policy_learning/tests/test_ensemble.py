"""§3.5 · L12 — temporal ensembling 의 모드 소멸과 처방 세 개.

핵심 테스트는 ``test_mean_annihilates_bimodal_analytically`` 다. 표준 ACT 평균이 이봉
입력을 얼마나 줄이는지는 **해석적으로 알 수 있으므로** (겹친 k 개 ±1 예측의 평균의 기댓값)
구현이 맞는지 근사가 아니라 정확히 판정할 수 있다.
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from rus_policy.ensemble import (  # noqa: E402
    SIGN_AXIS_Y,
    EnsembleConfig,
    TemporalEnsembler,
    bimodal_policy,
    chunk_mode,
)

K, DT, D_MM = 8, 0.2, 4.0
A_CHUNK = D_MM / (K * DT)          # chunk 한 장이 의도한 |a_y|


def _chunk(sign: float) -> np.ndarray:
    c = np.zeros((K, 3))
    c[:, SIGN_AXIS_Y] = sign * A_CHUNK
    return c


def _expected_abs_mean(n: int) -> float:
    """겹친 n 개의 독립 ±1 예측을 균등평균했을 때 |평균| 의 기댓값."""
    return sum(comb(n, j) * abs(2 * j - n) for j in range(n + 1)) / 2 ** n / n


def _run(cfg: EnsembleConfig, signs) -> np.ndarray:
    ens = TemporalEnsembler(K, DT, cfg)
    out = []
    for s in signs:
        ens.push(_chunk(s))
        out.append(ens.step()[0][SIGN_AXIS_Y])
    return np.asarray(out)


# --------------------------------------------------------------------------- L12 본체
def test_mean_annihilates_bimodal_analytically():
    """표준 ACT 평균은 이봉 입력을 E|S_k|/k 로 줄인다 — k=8 이면 0.2734."""
    rng = np.random.RandomState(0)
    signs = rng.choice([-1.0, 1.0], size=40_000)
    # m=0 (균등가중) 이라야 해석해와 정확히 비교된다
    e = _run(EnsembleConfig(mode="mean", m=0.0, deadband_mm=0.2 * D_MM), signs)
    retention = np.abs(e[K:]).mean() / A_CHUNK        # 워밍업 tick 제외
    assert abs(retention - _expected_abs_mean(K)) < 0.01
    assert retention < 0.30, "L12: 평균이 이봉 모드를 지운다는 주장 자체가 깨졌다"


def test_prescriptions_restore_full_amplitude():
    """처방 2·3 은 커밋된 모드 안에서만 평균하므로 진폭이 온전히 남는다."""
    rng = np.random.RandomState(1)
    signs = rng.choice([-1.0, 1.0], size=2000)
    for mode in ("cluster", "hysteresis"):
        e = _run(EnsembleConfig(mode=mode, deadband_mm=0.2 * D_MM), signs)
        retention = np.abs(e[K:]).mean() / A_CHUNK
        assert retention == pytest.approx(1.0, abs=1e-9), f"{mode}: 진폭이 줄었다 ({retention})"


def test_mean_keeps_amplitude_when_unimodal():
    """단봉 입력에서는 표준 평균도 진폭을 지킨다 — 문제는 평균이 아니라 이봉성이다."""
    e = _run(EnsembleConfig(mode="mean", deadband_mm=0.2 * D_MM), np.ones(200))
    assert np.abs(e[K:]).mean() / A_CHUNK == pytest.approx(1.0, abs=1e-9)


def test_majority_mode_survives_prescriptions():
    """약한 증거 (p=0.8) 가 있을 때 처방이 그 증거까지 뭉개면 안 된다."""
    rng = np.random.RandomState(2)
    signs = np.where(rng.rand(4000) < 0.8, 1.0, -1.0)
    e = _run(EnsembleConfig(mode="cluster", deadband_mm=0.2 * D_MM), signs)
    assert (np.sign(e[K:]) > 0).mean() > 0.9      # 다수 모드로 수렴


# --------------------------------------------------------------------------- 부품
def test_chunk_mode_deadband():
    assert chunk_mode(_chunk(1.0), SIGN_AXIS_Y, 0.2 * D_MM, DT) == 1
    assert chunk_mode(_chunk(-1.0), SIGN_AXIS_Y, 0.2 * D_MM, DT) == -1
    assert chunk_mode(_chunk(1.0), SIGN_AXIS_Y, 2.0 * D_MM, DT) == 0     # deadband 안


def test_act_weights_favour_older_predictions():
    """ACT 규약: w_i = exp(−m·i), i=0 이 가장 오래된 예측 → 오래된 쪽에 큰 가중."""
    # m 이 커야 순서가 뒤집혔을 때 실제로 부호가 바뀌어 판별력이 생긴다 (m=0.5 면 어느
    # 순서든 부호가 같아 테스트가 통과해 버린다).
    ens = TemporalEnsembler(K, DT, EnsembleConfig(mode="mean", m=2.0, deadband_mm=0.2 * D_MM))
    for _ in range(K):
        ens.push(_chunk(1.0))
        ens.step()
    ens.push(_chunk(-1.0))                      # 가장 최신 하나만 부호가 다르다
    a, diag = ens.step()
    assert diag["n_alive"] == K
    assert a[SIGN_AXIS_Y] > 0, "최신 예측이 과대가중되고 있다 (ACT 규약과 반대)"


def test_hysteresis_switches_only_on_chunk_boundary():
    cfg = EnsembleConfig(mode="hysteresis", boundary_only=True, switch_margin=0.0,
                         deadband_mm=0.2 * D_MM)
    ens = TemporalEnsembler(K, DT, cfg)
    for _ in range(K):                          # +모드 커밋
        ens.push(_chunk(1.0))
        ens.step()
    assert ens.committed == 1
    switched_at = None
    for i in range(K + 1):                      # 이후 전부 −
        ens.push(_chunk(-1.0))
        ens.step()
        if ens.committed == -1 and switched_at is None:
            switched_at = ens.tick - 1
    assert switched_at is not None, "증거가 뒤집혔는데 끝내 전환하지 않았다"
    assert (switched_at - ens.committed_since) % K == 0 or switched_at == ens.committed_since


def test_flip_rate_diagnostic():
    """§3.5 의 필수 진단 — 부호 전환율. 교대 입력은 경보선을 넘고 단조 입력은 0 이다."""
    ens_alt = TemporalEnsembler(K, DT, EnsembleConfig(mode="none", deadband_mm=0.2 * D_MM))
    for i in range(60):
        ens_alt.push(_chunk(1.0 if i % 2 == 0 else -1.0))
        ens_alt.step()
    assert ens_alt.oscillating and ens_alt.flip_rate_hz > 1.0

    ens_one = TemporalEnsembler(K, DT, EnsembleConfig(mode="none", deadband_mm=0.2 * D_MM))
    for _ in range(60):
        ens_one.push(_chunk(1.0))
        ens_one.step()
    assert ens_one.flip_rate_hz == 0.0 and not ens_one.oscillating


def test_expired_chunks_are_dropped():
    ens = TemporalEnsembler(K, DT, EnsembleConfig(mode="mean", deadband_mm=0.2 * D_MM))
    for _ in range(3 * K):
        ens.push(_chunk(1.0))
        _, diag = ens.step()
    assert diag["n_alive"] == K and len(ens.entries) == K


def test_step_without_push_is_zero():
    ens = TemporalEnsembler(K, DT, EnsembleConfig())
    a, diag = ens.step()
    assert np.all(a == 0.0) and diag["n_alive"] == 0


def test_validation():
    with pytest.raises(ValueError):
        EnsembleConfig(mode="평균")
    ens = TemporalEnsembler(K, DT, EnsembleConfig())
    with pytest.raises(ValueError):
        ens.push(np.zeros((K + 1, 3)))


def test_bimodal_policy_is_balanced_and_hits_target_displacement():
    rng = np.random.RandomState(3)
    chunks = [bimodal_policy(rng, K, DT, D_MM) for _ in range(2000)]
    nets = np.array([c[:, SIGN_AXIS_Y].sum() * DT for c in chunks])
    assert np.allclose(np.abs(nets), D_MM)
    assert abs((nets > 0).mean() - 0.5) < 0.05
