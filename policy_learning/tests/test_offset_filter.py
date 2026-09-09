"""§3.9 오프셋 필터 — 대칭이 한 번의 이동으로 풀리는지, 사각지대에서 멈추는지."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import small_config
from rus_policy.config import OffsetFilterConfig
from rus_policy.ensemble import EnsembleConfig, TemporalEnsembler
from rus_policy.offset_filter import OffsetFilter, ellipsoid_h, replay


def make_filter(**kw) -> OffsetFilter:
    cfg = OffsetFilterConfig(**kw)
    return OffsetFilter(cfg)


def observe(d_true: float, R: float = 50.0, A0: float = 1.0, noise: float = 0.0, rng=None) -> float:
    y = A0 * float(ellipsoid_h(abs(d_true), R))
    if rng is not None and noise > 0:
        y += rng.normal(0.0, noise)
    return y


def test_grid_and_prior():
    f = make_filter(d_range_mm=10.0, d_step_mm=1.0)
    assert f.d.size == 21 and f.d[0] == -10.0 and f.d[-1] == 10.0
    st = f.stats()
    assert abs(st["mean"]) < 1e-9 and st["p_neg"] == pytest.approx(st["p_pos"])


def test_single_observation_is_symmetric():
    """관측 하나로는 |d| 만 알고 부호는 모른다 — 이봉이 정직하게 남는다 (L6 의 재현)."""
    f = make_filter(sigma_q=0.01)
    f.update(observe(6.0, R=50.0))
    st = f.stats()
    assert st["p_neg"] == pytest.approx(st["p_pos"], abs=1e-6)
    assert abs(st["mean"]) < 1e-6
    assert st["d_pos"] > 2.0 and st["d_neg"] < -2.0
    act = f.action()
    assert act.mode in ("probe", "hold")


def dead_zone_mm(sigma_q: float, R: float, probe_snr: float, max_probe: float) -> float:
    """탐침 Δ_max 로 ±d 를 가를 수 있는 최소 |d|:  2dΔ/R² ≥ snr·σ_q  (h ≈ 1 − d²/2R² 근사, 서보잉 §5.2)."""
    return probe_snr * sigma_q * R ** 2 / (2.0 * max_probe)


@pytest.mark.parametrize("d0", [+6.0, -6.0, +3.0, -3.0])
@pytest.mark.parametrize("sigma_q", [0.01, 0.003])
def test_one_move_breaks_symmetry_and_converges(d0, sigma_q):
    """대칭은 한 번 움직이면 풀린다 (§3.9). 필터 자신의 행동 규칙으로 사각지대 안까지 간다.

    사각지대는 면적의 2 차 형상이 주는 물리 한계 (dh/dd = d/R²) 라 σ_q 에 비례한다: σ_q = 0.01 이면 ≈ 3 mm,
    0.003 이면 ≈ 1 mm (R = 50, Δ_max = 8). 필터는 그 안에서 'hold' 를 내야지 헛탐침을 하면 안 된다.
    """
    rng = np.random.RandomState(0)
    R_true, A0_true = 50.0, 1.0
    max_probe = 8.0
    f = make_filter(sigma_q=sigma_q, process_floor_mm=0.3, process_coeff_mm=0.2, converge_std_mm=1.5,
                    max_probe_mm=max_probe, R_grid_mm=[40.0, 50.0, 60.0], A0_grid=[1.0])
    dz = dead_zone_mm(sigma_q, R_true, f.cfg.probe_snr, max_probe)
    d = d0
    f.update(observe(d, R_true, A0_true, sigma_q * 0.5, rng))
    history = []
    for step in range(15):
        act = f.action()
        history.append((round(d, 2), act.mode, round(act.u_mm, 2)))
        if act.mode == "hold" and step >= 2:
            break
        u = act.u_mm + rng.normal(0.0, 0.2)          # 실현치 = 지령 + 실행 오차
        d = d + u
        f.predict(u)                                 # L14: 실현치를 넣는다
        f.update(observe(d, R_true, A0_true, sigma_q * 0.5, rng))
    assert abs(d) < dz + 1.0, (dz, history)
    modes = [h[1] for h in history]
    assert modes[0] in ("probe", "hold") and ("converge" in modes or abs(d0) < dz), history
    if sigma_q <= 0.003:
        assert abs(d) < 2.0, history


def test_probe_direction_tests_major_hypothesis():
    f = make_filter(sigma_q=0.01, R_grid_mm=[50.0], A0_grid=[1.0])
    f.update(observe(6.0, 50.0))
    # 사전분포를 살짝 + 쪽으로 기울인다
    prior = np.ones_like(f.d)
    prior[f.d > 0] *= 3.0
    f.reset(prior)
    f.update(observe(6.0, 50.0))
    act = f.action()
    assert act.mode == "probe"
    assert act.d_major > 0 and act.u_mm < 0            # +d 가설을 시험: −y 로 간다
    assert act.sign == -1


def test_dead_zone_holds():
    """면적이 두 가설을 못 가르면 (잡음 큼) 탐침하지 않는다 — 서보잉 §5.2 의 사각지대."""
    f = make_filter(sigma_q=0.2, probe_snr=2.0, max_probe_mm=4.0, R_grid_mm=[60.0], A0_grid=[1.0])
    f.update(observe(1.0, 60.0))
    act = f.action()
    assert act.mode == "hold" and act.u_mm == 0.0 and act.sign == 0


def test_predict_shifts_and_widens():
    f = make_filter(sigma_q=0.005, R_grid_mm=[50.0], A0_grid=[1.0], process_floor_mm=0.5)
    prior = np.exp(-0.5 * ((f.d - 4.0) / 0.5) ** 2)
    f.reset(prior)
    m0, s0 = f.stats()["mean"], f.stats()["std"]
    f.predict(-3.0, sigma_mm=1.0)
    st = f.stats()
    assert st["mean"] == pytest.approx(m0 - 3.0, abs=0.05)
    assert st["std"] > s0


def test_update_ignores_missing():
    f = make_filter()
    p0 = f.p.copy()
    f.update(None)
    f.update(float("nan"))
    assert np.allclose(f.p, p0) and f.n_updates == 0


def test_replay_records_rows():
    f = make_filter(R_grid_mm=[50.0], A0_grid=[1.0])
    rows = replay(f, u_seq=[0.0, -2.0, -2.0], y_seq=[observe(5.0, 50.0), observe(3.0, 50.0), observe(1.0, 50.0)])
    assert len(rows) == 3 and "act_mode" in rows[-1] and "std" in rows[-1]


def test_filter_drives_ensembler_mode():
    """필터의 부호가 앙상블러 커밋 모드가 된다 (2026-09-08, 잠금 제거)."""
    cfg = small_config()
    k, dt = cfg.timing.chunk_steps, cfg.timing.chunk_dt
    ens = TemporalEnsembler(k, dt, EnsembleConfig(mode="hysteresis", boundary_only=True, switch_margin=1.0))
    plus = np.zeros((k, 3)); plus[:, 1] = +5.0 / (k * dt)
    minus = np.zeros((k, 3)); minus[:, 1] = -5.0 / (k * dt)
    # 점수는 + 가 압도적 → 점수 규칙이면 + 에 잠긴다
    for _ in range(k):
        ens.push(plus, score=10.0)
        ens.push(minus, score=0.1)
        ens.step()
    assert ens.committed == +1
    # 필터가 − 라고 하면 즉시 − 로 커밋된다 (경계·마진 규칙보다 우선)
    ens.push(minus, score=0.1)
    a, diag = ens.step(forced_mode=-1)
    assert ens.committed == -1 and a[1] < 0
    # forced_mode=0 은 "모름" → 점수 규칙으로 복귀 (여기서는 − 유지, 마진 1.0 이 걸려 있음)
    ens.push(plus, score=10.0)
    ens.step(forced_mode=0)
    assert ens.committed in (-1, +1)


def test_config_roundtrip_has_filter_section():
    from rus_policy.config import PolicyConfig

    cfg = PolicyConfig.from_dict({"offset_filter": {"sigma_q": 0.02, "R_grid_mm": [40, 60]}})
    assert cfg.offset_filter.sigma_q == 0.02 and list(cfg.offset_filter.R_grid_mm) == [40, 60]
    assert "offset_filter" in cfg.to_dict() and "dataset" in cfg.to_dict()
