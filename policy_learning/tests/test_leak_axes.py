"""누설 진단이 **신호가 있는 축**을 본다는 것 — 2026-09-12 회귀 방지.

옛 제어기는 y 병진 하나만 봤다. 그 축은 가속도 이중적분 라벨이라 흘려보낼 신호가 적어
누설이 구조적으로 작게 나오고(실측 0.35σ), 그동안 θy 는 0.84σ 로 통째로 샜다.
"""

import numpy as np
import pytest

from rus_policy.config import PolicyConfig
from rus_policy.train import LEAK_AXIS_SETS, Trainer, _worst_leak


def _res(**leaks):
    return {f"leak_{k}_sigma": v for k, v in leaks.items()}


def test_rot_ignores_translation():
    """기본값은 회전 3 축 — 병진이 아무리 커도 고르지 않는다."""
    got = _worst_leak(_res(x=9.0, y=0.35, thx=0.79, thy=0.84, thz=0.09), "rot")
    assert got == ("thy", 0.84)


def test_all_includes_translation():
    assert _worst_leak(_res(x=9.0, thy=0.84), "all") == ("x", 9.0)


def test_y_reproduces_old_rule():
    assert _worst_leak(_res(x=9.0, y=0.35, thy=0.84), "y") == ("y", 0.35)


def test_missing_axes_are_skipped():
    assert _worst_leak(_res(thy=0.5), "rot") == ("thy", 0.5)
    assert _worst_leak(_res(thy=np.nan), "rot") is None
    assert _worst_leak({}, "rot") is None


def test_unknown_set_falls_back_to_rot():
    assert _worst_leak(_res(x=9.0, thy=0.4), "엉뚱한값") == ("thy", 0.4)


def test_axis_sets_name_real_axes():
    from rus_policy.model import AXIS_NAMES
    for names in LEAK_AXIS_SETS.values():
        assert set(names) <= set(AXIS_NAMES)


def _trainer(monkeypatch, leak_axes="rot"):
    """adapt_beta 만 부르기 위한 최소 Trainer — __init__ 을 거치지 않는다."""
    t = Trainer.__new__(Trainer)
    cfg = PolicyConfig()
    cfg.train.leak_axes = leak_axes
    cfg.train.beta_adapt = True
    cfg.train.beta_warmup_epochs = 0
    t.cfg, t.beta_mult, t.epoch = cfg, 1.0, 10

    class _M:
        head_type = "cvae"
    t.model = _M()
    return t


@pytest.mark.parametrize("leak_axes,expect", [("rot", "up"), ("y", "down")])
def test_adapt_beta_follows_the_chosen_axes(monkeypatch, leak_axes, expect):
    """2026-09-12 실측값 그대로: 회전을 보면 조이고, y 만 보면 여유로 읽어 푼다.

    목표가 0.3 이라 θy 0.84 는 누설, y 0.10 은 0.5×목표 아래라 하향 조건에 걸린다.
    """
    t = _trainer(monkeypatch, leak_axes)
    va = _res(x=0.85, y=0.10, z=0.26, thx=0.79, thy=0.84, thz=0.09)
    va.update({"sigma_net_y_mm": 30.0, "mode_collapse_vy_std": 30.0,
               "select_vy_sign_acc": 0.9})
    t.adapt_beta(va)
    assert (t.beta_mult > 1.0) if expect == "up" else (t.beta_mult < 1.0)


def test_adapt_beta_needs_per_axis_metrics():
    """옛 leak_gap_mm 만 있으면 **아무것도 하지 않는다** — 척도가 달라 섞으면 안 된다.

    sigma_net 은 라벨 불확도(회전 0.1°)라 그 단위의 gap 을 라벨 퍼짐 기준 목표와 비교하면
    언제나 "누설" 로 읽혀 β 가 상한까지 간다.
    """
    t = _trainer(None, "rot")
    t.adapt_beta({"sigma_net_y_mm": 0.89, "leak_gap_mm": 9.21,
                  "mode_collapse_vy_std": 30.0, "select_vy_sign_acc": 0.9})
    assert t.beta_mult == 1.0


def test_target_is_reachable_in_spread_units():
    """목표는 라벨 퍼짐의 배수여야 한다 — 1.0 이면 '신호를 다 잃어도 통과' 라 무의미하다."""
    from rus_policy.config import TrainConfig
    assert 0.0 < TrainConfig().beta_adapt_target_sigma < 1.0


# --- 방향 지표 ---------------------------------------------------------------

def test_masked_mean_ignores_unmasked_and_reports_nan_when_empty():
    import torch
    from rus_policy.train import _masked_mean
    ok = torch.tensor([True, False, True, True])
    assert _masked_mean(ok, torch.tensor([True, True, False, False])) == pytest.approx(0.5)
    assert np.isnan(_masked_mean(ok, torch.zeros(4, dtype=torch.bool)))


def test_direction_metric_sees_what_mae_barely_shows():
    """방향을 배워도 MAE 는 거의 안 움직인다 — 체크포인트 기준을 MAE 로 두면 못 고른다.

    2026-09-12 실측이 근거다. centroid_dx → θy 선형 프로브는 test 상관 0.575 를 내면서
    MAE 는 2.36 → 2.37 로 제자리였다. 크기는 0 근처 표본이 정하고, 방향은 크게 움직인
    표본에만 있기 때문이다. 여기서는 그 구조(약한 상관 + 큰 잡음)를 흉내내, 같은 두 예측을
    MAE 와 부호 지표가 각각 얼마나 갈라내는지 본다.
    """
    import torch
    from rus_policy.train import _masked_mean
    rng = np.random.default_rng(0)
    feat = rng.normal(0, 1, 20000)
    lab = 0.5 * feat + rng.normal(0, 1.5, 20000)      # 상관 ρ ≈ 0.32
    informed = 0.5 * feat                              # 방향을 아는 예측
    null = 0.5 * rng.permutation(feat)                 # 같은 분포, 관측과 무관

    mae_gain = 1 - np.abs(lab - informed).mean() / np.abs(lab - null).mean()
    big = torch.from_numpy(np.abs(lab) > lab.std())
    acc = lambda p: _masked_mean(torch.from_numpy(np.sign(p) == np.sign(lab)), big)
    dir_gain = acc(informed) - acc(null)

    assert mae_gain < 0.15, "MAE 로는 방향 학습이 거의 안 보인다는 전제가 깨졌다"
    assert dir_gain > 0.15, "부호 지표는 갈라내야 한다"
    assert dir_gain > 2 * mae_gain          # 부호 지표가 훨씬 민감하다
