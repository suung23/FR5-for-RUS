"""모델 · loss · 학습 루프 (CPU, 몇 초)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from conftest import small_config
from rus_policy.dataset import OBS_VEC_DIM, PolicyH5Dataset
from rus_policy.losses import compute_loss, huber, soft_bin_targets, trajectory_shape
from rus_policy.model import ACTION_DIM, build_policy
from rus_policy.perception import STATE_DIM


def fake_batch(cfg, B=3, with_force=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    m, k = cfg.timing.obs_frames, cfg.timing.chunk_steps
    H, W = cfg.perception.frame_size
    P = torch.cumsum(torch.randn(B, k + 1, 3, generator=g) * 2, dim=1)
    P[:, 0] = 0
    valid = torch.ones(B, m, dtype=torch.bool)
    valid[0, :2] = False
    return {
        "frames": torch.rand(B, m, H, W, generator=g),
        "frame_dt": -torch.linspace(2.0, 0.0, m)[None].repeat(B, 1),
        "frame_valid": valid,
        "state": torch.randn(B, m, STATE_DIM, generator=g),
        "vec": torch.randn(B, OBS_VEC_DIM, generator=g),
        "P": P,
        "sigma_net": torch.tensor([[1.4, 1.4, 0.1]]).repeat(B, 1),
        "sigma_shape": torch.tensor([[0.4, 0.4, 0.03]]).repeat(B, 1),
        "Q": torch.rand(B, k, generator=g),
        "Q_valid": torch.rand(B, k, generator=g) > 0.3,
        "Q_area": torch.rand(B, k, generator=g),
        "Q_area_valid": torch.rand(B, k, generator=g) > 0.3,
        "F": torch.rand(B, k, generator=g) * 3 if with_force else torch.zeros(B, k),
        "F_valid": torch.ones(B, k, dtype=torch.bool) if with_force else torch.zeros(B, k, dtype=torch.bool),
        "Fn_star": torch.full((B,), 2.0),
        "source": torch.zeros(B, dtype=torch.long),
        "index": torch.arange(B),
    }


@pytest.mark.parametrize("head", ["cvae", "discrete"])
def test_forward_loss_backward(head):
    cfg = small_config(**{"model.head": head})
    model = build_policy(cfg.model, cfg.timing)
    batch = fake_batch(cfg)
    out = model(batch, use_posterior=True)
    B, k = batch["P"].shape[0], cfg.timing.chunk_steps
    assert out.a_hat.shape == (B, k, ACTION_DIM) and out.P_hat.shape == (B, k + 1, ACTION_DIM)
    assert torch.all(out.P_hat[:, 0] == 0)
    assert torch.allclose(out.P_hat[:, 1:], torch.cumsum(out.a_hat, 1) * cfg.timing.chunk_dt, atol=1e-5)
    if head == "cvae":
        assert out.mu is not None and out.logits is None
    else:
        assert out.mu is None and out.logits.shape == (B, 3, cfg.model.discrete_bins)
    loss, logs = compute_loss(model, out, batch, cfg.loss)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert logs["force"] == 0.0 and logs["feas"] == 0.0 and logs["risk"] == 0.0   # 힘 라벨 없음 → 마스크
    assert 0.0 < logs["qual_frac"] < 1.0


def test_force_terms_active_only_with_labels():
    cfg = small_config()
    model = build_policy(cfg.model, cfg.timing)
    batch = fake_batch(cfg, with_force=True)
    out = model(batch)
    _, logs = compute_loss(model, out, batch, cfg.loss)
    assert logs["force"] > 0.0 and logs["risk"] > 0.0


def test_quality_loss_masked_when_no_Q():
    cfg = small_config()
    model = build_policy(cfg.model, cfg.timing)
    batch = fake_batch(cfg)
    batch["Q_valid"][:] = False
    out = model(batch)
    _, logs = compute_loss(model, out, batch, cfg.loss)
    assert logs["qual"] == 0.0


def test_huber_and_shape_helpers():
    x = torch.tensor([0.5, -3.0])
    d = torch.tensor(1.0)
    assert torch.allclose(huber(x, d), torch.tensor([0.125, 2.5]))
    P = torch.arange(9.0)[None, :, None].repeat(1, 1, 3)         # 선형 궤적 → 형상 0
    assert torch.allclose(trajectory_shape(P), torch.zeros(1, 7, 3))
    centers = torch.linspace(-20, 20, 21)[None].repeat(3, 1)
    tgt = soft_bin_targets(torch.tensor([[0.0, 25.0, -4.0]]), torch.tensor([[1.0, 1.0, 1.0]]), centers)
    assert torch.allclose(tgt.sum(-1), torch.ones(1, 3))
    assert tgt[0, 0].argmax() == 10 and tgt[0, 1].argmax() == 20 and tgt[0, 2].argmax() == 8


@pytest.mark.parametrize("head", ["cvae", "discrete"])
def test_select_action(head):
    cfg = small_config(**{"model.head": head})
    model = build_policy(cfg.model, cfg.timing).eval()
    batch = fake_batch(cfg, B=2)
    sel = model.select_action(batch, n_samples=5, gamma=0.2, prev_dy=torch.tensor([1.0, -1.0]))
    k = cfg.timing.chunk_steps
    assert sel["P"].shape == (2, k + 1, 3) and sel["a"].shape == (2, k, 3) and sel["Q_hat"].shape == (2, k)
    assert sel["net"].shape == (2, 3) and torch.isfinite(sel["net"]).all()
    if head == "cvae":
        assert sel["candidates"].shape == (2, 5, k + 1, 3)
        # 후보들이 서로 다르다 (z 가 실제로 출력에 들어간다)
        assert sel["vy_spread"].mean() > 0
    else:
        assert sel["prob"].shape == (2, 3, cfg.model.discrete_bins)


def test_quality_input_is_independent_of_z():
    """2026-09-08: z 는 디코더에만 들어간다 — Q̂ 의 관측 요약은 z 와 무관해야 한다 (누설 차단)."""
    cfg = small_config()
    model = build_policy(cfg.model, cfg.timing).eval()
    batch = fake_batch(cfg, B=2)
    z1 = torch.randn(2, cfg.model.z_dim)
    z2 = torch.randn(2, cfg.model.z_dim)
    with torch.no_grad():
        o1 = model(batch, z=z1, use_posterior=False)
        o2 = model(batch, z=z2, use_posterior=False)
    assert torch.allclose(o1.memory_pooled, o2.memory_pooled)          # 관측 요약 동일
    assert not torch.allclose(o1.P_hat, o2.P_hat)                      # 행동은 z 에 따라 다르다
    # 사후분포 z 로도 관측 요약은 같다 (학습시 Q̂ 가 라벨을 못 본다)
    with torch.no_grad():
        o3 = model(batch, use_posterior=True)
    assert torch.allclose(o3.memory_pooled, o1.memory_pooled)


def test_select_action_encodes_observation_once(monkeypatch):
    cfg = small_config()
    model = build_policy(cfg.model, cfg.timing).eval()
    batch = fake_batch(cfg, B=2)
    calls = {"enc": 0, "dec": 0}
    enc, dec = model.encode_observation, model.decode

    def enc_wrapped(*a, **k):
        calls["enc"] += 1
        return enc(*a, **k)

    def dec_wrapped(*a, **k):
        calls["dec"] += 1
        return dec(*a, **k)

    monkeypatch.setattr(model, "encode_observation", enc_wrapped)
    monkeypatch.setattr(model, "decode", dec_wrapped)
    model.select_action(batch, n_samples=7)
    assert calls["enc"] == 1 and calls["dec"] == 1


def test_beta_warmup_scales_kl():
    cfg = small_config()
    model = build_policy(cfg.model, cfg.timing).eval()      # dropout 을 꺼야 두 호출이 같은 Q̂ 를 낸다
    batch = fake_batch(cfg)
    out = model(batch, use_posterior=True)
    _, l0 = compute_loss(model, out, batch, cfg.loss, beta_scale=0.0)
    _, l1 = compute_loss(model, out, batch, cfg.loss, beta_scale=1.0)
    assert l0["beta"] == 0.0 and l1["beta"] == cfg.loss.beta_kl
    assert l1["total"] - l0["total"] == pytest.approx(cfg.loss.beta_kl * l1["kl"], rel=1e-4, abs=1e-5)


def test_bimodal_sanity_of_discrete_head():
    """이봉 분포에서 회귀 헤드는 평균(≈0)을, 이산 헤드는 두 봉을 유지한다 (§5.3g 표의 근거)."""
    torch.manual_seed(0)
    cfg = small_config(**{"model.head": "discrete"})
    model = build_policy(cfg.model, cfg.timing)
    centers = model.bin_centers()
    d = 10.0
    net = torch.tensor([[0.0, d, 0.0], [0.0, -d, 0.0]] * 8)
    sigma = torch.full((16, 3), 1.0)
    tgt = soft_bin_targets(net, sigma, centers)
    avg = tgt.mean(0)                       # 관측이 같으면 CE 의 최적 예측은 타깃 평균 = 이봉
    p = avg[1]
    peaks = (p > p.roll(1)) & (p > p.roll(-1))
    assert peaks.sum() == 2 and p[10] < p.max() * 0.2      # 가운데(0) 는 낮고 두 봉이 산다


def test_trainer_runs_and_checkpoints(built_dataset, tmp_path):
    from rus_policy.train import Trainer, load_policy

    cfg = small_config()
    cfg.train.epochs = 2
    tr = Trainer(cfg, dataset_path=built_dataset, output_dir=tmp_path / "run")
    hist = tr.fit()
    assert len(hist) == 2 and (tmp_path / "run" / "best.pt").is_file() and (tmp_path / "run" / "last.pt").is_file()
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert rows[-1]["epoch"] == 2 and np.isfinite(rows[-1]["train/total"])
    assert "val/mode_collapse_vy_std" in rows[-1] and "val/mae_y_mm" in rows[-1]
    assert "val/leak_gap_mm" in rows[-1] and "train/beta" in rows[-1]
    # KL 워밍업: epoch 1 의 β 가 epoch 2 보다 작다 (beta_warmup_epochs=5 기본)
    assert rows[0]["train/beta"] < rows[1]["train/beta"] <= cfg.loss.beta_kl
    model, cfg2 = load_policy(tmp_path / "run" / "best.pt", device="cpu")
    assert cfg2.model.d_model == cfg.model.d_model
    ds = PolicyH5Dataset(built_dataset, split="val")
    item = {k: (v[None] if torch.is_tensor(v) else v) for k, v in ds[0].items()}
    sel = model.select_action(item, n_samples=3)
    assert sel["P"].shape == (1, cfg.timing.chunk_steps + 1, 3)
    # 재개
    tr2 = Trainer(cfg, dataset_path=built_dataset, output_dir=tmp_path / "run2")
    tr2.resume(tmp_path / "run" / "last.pt")
    assert tr2.epoch == 2
