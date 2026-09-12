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
    P = torch.cumsum(torch.randn(B, k + 1, ACTION_DIM, generator=g) * 2, dim=1)
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
        "sigma_net": torch.tensor([[1.4, 1.4, 1.4, 0.1, 0.1, 0.1]]).repeat(B, 1),
        "sigma_shape": torch.tensor([[0.4, 0.4, 0.4, 0.03, 0.03, 0.03]]).repeat(B, 1),
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
        assert out.mu is None and out.logits.shape == (B, ACTION_DIM, cfg.model.discrete_bins)
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
    P = torch.arange(9.0)[None, :, None].repeat(1, 1, ACTION_DIM)   # 선형 궤적 → 형상 0
    assert torch.allclose(trajectory_shape(P), torch.zeros(1, 7, ACTION_DIM))
    centers = torch.linspace(-20, 20, 21)[None].repeat(ACTION_DIM, 1)
    net = torch.tensor([[0.0, 25.0, -4.0, 0.0, 0.0, 0.0]])          # 0 / 범위 밖 / −4mm
    tgt = soft_bin_targets(net, torch.ones(1, ACTION_DIM), centers)
    assert torch.allclose(tgt.sum(-1), torch.ones(1, ACTION_DIM))
    assert tgt[0, 0].argmax() == 10 and tgt[0, 1].argmax() == 20 and tgt[0, 2].argmax() == 8


@pytest.mark.parametrize("head", ["cvae", "discrete"])
def test_select_action(head):
    cfg = small_config(**{"model.head": head})
    model = build_policy(cfg.model, cfg.timing).eval()
    batch = fake_batch(cfg, B=2)
    sel = model.select_action(batch, n_samples=5, gamma=0.2, prev_dy=torch.tensor([1.0, -1.0]))
    k = cfg.timing.chunk_steps
    assert sel["P"].shape == (2, k + 1, ACTION_DIM) and sel["a"].shape == (2, k, ACTION_DIM) \
        and sel["Q_hat"].shape == (2, k)
    assert sel["net"].shape == (2, ACTION_DIM) and torch.isfinite(sel["net"]).all()
    if head == "cvae":
        assert sel["candidates"].shape == (2, 5, k + 1, ACTION_DIM)
        # 후보들이 서로 다르다 (z 가 실제로 출력에 들어간다)
        assert sel["vy_spread"].mean() > 0
    else:
        assert sel["prob"].shape == (2, ACTION_DIM, cfg.model.discrete_bins)


def test_quality_residual_vanishes_at_zero_action():
    """Q̂ = b(o) + [g(o,A) − g(o,0)] — A=0 에서 잔차가 정확히 0, 그리고 A 가 바뀌면 움직인다."""
    cfg = small_config()
    model = build_policy(cfg.model, cfg.timing).eval()
    batch = fake_batch(cfg)
    out = model(batch, use_posterior=True)
    P = batch["P"]
    base, resid = model.quality_parts(out.memory_pooled, P)
    _, resid0 = model.quality_parts(out.memory_pooled, torch.zeros_like(P))
    assert torch.allclose(resid0, torch.zeros_like(resid0), atol=1e-6)
    assert torch.allclose(model.predict_quality(out.memory_pooled, torch.zeros_like(P)), base, atol=1e-6)
    assert resid.abs().mean() > 0                      # 행동이 실제로 출력을 움직인다

    # 잔차 항의 그래디언트가 기저로 새지 않아야 한다 (b 가 행동 몫을 도로 흡수하는 것을 막는다)
    model.zero_grad(set_to_none=True)
    base2, resid2 = model.quality_parts(out.memory_pooled, P)
    (base2.detach() + resid2).sum().backward(retain_graph=True)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.quality_base.parameters())


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
    # total 이 수백~수천이라 두 값의 차를 float32 로 재면 상쇄 오차가 1e-4 수준까지 뜬다
    assert l1["total"] - l0["total"] == pytest.approx(cfg.loss.beta_kl * l1["kl"], rel=1e-3, abs=1e-3)


def test_bimodal_sanity_of_discrete_head():
    """이봉 분포에서 회귀 헤드는 평균(≈0)을, 이산 헤드는 두 봉을 유지한다 (§5.3g 표의 근거)."""
    torch.manual_seed(0)
    cfg = small_config(**{"model.head": "discrete"})
    model = build_policy(cfg.model, cfg.timing)
    centers = model.bin_centers()
    width = float(centers[0, 1] - centers[0, 0])
    mid = model.n_bins // 2
    d = 5 * width                           # 두 봉이 bin 중심에 정확히 놓이도록
    row = torch.zeros(ACTION_DIM)
    net = torch.stack([row.clone().index_fill_(0, torch.tensor([1]), s * d) for s in (1.0, -1.0)] * 8)
    sigma = torch.full((16, ACTION_DIM), width * 0.4)
    tgt = soft_bin_targets(net, sigma, centers)
    avg = tgt.mean(0)                       # 관측이 같으면 CE 의 최적 예측은 타깃 평균 = 이봉
    p = avg[1]
    peaks = (p > p.roll(1)) & (p > p.roll(-1))
    assert peaks.sum() == 2 and p[mid] < p.max() * 0.2     # 가운데(0) 는 낮고 두 봉이 산다


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
    assert sel["P"].shape == (1, cfg.timing.chunk_steps + 1, ACTION_DIM)
    # 재개
    tr2 = Trainer(cfg, dataset_path=built_dataset, output_dir=tmp_path / "run2")
    tr2.resume(tmp_path / "run" / "last.pt")
    assert tr2.epoch == 2


def test_checkpoint_metric_is_leak_immune_and_beta_adapts(built_dataset, tmp_path):
    """best 는 실행시 경로(select_nmae) 기준이어야 하고, β 는 leak_gap 을 보고 움직여야 한다."""
    from rus_policy.train import Trainer

    cfg = small_config()
    cfg.train.epochs, cfg.train.beta_warmup_epochs, cfg.train.beta_adapt_rate = 3, 1, 2.0
    tr = Trainer(cfg, dataset_path=built_dataset, output_dir=tmp_path / "run")
    tr.fit()
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    for key in ("val/select_nmae", "val/select_mae_x_mm", "val/select_mae_z_mm",
                "val/select_mae_thy_deg", "val/sigma_net_y_mm",
                "val/leak_thy_sigma", "val/leak_worst_sigma", "val/spread_thy_deg",
                "val/select_dir_thy", "val/select_dir_err"):
        assert key in rows[-1] and np.isfinite(rows[-1][key]), key
    # σ_net,y 는 라벨 설정에서 나온다: 0.7 · τ^1.5, τ = k / f_p
    tau = cfg.timing.chunk_steps / cfg.timing.policy_hz
    assert rows[-1]["val/sigma_net_y_mm"] == pytest.approx(
        max(cfg.labels.sigma_translation_coeff_mm * tau ** 1.5, cfg.labels.sigma_floor_mm), rel=0.05)
    # best 는 total 이 아니라 select_nmae 의 최소값
    assert tr.best == pytest.approx(min(r["val/select_nmae"] for r in rows))

    # 기준이 바뀌면 이전 best 는 단위가 달라 비교 불가 → 초기화, β 배율은 승계
    cfg2 = small_config()
    cfg2.train.checkpoint_metric = "total"
    tr2 = Trainer(cfg2, dataset_path=built_dataset, output_dir=tmp_path / "run2")
    tr2.resume(tmp_path / "run" / "last.pt")
    assert tr2.best == float("inf")
    assert tr2.beta_mult == pytest.approx(tr.beta_mult)

    # 조정 규칙 자체를 직접 검증한다 (합성 데이터의 우연에 기대지 않는다)
    # 누설은 축별 leak_<축>_sigma (라벨 퍼짐 배수) 로 준다 — leak_gap_mm 은 척도가 달라 안 쓴다.
    sig, rate = 1.0, cfg.train.beta_adapt_rate
    tr.epoch = max(1, cfg.train.beta_warmup_epochs)
    base = {"sigma_net_y_mm": sig, "mode_collapse_vy_std": 0.01 * sig}
    over = 2.0 * cfg.train.beta_adapt_target_sigma               # 목표를 넘는 누설
    under = 0.1 * cfg.train.beta_adapt_target_sigma              # 0.5×목표 아래 = 여유

    tr.beta_mult = 1.0                                          # 누설 → 좁힌다
    tr.adapt_beta({**base, "leak_thy_sigma": over, "select_vy_sign_acc": 0.9})
    assert tr.beta_mult == pytest.approx(rate)

    tr.beta_mult = 1.0                                          # 선택기가 우연 수준이면 넓히지 않는다
    tr.adapt_beta({**base, "leak_thy_sigma": under, "select_vy_sign_acc": 0.5})
    assert tr.beta_mult == pytest.approx(1.0)

    tr.beta_mult = 1.0                                          # 선택기가 쓸 만하면 최소 β 쪽으로
    tr.adapt_beta({**base, "leak_thy_sigma": under, "select_vy_sign_acc": 0.9})
    assert tr.beta_mult == pytest.approx(1.0 / rate)


def test_beta_adapt_off_keeps_beta_fixed(built_dataset, tmp_path):
    """고정 β 스윕용 — beta_adapt=false 면 워밍업 뒤 β 가 beta_kl 에 머문다."""
    from rus_policy.train import Trainer

    cfg = small_config()
    cfg.train.epochs, cfg.train.beta_warmup_epochs, cfg.train.beta_adapt = 3, 1, False
    tr = Trainer(cfg, dataset_path=built_dataset, output_dir=tmp_path / "run")
    tr.fit()
    rows = [json.loads(l) for l in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
    assert tr.beta_mult == 1.0
    assert all(r["train/beta"] == pytest.approx(cfg.loss.beta_kl) for r in rows)
