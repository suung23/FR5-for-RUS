"""품질 헤드만 다시 학습하기 — 행동 후보는 그대로, 줄 세우는 기준만 바뀌어야 한다.

frame_encoder 에 BatchNorm2d 가 8 개 있다. requires_grad 를 꺼도 train 모드면 running
통계가 바뀌어 얼린 인코더가 조용히 변하고, 그 위의 행동 디코더 출력도 따라 변한다.
그러면 재학습 전후 비교가 "기준만 바뀐" 비교가 아니게 된다.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch

from conftest import small_config
from rus_policy.dataset import PolicyH5Dataset
from rus_policy.train import Trainer, load_policy

QUALITY = ("quality_head", "quality_base")


def _state(model):
    """파라미터와 버퍼(BN running 통계 포함)를 모두 복사한다."""
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _cfg():
    """CPU 로 고정한다 — 로봇 실험이 같은 GPU 로 지각을 돌린다."""
    cfg = small_config()
    cfg.train.device = "cpu"
    return cfg


def _base_run(built_dataset, tmp_path):
    cfg = _cfg()
    cfg.train.epochs = 1
    tr = Trainer(cfg, dataset_path=built_dataset, output_dir=tmp_path / "base")
    tr.fit()
    return tmp_path / "base" / "last.pt"


def _with_area_labels(built_dataset, tmp_path):
    """합성 데이터에는 유효한 품질 라벨이 하나도 없다 (Q_valid·Q_area_valid 합 0). 그러면
    품질 손실이 정확히 0 이라 헤드가 배울 것이 없다 — 복사본에 Q_area 를 채운다."""
    import shutil
    path = tmp_path / "with_area.h5"
    shutil.copy(built_dataset, path)
    with h5py.File(path, "r+") as f:
        rng = np.random.default_rng(0)
        f["label/Q_area"][...] = rng.uniform(0.0, 1.0, f["label/Q_area"].shape).astype(np.float32)
        f["label/Q_area_valid"][...] = True
    return path


def test_head_only_retrain_changes_only_the_quality_head(built_dataset, tmp_path):
    data = _with_area_labels(built_dataset, tmp_path)
    base_ckpt = _base_run(data, tmp_path)
    base_model, _ = load_policy(base_ckpt, device="cpu")
    before = _state(base_model)

    cfg = _cfg()
    cfg.train.epochs = 2
    cfg.train.init_from = str(base_ckpt)
    cfg.train.train_only = "quality"
    cfg.loss.quality_target = "Q_area"
    tr = Trainer(cfg, dataset_path=data, output_dir=tmp_path / "head")
    assert tr.cfg.train.checkpoint_metric == "qual", "행동이 얼면 select_nmae 는 상수다"
    assert tr.cfg.train.beta_adapt is False
    tr.fit()
    after = _state(tr.model)

    changed = [k for k in before if not torch.equal(before[k], after[k])]
    outside = [k for k in changed if k.split(".")[0] not in QUALITY]
    assert not outside, f"품질 헤드 밖이 바뀌었다: {outside[:5]}"
    bn = [k for k in before if "running_mean" in k or "running_var" in k]
    assert bn, "BatchNorm 버퍼를 못 찾았다 — 이 테스트가 지키려는 것이 없다"
    assert all(torch.equal(before[k], after[k]) for k in bn), "얼린 BN 통계가 바뀌었다"
    assert any(k.split(".")[0] in QUALITY for k in changed), "품질 헤드가 전혀 안 바뀌었다"


def test_action_candidates_are_identical_after_head_retrain(built_dataset, tmp_path):
    """같은 관측·같은 z 면 행동 후보가 비트 단위로 같아야 한다 — 바뀌는 것은 순위뿐이다."""
    data = _with_area_labels(built_dataset, tmp_path)
    base_ckpt = _base_run(data, tmp_path)
    cfg = _cfg()
    cfg.train.epochs = 1
    cfg.train.init_from = str(base_ckpt)
    cfg.train.train_only = "quality"
    cfg.loss.quality_target = "Q_area"
    tr = Trainer(cfg, dataset_path=data, output_dir=tmp_path / "head")
    tr.fit()
    base_model, _ = load_policy(base_ckpt, device="cpu")
    new_model = tr.model.to("cpu").eval()

    ds = PolicyH5Dataset(data, split="val")
    item = {k: (v[None] if torch.is_tensor(v) else v) for k, v in ds[0].items()}
    with torch.no_grad():
        torch.manual_seed(3)
        a = base_model.select_action(item, n_samples=8, gamma=0.0)["candidates"]
        torch.manual_seed(3)
        b = new_model.select_action(item, n_samples=8, gamma=0.0)["candidates"]
    # 비트 동일이 아니라 반올림 오차 이내로 본다. 가중치가 비트까지 같아도 두 모델 객체는
    # 메모리 배치가 달라 CPU BLAS 가 다른 커널 경로를 탈 수 있고, 그러면 후보 몇 개가 float32
    # 한 칸(실측 9.5e-7) 어긋난다 — 같은 객체로 두 번이면 비트까지 같다. 행동이 학습됐거나 BN
    # 통계가 흘렀다면 차이는 이보다 몇 자릿수 크다.
    torch.testing.assert_close(a, b, rtol=0.0, atol=1e-5,
                               msg="행동 후보가 바뀌었다 — 행동 쪽이 학습됐거나 BN 통계가 흘렀다")


def test_unknown_quality_target_is_rejected():
    import pytest

    from rus_policy.losses import compute_loss
    from test_model_train import fake_batch
    from rus_policy.model import build_policy

    cfg = small_config()
    cfg.loss.quality_target = "Q_seg"
    model = build_policy(cfg.model, cfg.timing)
    batch = fake_batch(cfg)
    out = model(batch, use_posterior=True)
    with pytest.raises(ValueError, match="quality_target"):
        compute_loss(model, out, batch, cfg.loss)


def test_quality_eval_runs_and_reports_the_selection_metrics(built_dataset, tmp_path):
    """평가기가 도는지 — 재학습 전후 비교가 이것에 달려 있다."""
    from rus_policy.quality_eval import evaluate_quality_head, verdict

    data = _with_area_labels(built_dataset, tmp_path)
    model, _ = load_policy(_base_run(data, tmp_path), device="cpu")
    ds = PolicyH5Dataset(data, split="val")
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    r = evaluate_quality_head(model, loader, target="Q_area", M=8)
    for key in ("fit_r2", "effect_spearman", "gap_median", "consistency", "decided_frac"):
        assert key in r
    assert r["n_obs"] > 0
    assert isinstance(verdict(r, r), str)


def test_verdict_flags_a_head_that_cannot_choose_a_direction():
    """사전 판정이 예상한 결과를 '개선' 으로 잘못 읽지 않아야 한다."""
    from rus_policy.quality_eval import verdict

    old = {"consistency": 0.667, "effect_spearman": 0.02}
    new = {"consistency": 0.55, "effect_spearman": 0.03}
    assert "가르지 못한다" in verdict(old, new)
    good = {"consistency": 0.80, "effect_spearman": 0.35}
    assert "나아졌다" in verdict(old, good)


def test_cached_loss_matches_the_trainer_loss():
    """빠른 경로(특징 캐시)와 Trainer 경로가 **같은 손실**을 최소화해야 한다.

    둘이 어긋나면 CPU 에서 빠르게 돌린 결과가 GPU 본 학습의 예고가 되지 못한다.
    """
    from rus_policy.losses import compute_loss
    from rus_policy.model import build_policy
    from rus_policy.quality_eval import quality_loss
    from test_model_train import fake_batch

    for target in ("Q", "Q_area"):
        cfg = small_config()
        cfg.loss.quality_target = target
        model = build_policy(cfg.model, cfg.timing).eval()
        batch = fake_batch(cfg, B=4)
        batch["Q_area"] = torch.rand_like(batch["Q"])
        batch["Q_area_valid"] = torch.ones_like(batch["Q_valid"])
        batch["Q_valid"] = torch.ones_like(batch["Q_valid"])
        with torch.no_grad():
            out = model(batch, use_posterior=True)
            _, logs = compute_loss(model, out, batch, cfg.loss)
            ours = quality_loss(model, out.memory_pooled, batch["P"], batch[target],
                                batch[f"{target}_valid"])
        assert abs(float(ours) - logs["qual"]) < 1e-6, (target, float(ours), logs["qual"])


def test_cached_training_touches_only_the_quality_head(built_dataset, tmp_path):
    from rus_policy.quality_eval import cache_features, train_head_cached

    data = _with_area_labels(built_dataset, tmp_path)
    model, _ = load_policy(_base_run(data, tmp_path), device="cpu")
    before = _state(model)
    tr = torch.utils.data.DataLoader(PolicyH5Dataset(data, split="train"), batch_size=8)
    va = torch.utils.data.DataLoader(PolicyH5Dataset(data, split="val"), batch_size=8)
    res = train_head_cached(model, cache_features(model, tr, "Q_area"),
                            cache_features(model, va, "Q_area"), epochs=5, log=lambda *_: None)
    after = _state(model)
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert changed and all(k.split(".")[0] in QUALITY for k in changed), changed[:5]
    assert np.isfinite(res["best_val"]) and 1 <= res["best_epoch"] <= 5
