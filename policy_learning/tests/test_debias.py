"""Q̂ 의 관측과 무관한 행동 기울어짐 빼기.

2026-09-12 실측 (test 320 개): Q̂ 를 행동으로 미분하면 부호가 관측과 무관하게 거의 같다
(θx 98 % · θz 95 %). 그 상수가 만드는 점수 차 0.012 가 후보 사이의 실제 차(중앙 0.021)와
맞먹어, 64 개 중 최댓값은 거의 늘 같은 방향이 된다 — 후보는 θx 양수 51 % 인데 고른 뒤에는
32 %. 빼고 나면 55 % 로 돌아온다.
"""

import numpy as np
import torch

from conftest import small_config
from rus_policy.debias import MIN_SIGN_AGREEMENT, gate_by_agreement
from rus_policy.model import build_policy
from test_model_train import fake_batch


def test_low_agreement_axes_are_not_subtracted():
    """부호가 관측마다 뒤집히는 축의 평균은 상수가 아니다 — 빼면 신호를 지운다.

    실측: θy 는 부호 일치 65 % 인데 평균을 빼자 고른 방향이 49 % → 34 % 로 오히려 치우쳤다.
    """
    mean = np.array([0.001, 0.002, 0.003, -0.003, 0.0016, 0.0029])
    agree = np.array([1.00, 0.98, 0.89, 0.98, 0.65, 0.95])
    out = gate_by_agreement(mean, agree)
    assert out[2] == 0.0 and out[4] == 0.0, "일치율이 낮은 축을 뺐다"
    assert out[3] == mean[3] and out[5] == mean[5], "일치율이 높은 축을 안 뺐다"
    assert MIN_SIGN_AGREEMENT == 0.90


def test_action_bias_moves_the_choice_the_other_way():
    """점수에서 bias·net 을 빼면 선택이 그 축의 반대쪽으로 간다 — 실제 select_action 으로."""
    cfg = small_config()
    model = build_policy(cfg.model, cfg.timing).eval()
    batch = fake_batch(cfg, B=4)
    axis = 3                                   # θx

    def chosen(bias):
        torch.manual_seed(7)
        with torch.no_grad():
            return model.select_action(batch, n_samples=32, gamma=0.0,
                                       action_bias=bias)["net"][:, axis].numpy()

    base = chosen(None)
    push_neg = np.zeros(6); push_neg[axis] = +1e3    # +방향에 큰 벌점 → −쪽을 고른다
    push_pos = np.zeros(6); push_pos[axis] = -1e3
    neg, pos = chosen(push_neg), chosen(push_pos)
    assert (neg < base + 1e-6).all(), (base, neg)
    assert (pos > base - 1e-6).all(), (base, pos)
    assert (pos > neg).all(), "부호를 뒤집어도 선택이 안 바뀐다"


def test_runner_loads_the_bias_next_to_the_checkpoint():
    from pathlib import Path

    from rus_policy.debias import default_bias_path, load_action_bias, save_action_bias

    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text()
    assert 'p.add_argument("--action-bias"' in src
    assert "action_bias=bias" in src, "선택에 실제로 넘기지 않는다"
    assert "bias = self.online_bias.bias() if self.online_bias is not None else self.action_bias" in src
    assert '"action_bias": None if self.action_bias is None' in src, "쓴 값이 meta 에 안 남는다"
    assert str(default_bias_path("x.pt")).endswith("x.pt.action_bias.json")
    assert load_action_bias(Path("/nonexistent/none.json")) is None, "없으면 None 이어야 한다"


def test_online_bias_recovers_a_planted_constant_tilt():
    """상수 기울어짐을 심어 두면 온라인 추정이 그것을 찾아낸다 — 관측 고유의 몫은 상쇄된다."""
    from rus_policy.debias import OnlineActionBias

    rng = np.random.default_rng(0)
    true_tilt = np.array([0.0, 0.0, 0.0, -0.0031, 0.0, 0.0029])
    est = OnlineActionBias(window=200, min_count=50)
    for _ in range(200):
        net = rng.normal(0, 2.0, size=(64, 6))                  # 후보들 (±2° 남짓)
        # 관측 고유의 몫을 실측 수준으로 둔다: θx 는 평균 −0.00307 · 표준편차 0.00312 인데
        # 부호 일치가 98 % 였다 — 분포가 한쪽으로 몰려 있어서다. 정규분포로 흉내 내려면
        # 표준편차를 평균의 절반쯤으로 잡아야 같은 일치율이 나온다.
        per_obs = rng.normal(0, 0.0015, size=6)
        q = net @ (true_tilt + per_obs) + rng.normal(0, 1e-4, 64)
        est.update(net, q)
    b = est.bias()
    assert b is not None
    assert abs(b[3] - true_tilt[3]) < 0.0008, (b[3], true_tilt[3])
    assert abs(b[5] - true_tilt[5]) < 0.0008, (b[5], true_tilt[5])
    assert b[4] == 0.0, "일관되지 않은 축을 뺐다"      # 참값 0 → 부호가 엎치락뒤치락


def test_online_bias_waits_until_it_has_enough():
    """모자란 표본으로 빼면 관측 고유의 몫을 상수로 착각한다."""
    from rus_policy.debias import OnlineActionBias

    rng = np.random.default_rng(1)
    est = OnlineActionBias(window=150, min_count=50)
    for _ in range(10):
        net = rng.normal(0, 2.0, size=(64, 6))
        est.update(net, net @ rng.normal(0, 0.004, 6))
    assert est.bias() is None and est.n == 10


def test_runner_uses_pre_correction_scores_for_the_estimate():
    """보정된 값으로 다시 재면 추정이 0 으로 수렴해 보정이 스스로 풀린다."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "scripts" / "run_policy.py").read_text()
    assert 'self.online_bias.update(sel["candidates"][0, :, -1, :].cpu().numpy(),' in src
    assert 'sel["q_sum"][0].cpu().numpy())' in src, "보정 전 Σ Q̂ 를 써야 한다"
    model = (Path(__file__).resolve().parents[1] / "rus_policy" / "model.py").read_text()
    i_q = model.index("q_sum = score.reshape(B, n_samples).clone()")
    i_bias = model.index("if action_bias is not None:")
    assert i_q < i_bias, "q_sum 을 보정 뒤에 뜨고 있다"
