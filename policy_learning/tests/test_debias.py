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
    assert "action_bias=self.action_bias" in src, "선택에 실제로 넘기지 않는다"
    assert '"action_bias": None if self.action_bias is None' in src, "쓴 값이 meta 에 안 남는다"
    assert str(default_bias_path("x.pt")).endswith("x.pt.action_bias.json")
    assert load_action_bias(Path("/nonexistent/none.json")) is None, "없으면 None 이어야 한다"
