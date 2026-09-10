"""지각의 스트리밍 경로 — 실시간 한 장씩과 배치 순차가 같은 결과를 내야 한다.

U-Net 없이 검증하려고 주입되는 콜러블(predictor · extract_control_state · resize_image)만
가짜로 둔다. 확인하려는 것은 모델이 아니라 **previous_state 가 프레임 사이로 이어지는가** 이다.
2026-09-10: 실시간 노드가 매 프레임 run(frame[None]) 을 불러 모든 프레임이 "첫 프레임" 으로
처리되던 결함을 잡고 넣은 시험이다 (state 안의 quality 가 직접 틀어졌다).
"""

import sys
import types

import numpy as np
import pytest

from rus_policy.perception import STATE_DIM, UnetPerception


@pytest.fixture
def fake_rus_perception():
    """rus_perception.data.io.resize_image 만 있으면 스트리밍 로직은 돈다."""
    saved = {k: sys.modules.get(k) for k in
             ("rus_perception", "rus_perception.data", "rus_perception.data.io")}
    root = types.ModuleType("rus_perception")
    data = types.ModuleType("rus_perception.data")
    io = types.ModuleType("rus_perception.data.io")
    io.resize_image = lambda img, size: img
    data.io = io
    root.data = data
    sys.modules.update({"rus_perception": root, "rus_perception.data": data,
                        "rus_perception.data.io": io})
    yield
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def _make(monkeypatch):
    """__init__ 을 우회해 스트리밍 로직만 있는 인스턴스를 만든다."""
    p = object.__new__(UnetPerception)
    p.predictor = types.SimpleNamespace(predict_probability=lambda image: (image, None))
    p.feature_config = p.roi = None
    p.image_size = (8, 8)
    p.beam_axis_px, p.hold_deadband_px = 4.0, 1.0
    p.checkpoint_id = "fake"
    p._prev = None
    # 상태를 이어받는지 보이게: 직전 상태의 depth 를 1 씩 늘린다
    p._extract = lambda prob, image, previous_state, config, roi_mask: {
        "depth": 0 if previous_state is None else previous_state["depth"] + 1}
    monkeypatch.setattr("rus_policy.perception.control_state_to_vector",
                        lambda cs, *a, **k: (np.full(STATE_DIM, cs["depth"], np.float32),
                                             float(cs["depth"]), 0, 0.0))
    return p


def test_step_carries_previous_state(fake_rus_perception, monkeypatch):
    p = _make(monkeypatch)
    frames = np.zeros((5, 8, 8), np.uint8)
    assert [p.step(f)[0][0] for f in frames] == [0, 1, 2, 3, 4]   # 이어받으면 단조 증가
    p.reset()
    assert p.step(frames[0])[0][0] == 0                            # reset 이면 다시 첫 프레임


def test_batch_run_equals_repeated_step(fake_rus_perception, monkeypatch):
    frames = np.arange(5 * 8 * 8, dtype=np.uint8).reshape(5, 8, 8)
    batch = _make(monkeypatch).run(frames, progress=False)
    b = _make(monkeypatch)
    stream = np.stack([b.step(f)[0] for f in frames])
    assert np.allclose(batch.state, stream)        # 학습(배치) 과 실시간(스트림) 이 일치
    assert batch.state[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert np.allclose(batch.quality, [0, 1, 2, 3, 4])
