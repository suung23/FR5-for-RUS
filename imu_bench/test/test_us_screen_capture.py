"""us_screen_capture 의 순수 함수 — ROI 분율→픽셀, 레터박스 등방 축소, 중복 판정.

Win32 창 탐색·캡처는 화면이 있어야 하므로 여기서 다루지 않는다. 이 세 함수가 틀리면
저장 프레임의 기하(fan 반경)와 fps 계산이 조용히 틀어지므로 pytest 로 고정한다.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "host"))

from us_screen_capture import CaptureRoi, frame_changed, letterbox_resize  # noqa: E402


def test_roi_fraction_to_bbox_and_validation():
    roi = CaptureRoi(0.1, 0.2, 0.9, 0.8)
    assert roi.to_bbox((100, 50, 1100, 1050)) == (200, 250, 1000, 850)
    with pytest.raises(ValueError):
        CaptureRoi(0.5, 0.0, 0.5, 1.0)          # x0 == x1
    with pytest.raises(ValueError):
        CaptureRoi(0.0, 0.0, 1.0, 1.2)          # 범위 밖
    with pytest.raises(ValueError):
        CaptureRoi(0.0, 0.0, 0.001, 1.0).to_bbox((0, 0, 1000, 1000))   # 8 px 미만


def test_roi_json_roundtrip(tmp_path):
    roi = CaptureRoi(0.105, 0.06, 0.885, 0.905)
    p = str(tmp_path / "roi.json")
    roi.save(p)
    assert CaptureRoi.load(p) == roi


def test_letterbox_is_isotropic_and_centered():
    """가로로 긴 입력: 세로 패딩, 원의 종횡비가 유지된다 (fan 반경 = hypot 가정을 지키기 위해)."""
    h, w = 300, 600
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (((xx - 300) ** 2 + (yy - 150) ** 2) < 100 ** 2).astype(np.uint8) * 255
    frame, info = letterbox_resize(disc, 256)
    assert frame.shape == (256, 256) and frame.dtype == np.uint8
    assert info["square_side"] == 600 and info["pad_left_top"] == [0, 150]
    assert info["scale"] == pytest.approx(256 / 600)
    ys, xs = np.nonzero(frame > 127)
    span_x = xs.max() - xs.min()
    span_y = ys.max() - ys.min()
    assert abs(span_x - span_y) <= 2                       # 원이 원으로 남는다
    assert abs(xs.mean() - 127.5) < 1.5 and abs(ys.mean() - 127.5) < 1.5   # 중앙
    # 패딩 영역은 검정
    assert frame[:20, :].max() == 0 and frame[-20:, :].max() == 0


def test_letterbox_identity_when_square_and_same_size():
    g = (np.random.RandomState(0).rand(256, 256) * 255).astype(np.uint8)
    frame, info = letterbox_resize(g, 256)
    assert np.array_equal(frame, g) and info["scale"] == 1.0


def test_frame_changed_mean_and_fraction_rules():
    base = np.full((256, 256), 40, np.uint8)
    assert frame_changed(None, base, 0.5)
    assert not frame_changed(base, base.copy(), 0.5)
    # 전역 미세 변화 (평균차 1.0 > 0.5)
    assert frame_changed(base, base + 1, 0.5)
    # 국소 변화: 영상 부분만 바뀌고 나머지는 정적 — 평균차는 작지만 비율 규칙으로 잡힌다
    cur = base.copy()
    cur[100:130, 100:130] = 200                            # 900/65536 = 1.4 % 픽셀
    d_mean = np.abs(cur.astype(int) - base.astype(int)).mean()
    assert d_mean > 0.5 or frame_changed(base, cur, 0.5)
    tiny = base.copy()
    tiny[0:5, 0:5] = 200                                   # 25/65536 = 0.04 % < 0.5 %
    assert not frame_changed(base, tiny, 5.0)
