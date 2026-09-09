"""C10UR 극좌표 세션 → 부채꼴 B-mode 변환 (rus_policy.bmode)."""
import numpy as np

from rus_policy.bmode import BmodeConverter, is_polar_session, letterbox_square


def test_identity_for_sl2c():
    conv = BmodeConverter({"us": {"probe": "sl2c"}}, (256, 256), 256)
    assert not conv.polar and conv.describe()["mode"] == "identity"
    f = np.random.randint(0, 255, (3, 256, 256), np.uint8)
    assert conv.convert_all(f) is f or np.array_equal(conv.convert_all(f), f)


def test_polar_detect_and_convert():
    meta = {"us": {"probe": "c10ur", "fan_geometry": {"radius_mm": 59, "half_angle_deg": 28, "depth_mm": 220}}}
    assert is_polar_session(meta, (160, 512))
    assert is_polar_session({"us": {}}, (160, 512))
    assert is_polar_session({"us": {}}, (320, 256))     # 2026-09-09 이전 메타 (잘못된 배치) 도 극좌표로 본다
    conv = BmodeConverter(meta, (160, 512), 128)
    polar = np.zeros((160, 512), np.uint8)
    polar[:, 200:220] = 255            # 모든 A-line 의 같은 깊이 → 부채꼴에서는 원호
    out = conv(polar)
    assert out.shape == (128, 128) and out.dtype == np.uint8
    assert out.max() > 200 and (out > 0).mean() < 0.5
    d = conv.describe()
    assert d["mode"] == "scan_convert+letterbox" and d["out_size"] == 128
    batch = conv.convert_all(np.stack([polar, polar]))
    assert batch.shape == (2, 128, 128)


def test_letterbox_square_isotropic():
    img = np.zeros((100, 200), np.uint8); img[:] = 200
    out = letterbox_square(img, 64)
    assert out.shape == (64, 64)
    assert out[:16].max() == 0 and out[32].min() > 0     # 위·아래 검정, 가운데 채움
