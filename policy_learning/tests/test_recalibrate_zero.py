"""전원 재기동 뒤 전기 영점만 다시 잡기 — 변환을 거꾸로 되돌리는 수식이 맞는가.

전원을 껐다 켜면 스트레인게이지의 전기 영점이 바뀐다. 그것은 센서 프레임의 상수이고,
센서→프로브 회전이 자세와 무관한 고정 회전이므로 프로브 프레임에서도 상수로 나타난다.
자세를 아는 경로에서 실제로 값을 바꾸는 것은 중력 모델의 residual_bias 하나뿐이다
(bias.bias 는 compensate 안에서 상쇄된다).
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "policy_learning" / "scripts"))
sys.path.insert(0, str(ROOT / "fr5_control" / "fr5_control"))


def _module():
    spec = importlib.util.spec_from_file_location(
        "recalibrate_zero", ROOT / "policy_learning" / "scripts" / "recalibrate_zero.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reg(mod):
    return mod._registration({
        "mounting_angle_deg": 43.0, "axial_flip": False,
        "r_sensor_to_probe_m": [0.0, 0.0, 0.19891], "flange_to_sensor_rpy": [0.0, 0.0, 0.8203]})


def test_sensor_shift_inverts_the_transform():
    mod = _module()
    from fr5_control.wrench_frames import wrench_rotate, wrench_translate

    reg = _reg(mod)
    rot, r_sp = reg.rotation_probe_from_sensor(), np.asarray(reg.r_sensor_to_probe_m, float)
    delta = np.array([0.7, -1.3, 3.9, 0.02, -0.05, 0.01])          # 센서 프레임의 영점 이동
    seen = wrench_translate(wrench_rotate(rot, delta), r_sp)        # 프로브 프레임에서 이렇게 보인다
    np.testing.assert_allclose(mod.sensor_shift(seen, reg), delta, atol=1e-9)


def test_fix_cancels_a_power_cycle_offset_in_compensate():
    """실제 compensate 로 왕복: 영점이 4.2 N 옮겨진 센서를 고치면 공중에서 0 이 된다."""
    mod = _module()
    from fr5_control.wrench_calibration import BiasResult, GravityModel
    from fr5_control.wrench_profile import CalibrationProfile, compensate

    reg = _reg(mod)
    g = GravityModel(mass_kg=0.2306, com_sensor_m=np.array([-0.005, -0.002, 0.116]),
                     residual_bias=np.array([48.1, -11.8, -836.8, -0.18, 0.45, 1.02]),
                     rms_force_n=0.19, rms_torque_nm=0.003,
                     per_axis_force_n=np.zeros(3), per_axis_torque_nm=np.zeros(3),
                     coverage=0.45, poses=6, valid=True,
                     rotation_sensor_from_flange=np.eye(3))
    prof = CalibrationProfile(
        registration=reg, gravity=g,
        bias=BiasResult(bias=g.residual_bias.copy(), std=np.zeros(6), samples=0,
                        duration_s=0.0, accepted=True, reason="시험"))
    rot_bf = np.eye(3)                                   # 자세 하나면 충분하다 (상수 오차다)
    g_s = g.rotation_sensor_from_flange @ (rot_bf.T @ np.array([0.0, 0.0, -9.80665]))
    raw_ok = g.predict(g_s)                              # 영점이 맞을 때의 원값 = 예측값
    assert np.linalg.norm(compensate(prof, raw_ok, rot_bf).contact_probe[:3]) < 1e-9

    drift = np.array([1.1, -0.9, 4.0, 0.01, -0.02, 0.005])          # 전원 재기동으로 이만큼 밀림
    raw_bad = raw_ok + drift
    seen = compensate(prof, raw_bad, rot_bf).contact_probe
    assert np.linalg.norm(seen[:3]) > 3.0, "시험 조건이 안 만들어졌다"

    g.residual_bias = g.residual_bias + mod.sensor_shift(seen, reg)  # 이 스크립트가 하는 일
    fixed = compensate(prof, raw_bad, rot_bf).contact_probe
    assert np.linalg.norm(fixed[:3]) < 1e-9, fixed


def test_refuses_without_a_gravity_model(tmp_path, monkeypatch, capsys):
    """중력 모델이 없으면 전기 영점만으로는 보상이 성립하지 않는다."""
    mod = _module()
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"registration": {
        "mounting_angle_deg": 43.0, "axial_flip": False,
        "r_sensor_to_probe_m": [0, 0, 0.2], "flange_to_sensor_rpy": [0, 0, 0.82]}, "gravity": None}))
    monkeypatch.setattr(sys, "argv", ["x", "--profile", str(path)])
    assert mod.main() == 1
    assert "중력 모델이 없다" in capsys.readouterr().out


def test_missing_profile_is_reported(tmp_path, monkeypatch, capsys):
    mod = _module()
    monkeypatch.setattr(sys, "argv", ["x", "--profile", str(tmp_path / "nope.json")])
    assert mod.main() == 1
    assert "교정 파일이 없다" in capsys.readouterr().out
