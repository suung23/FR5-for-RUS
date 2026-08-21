#!/usr/bin/env python3
"""qc_track 자체 시험 — 센서도 로봇도 없이 돌아간다.

    python3 -m pytest imu_bench/qc_track/test_qc_track.py -q

두 층으로 본다.

  1. **단위**   회전 규약, 시계 정합, ZUPT, 정지 판정, 록인, 상호상관.
  2. **끝에서 끝까지**  시뮬레이터로 **알려진 값을 심은** 세션을 만들고 분석기가
     그것을 되찾아 오는지. 되찾지 못하면 센서 이야기가 아니라 분석기 버그이고,
     실기 데이터로는 그 둘을 구분할 수 없다.

2 번이 이 파일의 핵심이다. 실제로 개발 중에 세 개를 여기서 잡았다 —
가속도를 IMU world 에 둔 채 base 변위와 뺀 것, 축각 보간이 180 deg 에서 발산한 것,
지렛대를 스케일 오차와 함께 풀지 않아 엉뚱한 값이 나온 것.
"""
import inspect
import os
import subprocess
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "host"))

import analyze_track as at             # noqa: E402
import protocol                        # noqa: E402
import qc_common as qc                 # noqa: E402
import reference as ref                # noqa: E402

_HERE_RAW = qc.RAW                     # use_raw_dir 로 옮긴 뒤 되돌릴 자리


# ------------------------------------------------------------------ 단위
def test_probe_axis_is_plus_z():
    """프로브 침투축은 tool +z 다. Surgilogger 의 샤프트(-z)와 부호가 반대라
    복사해 오면 하방각과 롤이 통째로 뒤집힌다."""
    R = np.eye(3)[None]
    assert np.allclose(qc.probe_axis(R)[0], [0, 0, 1])


def test_tilt_is_complement_of_depression():
    """tilt + depression = 90 deg. 기준 보고서의 하방각과 바로 비교하기 위한 항등식."""
    rng = np.random.default_rng(0)
    u = rng.normal(size=(50, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    depression = np.degrees(np.arcsin(np.clip(-u[:, 2], -1, 1)))
    assert np.allclose(qc.tilt_deg(u) + depression, 90.0, atol=1e-9)


def test_tip_xy_survives_vertical_probe():
    """프로브가 정확히 연직일 때도 tip_x/tip_y 는 정의된다 (0 이 나와야 한다).
    '기울기 크기 + 방위' 로 쪼갰다면 여기서 0/0 이 된다."""
    tx, ty = qc.tip_xy_deg(np.array([[0.0, 0.0, -1.0]]))
    assert abs(tx[0]) < 1e-9 and abs(ty[0]) < 1e-9


def test_spin_reference_survives_vertical_probe():
    """프로브가 연직이어도 spin 이 무너지지 않아야 한다 — 지평선 기준이면 여기서
    투영이 0 벡터가 된다 (그래서 base +x 를 기준선으로 쓴다)."""
    th = np.radians(37.0)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    down = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
    got = qc.spin_deg((Rz @ down)[None])[0]
    assert np.isfinite(got)
    assert abs(abs(got) - 37.0) < 1e-6


def test_fit_clock_recovers_skew():
    """하단 포락선 정합. pc_ts 가 계단으로 양자화돼 있어도 skew 를 되찾아야 한다."""
    n = 20000
    dev = np.arange(n) / 250.0
    skew_ppm = -25.0
    pc = 1e9 + dev * (1.0 + skew_ppm * 1e-6)
    pc_q = 1e9 + np.ceil((pc - 1e9) / 0.005) * 0.005      # 시리얼 read 계단
    a, b, info = qc.fit_clock(dev, pc_q)
    assert info["ok"]
    # pc = dev*(1+s) 로 만들었으므로 a = 1+s, 즉 보고되는 skew 는 +s 다.
    # 계단 양자화가 하단 포락선을 조금 밀어 올리므로 몇 ppm 은 남는다 —
    # 100 s 구간에서 1 ms 이하라 지연 추정에 영향이 없다 (Exp-Latency §3.2).
    assert abs(info["skew_ppm"] - skew_ppm) < 8.0
    _ = (a, b)


def test_zupt_removes_constant_bias():
    """ZUPT 는 상수 가속도 바이어스를 **원리적으로** 지운다. 이 성질이 깨지면
    변위 QC 전체가 뜻을 잃는다."""
    dt, n = 0.005, 400
    bias = np.tile([0.03, -0.02, 0.05], (n, 1))
    d = at.zupt_integrate(bias, dt, zupt=True)[-1]
    assert np.linalg.norm(d) < 1e-9
    d_no = at.zupt_integrate(bias, dt, zupt=False)[-1]
    assert np.linalg.norm(d_no) > 1e-3


def test_xcorr_recovers_known_lag():
    dt = 0.005
    t = np.arange(0, 20, dt)
    x = np.sin(2 * np.pi * 0.3 * t) + 0.4 * np.sin(2 * np.pi * 0.7 * t + 1.0)
    lag = 0.035
    y = np.interp(t - lag, t, x)
    got, corr = at.xcorr_lag(dt, x, y)
    assert abs(got - lag) < 0.003 and corr > 0.99


def test_xcorr_recovers_negative_lag():
    """센서가 GT 보다 **앞선** 경우도 읽어야 한다.

    GT 쪽에는 컨트롤러 UDP 상태 패키지가 PC 에 닿는 몫이 이미 실려 있어, 그 몫이
    IMU 자신의 지연보다 크면 센서가 앞선 것으로 보인다. 한쪽만 뒤지던 판에서는
    이 경우가 0 으로 잘려 "지연 0 ms — 통과" 가 나왔다. 실기 첫 run 이 네 채널
    모두 정확히 0.0 ms 였고, 양쪽을 열자 -15 ~ -50 ms 로 드러났다.
    """
    dt = 0.005
    t = np.arange(0, 20, dt)
    x = np.sin(2 * np.pi * 0.3 * t) + 0.4 * np.sin(2 * np.pi * 0.7 * t + 1.0)
    lag = -0.035                                   # 센서가 GT 보다 35 ms 앞선다
    y = np.interp(t - lag, t, x)
    got, corr = at.xcorr_lag(dt, x, y)
    assert abs(got - lag) < 0.003 and corr > 0.99


def test_zero_bias_key_contract():
    """분석기가 읽는 키에 **실기 로거가 쓰는 키**가 들어 있어야 한다.

    이것이 갈라져 있었다. 분석기는 b_E 만 봤고 실기 로거는 accel_bias_earth 로
    썼다 — 시뮬레이터도 b_E 를 써서 자체 시험 19 개가 전부 통과하는 채로, 실기
    에서만 영점이 조용히 블록 내 평균으로 대체됐다. 그래서 이 시험은 문자열을
    비교하지 않고 **ZeroReference 가 실제로 내놓는 키**를 묻는다.
    """
    from zero_ref import ZeroReference
    keys = set(ZeroReference(q0=[1.0, 0.0, 0.0, 0.0], b_E=np.zeros(3),
                             gyro_bias=np.zeros(3), quality={},
                             quat_convention="R").to_dict())
    assert keys & set(at.ZERO_BIAS_KEYS), (
        f"실기 로거가 쓰는 키 {sorted(keys)} 중 어느 것도 "
        f"분석기의 {at.ZERO_BIAS_KEYS} 에 없다")


def test_lockin_recovers_phase():
    t = np.arange(0, 20, 0.004)
    f, lag = 0.6, 0.030
    g = 3.0 * np.sin(2 * np.pi * f * t) + 0.5
    s = 3.0 * np.sin(2 * np.pi * f * (t - lag)) + 0.5
    ag, pg, _ = at.lockin(t, g, f, 0.0)
    as_, ps, _ = at.lockin(t, s, f, 0.0)
    got = -float(at.wrap_pi(ps - pg)) / (2 * np.pi * f)
    assert abs(ag - 3.0) < 1e-6 and abs(as_ - 3.0) < 1e-6
    assert abs(got - lag) < 1e-3


def test_fit_LT_separates_pure_delay():
    """순수지연만 있으면 T ~ 0 이어야 한다. IMU 가 이 모양이어야 정상이다."""
    f = np.array(protocol.FREQS["roll"])
    L = 0.026
    w = 2 * np.pi * f
    fit = at.fit_LT(f, -w * L, np.ones_like(f), np.ones_like(f))
    assert fit["ok"] and abs(fit["L"] - L) < 0.004 and fit["T"] < 0.006


def test_solve_AB_recovers_fixed_rotations():
    """X = A Y B. 정지 자세가 여러 축으로 흩어져 있으면 유일하게 풀려야 한다."""
    rng = np.random.default_rng(3)

    def rot(v):
        th = np.linalg.norm(v)
        k = v / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)

    A = rot(np.array([0.1, -0.2, 0.9]))
    B = rot(np.array([0.5, 0.3, -0.2]))
    X = [rot(rng.normal(size=3) * 0.6) for _ in range(8)]
    Y = [A.T @ x @ B.T for x in X]
    Ah, Bh, obs, resid = at.solve_AB(X, Y)
    assert resid.max() < 1e-6
    assert qc.geodesic_deg(Ah, A) < 1e-6 and qc.geodesic_deg(Bh, B) < 1e-6
    assert obs > 1e-3


def test_still_mask_rejects_sine_turning_point():
    """사인 운동의 변곡점은 속도가 0 이지만 정지가 아니다. 가속도 산포로 걸러야
    한다 — 안 걸러지면 ZUPT 구간이 이동을 물고 들어간다."""
    t = np.arange(0, 10, 0.004)
    w = 2 * np.pi * 0.5
    gyr = np.column_stack([0.4 * np.cos(w * t), np.zeros_like(t), np.zeros_like(t)])
    acc = np.column_stack([np.zeros_like(t), np.zeros_like(t),
                           qc.G0 - 0.4 * w * np.sin(w * t)])
    m = qc.still_mask(t, gyr, acc)
    assert m.mean() < 0.05


def test_reference_table_is_complete():
    """분석기가 기준값을 결과에 박는다. 표가 비면 우리 숫자를 혼자 읽게 된다."""
    d = ref.as_dict()
    for k in ("latency", "alignment", "displacement_floor_mm", "pass"):
        assert d[k]
    assert ref.latency_row("tilt")["eff_ms"] == 12       # depression
    assert ref.latency_row("spin")["eff_ms"] == 28       # roll
    assert ref.latency_row("tip_x") is None


# ------------------------------------------------------- 끝에서 끝까지
@pytest.fixture(scope="module")
def sim_run(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("sim") / "raw_data_sim")
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "simulate_session.py"), "--out", out,
         "--probe-seconds", "70", "--still-seconds", "20", "--no-sweeps"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


@pytest.fixture(scope="module")
def sim_result(sim_run):
    qc.use_raw_dir(sim_run)
    return at.analyse(sources=("chip",), primary="probe"), sim_run


def test_end_to_end_recovers_latency(sim_result):
    res, run = sim_result
    truth = qc.load_json(os.path.join(run, "meta", "program.json"))["truth"]
    got = res["rotation"]["chip"]["geodesic"]["lag_ms"]
    assert abs(got - truth["imu_lag_ms"]) < 3.0


def test_end_to_end_alignment_is_exact(sim_result):
    res, _ = sim_result
    assert res["alignment"]["resid_rms_deg"] < 0.05
    assert res["alignment"]["n"] >= protocol.ALIGN_MIN_POSES


def test_end_to_end_rotation_error_is_small(sim_result):
    res, _ = sim_result
    g = res["rotation"]["chip"]["geodesic"]
    assert g["rms_delay_corrected_deg"] < 0.5
    for ch in at.ANGLE_CHANNELS:
        assert res["rotation"]["chip"]["channels"][ch]["corr"] > 0.99


def test_end_to_end_recovers_lever_arm_and_scale(sim_result):
    """지렛대와 스케일은 **함께** 풀어야 나온다 (identify_accel_model 문서주석)."""
    res, run = sim_result
    truth = qc.load_json(os.path.join(run, "meta", "program.json"))["truth"]
    m = res["model"]
    assert m is not None
    err = np.linalg.norm(np.array(m["lever_mm"]) - np.array(truth["lever_mm"]))
    assert err < 12.0, f"지렛대 {m['lever_mm']} vs {truth['lever_mm']}"
    # C = S^-1 이므로 스케일은 -(심은 스케일-1) 근처여야 한다
    want = 100.0 * (1.0 / truth["accel_scale"] - 1.0)
    assert all(abs(v - want) < 1.0 for v in m["scale_pct"]), m["scale_pct"]
    assert m["accel_resid_after_ms2"] < 0.3 * m["accel_resid_before_ms2"]


def test_end_to_end_error_floor_matches_reference(sim_result):
    """정지 구간 오차 바닥이 imu_bench 실측(QC_PLAN §9) 언저리여야 한다.
    크게 벗어나면 파이프라인이 그때와 다른 것을 재고 있다는 뜻이다."""
    res, _ = sim_result
    fl = res["translation"]["still"]["error_floor_mm"]
    for W in ("0.5", "1.0", "2.0"):
        got = fl[W]["C"]
        want = ref.DISPLACEMENT_FLOOR_MM[float(W)]["C"]
        assert 0.2 * want < got < 4.0 * want, (W, got, want)


def test_end_to_end_calibration_beats_uncalibrated(sim_result):
    """검증 블록에서 D(캘리브 후)가 C 보다 나아야 한다. 안 나아지면 모형이
    적합 블록에만 맞은 것이다."""
    res, _ = sim_result
    vt = res["translation"][res["validation_block"]]
    rows = [r for r in vt["segments"] if "D" in r["eps_mm"]]
    assert rows
    c = np.median([r["eps_mm"]["C"] for r in rows])
    d = np.median([r["eps_mm"]["D"] for r in rows])
    assert d < c, (c, d)


def test_disjoint_block_is_reported_not_silently_dropped(sim_run, tmp_path):
    """두 로거가 **다른 시간대**를 담은 블록은 소리내어 잡혀야 한다.

    실기 첫 run 에서 still 을 다시 딸 때 로봇 로거가 pose 를 한 건도 못 받아
    저장을 건너뛰었고(그러고도 종료코드 0), 한 시간 전 파일이 그대로 남았다.
    겹치는 격자가 없으니 그 블록은 분석에서 통째로 빠졌는데 표에는 짝비율 100 %
    로 멀쩡히 찍혀 있었다. 빠진 것이 눈에 보여야 한다.
    """
    import shutil

    import h5py

    run = str(tmp_path / "raw_data")
    shutil.copytree(sim_run, run)
    with h5py.File(os.path.join(run, "robot", "still.h5"), "r+") as f:
        for k in ("pose/t", "pose/pc_ts", "joints/t", "joints/pc_ts"):
            if k in f:
                f[k][...] = f[k][...] - 10_000.0
    try:
        qc.use_raw_dir(run)
        res = at.analyse(sources=("chip",), primary="probe")
    finally:
        qc.use_raw_dir(sim_run)          # 모듈 픽스처가 이 전역을 공유한다
    assert res["timebase"]["still"]["overlap_s"] < 0
    assert "still" not in res["translation"]


def test_runner_refuses_stale_block(tmp_path):
    """수집 쪽 방어선. 분석기가 잡기 전에 **러너가** 먼저 잡아야 한다.

    2026-08-20 세션에서 log_robot 이 pose 를 못 받아 저장을 건너뛰고도 0 으로
    끝났고, 러너가 그 0 을 믿고 다음 블록으로 넘어갔다. 그 자리에 한 시간 전
    파일이 남아 있어 분석기까지 조용히 그것을 썼다. 세 가지를 지킨다 —
    (1) 종료코드가 0 이 아니면 문제로 잡고, (2) 파일이 안 바뀌었으면 잡고,
    (3) 잡힌 낡은 파일은 분석기가 못 집는 이름으로 치운다.
    """
    import run_session as rs

    raw = str(tmp_path / "raw_data")
    try:
        qc.use_raw_dir(raw)
        qc.ensure_dirs()

        def touch(block):
            for d in (qc.DIR_ROBOT, qc.DIR_IMU):
                open(os.path.join(d, block + ".h5"), "w").close()

        # 정상 — 블록이 두 파일을 새로 썼다
        before = rs.snapshot("probe")
        touch("probe")
        assert rs.check_block("probe", {"returncode": {"robot": 0, "imu": 0}}, before) == []

        # 이미 있던 파일을 다시 쓴 경우(재수집)도 정상이어야 한다.
        # 벽시계와 비교하면 여기서 새는데, 파일시스템 시각이 time.time() 보다
        # ms 단위로 앞설 수 있기 때문이다. 전후 비교라 그 문제가 없다.
        before = rs.snapshot("probe")
        time.sleep(0.01)
        touch("probe")
        assert rs.check_block("probe", {"returncode": {"robot": 0, "imu": 0}}, before) == []

        # 사고 재현 — robot 이 2 로 끝나고 이전 세션 파일만 남아 있다
        open(os.path.join(qc.DIR_ROBOT, "still.h5"), "w").close()
        before = rs.snapshot("still")
        open(os.path.join(qc.DIR_IMU, "still.h5"), "w").close()
        problems = rs.check_block("still", {"returncode": {"robot": 2, "imu": 0}}, before)
        assert problems and "robot" in problems[0]
        assert not os.path.exists(os.path.join(qc.DIR_ROBOT, "still.h5"))
        assert os.path.exists(os.path.join(qc.DIR_ROBOT, "still.stale.h5"))

        # 종료코드가 0 이어도 파일이 안 바뀌었으면 잡는다
        touch("align")
        before = rs.snapshot("align")
        problems = rs.check_block("align", {"returncode": {"robot": 0, "imu": 0}}, before)
        assert len(problems) == 2
    finally:
        qc.use_raw_dir(_HERE_RAW)


def test_census_counts_short_moves(tmp_path):
    """수집 직후의 구간 통계. '짧은 이동이 없다' 를 그 자리에서 알아야 한다.

    2026-08-20 run 은 probe 90 초를 다 쓰고도 0.5~1.5 s 이동이 하나뿐이었다.
    변위 오차는 구간 길이에 t^2 로 자라므로 그 칸이 곧 사용가능 영역인데,
    비었다는 사실을 한 시간 뒤 분석에서야 알았다.
    """
    import run_session as rs

    fs = 200.0
    plan = [("still", 2.0), ("move", 1.0), ("still", 2.0), ("move", 5.0),
            ("still", 2.0), ("move", 1.2), ("still", 2.0)]
    t, gyr, acc = [], [], []
    now = 1_000_000.0
    for kind, dur in plan:
        n = int(dur * fs)
        tt = now + np.arange(n) / fs
        now = tt[-1] + 1 / fs
        t.append(tt)
        if kind == "still":
            gyr.append(np.zeros((n, 3)))
            acc.append(np.tile([0.0, 0.0, 9.81], (n, 1)))
        else:
            w = 0.8 * np.sin(2 * np.pi * 0.8 * (tt - tt[0]))
            gyr.append(np.column_stack([w, 0.3 * w, 0.1 * w]))
            acc.append(np.column_stack([2.0 * w, 0.5 * w, 9.81 + 0.5 * w]))
    cols = {"t": np.concatenate(t), "gyr": np.vstack(gyr), "acc": np.vstack(acc)}

    raw = str(tmp_path / "raw_data")
    try:
        qc.use_raw_dir(raw)
        qc.ensure_dirs()
        qc.save_table(os.path.join(qc.DIR_IMU, "probe"), cols, {"block": "probe"})
        c = rs.census("probe")
        assert c is not None
        # 1.0 s 와 1.2 s 는 짧은 이동, 5 s 는 아니다
        assert c["short"] == 2, c
        assert c["moves"] >= 3, c
        # 모자란 것이 경고로 나와야 한다 (기본 요구는 8 개)
        assert any("짧은 이동" in w for w in rs.print_census("probe", c))
    finally:
        qc.use_raw_dir(_HERE_RAW)


def test_result_json_carries_reference(sim_result):
    res, _ = sim_result
    assert res["reference"]["latency"]["depression"]["eff_ms"] == 12
    assert res["verdict"] and all("limit" in v for v in res["verdict"])


def test_cal_gate_catches_unconverged_chip(monkeypatch, capsys):
    """2026-08-20 세션을 못 쓰게 만든 것을 **수집 전에** 잡는다.

    그때는 칩 보정 정확도를 아예 버리고 있어서, 자력계가 자세를 악화시킨 원인이
    '로봇이 만드는 장' 인지 '보정이 애초에 수렴한 적이 없는 것' 인지 가를 수
    없었다. 지금은 기록에 남지만 기록만으로는 세션이 끝난 뒤에야 안다 —
    90 초를 다 쓰고 rv=0 이었음을 아는 것은 늦다.

    세 가지를 지킨다 — (1) 문턱 미만이면 잡고, (2) 6 축으로 돌 때는 자력계
    채널을 안 보고, (3) 못 읽은 채널을 '통과' 로 치지 않는다.
    """
    import run_session as rs

    class Args:
        port = "/dev/null"
        no_mag = False
        no_cal_check = False
        yes = True                       # 사람 입력 없이 한 바퀴만 돌게 한다

    ok = {"cal_status": {"acc": 3, "gyr": 3, "mag": 2, "rv": 3}}
    bad = {"cal_status": {"acc": 3, "gyr": 0, "mag": 1, "rv": 0}}
    missing = {"cal_status": {"acc": 3, "gyr": 3, "mag": None, "rv": 3}}

    assert rs.cal_shortfall(ok, use_mag=True) == []
    assert {k for k, _, _ in rs.cal_shortfall(bad, use_mag=True)} == {"mag", "rv"}
    # 6 축이면 자력계는 안 보지만 칩 RV 는 어차피 9 축이라 그대로 걸린다
    assert {k for k, _, _ in rs.cal_shortfall(bad, use_mag=False)} == {"rv"}
    # gyr 은 **절대 게이트가 되면 안 된다.** 2026-08-21 세션 90 분 내내 0 이었고
    # (rv 는 2~3, mag 은 2), 조작으로 올릴 방법이 없었다. 여기에 문턱을 걸면
    # 사람이 매번 [s] 로 넘기게 되고 게이트 전체가 무력해진다 — 실제로 그랬다.
    stuck = {"cal_status": {"acc": 3, "gyr": 0, "mag": 2, "rv": 3}}
    assert rs.cal_shortfall(stuck, use_mag=True) == []
    assert rs.cal_shortfall(stuck, use_mag=False) == []
    # 못 읽은 채널을 통과로 치면 게이트가 있으나 마나 해진다
    assert {k for k, _, _ in rs.cal_shortfall(missing, use_mag=True)} == {"mag"}

    monkeypatch.setattr(rs, "read_cal", lambda args: bad)
    got = rs.cal_gate("probe", Args())
    out = capsys.readouterr().out
    assert "모자람" in out and "rv" in out
    # 막고 끝내는 게 아니라 **알리고 기록에 남긴다**
    assert got["overridden"] is True

    monkeypatch.setattr(rs, "read_cal", lambda args: ok)
    rs.cal_gate("probe", Args())
    assert "모자람" not in capsys.readouterr().out

    # --no-cal-check 는 포트를 아예 안 연다
    def boom(args):
        raise AssertionError("--no-cal-check 인데 포트를 열었다")
    monkeypatch.setattr(rs, "read_cal", boom)
    Args.no_cal_check = True
    assert rs.cal_gate("probe", Args()) is None


def test_monitor_counts_match_the_analyser(tmp_path):
    """수집 중 화면이 분석기와 **같은 답**을 내야 한다.

    이 화면의 존재 이유가 '분석에서 0 개로 드러나기 전에 현장에서 알려 준다' 인데,
    화면이 분석기보다 느슨하면 사람을 안심시키는 일만 하게 된다 — 2026-08-21 의
    census 가 정확히 그 실패였다 (0.8 s 로 세어 "정지 9 개", 분석기는 2.5 s 로 0 개).

    CSV 를 조각조각 이어붙이며 따라 읽어도(실제 수집과 같은 모양) 전체를 한 번에
    돌린 것과 구간 개수가 같은지 본다. 조각 크기를 일부러 들쭉날쭉하게 준다 —
    창 폭이 조각마다 흔들리면 앞뒤가 다른 창으로 계산돼 여기서 걸린다.
    """
    import monitor as mo

    rng = np.random.default_rng(0)
    fs, n = 250.0, 6000
    t = np.arange(n) / fs
    gyr = np.zeros((n, 3))
    acc = np.tile([0.0, 0.0, 9.81], (n, 1))
    # 정지 - 이동 - 정지 - 짧은 이동 - 정지
    for a, b in ((1200, 2000), (3600, 3900)):
        gyr[a:b] = rng.normal(0, 0.5, (b - a, 3))
        acc[a:b] += rng.normal(0, 1.5, (b - a, 3))
    gyr += rng.normal(0, 1e-4, gyr.shape)
    acc += rng.normal(0, 1e-3, acc.shape)

    raw = str(tmp_path / "raw_data")
    try:
        qc.use_raw_dir(raw)
        qc.ensure_dirs()
        path = os.path.join(qc.DIR_IMU, "probe_raw_20260101_000000.csv")
        from imu_log import COLUMNS
        rows = []
        for i in range(n):
            r = [""] * len(COLUMNS)
            r[mo.COL["dev_us"]] = "%d" % round(t[i] * 1e6)
            for k, name in enumerate(("gx", "gy", "gz")):
                r[mo.COL[name]] = "%.6f" % gyr[i, k]
            for k, name in enumerate(("ax", "ay", "az")):
                r[mo.COL[name]] = "%.6f" % acc[i, k]
            for name in ("cal_acc", "cal_gyr", "cal_mag", "cal_rv"):
                r[mo.COL[name]] = "3"
            rows.append(",".join(r) + "\n")

        with open(path, "w") as fh:
            fh.write(",".join(COLUMNS) + "\n")
        st = mo.BlockState("probe", path)
        i = 0
        while i < n:                                    # 들쭉날쭉하게 흘려 넣는다
            j = min(i + int(rng.integers(120, 1500)), n)
            with open(path, "a") as fh:
                fh.writelines(rows[i:j])
            st.poll()
            i = j

        assert len(st.csv.rows) == n
        full = qc.still_mask(st.t, st.gyr, st.acc, dt=st._dt)
        for min_s in (0.5, 0.8, 1.5, 2.5):
            assert (len(qc.segments_from_mask(st.t, st.mask, min_s))
                    == len(qc.segments_from_mask(st.t, full, min_s))), min_s
        assert len(st.moves()) == len(qc.segments_from_mask(st.t, ~full, 0.2))
        # 블록마다 분석기가 쓰는 문턱을 그대로 써야 한다
        assert mo.BlockState("align", path).hold_min_s == protocol.ALIGN_HOLD_S
        assert mo.BlockState("mag_map", path).hold_min_s == protocol.MAG_MAP_HOLD_S
        assert st.hold_min_s == protocol.ZUPT_MIN_STILL_S
        assert st.cal == [3, 3, 3, 3]
    finally:
        qc.use_raw_dir(_HERE_RAW)


def test_monitor_survives_a_file_that_has_no_header_yet(tmp_path):
    """생성 직후의 CSV 를 열어도 죽지 않아야 한다.

    2026-08-21 13:45, align 을 다시 딴 그 순간 모니터가 죽었다. 러너가 새 CSV 를
    만들자마자 모니터가 그것을 열었는데 헤더가 아직 파이썬 버퍼에 있어 파일이
    비어 있었다. '첫 줄을 건너뛴다' 는 방식은 그 빈 파일에서 아무것도 못 건너뛰고,
    나중에 도착한 헤더를 데이터로 먹어 float('dev_us') 로 터졌다.

    모니터가 조용히 멈추면 낡은 값을 계속 보여주므로 **없는 것보다 나쁘다.**
    """
    import monitor as mo
    from imu_log import COLUMNS

    path = str(tmp_path / "align_raw_20260101_000000.csv")
    open(path, "w").close()                             # 빈 파일 — 헤더가 아직 없다
    tail = mo.CsvTail(path)
    assert tail.poll() == 0

    def row(dev_us):
        r = [""] * len(COLUMNS)
        r[mo.COL["dev_us"]] = str(dev_us)
        for name in ("gx", "gy", "gz", "ax", "ay", "az"):
            r[mo.COL[name]] = "0.0"
        return ",".join(r)

    with open(path, "a") as fh:                         # 헤더가 뒤늦게 도착한다
        fh.write(",".join(COLUMNS) + "\n" + row(1000) + "\n" + row(5000) + "\n")
    assert tail.poll() == 2                             # 헤더는 안 세고 데이터만
    assert mo._col(tail.rows, "dev_us").tolist() == [1000.0, 5000.0]


def test_mag_map_plan_actually_produces_usable_pairs():
    """스크립트 mag_map 의 계획이 **설계상** 쓸 수 있는 쌍을 낸다는 것.

    손으로 두 번 하고 두 번 다 쌍이 0 개였다 (2026-08-21: 정지 18 개 / 153 쌍 중
    자세차 3 deg 이하가 2 개, 그 둘은 1 mm 와 39 mm 떨어져 있었다). 계획을 바꾼
    이상, 그 계획이 실제로 조건을 만족하는지는 코드가 답해야 한다 — 로봇을 세워
    놓고 알아내는 것이 아니라.

    자세 묶음 **안에서만** 쌍이 성립해야 한다 (묶음 사이는 일부러 25 deg 돌린다).
    """
    legs = protocol.plan_mag_map()
    groups, cur = [[]], np.zeros(3)
    for kind, payload in legs:
        if kind == "move":
            cur = np.array(payload, float)
        elif kind == "turn":
            groups.append([])
        else:
            groups[-1].append(cur.copy())

    assert len(groups) == protocol.MAG_MAP_POSES
    assert all(len(g) == protocol.MAG_MAP_POSITIONS for g in groups)

    usable = 0
    for g in groups:                                    # 자세가 같은 자리들끼리
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                if np.linalg.norm(g[i] - g[j]) >= protocol.MAG_MAP_MIN_DP_MM:
                    usable += 1
    assert usable >= protocol.MAG_MAP_MIN_PAIRS, usable
    # 같은 자리를 두 번 세는 계획이면 쌍이 부풀 뿐 정보가 안 는다
    for g in groups:
        d = [np.linalg.norm(g[i] - g[j])
             for i in range(len(g)) for j in range(i + 1, len(g))]
        assert min(d) >= protocol.MAG_MAP_MIN_DP_MM, min(d)
    # 묶음 사이는 자세가 달라야 하므로 회전이 실제로 들어 있어야 한다
    assert sum(1 for k, _ in legs if k == "turn") == protocol.MAG_MAP_POSES - 1


def test_monitor_reports_the_same_observability_as_the_analyser():
    """정렬 관측도를 수집 중에 낸다. 분석기와 같은 값이어야 한다.

    유지 시간만 보여주는 화면은 절반짜리였다. 2026-08-21 재수집은 유지 시간을
    1.3 s -> 2.2 s 로 고쳤는데도 못 썼다 — 자세들이 서로 5~18 deg 밖에 안 벌어져
    관측도가 0.12 였기 때문이다 (지시문은 아래/위/+-x/+-y, 즉 90~180 deg 를
    요구한다).

    **관측도가 낮으면 잔차는 어느 쪽으로도 간다.** 실측 08-21 은 자세 3 개에
    잔차 1.16 deg 로 작게 나왔고 (미지수가 더 많아 아무 데나 맞는다), 아래 합성
    예는 같은 상황에서 13 deg 로 크게 나온다 (영벡터를 엉뚱하게 고른다). 어느
    쪽이든 그 잔차로는 아무것도 못 말한다. 그래서 봐야 할 값은 관측도다.
    """
    import monitor as mo

    rng = np.random.default_rng(3)
    A = qc.project_SO3(np.eye(3) + 0.3 * rng.normal(size=(3, 3)))
    B = qc.project_SO3(np.eye(3) + 0.3 * rng.normal(size=(3, 3)))

    def block(axes_deg):
        """자세 목록을 주면 (FakeState, FakeGt) 를 만든다."""
        fs, hold, gap = 250.0, 3.0, 1.0
        t, gyr, acc, cq, gt_t, gt_q, gt_p = [], [], [], [], [], [], []
        now = 1_700_000_000.0
        cur = now
        for ang, axis in axes_deg:
            th = np.radians(ang)
            k = np.array(axis, float); k /= np.linalg.norm(k)
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            X = qc.project_SO3(np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K)
            Y = qc.project_SO3(A.T @ X @ B.T)
            n = int(hold * fs)
            t.append(cur + np.arange(n) / fs)
            gyr.append(np.full((n, 3), 1e-5))
            acc.append(np.tile([0.0, 0.0, 9.81], (n, 1)))
            w = np.sqrt(max(1e-12, 1 + np.trace(Y))) / 2
            cq.append(np.tile([w, (Y[2, 1] - Y[1, 2]) / (4 * w),
                               (Y[0, 2] - Y[2, 0]) / (4 * w),
                               (Y[1, 0] - Y[0, 1]) / (4 * w)], (n, 1)))
            m = int(hold * 150)
            gt_t.append(cur + np.arange(m) / 150.0)
            wx = np.sqrt(max(1e-12, 1 + np.trace(X))) / 2
            gt_q.append(np.tile([(X[2, 1] - X[1, 2]) / (4 * wx),
                                 (X[0, 2] - X[2, 0]) / (4 * wx),
                                 (X[1, 0] - X[0, 1]) / (4 * wx), wx], (m, 1)))
            gt_p.append(np.zeros((m, 3)))
            cur += hold + gap
            n2 = int(gap * fs)                          # 사이의 '이동'
            t.append(cur - gap + np.arange(n2) / fs)
            gyr.append(rng.normal(0, 0.5, (n2, 3)))
            acc.append(np.tile([0.0, 0.0, 9.81], (n2, 1)) + rng.normal(0, 1.5, (n2, 3)))
            cq.append(np.tile(cq[-1][0], (n2, 1)))

        class FakeState:
            block = "align"
            hold_min_s = protocol.ALIGN_HOLD_S

            def __init__(self):
                self.t = np.concatenate(t)
                self.gyr = np.vstack(gyr)
                self.acc = np.vstack(acc)
                self.chip_q = np.vstack(cq)
                self._m = qc.still_mask(self.t, self.gyr, self.acc)

            def holds(self):
                return qc.segments_from_mask(self.t, self._m, self.hold_min_s)

            def now(self):
                return False, 0.0

        st = FakeState()
        shift = time.time() - st.t[-1]

        class FakeGt:
            ok = True

            def snapshot(self):
                return (np.concatenate(gt_t) + shift, np.vstack(gt_p), np.vstack(gt_q))

        return st, FakeGt()

    # 크게 갈린 자세 — 관측도가 살아 있어야 한다
    wide = [(0, (0, 0, 1)), (80, (1, 0, 0)), (80, (0, 1, 0)), (-70, (1, 0, 0)),
            (60, (1, 1, 0))]
    st, gt = block(wide)
    a = mo.align_state(st, gt)
    assert a["n"] == len(wide), a["n"]
    assert a["obs"] > 0.5, a["obs"]
    assert a["max_pair_deg"] > 60.0, a["max_pair_deg"]

    # 2026-08-21 을 닮은 것 — 한 축으로만 조금씩. 관측도가 무너져야 한다
    narrow = [(0, (0, 0, 1)), (6, (1, 0, 0)), (12, (1, 0, 0)), (18, (1, 0, 0))]
    st2, gt2 = block(narrow)
    a2 = mo.align_state(st2, gt2)
    assert a2["n"] == len(narrow)
    assert a2["obs"] < a["obs"] / 3, (a2["obs"], a["obs"])
    assert a2["max_pair_deg"] < 25.0, a2["max_pair_deg"]
    # 잔차에는 아무 문턱도 걸지 않는다 — 위 docstring 참조. 값이 나오기는 해야
    # 하지만(화면에 찍으므로), 그 크기로 통과/불통을 가르면 안 된다.
    assert a2["resid_rms"] is not None


def test_exciter_waits_instead_of_giving_up_instantly(tmp_path):
    """자극이 (1) 상태를 기다리고 (2) IMU 영점이 끝나기를 기다린다.

    2026-08-21 첫 실행이 두 가지로 죽었다.

    (1) 대기 시간을 `node.t_pose` 로 쟀는데 초기값이 0.0(=1970 년)이라
        `time.time() - node.t_pose` 가 17 억이 되어 **루프가 한 번도 안 돌고**
        곧장 '로봇 자세를 못 받았다' 로 끝났다. 같은 시각 로봇 로거는 같은
        토픽을 멀쩡히 받고 있었다.
    (2) 그것만 고치면 자극이 로거 2 초 뒤에 움직이기 시작하는데, 그때 log_imu 는
        3 초 영점 캘리브레이션 중이다. 영점이 깨지면 변위의 B/C 단계가 통째로
        날아간다. 그래서 IMU 의 CSV 가 생기는 것을 보고 시작한다.
    """
    import excite_magmap as ex

    class Stub:
        """`wait_*` 가 만지는 것만 갖춘 대역. 이 둘은 ROS 를 안 만진다."""

        def __init__(self, pose_after=5):
            self.p = None
            self.t_pose = 0.0                           # 사고 당시의 초기값 그대로
            self.t_launch = time.time()
            self.spins = 0
            self.pose_after = pose_after

        def _spin(self):
            self.spins += 1
            if self.spins > self.pose_after:
                self.p = np.array([0.1, 0.2, 0.3])

    t0 = time.time()
    st = Stub()
    assert ex.wait_for_pose(st, seconds=3.0) is True     # 사고 당시에는 False 였다
    assert time.time() - t0 < 3.0
    assert st.spins > 5                                  # 실제로 기다렸다는 것

    # 상태가 끝내 안 오면 기다린 뒤에 False 여야 한다 (즉시 포기가 아니라)
    t1 = time.time()
    assert ex.wait_for_pose(Stub(pose_after=10**9), seconds=0.5) is False
    assert time.time() - t1 >= 0.4

    try:
        qc.use_raw_dir(str(tmp_path / "raw_data"))
        qc.ensure_dirs()
        st2 = Stub(pose_after=0)
        assert ex.wait_for_imu(st2, "mag_map", seconds=0.4) is False   # 파일이 없다
        with open(os.path.join(qc.DIR_IMU, "mag_map_raw_20260101_000000.csv"), "w") as fh:
            fh.write("x\n")
        assert ex.wait_for_imu(st2, "mag_map", seconds=2.0) is True
        # 다른 블록 파일에 속으면 안 된다
        assert ex.wait_for_imu(Stub(0), "probe", seconds=0.4) is False
    finally:
        qc.use_raw_dir(_HERE_RAW)



def test_cal_gate_treats_enter_as_proceed(monkeypatch, capsys):
    """막힌 사람이 누르는 키는 언제나 Enter 다.

    처음에는 Enter 를 '다시 읽기' 로 뒀다. 두 번의 세션이 이 프롬프트에서 멈췄고,
    두 번 다 사용자가 "enter 로 진행이 안 된다" 고 했다. 이 게이트의 목적은 막는
    것이 아니라 **알리고 program.json 에 남기는 것**이므로, 자연스러운 키가 진행
    이어야 한다. 다시 읽기는 [r] 로 옮겼다.
    """
    import run_session as rs

    class Args:
        port = "/dev/null"
        no_mag = False
        no_cal_check = False
        yes = False

    bad = {"cal_status": {"acc": 3, "gyr": 0, "mag": 1, "rv": 0}}
    monkeypatch.setattr(rs, "read_cal", lambda args: dict(bad))

    monkeypatch.setattr("builtins.input", lambda _p="": "")      # Enter
    got = rs.cal_gate("probe", Args())
    assert got["overridden"] is True
    assert "[Enter] 이대로 진행" in capsys.readouterr().out

    reads = []
    def again(args):
        reads.append(1)
        return dict(bad)
    monkeypatch.setattr(rs, "read_cal", again)
    answers = iter(["r", "r", ""])                               # 두 번 다시 읽고 진행
    monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
    got = rs.cal_gate("probe", Args())
    assert len(reads) == 3                                       # 최초 1 + 다시 2
    assert got["overridden"] is True

    monkeypatch.setattr("builtins.input", lambda _p="": "q")
    with pytest.raises(SystemExit):
        rs.cal_gate("probe", Args())


def test_mag_map_step_still_yields_pairs_and_is_bounded():
    """보폭을 줄여도 쌍이 살아 있어야 한다.

    처음 160 mm 로 잡았다가 실기에서 관절 한계 근처까지 갔다 (2026-08-21).
    줄이는 것은 맞지만, 쌍의 위치차 문턱(100 mm) 아래로 내리면 이 블록이 답할 수
    있는 것이 **하나도 없어진다** — 그러면 안전해진 것이 아니라 무의미해진 것이다.
    """
    assert protocol.MAG_MAP_STEP_MM >= protocol.MAG_MAP_MIN_DP_MM
    legs = protocol.plan_mag_map()
    groups, cur = [[]], np.zeros(3)
    for kind, payload in legs:
        if kind == "move":
            cur = np.array(payload, float)
        elif kind == "turn":
            groups.append([])
        else:
            groups[-1].append(cur.copy())
    d = [np.linalg.norm(g[i] - g[j])
         for g in groups for i in range(len(g)) for j in range(i + 1, len(g))]
    assert min(d) >= protocol.MAG_MAP_MIN_DP_MM, min(d)
    assert len(d) >= protocol.MAG_MAP_MIN_PAIRS
    # 시작점에서 가장 멀리 나가는 거리 — 사람이 "이만큼 비어 있나" 를 볼 값이다
    reach = max(np.linalg.norm(np.array(p, float))
                for k, p in legs if k == "move")
    assert reach == protocol.MAG_MAP_STEP_MM


def test_joint_margin_reports_none_when_it_cannot_watch():
    """감시를 못 할 때 '여유 넉넉' 으로 보이면 안 된다.

    한계를 못 읽었거나 관절 상태가 아직 없으면 None 을 준다 — 큰 수를 주면
    부르는 쪽이 조용히 통과시킨다. 이 QC 가 반복해서 당한 실패 방식이다.
    """
    import excite_magmap as ex

    src = inspect.getsource(ex)
    # 한계 숫자를 여기에 베껴 두면 언젠가 probe.yaml 과 갈라진다
    assert "-265.0" not in src and "175.0" not in src
    assert "probe.yaml" in src

    class Node:
        limits = None
        q_deg = None
        joint_margin = ex.make_node.__doc__ and None

    # joint_margin 은 make_node 안의 클래스 메서드라 직접 못 꺼낸다.
    # 계약만 검사한다: limits 나 q_deg 가 없으면 None 을 돌려주도록 쓰여 있는가.
    body = src[src.index("        def joint_margin(self):"):
               src.index("        def _on_retreat(self, msg):")]
    assert "if self.limits is None or self.q_deg is None:" in body
    assert "return None" in body
    # 그리고 부르는 쪽이 None 을 '통과' 로 읽지 않는가
    caller = src[src.index("                jm = self.joint_margin()"):]
    assert "jm is not None and jm[0] < protocol.MAG_MAP_JOINT_MARGIN_DEG" in caller
