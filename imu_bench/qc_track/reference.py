#!/usr/bin/env python3
"""기준값 — 이 QC 의 숫자를 **혼자 읽지 않기 위한** 표.

두 갈래가 있다.

1. **Surgilogger QC / Exp-Latency (2026-07-12 펌웨어, FR3 리그)**
   같은 BNO085 를, 같은 정렬 해법(X=A·Y·B)과 같은 추정기(록인 / L·T 피팅 /
   상호상관)로 잰 값이다. 우리 회전 채널은 그 표와 **직접 비교된다** — 로봇이
   FR3 에서 FR5 로, 도구가 복강경 샤프트에서 초음파 프로브로 바뀌었을 뿐
   측정 대상(BNO085 rotation vector 의 지연·정확도)은 같다.
   출처: ExpLatency_관측레이턴시_보고서.pdf §1, §3.3, §4.

2. **imu_bench 변위 오차 바닥 (2026-08-19, 이 벤치 실측)**
   IMU 단독 변위의 상한. 로봇 GT 없이 정지 구간에 같은 파이프라인을 돌려
   "0 이 나와야 하는데 얼마가 나오는가" 로 잰 값이다.
   출처: imu_bench/QC_PLAN.md §9, host/displacement_check.py.

분석기는 이 값을 결과 JSON 과 보고서 표에 **그대로 같이 박는다.** 우리 숫자만
있으면 "26 ms 가 좋은 것인가" 를 판단할 근거가 없기 때문이다.

주의 — 무엇이 비교 가능한가
---------------------------
* 회전(하방각/롤) 지연·상관: **비교 가능**. 같은 칩, 같은 추정기.
* 삽입깊이(ToF): **비교 대상 아님.** 이 리그에는 ToF 가 없다. 그 줄은 "펌웨어
  스무딩이 걸리면 어떤 모양이 되는가" 의 대조군으로만 싣는다.
* 정렬 잔차·관측도: 비교 가능하되 자세 다양성에 딸린 값이다. 우리 align 자세가
  더 좁으면 관측도가 낮게 나오는 것이 정상이고, 그때는 자세를 넓혀야 한다.
* 변위: Surgilogger 쪽에 대응하는 값이 없다. imu_bench 바닥과 비교한다.
"""

SOURCE = {
    "latency": ("Surgilogger QC — 관측 레이턴시 (Exp-Latency), 2026-07-12 펌웨어 v1, "
                "board-6-qc, FR3 tip_pose 179 Hz GT"),
    "displacement": ("imu_bench QC_PLAN §9 — 정지 구간 변위 오차 바닥, "
                     "2026-08-19, XIAO nRF52840 + BNO085"),
}

# --- 1) Exp-Latency §1 요약표 --------------------------------------------
#   eff_ms : 자유동작 90 s 상호상관 실효 지연
#   L_ms   : 순수지연,  T_ms : 1차 지체,  LT_ms : 저주파 군지연 L+T
#   corr   : 상호상관 최대점의 상관계수
LATENCY = {
    "depth":      {"eff_ms": 341, "L_ms": 22, "T_ms": 326, "LT_ms": 349, "corr": 0.992,
                   "comparable": False,
                   "note": "VL53L0X + 펌웨어 EMA(a=0.1, ToF 29 Hz). 이 리그엔 ToF 가 없다"},
    "depression": {"eff_ms": 12, "L_ms": 14, "T_ms": 13, "LT_ms": 26, "corr": 0.999,
                   "comparable": True,
                   "note": "BNO085 RV. 우리 tilt 채널의 정확한 여각 (depression = 90 - tilt) — "
                           "상수 오프셋이라 지연·이득·오차는 그대로 비교된다"},
    "roll":       {"eff_ms": 28, "L_ms": 1, "T_ms": 23, "LT_ms": 24, "corr": 0.999,
                   "comparable": True,
                   "note": "BNO085 RV. 우리 spin 과 같은 '축 둘레 회전' — 기준선만 다르다"},
}

# 록인 주파수별 지연 [Hz -> ms] (Exp-Latency §4.1). 우리 값이 '평평한가' 를
# 판단할 때 눈금이 된다 — 자세 채널은 평평(22~45 ms)해야 정상이다.
LOCKIN_MS = {
    "depth":      {0.10: 362, 0.20: 351, 0.35: 306, 0.60: 255, 1.00: 188},
    "depression": {0.15: 9, 0.30: 23, 0.60: 24, 1.00: 45, 1.60: 26},
    "roll":       {0.15: 42, 0.30: 22, 0.60: 24, 1.00: 24, 1.60: 23},
}

# 록인 이득 (같은 절). 자세 채널은 1 근처를 유지해야 한다 = 대역폭 손실 없음.
LOCKIN_GAIN = {
    "depth":      {0.10: 1.066, 0.20: 0.991, 0.35: 0.852, 0.60: 0.631, 1.00: 0.417},
    "depression": {0.15: 0.987, 0.30: 0.996, 0.60: 0.995, 1.00: 1.064, 1.60: 0.990},
    "roll":       {0.15: 0.970, 0.30: 0.985, 0.60: 0.987, 1.00: 0.983, 1.60: 0.981},
}

# --- 2) IMU <-> 로봇 고정회전 정렬 (Exp-Latency §3.3) ---------------------
ALIGNMENT = {
    "n_poses": 10, "n_poses_min": 3,
    "resid_rms_deg": 1.26, "resid_max_deg": 2.39,
    "resid_rms_pass_deg": 3.0,
    "observability": 0.327,
    "observability_note": "ALIGN_POSES 로 합성한 기준값이 0.32 근처. 이보다 훨씬 낮으면 자세가 한 축으로만 기울었다",
}

# --- 3) 시간축 품질 (Exp-Latency §3.2) -----------------------------------
TIMEBASE = {
    "skew_ppm_abs_max": 30.0,
    "resid_p95_ms": 1.44,
    "bad_lines": 0,
    "note": "|skew| < 30 ppm 이면 90 s 구간에서 2.7 ms 미만 — 지연 추정에 영향 없음",
}

# --- 4) 센서만으로 추정한 팁 위치 (Exp-Latency §4.4) ----------------------
# 우리 쪽 대응물은 '센서 자세 + GT 위치' 가 아니라 '센서 자세 + IMU 변위' 다.
# 직접 비교는 못 하지만, 자세 오차가 위치 오차로 얼마나 번지는지의 눈금이 된다.
TIP_RECON = {"rms_mm": 6.1, "p95_mm": 10.2,
             "rms_delay_corrected_mm": 5.6, "p95_delay_corrected_mm": 8.2,
             "note": "p = 트로카 + 샤프트방향(IMU) x 삽입깊이(ToF). 채널별 제 지연으로 보정"}

# --- 5) imu_bench 변위 오차 바닥 (QC_PLAN §9) ----------------------------
#   A 무보정 / B 영점 바이어스 보정 / C +ZUPT.  창 길이 [s] -> mm
DISPLACEMENT_FLOOR_MM = {
    0.5: {"A": 28.0, "B": 0.50, "C": 0.23},
    1.0: {"A": 112.0, "B": 1.5, "C": 0.67},
    2.0: {"A": 446.0, "B": 4.5, "C": 1.7},
    5.0: {"A": 2780.0, "B": 19.0, "C": 7.0},
}

# 칩 가속도계 보정 상태가 스케일 오차를 14 배 움직인다 (QC_PLAN §8).
SCALE_ERROR_PCT = {"uncalibrated": 2.28, "calibrated": 0.16,
                   "note": "보정 status 를 시행마다 기록하지 않으면 이 차이가 결과에 조용히 섞인다"}

# --- 6) 통과 기준 --------------------------------------------------------
#
# 변위 쪽은 QC_PLAN §10 그대로다. 회전 쪽은 Exp-Latency 의 실측값에서 끌어왔다 —
# 같은 칩이 같은 조건에서 낸 값이 있으므로, 그보다 크게 나쁘면 이 리그의 문제다.
PASS = {
    "rot_static_resid_rms_deg": 3.0,      # 정렬 잔차 (Exp-Latency 실측 1.26)
    "rot_dynamic_rms_deg": 2.0,           # teleop 구간 지연보정 후 geodesic RMS
    "rot_latency_ms": 50.0,               # 상호상관 실효 지연 (실측 12~28)
    "rot_corr_min": 0.99,                 # (실측 0.999)
    "yaw_drift_deg_per_min": 1.0,         # 자기환경이 나쁘면 여기서 먼저 걸린다
    # 정지 구간 오차 바닥이 §9 실측의 몇 배까지 허용되는가. **리그가 성한가** 를
    # 묻는 항목이지 응용 요구가 아니다 — 같은 칩·같은 파이프라인의 실측이 있으므로
    # 그보다 크게 나쁘면 이 리그(장착·영점·시간축)의 문제다.
    "disp_floor_ratio": 2.0,
    "note": "IMU 용도(FK 검증 / 프리핸드 기록 / 영상 태깅)가 확정되면 재조정한다",
}

# 이동 구간의 변위 오차에는 **일부러 문턱을 걸지 않는다.**
#
# 지금 이 QC 가 묻는 것은 '초음파 프로브의 움직임을 이 센서로 얼마나 정확히
# 기록할 수 있는가' 하나다. 실제 시술에서 한 번에 얼마를 움직이는지 확언할 수
# 없으므로, 몇 mm 를 통과선으로 삼든 그 숫자는 근거가 아니라 임의값이 된다.
# 임의값에 대고 '불합격' 을 찍으면 표는 있으나 마나 한 것이 되고, 반대로 통과가
# 찍히면 근거 없는 안심을 준다.
#
# 그래서 이 축은 **판정이 아니라 곡선으로** 낸다 — 구간 길이 T 마다 '얼마나
# 틀리는가' 를 그대로 싣는다. 용도가 정해지면 그때 그 표에서 필요한 칸을 읽어
# 문턱을 정하면 된다. 그 곡선의 모양은 물리로 정해져 있다: ZUPT 가 상수 바이어스를
# 지우므로 남는 것은 자세오차가 만드는 중력 누설 a 이고, 오차는 대략 a·T^2/8 로
# 자란다. 즉 짧은 구간일수록 급격히 정확해진다.
REPORT_ONLY = {
    "disp_by_duration": "구간 길이별 변위 오차 — 판정 없음 (용도 미확정)",
    "disp_rel_pct": "이동거리 대비 상대오차 — 판정 없음 (용도 미확정)",
}


def latency_row(channel):
    """우리 채널 이름 -> 비교 가능한 기준 행. 없으면 None."""
    # tilt 는 기준 보고서의 하방각과 정확히 여각이고, spin 은 같은 '축 둘레 회전'
    # 이다 (기준선만 다르다 — 상수 오프셋). tip_x/tip_y 는 기울기를 두 성분으로
    # 쪼갠 것이라 기준에 대응하는 행이 없다.
    return LATENCY.get({"tilt": "depression", "spin": "roll",
                        "tip_x": None, "tip_y": None}.get(channel, channel))


def as_dict():
    """결과 JSON 에 통째로 박을 한 벌."""
    return {"source": SOURCE, "latency": LATENCY, "lockin_ms": LOCKIN_MS,
            "lockin_gain": LOCKIN_GAIN, "alignment": ALIGNMENT, "timebase": TIMEBASE,
            "tip_recon": TIP_RECON, "displacement_floor_mm": DISPLACEMENT_FLOOR_MM,
            "scale_error_pct": SCALE_ERROR_PCT, "pass": PASS,
            "report_only": REPORT_ONLY}
