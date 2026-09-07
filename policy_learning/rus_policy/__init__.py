"""rus_policy — 초음파 영상 + IMU 시연으로부터 프로브 운동 정책을 학습하는 파이프라인.

설계 근거: ``docs/POLICY_LEARNING_MATH.md`` (ACT · 이종분산 loss · Q̂ 헤드).
입력 데이터: ``imu_bench/host/us_imu_collect.py`` 가 남기는 세션 디렉터리.

모듈 구성 (데이터가 흐르는 순서):

    session     us_imu_<stamp>/ 디렉터리 → 프레임 + IMU 표 + 시간축
    imu_labels  정지-이동-정지 분절 → ZUPT 이중적분 → chunk 라벨 (P̃, R̃, σ)
    perception  U-Net → ControlState → 프레임별 특징 s_t, Q_seg, 액션 토큰
    dataset     세션 묶음 → HDF5 샘플 파일, torch Dataset
    model       ACT-lite (CVAE 또는 이산 헤드) + Q̂ 헤드
    losses      §5.3 loss
    train       학습 루프 · 진단
    ensemble    실행시 chunk 혼합 — 모드 보존 temporal ensembling (§3.5, L12)
    synth       하드웨어 없이 파이프라인을 검증하는 합성 세션 생성기

**범위: Stage 2 (면내 3축) 전용.** policy 가 내는 것은 (v_x, v_y, ω_z) 뿐이다. 힘축
(v_z, ω_x, ω_y) 은 F/T 센서 + admittance 가 100 Hz 로 닫고, Stage 1 의 힘 탐색 (Q_raw
격자) 도 학습 대상이 아니다 (DESIGN_NOTES §7). **이 분리는 결손이 아니라 설계다** —
프리핸드 시연에서 힘 라벨을 얻을 방법이 없으므로 힘축을 학습에서 떼어낸 것이다.
따라서 ``label/F``·``label/Fn_star`` 와 ``loss.lambda_force/feas/risk`` 는 쓰지 않는
확장점이고, ``dataset.OBS_VEC_NAMES`` 의 wrench 7 칸도 **학습·추론 양쪽에서 0 이다**
(추론 때만 채우면 학습에서 본 적 없는 입력이 되고, 방금 분리한 힘축을 관측으로 다시
묶게 된다).
"""

from __future__ import annotations

__version__ = "0.1.0"
