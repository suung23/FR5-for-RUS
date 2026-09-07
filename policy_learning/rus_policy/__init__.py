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
    synth       하드웨어 없이 파이프라인을 검증하는 합성 세션 생성기
"""

from __future__ import annotations

__version__ = "0.1.0"
