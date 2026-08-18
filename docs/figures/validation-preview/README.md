# 검증 그림 미리보기 — 합성 데이터

여기 있는 13장은 **합성 데이터로 그린 레이아웃 미리보기**다. 어떤 수치도 실제 측정값이
아니며, 어떤 정확도 주장도 뒷받침하지 않는다. 목적은 하나 — 실제 데이터가 오기 전에
그림의 형태·축·주석·판정 문구를 확정해 두는 것.

합성 데이터에는 검증이 잡아내야 할 실패를 일부러 심어 두었다:

- `P03` 환자만 Dice가 무너진다 → **B7**이 그것만 붉게 드러낸다
- `border_penalty`는 순수 잡음, `lumen_contrast`는 역상관 → **B5**의 막대가 0 왼쪽으로 간다
- `excessive_centroid_jump`는 아무 프레임에나 발화 → **B6**에서 두 점이 겹치고 lift ≈ 0.9
- 에포크 17에 비유한 배치 → **A4**에 세로선
- 힘 5.0 N 레벨은 유효 샘플 부족 → **B8**에 "measurement failure"

실제 실행:

    python scripts/plot_report.py --run-dir runs/<이름> \
        --accuracy-floor 0.70 --target-bad-rate 0.02

읽는 법과 합격 기준은 `docs/VALIDATION_PLAN.md`.
