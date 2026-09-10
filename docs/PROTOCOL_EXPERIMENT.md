# 실행 프로토콜 — 정책 검증 실험

2026-09-11. **무엇을 검증하는지**는 `EVAL_PLAN_POLICY_RESCUE.md`(사전 등록)에, **시스템의
세부**는 `RUNBOOK_POLICY_INFERENCE.md`에 있다. 이 문서는 **실제로 무엇을 어떤 순서로
치는가**만 담는다. 셋이 어긋나면 사전 등록이 이긴다.

---

## 0. 시스템은 이렇게 돌아간다

| | |
|---|---|
| **모드** | 하나뿐이다 — 정책 추론 모드. 접촉력으로 régime 을 가르지 않는다 |
| 기본 상태 | teleop, freespace 상한(빠름), 조작자가 여섯 축, z 자유 |
| 정책 모드 | 콘솔 버튼으로만 진입. 접촉 상한 10 mm/s, **로봇이 z**, 조작자 축 0 |
| **정책이 움직이는 축** | **회전만** (θx, θy, θz). 병진은 학습되지 않았다 |
| **힘** | 고정 목표가 없다. `force_search` 가 **Q_raw** 경사로 설정값을 옮긴다 |
| 안전 | ‖F‖ ≥ **5 N** 강제 후퇴. 이것이 régime 안의 유일한 힘 규약이다 |

> ⚠️ 버튼을 누르기 전 teleop 구간에는 접촉 시 자동 감속이 **없다**. 5 N 후퇴가 유일한
> 보호다 (의도된 설계, 되돌리려면 `contact_probing_force_trigger: true`).

**Q_seg 와 Q_raw 는 다른 신호다.** Q_seg 는 분할 기반("방광을 제대로 보이게") 으로 성공
판정의 재료이고, Q_raw 는 세그 비의존("일단 제대로 닿게") 으로 힘 축을 맡는다. 콘솔이 둘을
따로 100 점으로 보여준다 — 갈라질 때가 진단이다.

## 1. 준비 (하루에 한 번)

```bash
source ~/FR5-for-RUS/env.sh          # 모든 터미널의 첫 줄
cd ~/FR5-for-RUS
colcon build --packages-select fr5_control fr5_ik --symlink-install
source ~/FR5-for-RUS/env.sh
```

교정이 없으면 힘 유지가 열리지 않는다. 안 했으면 `./scripts/start_session.sh --calib` 로
전자영점·다자세 중력을 먼저 마친다.

## 2. 기동 (터미널 셋)

```bash
# ① 세션 + GUI. --seg 를 붙이지 않는다 (러너가 자기 U-Net 을 돌린다)
./scripts/start_session.sh

# ② 힘 탐색. 이것이 없으면 힘 설정값이 출발값에 고정된다
source ~/FR5-for-RUS/env.sh
ros2 run fr5_control force_search                       # 관찰: 2.5 s 마다 부호가 뒤집히는지
ros2 run fr5_control force_search --ros-args -p execute:=true

# ③ 러너 / 세션 드라이버
source ~/FR5-for-RUS/env.sh && cd ~/FR5-for-RUS/policy_learning
```

콘솔 확인: 상단이 `NO TELEMETRY` 가 아니고, **Monitoring** 하단 오른쪽에 품질 두 눈금과
`Policy inference` 버튼이 보인다.

## 3. DRY-RUN — 실험 전 한 번 (필수)

```bash
python3 scripts/run_policy.py runs/qres2_ep25.pt
```

**접촉을 먼저 만들고** 손을 놓은 뒤 콘솔 `Start inference`. 누르면 teleop 이 사라지므로
순서를 바꾸면 접촉을 만들 수단이 없다.

| 확인 | 기대 | 어디서 |
|---|---|---|
| B-mode 변환 | `scan_convert+letterbox` | 러너 로그 |
| 지각 속도 | **< 125 ms/장** | 러너 로그 |
| 중력 채널 | 프로브 수직일 때 예상값 | 러너 로그 |
| **각속도 축** | `ω` 부호와 실제 회전 방향이 맞는가 | 눈으로 |
| 힘 탐색 | `f̄` 가 움직인다 (무음 경고가 없다) | 러너 로그 |

각속도 축은 코드가 검증할 수 없는 가정이다. 의심되면:

```bash
ros2 topic pub --once /fr5_right/desired_twist geometry_msgs/Twist "{angular: {y: 0.02}}"
```

**이 확인 전에는 `--execute` 를 붙이지 않는다.**

## 4. 파일럿 — 6 자세 × 3 조건 = 18 (약 1 시간)

목적은 결과가 아니라 본 시험을 설계할 수 있게 하는 것이다.

```bash
python3 scripts/run_experiment.py runs/qres2_ep25.pt --out runs/pilot \
        --poses 6 --conditions hold,placebo,policy \
        --duration 90 --blind --execute --max-deg-s 3
```

조건 순서는 자세마다 섞인다. 조작자는 조건을 모른다 (`--blind`). 끊기면 같은 `--out` 으로
이어서 한다.

### 에피소드 하나

```
1. teleop 으로 접촉을 만든다                  Touch 데드맨
2. 손을 놓는다
3. 터미널 ③ Enter                            드라이버가 그 에피소드를 띄운다
4. 콘솔 Start inference                       ← 여기서 teleop 이 사라진다
5. HANDOVER 로 채워지는지 확인                 PENDING 이면 스택이 안 받았다 = 안 누른 것
6. 90 s 대기                                  러너가 스스로 멈추고 판정을 인쇄한다
7. 콘솔 Stop                                  눌러야 Touch 가 돌아온다
```

조건이 `hold` 나 `placebo` 여도 **4 번을 똑같이 누른다.** 조작자가 조건을 모르는 것이
위약 대조의 전제다.

### 시작 조건과 성공 판정 (러너가 자동으로 잰다)

| | 기준 | 근거 |
|---|---|---|
| **시작** | 면적비 < 2 % **이면서 Q_raw ≥ 0.6**, 2 s 연속 | "접촉은 좋은데 방광이 없다" |
| **성공** | 면적비 ≥ 8 % · 최대 연결성분 ≥ 80 % · \|중심−0.5\| ≤ 0.30, **연속 3 s** | 진단 가능 뷰 |

시작 조건에 마스크 존재는 요구하지 않는다 — 면적비 0 이 "안 보인다" 의 가장 극단이다.
성공 판정은 반대로 마스크가 있어야 한다.

조건 미충족이면 그 에피소드는 **폐기**로 기록된다. 폐기율 자체가 파일럿의 산출물이다.

### 러너가 조용하면 이유를 읽는다 (5 s 마다)

`렌치가 중력 보상되지 않았다` · `접촉 부족 ‖F‖=…` · `정책이 꺼져 있다` ·
`시작 조건 미충족 — 면적비 … Q_raw …` · `힘 탐색이 보이지 않는다`

## 5. 파일럿 분석 — 네 산출물

```bash
python3 scripts/analyze_experiment.py runs/pilot --sweep
```

1. **판정 임계** — `--sweep` 이 면적비 4~12 % × 연결성분 70~90 % 를 훑는다. 조건이 가장 잘
   갈리는 자리가 아니라 **성공률이 평탄한 자리**를 고른다. 민감한 지점을 고르면 그 선택이
   결과를 만든다.
2. **p_B** — 위약 성공률. 본 시험 규모가 여기서 나온다.
3. **도달 불가** — 시작조차 못 한 비율.
4. **폐기율** — 30 % 를 넘으면 "teleop 실패" 를 만들지 못한 것이다. 설계를 고칠 일이지
   결과를 보고할 일이 아니다 (계획서 §9).

비율에는 Wilson 구간을 쓴다 — n=6 에서 정규근사는 음수 하한을 낸다.

## 6. 본 시험

파일럿의 임계와 p_B 를 **고정한 뒤** 돌린다. 규모는 p_B 가 정한다 (계획서 §6 은 15 자세,
분석 스크립트가 필요한 수를 계산해 준다).

```bash
python3 scripts/analyze_experiment.py runs/pilot          # 필요한 자세 수를 확인
python3 scripts/run_experiment.py runs/qres2_ep25.pt --out runs/main \
        --poses <N> --conditions hold,placebo,policy,expert \
        --duration 90 --blind --execute \
        --success-area-min <파일럿 값> --success-component-min <파일럿 값>
```

`expert` 는 사람이 계속 지령한다 — 러너는 기록만 한다. 그 에피소드만 조건이 콘솔에 뜬다.

**1 차 분석은 짝지어진 C 대 B (McNemar)** 다. 같은 자세에서 네 조건을 모두 돌리므로 짝
비교가 성립한다. 부트스트랩 95 % CI 를 함께 낸다 (계획서 §8).

## 7. 중단

콘솔 `Stop` → Touch 데드맨 놓기(워치독 후퇴) → 터미널 ① Ctrl-C → `./scripts/stop_all.sh`

## 8. 결과와 함께 반드시 보고할 것

계획서 §10 의 여섯에 더해, 오늘 확정된 둘:

7. **힘 축은 Q_raw 극값 탐색이 잡는다** — 고정 3 N 유지가 아니다. `force_search` 가 떠
   있지 않은 에피소드가 있으면 그 에피소드의 힘은 출발값에 고정된 것이므로 **분리해서
   보고**해야 한다 (`decisions.csv` 의 `f_bar` 가 움직였는지로 가른다).
8. **접촉 전 자동 감속이 없다** — teleop 구간의 접촉은 freespace 상한에서 일어난다.
   안전 지표(밴드 이탈·후퇴 발동)를 볼 때 이 구간과 정책 구간을 나눠서 센다.
