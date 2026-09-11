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

## 2. 기동 (터미널 둘)

```bash
# ① 세션 + GUI. --seg 를 붙이지 않는다 (러너가 자기 U-Net 을 돌린다)
~/FR5-for-RUS/scripts/start_session.sh

# ② 힘 탐색 + 러너. 한 명령이 둘 다 띄우고 Ctrl-C 하나로 둘 다 내린다
~/FR5-for-RUS/scripts/start_policy_eval.sh --pilot          # 또는 --main N, 또는 인자 없이 루프 평가
```

힘 탐색을 따로 띄우지 않는다. 그것이 없으면 힘 설정값이 출발값에 고정된 채 돌고
(Q_raw 기반 제어가 죽은 상태), 그 에피소드들은 나중에 **분리해서 보고**해야 한다.
한 명령이 소유하면 빠뜨릴 수가 없다.

콘솔 확인: 상단이 `NO TELEMETRY` 가 아니고, **Monitoring** 하단 오른쪽에 품질 두 눈금과
`Policy inference` 버튼이 보인다.

## 3. 시스템 검증 — 데이터를 찍기 전에 (한 번, 순서대로)

각 항목은 **무엇을 치고 · 무엇이 나와야 하고 · 아니면 무슨 뜻인가** 다. 하나라도 통과
못 하면 그 아래로 내려가지 않는다 — 통과 못 한 층 위에서 잰 값은 해석할 수 없다.

### V1 배선 — 토픽이 다 있는가

```bash
ros2 topic list | grep -E "fr5_right|/us/image"
```

`wrench_px6d · desired_twist · ee_wrt_base · probing_mode · policy_enable ·
image_quality_seg · image_quality_raw · force_setpoint_bar · /us/image` 가 보여야 한다.
`image_quality_*` 는 러너가, `force_setpoint_bar` 는 힘 탐색이 낸다 — 없으면 그 노드가 안 떠 있다.

### V2 교정 — 렌치가 중력 보상됐는가

```bash
ros2 topic echo /fr5_right/wrench_px6d --once --field header.frame_id
```

**`_probe` 로 끝나야 한다.** 아니면 보상 전 값이고, 거기에는 마운트·프로브 자중이 자세에
따라 10 N 실려 있다 — 그것을 접촉력으로 믿으면 닿지도 않았는데 게이트가 열린다.
러너도 이 상태를 거부하고 이유를 찍는다.

### V3 지각 — 따라가는가, 값이 그럴듯한가

러너를 띄우고(§3 DRY-RUN) 로그에서:

* `지각 XX ms/장` 이 **125 ms 미만** — 8 fps 를 못 따라가면 관측 창이 오래된 프레임으로 찬다.
* 콘솔 **Image quality** 눈금 둘이 움직인다. 프로브를 떼면 `Q_raw` 가 떨어지고, 방광을
  화면에 넣으면 `Q_seg` 가 오른다. **둘이 같이 움직이면 배선이 틀린 것이다** — 서로 다른
  신호이므로 그렇게 될 이유가 없다.
* 대시(—)는 "재지 못했다" 다. 계속 대시면 ROI 나 프레임 방향을 의심한다.

### V4 힘 탐색 — 디더가 돌고 설정값이 움직이는가

```bash
ros2 run fr5_control force_search                     # 관찰 모드
```

`← 3.250 N` 다음 **2.5 s 뒤 `← 2.750 N`**. 반주기마다 부호가 뒤집혀야 한다. 주기가 다르면
시간 규모가 어긋난 것이고, 그러면 복조가 "힘의 차이에 대한 품질의 차이" 를 재지 못한다.

`execute:=true` 로 바꾼 뒤 접촉 상태에서 `ros2 topic echo /fr5_right/force_setpoint_bar`
가 **천천히 움직이면** (한 주기에 ≤ 0.1 N) 탐색이 살아 있는 것이다. 고정이면 품질이
안 들어오고 있다.

### V5 인계 — 버튼이 실제로 régime 을 여는가

접촉을 만든 뒤 콘솔 `Start inference`.

```bash
ros2 topic echo /fr5_right/probing_mode --once
```

**`contact_probing_policy`** 가 나와야 한다. 콘솔 패널도 `HANDOVER` 로 채워진다.
`PENDING` 에 머물면 요청만 가고 스택이 받지 않은 것이다 — 누르지 않은 것과 같다.

`Stop` 을 누르면 `approach` 로 돌아오고 Touch 가 살아난다. **돌아오지 않으면 다음 자세로
갈 수 없다.**

### V6 각속도 축 — 코드가 검증할 수 없는 가정

```bash
ros2 topic pub --once /fr5_right/desired_twist geometry_msgs/Twist "{angular: {y: 0.02}}"
```

프로브가 **영상면을 유지한 채** 도는지 본다 (y 는 영상면 법선이므로 그 둘레 회전만 면을
자기 자신으로 옮긴다 — DESIGN_NOTES §1811). x·z 도 같은 식으로 하나씩 확인한다.

여기서 틀리면 정책이 의도와 다른 축으로 돈다. **이 항목을 통과하기 전에는 `--execute` 를
붙이지 않는다.**

### V7 안전 — 후퇴가 실제로 걸리는가

정책을 끈 상태에서 teleop 으로 천천히 눌러 ‖F‖ 를 올린다. 5 N 부근에서 `/diag/retreating`
이 뜨고 로봇이 물러나야 한다.

```bash
ros2 topic echo /diag/retreating
```

**이 항목은 실제로 해 본다.** 한 번도 확인하지 않은 안전 장치는 없는 것과 같다.

### V8 에피소드 틀 — 기록이 남는가

지령 없이 한 에피소드를 돌린다.

```bash
cd ~/FR5-for-RUS/policy_learning
python3 scripts/run_experiment.py runs/qres2_ep25.pt --out runs/_verify \
        --poses 1 --conditions hold,policy --duration 30
```

`--execute` 가 없으므로 로봇은 안 움직인다. 확인할 것:

* 시작 조건이 **열린다**. 안 열리면 면적비·Q_raw 문턱이 이 팬텀에 안 맞는 것이다.
* `runs/_verify/ep0001_*/` 에 `decisions.csv` · `states.npz` · `meta.json` 이 남는다.
* `meta.json` 의 `verdict` 가 채워진다.
* `summary.csv` 에 두 줄이 쌓인다.

`states.npz` 가 없으면 나중에 임계를 다시 훑을 수 없다 — 파일럿의 목적이 그것이므로
여기서 반드시 확인한다. 확인이 끝나면 `rm -rf ~/FR5-for-RUS/policy_learning/runs/_verify`.

---

## 4. 파일럿 — 6 자세 × 3 조건 = 18 (약 1 시간)

목적은 결과가 아니라 본 시험을 설계할 수 있게 하는 것이다.

```bash
~/FR5-for-RUS/scripts/start_policy_eval.sh --pilot
```

`~/FR5-for-RUS/policy_learning/runs/pilot` 에 쌓인다. 조건 순서는 자세마다 섞이고, 조작자는 조건을 모른다.
**끊기면 같은 명령을 다시 친다** — 같은 폴더로 이어서 하고, 이미 있으면 그렇게 알린다.

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
python3 ~/FR5-for-RUS/policy_learning/scripts/analyze_experiment.py \
        ~/FR5-for-RUS/policy_learning/runs/pilot --sweep
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
# ① 몇 자세가 필요한지 확인한다 — 숫자가 인쇄된다
python3 ~/FR5-for-RUS/policy_learning/scripts/analyze_experiment.py \
        ~/FR5-for-RUS/policy_learning/runs/pilot

# ② 그 숫자를 그대로 넣는다 (아래 25 는 예시다 — ① 이 알려준 수로 바꾼다)
~/FR5-for-RUS/scripts/start_policy_eval.sh --main 25 -- \
        --success-area-min 0.08 --success-component-min 0.80
```

`--` 뒤는 드라이버로 그대로 넘어간다. 판정 임계는 **파일럿이 정한 값**으로 바꿔 넣는다
(위 0.08 · 0.80 은 기본값이다). `--main` 뒤가 숫자가 아니면 스크립트가 거부한다 —
자리표시자를 그대로 붙여넣어 bash 가 리다이렉션으로 읽는 일을 막는다.

`expert` 는 사람이 계속 지령한다 — 러너는 기록만 한다. 그 에피소드만 조건이 콘솔에 뜬다.

**1 차 분석은 짝지어진 C 대 B (McNemar)** 다. 같은 자세에서 네 조건을 모두 돌리므로 짝
비교가 성립한다. 부트스트랩 95 % CI 를 함께 낸다 (계획서 §8).

## 7. 중단

콘솔 `Stop` → Touch 데드맨 놓기(워치독 후퇴) → 터미널 ② Ctrl-C → 터미널 ① Ctrl-C
→ `./scripts/stop_all.sh`

## 8. 결과와 함께 반드시 보고할 것

계획서 §10 의 여섯에 더해, 오늘 확정된 둘:

7. **힘 축은 Q_raw 극값 탐색이 잡는다** — 고정 3 N 유지가 아니다. `force_search` 가 떠
   있지 않은 에피소드가 있으면 그 에피소드의 힘은 출발값에 고정된 것이므로 **분리해서
   보고**해야 한다 (`decisions.csv` 의 `f_bar` 가 움직였는지로 가른다).
8. **접촉 전 자동 감속이 없다** — teleop 구간의 접촉은 freespace 상한에서 일어난다.
   안전 지표(밴드 이탈·후퇴 발동)를 볼 때 이 구간과 정책 구간을 나눠서 센다.
