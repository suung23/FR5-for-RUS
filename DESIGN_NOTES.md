# FR5 로봇 초음파(RUS) 제어 설계 노트

> 복강경 봉합용 듀얼암 스택을 **오른팔 단일 초음파 프로브 제어**로 전환하는 설계 문서.
> 최종 갱신: 2026-08-14

## 표기 범례

| 기호 | 의미 |
|---|---|
| ✅ | **확정** — 사용자 결정 |
| 🟡 | **제안값** — 제안 후 승인(2026-08-14). 근거는 공학적 추정이며 튜닝/재검토 대상 |
| ⏳ | **실측대기** — 하드웨어 확인 후 확정 |
| ❓ | **미결** — 아직 논의되지 않음 |

🟡 항목은 "일단 이 값으로 시작한다"는 뜻이지 검증된 값이 아닙니다. 실험 후 반드시 갱신하십시오.

---

# 1. 전환의 성격

복강경 가정이 코드에 4개 층위로 박혀 있고, 층위마다 대응이 다릅니다.

| 층위 | 복강경 | 초음파 | 대응 |
|---|---|---|---|
| 표면 | 그리퍼 토픽, 듀얼암 네이밍 | 없음 | 삭제 |
| 기하 | 로드 0.64 m, RCM 거리 0.19 m, 베이스 간격 0.61 m | 프로브 마운트 | 상수 교체 (⏳ CAD) |
| 구속 | RCM (트로카 고정점 통과) | 표면 접촉 | **구속 삭제**, 대체물은 힘/모멘트 |
| 패러다임 | 접촉 = 이상(異常) | **접촉 = 정상 상태** | 제어 구조 변경 |

## 1.1 패러다임 변경이 핵심

현재 서보 루프는 명령 속도를 그대로 적분하는 **오픈루프 위치 적분**입니다
(`fr5_servo_joint_control_node.py:205-208`).

```python
self.current_joint_pos_deg[i] += vel_deg_s * dt
```

자유공간에서는 문제없지만 접촉 상태에서는 조직이 밀어내도 적분기가 계속 전진합니다.
admittance가 twist를 공급하는 정상 동작에서는 안전하지만, **admittance 루프가 죽으면**
(노드 크래시 / US 끊김 / F/T 드롭아웃) 적분기는 마지막 속도를 워치독 시간만큼 유지합니다.

> **`워치독 시간 × 접근 속도 = 최악 침투 깊이`.** 이 시스템의 안전 설계를 지배하는 수식입니다.
> §12 참조.

---

# 2. 하드웨어 전제

| 항목 | 상태 | 값 / 비고 |
|---|---|---|
| 로봇 | ✅ | FR5 **오른팔 단일**, `192.168.58.3` |
| 왼팔 | ❓ | 셀에 물리적으로 남아 있는지 확인 필요. 남아 있으면 정적 장애물 |
| 프로브 마운트 CAD | ⏳ | J6→프로브 접촉면 변환. §4.2 |
| F/T 센서 | ⏳ | 모델 / 장착 위치 / 샘플링 레이트 |
| US 프레임그래버 | ⏳ | 인터페이스 / **엔드투엔드 지연** |
| US 영상 기하 | ⏳ | linear / convex, 깊이 스케일, 부채꼴 ROI 마스크 |

## 2.1 F/T 경로 — SDK 조사 결과

`robot_state_pkg`에 필드가 존재합니다:

```
ft_sensor_raw_data[6]    원시
ft_sensor_data[6]        보상 후
ft_sensor_active         활성 플래그
```

**중요:** `Robot.py:5415`의 `FT_GetForceTorqueRCS()`는 XML-RPC를 호출하지 않고 state pkg를
직접 읽습니다. 따라서 F/T 읽기에 RPC 왕복 지연이 없고, 실질 레이트 = 컨트롤러 UDP RT
스트림 레이트(FR 시리즈 통상 8 ms)입니다.

→ **100 Hz admittance 루프는 실현 가능성이 높습니다.** 단 ⏳ 실측 확인 대상.

관련 API: `FT_Activate`, `FT_SetZero`, `FT_SetRCS`, `ForceSensorAutoComputeLoad`,
`SetForceSensorPayload` / `SetForceSensorPayloadCog`.

## 2.2 컨트롤러 내장 힘 모드는 사용하지 않음 ✅

SDK는 `FT_Control`, `SetAdmittanceParams`, `ImpedanceControlStartStop`,
`EndForceDragControl` 등 컨트롤러측 힘 모드를 제공하지만 **사용하지 않습니다.**

이유: 내장 힘 모드는 ServoJ 스트리밍과 명령 경로를 다툽니다. **명령 경로는 하나여야 합니다.**
wrench를 state pkg에서 읽어 ROS 노드에서 admittance를 계산하고 twist로 내보냅니다.

## 2.3 중력 / 페이로드 보상 — 현재 전무

전체 코드베이스에 `SetLoadWeight` / `SetLoadCoord` / `SetForceSensorPayload` **호출이 하나도
없습니다.** 프로브 자중 보상 없이는 Mx/My 정렬(§8)이 원리적으로 불가능합니다.

Phase 1에서 `ForceSensorAutoComputeLoad` 기반 페이로드 식별 절차를 반드시 수행합니다.

---

# 3. 전체 아키텍처

```
                        ┌─────────────────────────────┐
                        │      US 프레임그래버         │
                        └──────────────┬──────────────┘
                                       │ 30 Hz
                        ┌──────────────▼──────────────┐
                        │       us_frame_node          │
                        │      /us/image (Image)       │
                        └──────────────┬──────────────┘
                          ┌────────────┴────────────┐
                          ▼                         ▼
              ┌───────────────────────┐  ┌──────────────────────┐
              │   quality_raw_node    │  │   perception_node    │
              │  고전 영상처리 (신규)  │  │  Slim U-Net (기존)    │
              │  세그멘테이션 비의존   │  │                      │
              │      → Q_raw          │  │   → ControlState     │
              └───────────┬───────────┘  │      → Q_seg         │
                          │              └──────────┬───────────┘
                          │ /us/quality_raw         │ /us/control_state
                          │      30 Hz              │      30 Hz
          ┌───────────────┼─────────────────────────┼───────────────┐
          │               │                         │               │
          ▼               ▼                         ▼               ▼
 ┌─────────────────┐  ┌────────────────┐  ┌──────────────┐  ┌────────────┐
 │ force_search    │  │  supervisor    │  │ policy_node  │  │  (로깅)     │
 │    _node        │◄─┤   상태머신      ├─►│   3-DoF      │  │            │
 │  Stage1 전용     │  │   실패복구      │  │  속도3+set3   │  └────────────┘
 │  Stage1 + 배경   │  └────────┬───────┘  └──────┬───────┘
 └────────┬────────┘           │                 │
          │ /control/f_normal_setpoint           │ /control/image_twist
          │        (F_n*)      │                 │       (x, y, rz)
          └──────────┬─────────┴─────────────────┘
                     ▼
        ┌────────────────────────────────┐
        │      admittance_node  100 Hz    │◄──── /fr5_right/wrench  (100 Hz+)
        │  ┌──────────────────────────┐  │
        │  │ z  ← Fz  → F_n*          │  │
        │  │ rx ← Mx  → 0             │  │
        │  │ ry ← My  → 0             │  │
        │  ├──────────────────────────┤  │
        │  │  twist arbiter           │  │  힘축 + 영상축 합성
        │  │  (QP 전환 지점)           │  │
        │  └──────────────────────────┘  │
        └────────────────┬───────────────┘
                         │ /fr5_right/desired_twist  (프로브 프레임, 6-DoF)
                         ▼
        ┌────────────────────────────────┐
        │      diff_ik_node   100 Hz      │
        │   solve(V_desired, q) → q̇      │
        │   DLS (λ=0.02)                  │
        │   + twist 추종 모니터            │──► /diag/twist_tracking_error
        └────────────────┬───────────────┘
                         │ /fr5_right/joint_velocity_cmds
                         ▼
        ┌────────────────────────────────┐
        │      servo_node     125 Hz      │──► /fr5_right/wrench
        │   q += q̇·dt → ServoJ           │──► /fr5_right/joint_states
        │   워치독 / 안전 클램프           │──► /fr5_right/ee_wrt_base
        └────────────────┬───────────────┘
                         ▼
                    FR5 (오른팔)
```

## 3.1 노드 목록

실행 파일 이름은 `ros2 run <패키지> <실행파일>` 의 실행파일이다.

| 노드 / 실행파일 | 패키지 | 주기 | 구현 | 기원 |
|---|---|---|---|---|
| `us_servo` | `fr5_control` | 125 Hz | ✅ | `fr5_servo_joint_control_node.py` |
| `us_diff_ik` | `fr5_ik` | 100 Hz | ✅ | `freespace_two_twist.py` |
| `us_admittance` | `fr5_control` | 100 Hz | ✅ | 신규 |
| `us_frame` | `fr5_vision` | 30 Hz | ✅ | `camera_node.py` |
| `us_perception` | `fr5_control` | 30 Hz | ✅ | 신규 — `rus_perception` 얇은 래퍼 (§13.3) |
| `us_force_search` | `fr5_control` | Stage 1 중에만 | ✅ | 신규 — 상시 루프 아님 (§8.4) |
| `us_supervisor` | `fr5_control` | 20 Hz | ✅ | 신규 |
| `policy_node` | — | ~5 Hz | ❌ Phase 6 | `fr5_inference/` 구조 참조 |
| `touch_twist` | `touch_teleop` | 50 Hz | ✅ 개조 | 프로파일 분리 (§10.4) |
| `data_collector` | `dataset` | 30 Hz | ❌ 개조 대기 | `fr5_h5_collector.py` |

### 지각 노드는 하나다 ✅ (2026-08-18 결정)

이 표는 원래 `perception_node` 와 `quality_raw_node` 를 **따로** 두었으나, 구현하면서
하나로 합쳤다.

이유: **두 품질함수가 같은 ROI 를 봐야 한다.** `Q_raw` 와 `Q_seg` 가 서로 다른 영역을
보면 §5.3 의 진단 분기가 무너진다 — "둘 다 열화"와 "한쪽만 열화"를 구별할 수 없게 된다.
`us_perception` 은 `Predictor.roi_for()` 가 만든 마스크를 `compute_raw_quality` 에
그대로 넘겨 이것을 보장한다. 노드를 나누면 ROI 를 두 번 만들거나 토픽으로 주고받아야
하는데, 둘 다 어긋날 여지를 만든다.

## 3.2 주요 토픽

토픽 그래프는 코드에서 추출해 대조했다 (24개, 발행자–구독자 전부 연결됨).

| 토픽 | 타입 | 발행자 | 구독자 |
|---|---|---|---|
| `/us/image` | `Image` (mono8) | `us_frame` | `us_perception` |
| `/us/quality_raw` | `Float32` | `us_perception` | `us_force_search`, `us_supervisor` |
| `/us/quality_seg` | `Float32` | `us_perception` | `us_force_search` |
| `/us/valid_for_control` | `Bool` | `us_perception` | `us_force_search`, `us_supervisor` |
| `/us/rejection_reasons` | `String` (구분자 pipe) | `us_perception` | `us_supervisor` |
| `/us/control_features` | `Float32MultiArray` | `us_perception` | (policy 대기) |
| `/fr5_right/wrench` | `WrenchStamped` | `us_servo` | admittance, diff_ik, force_search, supervisor |
| `/control/contact_setpoint` | `WrenchStamped` | `us_force_search` → policy | `us_admittance` |
| `/control/image_twist` | `Twist` (x, y, rz) | policy ❌ | `us_admittance` |
| `/control/teleop_twist` | `Twist` (6축) | `touch_twist` | `us_admittance` |
| `/supervisor/state` | `String` | `us_supervisor` | `us_force_search` |
| `/supervisor/mode` | `String` | `us_supervisor` | `us_admittance` |
| `/fr5_right/desired_twist` | `Twist` | `us_admittance` | `us_diff_ik` |
| `/fr5_right/joint_velocity_cmds` | `JointState` | `us_diff_ik` | `us_servo` |
| `/operator/engage`, `/operator/confirm` | `Bool` | 수동 ❌ | `us_supervisor` |

진단: `/diag/admittance`, `/diag/force_search`, `/diag/force_curve`,
`/diag/twist_tracking_error`, `/diag/retreating`, `/diag/perception_latency`

### setpoint 는 `WrenchStamped` 하나로 묶는다 ✅ (2026-08-18 결정)

`(F_n*, M*_x, M*_y)` 를 `Float32` 셋으로 쪼개면 세 값이 **서로 다른 시각에** 도착해
동기가 깨진다. `WrenchStamped` 는 의미가 정확히 맞고(목표 접촉 wrench) 헤더에
타임스탬프가 있다.

```
force.z  = F_n*      torque.x = M*_x      torque.y = M*_y
```

덕분에 `us_interfaces` 커스텀 메시지 패키지가 **덜 급해졌다**. `/us/control_features` 만
임시 형식으로 남아 있고, policy 를 만들 때 함께 확정한다.

### `Q_raw` 는 점수를 낼 수 없으면 NaN ✅

`0` 으로 내면 "품질 최악"과 "측정 불가"가 구별되지 않아 supervisor 가 잘못된 재탐색을
건다. 소비자 둘 다 NaN 을 명시적으로 처리한다 — supervisor 는 `None` 으로 바꾸고,
force_search 는 그 표본을 평균에서 제외한다.

---

# 4. 좌표계 정의

## 4.1 프레임 ✅

| 프레임 | 정의 |
|---|---|
| `base` | 오른팔 베이스 |
| `j6` | 6축 플랜지 |
| `probe` | **프로브 접촉면 중심.** `+z` = 조직 침투 방향(법선), `+x` = 트랜스듀서 배열 방향(영상면 내 lateral), `+y` = elevational (영상면 밖) |
| `image` | US B-mode 영상면 = `probe`의 x–z 평면 |

모든 twist 지령은 **`probe` 프레임 기준**입니다 (기존 `freespace_two_twist.py`의 body-frame
관례를 유지).

이 정의에서 policy의 3-DoF가 자연스럽게 떨어집니다:

- `x`, `y` 병진 = 피부 위 슬라이딩
- `rz` = 영상면 회전
- `z`, `rx`, `ry` = 힘 축 (§7)

## 4.2 툴 변환 — 단일화 필요 ⏳

현재 J6→툴 오프셋이 **파일마다 다릅니다**:

| 파일 | 오른팔 오프셋 |
|---|---|
| `freespace_two_twist.py:44-49` | 0.64 |
| `free_control_node.py:254-259` | 0.22 |
| `gt_sparse_depth_node.py` | 0.615 |

5개 파일에 하드코딩된 numpy 상수로 흩어져 있습니다.

✅ **결정: 툴 변환을 YAML 파라미터 하나로 단일화**하고 모든 노드가 거기서 읽습니다.
CAD 도착 시 한 줄 변경으로 끝나고, F/T wrench 기준점 이동(§4.3)도 같은 파일에서 정의됩니다.

```yaml
# config/probe.yaml
tool:
  j6_to_probe:            # ⏳ CAD 대기
    xyz: [0.0, 0.0, 0.0]
    rpy: [0.0, 0.0, 0.0]
ft_sensor:
  j6_to_sensor:           # ⏳ 장착 위치 확인
    xyz: [0.0, 0.0, 0.0]
    rpy: [0.0, 0.0, 0.0]
payload:                  # ⏳ 식별 절차로 채움
  mass_kg: null
  cog_xyz: [null, null, null]
```

## 4.3 wrench 기준점 이동 — 필수 ⚠️

모멘트 변환:

```
M_probe = M_sensor + r × F        (r = 센서 원점 → 프로브 접촉면)
```

`r ≈ 0.1 m`, `F_z ≈ 5 N`이면 **오프셋 0.5 N·m**입니다. 정렬 신호(수십 mN·m 수준)를 완전히
덮습니다. §8의 rx/ry 정렬은 이 변환 없이는 동작하지 않습니다.

✅ **`FT_SetRCS`로 컨트롤러에 맡기지 않고 우리 노드에서 직접 변환합니다.** 컨트롤러 상태에
의존하지 않고, `r`이 §4.2 YAML의 상수 하나로 결정되기 때문입니다.

## 4.4 힘 부호 규약 ⏳

`F_n` (접촉 법선력, **양수 = 압축**)을 정의하고 코드 전역에서 이 부호만 씁니다.

```
F_n = -F_z^probe        (⏳ 실제 부호는 하드웨어에서 검증)
```

Phase 1의 첫 검증 항목입니다. 부호가 반대면 admittance가 발산합니다.

## 4.5 회전 중심 — RCM의 대체물 ✅

복강경에서 회전 중심은 **트로카 고정점**으로 기구학적으로 못 박혀 있었습니다. 회전 지령을
어떻게 주든 도구는 그 점을 중심으로 돌았고, 복벽을 지렛대로 밀 수 없었습니다.

RCM을 삭제하면 회전 중심은 **툴 프레임 원점**이 됩니다 — 순수 각속도 지령의 순간회전중심이
곧 야코비안 기준점이기 때문입니다.

> **프로브 접촉면이 RCM을 대체하는 회전 중심입니다.** 다만 구속으로 강제되는 것이 아니라,
> §4.2의 툴 변환을 맞게 넣어야만 성립합니다.

툴 프레임 원점이 실제 접촉면에서 `δ`만큼 어긋나 있으면, 각속도 `ω` 지령이 접촉점에
의도치 않은 미끄러짐을 만듭니다:

```
v_slip = ω × δ
```

`ω = 0.2 rad/s`(§12.3 상한), `δ = 20 mm`(CAD 오차)이면 **v_slip = 4 mm/s** — 병진 예산
10 mm/s의 40%가 회전 부작용으로 새어 나갑니다. 접촉 패치가 밀리면 힘도 바뀌고 admittance가
그것을 다시 쫓으므로, 힘 루프와 영상 루프가 §5.3에서 분리해 둔 전제가 무너집니다.

→ **CAD 정확도가 회전 품질을 지배합니다.** teleop 데드존(§10.4)보다 훨씬 큰 영향입니다.

`us_diff_ik_node`는 KDL 체인 끝에 프로브 세그먼트를 붙여 프로브 원점에서 야코비안을
잡으므로 구조는 이미 맞습니다. 값만 들어오면 됩니다.

---

# 5. 제어 축 분담 ✅

**모든 축은 100 Hz admittance 또는 5 Hz policy가 직접 구동합니다. policy는 힘 축을
속도로 지령하지 않고 setpoint만 옮깁니다** (2026-08-17 결정, §5.4).

| 축 | 구동 | 추종 대상 | 대역 | setpoint 출처 |
|---|---|---|---|---|
| `z` | admittance | `F_n → F_n*` | 100 Hz | Stage 1 탐색 → Stage 2 **policy** |
| `rx` | admittance | `M_x → M*_x` | 100 Hz | **policy** (5 Hz) |
| `ry` | admittance | `M_y → M*_y` | 100 Hz | **policy** (5 Hz) |
| `x`, `y`, `rz` | **policy** | 속도 직접 지령 | 5 Hz | — |

policy 출력은 **속도 3 + setpoint 3**입니다:

```
(v_x, v_y, ω_z)          속도 — 즉시 반영
(M*_x, M*_y, F_n*)       setpoint — 100 Hz 힘 루프가 추종
```

`M* = 0`이면 순수 법선 정렬이고, 지금까지의 설계와 동일하게 동작합니다.

## 5.1 왜 이 분담인가 (쟁점 7)

프로브면을 표면 법선에 정렬하는 것은 rx/ry 회전이고, **Mx/My가 그 오정렬을 직접 측정합니다.**
모서리가 눌리면 모멘트가 생기고, 나란하면 0에 수렴합니다.

얻는 것:
- 영상 policy가 5-DoF → **3-DoF**. 학습 난이도와 데이터 요구량이 크게 감소
- 자세 정렬이 30 Hz 영상 루프가 아니라 **100 Hz 힘 루프**에서 동작 → 빠르고 안정적
- 자세 정렬은 영상으로 배우기 특히 어려운 축. 오정렬의 영상 신호(음영)가 다른 원인과 구별되지
  않음. 반면 모멘트는 모호하지 않음

비용: 중력·프로브 자중 보상(§2.3)과 wrench 기준점 이동(§4.3)을 제대로 해야 함.

## 5.2 단서 조항

`Mx, My → 0`은 **접촉 패치가 대칭일 때만** 법선 정렬과 등가입니다. 곡면에서는 "압력 중심이
프로브 중앙"을 의미하며 법선 정렬과 근사적으로만 일치합니다.

실용상 충분하지만, 논문에서는 이 구분을 명시해야 합니다.

## 5.4 모멘트 setpoint 편향 — policy가 기울임을 쓰는 방법 ✅

**결정 (2026-08-17): policy는 기울임을 쓸 수 있어야 합니다.**

### 왜 필요한가

영상면에서 표적을 옮기는 방법은 물리적으로 두 가지이고, 서로 다릅니다.

| 방법 | 축 | 효과 |
|---|---|---|
| **미끄러짐** | `x`, `y` 병진 | 접촉점 자체가 이동. 시야가 평행이동 |
| **기울임** | `rx`, `ry` 회전 | 접촉점 고정, 빔이 부채꼴로 쓸림 |

깊이 `d`의 표적에 대해 기울기 `θ`는 대략 `d·θ`의 미끄러짐과 등가입니다. 방광 깊이 60 mm에서
**5° 기울임 ≈ 5 mm 미끄러짐**.

**미세 조정에서는 기울임이 우월합니다.** 미끄러짐은 접촉 패치를 바꾸고 마찰의 stick-slip을
겪는데, 이것이 몇 mm 단위 보정에서 가장 나쁜 특성입니다. 기울임은 접촉점을 유지합니다.

쟁점 7에서 `rx, ry`를 힘 루프에 통째로 준 대가가 여기서 드러납니다 — policy에 남은 회전은
`rz`(영상면 자체 회전)뿐이고, `rz`는 centroid를 좌우로 옮기지 못합니다.

### 구조 — `F_n*`와 대칭

```
M_x → M*_x ,  M_y → M*_y        (M* = 0 이면 순수 법선 정렬)
```

policy가 `M*`를 편향시키면 admittance가 **의도적으로 기울어진 평형**에 정착합니다.

- **대역 분리 유지.** policy는 100 Hz로 회전을 지령하지 않고 5 Hz로 setpoint만 옮깁니다.
  실제 자세 추종은 여전히 100 Hz 힘 루프
- **학습 부담이 낮음.** setpoint는 느리고 부드러운 신호라 속도 지령보다 데이터 요구량이 적음
- **안전 한계가 물리적으로 옳음.** 과도한 편향 = 모서리 눌림이므로 `max_moment_nm`이 그대로
  편향 상한 역할

### 한계와 슬루율 🟡

| 항목 | 값 | 근거 |
|---|---|---|
| `max_moment_bias_nm` | **0.10** | `max_moment_nm` 0.3에 과도응답 여유를 남김 |
| `moment_bias_slew_nm_s` | **0.05** | 계단 입력 금지. 아래 참조 |
| `force_setpoint_slew_n_s` | **2.0** | Stage 1 램프 속도와 동일 |

⚠️ **setpoint 슬루 제한이 안전상 필수입니다.** `M*`가 계단으로 0.1 N·m 뛰면 admittance가
`ω = ΔM / B_r = 0.1 / 0.5 = 0.2 rad/s` — 각속도 상한 전체를 한 번에 소진합니다.
0.05 N·m/s로 제한하면 기여분이 0.1 rad/s로 묶입니다.

⏳ 편향 0.1 N·m에서 실제 몇 도가 나오는지는 접촉 회전강성에 달려 있어 **실측 전에는
모릅니다.** Phase 2에서 측정합니다.

## 5.3 품질함수와 액추에이터의 대응 ✅

사용자 결정(쟁점 6)에 따른 매핑:

```
Q_raw  ←──  힘 축 (z, rx, ry)      "일단 제대로 닿게"
Q_seg  ←──  면내 축 (x, y, rz)     "방광을 제대로 보이게"
```

두 최적화가 같은 축을 두고 다투지 않습니다. 이 분리가 전체 설계의 뼈대입니다.

**진단 분기:**

| `Q_raw` | `Q_seg` | 해석 | 조치 |
|---|---|---|---|
| 정상 | 정상 | 양호 | 유지 |
| 정상 | 열화 | 면내 위치 문제 | policy가 처리 (상태 전이 없음) |
| 열화 | 열화 | **접촉 문제** | 힘 재탐색 |
| 열화 | 정상 | 드묾 — 과압 가능성 | 힘 감소 방향 탐색 |

---

# 6. 품질함수

## 6.1 왜 두 개인가 (쟁점 5) ✅

`ControlState`의 지표는 전부 **방광을 찾은 것을 전제**합니다. Stage 1 시작 시점(접촉이 아직
나쁨)에는 방광이 안 보일 수 있고, 그러면 `valid_for_control=False`가 뜨면서 `Q` 자체가
정의되지 않습니다. **최적화할 대상이 없는 상태로 시작하는 것입니다.**

→ Stage 1에는 세그멘테이션과 무관한 원시 영상 품질 지표가 필요합니다. 이것은 U-Net이 아니라
고전 영상처리로 충분하고, 오히려 그쪽이 더 안정적입니다.

## 6.2 `Q_raw` — Stage 1용, 세그멘테이션 비의존 🟡

✅ **정의처는 `Unet_seg`입니다** (`rus_perception/control/raw_quality.py`, 2026-08-14 구현).
`quality_raw_node`는 계산 로직을 갖지 않고 이 함수를 호출하는 얇은 ROS 래퍼입니다. 근거는
§13.3.2.

전부 원시 B-mode에서 계산. 4개 하위점수의 가중평균, 각각 `[0,1]`.

| 하위점수 | 정의 | 가중치 🟡 |
|---|---|---|
| `near_field_echo` | 근거리 밴드(깊이 상위 15%) 평균 강도를 기준값으로 정규화 | 1.0 |
| `contact_continuity` | `1 −` (근거리가 어두운 A-line 비율). 공기층 → 접촉 불연속 | 1.5 |
| `total_echo_energy` | ROI 전체 평균 강도의 plateau 함수 | 0.5 |
| `shadow_penalty` | `1 −` (원거리 에너지가 붕괴한 A-line 비율) | 1.0 |

```
Q_raw = Σ wᵢ sᵢ / Σ wᵢ  ∈ [0, 1]
```

`contact_continuity`에 최대 가중치를 준 이유: 접촉 불량의 가장 직접적이고 모호하지 않은
신호입니다. 공기가 끼면 그 A-line 전체가 죽습니다.

⏳ **깊이 밴드 비율, 어두움 임계, ROI 부채꼴 마스크는 US 영상 기하 확인 후 확정.**

⚠️ **이 4개 지표 자체가 검증되지 않았습니다.** Phase 3에서 실제 팬텀 영상으로 각 지표가
접촉력에 대해 단조/단봉인지 먼저 확인해야 합니다. 아니면 Stage 1 전체가 무의미해집니다.

## 6.3 `Q_seg` — Stage 2용 🟡

기존 `Unet_seg/rus_perception/control/quality.py`를 **코드 수정 없이 설정만으로** 재사용합니다.

### 문제: 현행 `Q`를 그대로 쓰면 힘 탐색이 반대로 갑니다 ⚠️

`Q`의 8개 항 중 3개가 시간축 안정성 항입니다 (`quality.py:199-208`):

- `temporal_iou` — 이전 마스크와의 warped IoU
- `centroid_stability` — centroid 점프에 `exp(-·)`
- `area_stability` — 면적 변화율에 `exp(-·)`

힘을 바꾸면 영상이 **당연히** 변합니다. 그러면 이 3개 항이 전부 떨어집니다.
기본 가중치 합 `1.0+0.5+0.5 = 2.0` / 전체 `6.0` = **1/3**.

→ 힘 탐색기가 "안 움직이는 게 최고"라고 결론 내리고 **힘을 안 올립니다.**

### 해결 🟡

최적화 목적함수에서 시간축 항의 **가중치를 0으로** 두고, 시간축 판정은 `valid_for_control`
게이트에만 남깁니다. `validity.py`의 기본 임계는 이미 활성화되어 있습니다
(`min_temporal_warped_iou=0.50`, `max_centroid_jump=0.15`, `max_relative_area_change=0.50`).

```python
# Q_seg: 최적화용 — 시간축 항 제외
QualityConfig(weights={
    "segmentation_confidence": 1.0,
    "mask_completeness":       1.0,
    "lumen_contrast":          0.5,
    "border_penalty":          1.0,
    "component_quality":       1.0,
    "temporal_iou":            0.0,   # ← 게이트로 이관
    "centroid_stability":      0.0,   # ← 게이트로 이관
    "area_stability":          0.0,   # ← 게이트로 이관
})
```

**`quality.py` / `validity.py` 코드 변경 불필요 — 설정만으로 달성됩니다.**

### 남는 위험 ⚠️

시간축 항을 빼면 힘 탐색 중 영상이 튀는 것을 거를 수단이 `valid_for_control`밖에 안 남습니다.
게이트는 이진(binary)이라 "약간 불안정"을 표현하지 못합니다.

완화책: `Q̄` 계산 시 **윈도우 내 `valid_for_control=True` 프레임만** 평균에 넣고, 유효
프레임 비율이 🟡 60% 미만이면 그 힘 레벨을 **측정 실패**로 처리(점수 부여 안 함).

### `target_area_ratio` 문제 ⏳

`quality.py:77`의 `target_area_ratio = 0.15`는 **검증되지 않은 임의값**인데, 힘에 민감한 항이라
최적 힘을 직접 좌우합니다. 실제 방광 영상에서 재설정이 필요합니다.

---

# 7. Stage 1 — 힘 탐색

## 7.1 원칙 ✅

- 쟁점 1: **위치 탐색이 아닌 힘 탐색.** 탐색 변수는 스칼라 `F_n` 하나
- 쟁점 2: **구간 평균** `Q̄(F)`로 판정. 순간값 금지
- 쟁점 3: **최대 품질을 내는 가장 작은 힘.** 힘은 반드시 안전 구간 내

## 7.2 파라미터 🟡

| 항목 | 값 | 근거 |
|---|---|---|
| 힘 격자 | 1.0 → 7.0 N, **0.5 N 간격** (13점) | 하한은 접촉 확인 가능한 최소, 상한은 안전 한계 |
| 홀드 윈도우 | **1.0 s** (US 30 Hz 기준 30프레임) | 조직 점탄성 완화 + 통계 안정 |
| 램프 속도 | **≤ 2 N/s** | 안전 |
| ε (품질 동률 허용치) | **0.03** | 잡음 수준보다 크고 의미 있는 차이보다 작게 |
| 최소 유효 프레임 비율 | **60%** | §6.3 |

## 7.3 알고리즘 🟡

```
for F in [1.0, 1.5, ..., 7.0]:
    ramp_to(F, rate ≤ 2 N/s)
    hold(1.0 s), collect Q samples
    Q̄(F) = mean(유효 샘플)
    if 유효비율 < 60%: mark F as 측정실패, continue
    if 조기종료조건: break

F* = min{ F : Q̄(F) ≥ (1 − ε)·max Q̄ }
```

**최소 힘 선택이 핵심입니다.** `argmax Q̄`가 아니라 "최댓값과 통계적으로 구분되지 않는
가장 작은 힘"입니다. 환자 부담을 줄이는 방향으로 편향되어 있습니다 (쟁점 3).

**조기 종료 🟡:** 진행 최댓값 대비 `Q̄`가 15% 이상 낮은 레벨이 **3회 연속**이면 상승 중단.
불필요한 압박을 피하고 탐색 시간을 줄입니다.

## 7.4 2단 구조 ✅

| 단계 | 목적함수 | 목표 |
|---|---|---|
| **Stage 1a** | `Q_raw` | "일단 제대로 닿게" — 세그멘테이션 불필요 |
| **Stage 1b** | `Q_seg` | 방광이 보이기 시작하면 세그 기반으로 미세 최적화 → `F_n*` |

Stage 1a는 전체 격자를 훑고, Stage 1b는 1a 결과 주변 🟡 **±1.0 N을 0.25 N 간격**으로 재탐색
합니다 (전체 재탐색은 시간 낭비).

---

# 8. Admittance (100 Hz)

## 8.1 법선력 축 (z) 🟡

감쇠 지배형(damping-dominant) admittance:

```
v_z = clamp( (F_n* − F_n) / B_z ,  ±v_z_max )
```

| 파라미터 | 값 🟡 | 근거 |
|---|---|---|
| `B_z` | **1000 N·s/m** | 1 N 오차 → 1 mm/s. 부드럽고 안전 |
| `v_z_max` | **10 mm/s** | 최악 침투 속도 한계 |
| 힘 데드밴드 | **0.1 N** | ⏳ F/T 잡음 실측 후 조정 |

질량항(`M`)은 우선 생략합니다. 감쇠 지배형이 접촉 안정성 측면에서 가장 견고하고, 튜닝할
파라미터가 하나뿐입니다. 응답이 느리면 그때 `M` 도입을 검토합니다.

## 8.2 자세 정렬 축 (rx, ry) 🟡

```
ω_x = clamp( (M*_x − M_x) / B_r , ±ω_max )
ω_y = clamp( (M*_y − M_y) / B_r , ±ω_max )
```

`M*`는 policy가 5 Hz로 옮기는 편향 setpoint입니다 (§5.4). `M* = 0`이면 순수 법선 정렬.
**`M*`에는 슬루 제한이 반드시 걸려야 합니다** — 계단 입력이 각속도 예산을 한 번에
소진합니다.

| 파라미터 | 값 🟡 | 근거 |
|---|---|---|
| `B_r` | **0.5 N·m·s/rad** | 0.05 N·m → 0.1 rad/s (≈5.7°/s) |
| `ω_max` | **0.2 rad/s** | ≈11°/s |
| 모멘트 데드밴드 | **0.01 N·m** | ⏳ F/T 잡음 실측 후 조정 |
| `max_moment_bias_nm` | **0.10** | §5.4 |
| `moment_bias_slew_nm_s` | **0.05** | §5.4 |

⚠️ 데드밴드는 **오차 `M* − M`에 적용**해야 합니다. 측정값 `M`에 걸면 편향이 걸린 상태에서
정상 동작 지점이 데드밴드 밖으로 나가 항상 움직입니다.

부호는 ⏳ 하드웨어에서 검증 (§4.4와 동일한 위험).

**전제:** §2.3 페이로드 보상 + §4.3 wrench 기준점 이동. 둘 중 하나라도 빠지면 이 루프는
동작하지 않습니다.

## 8.3 twist arbiter 🟡

DLS는 축 우선순위를 표현하지 못하므로(§9.1) 합성 지점을 코드상 한 곳으로 명시합니다.

```
V_probe = [ v_x_img,  v_y_img,  v_z_adm,
            ω_x_adm,  ω_y_adm,  ω_z_img ]
```

- 힘축 3개(`v_z`, `ω_x`, `ω_y`)는 admittance가 100 Hz로 채움
- 영상축 3개(`v_x`, `v_y`, `ω_z`)는 policy가 ~5 Hz로 갱신, 그 사이 ZOH
- **US 지연 > 🟡 0.5 s이면 영상축을 0으로** (힘 루프는 계속 동작)

이 지점이 나중에 QP로 갈아끼울 때 목적함수/제약으로 바뀝니다.

## 8.4 배경 힘 적응 ✅ 쟁점 6

FORCE LOCK은 **완전 잠금이 아니라 setpoint 기능**입니다. Stage 1은 끝나는 것이 아니라
저대역 배경 루프로 잔존합니다.

이유: `F_n*`를 고정한 채 5-DoF로 움직이면 프로브가 곡률과 조직 특성이 다른 지점으로 이동
합니다. 그 지점의 최적 힘은 `F_n*`가 아닙니다. 프로브를 기울이면 접촉 패치 자체가 바뀝니다.

### 디더를 버리고 절충안으로 ✅ (2026-08-17 결정)

**폐기된 방식:** 5초 주기로 `F_n*`를 ±0.25 N 섭동시켜 `ΔQ̄_raw`로 경사를 추정.

폐기 이유 — **귀속(attribution)이 불가능합니다.** 디더 반주기 2.5 s 동안 policy가 프로브를
10 mm/s로 움직이면 **25 mm 이동**합니다. 25 mm 이동이 만드는 해부학적 `Q_raw` 변화가
±0.25 N이 만드는 변화보다 훨씬 큽니다. 경사 추정치가 잡음에 묻힙니다.

**채택된 절충안:**

| 단계 | `F_n*` 출처 | 성질 |
|---|---|---|
| **Stage 1a / 1b** | **명시적 격자 탐색** (§7.3) | 규칙 — `min{F : Q̄ ≥ (1−ε)Q̄max}` |
| **Stage 2** | **policy 출력** (5 Hz, 슬루 제한) | 학습 |

이렇게 하면:

- 쟁점 3의 **"최대 품질을 내는 최소 힘"이 Stage 1에서 규칙으로 확보**됩니다. policy가
  이어받는 시점의 출발값이 그 규칙의 결과입니다
- 스캔 중 적응은 policy가 하므로 **디더가 필요 없고, 귀속 문제가 성립하지 않습니다**
- 32 s의 탐색 비용은 스캔 시작 전 **한 번만** 발생합니다
- Stage 1이 만든 **힘–품질 곡선(`Q̄` vs `F` 표본)이 학습 데이터로 남습니다**. 버리지 말고
  에피소드에 기록해야 합니다

**부수 효과 — 0.2 Hz 루프가 사라집니다.** §11 타임스케일 표에서 `F_n*` 적응 행이 없어지고
policy 5 Hz에 흡수됩니다. 아키텍처가 루프 하나만큼 단순해집니다.

**안전망은 유지:** `Q̄_raw`가 Stage 1 종료 시점 대비 🟡 **25% 이상 열화된 상태가 3초
지속**되면 Stage 1b 재진입. policy의 `F_n*`가 나빠져도 명시적 탐색이 되찾아옵니다.

---

# 9. IK — DLS

## 9.1 결정 ✅

**DLS로 상위 아키텍처를 먼저 검증합니다.** QP는 보류하되 교체 가능하도록 인터페이스를
고정합니다.

```python
def solve(V_desired: np.ndarray, q: np.ndarray) -> np.ndarray:
    """프로브 프레임 6-DoF twist → 관절 속도 (rad/s)."""
```

DLS 감쇠 `λ = 0.02` (기존 `freespace_two_twist.py` 값 유지).

### DLS의 한계와 대응

6-DoF 로봇이 6-DoF twist를 받으므로 **특이점에서 멀면 DLS는 지령을 거의 그대로 재현합니다.**
아키텍처 검증 단계에서 충분합니다.

문제는 특이점·관절한계 근처에서 감쇠가 걸릴 때 **어느 축이 희생됐는지 알 수 없다**는 점입니다.
힘 축이 조용히 뭉개지면 접촉력이 어긋나는데 로그에 아무것도 안 남습니다.

## 9.2 twist 추종 모니터 🟡

DLS 해를 순전파해 축별 오차를 계산합니다.

```
V_achieved = J · q̇_dls
e = V_desired − V_achieved          # 6개
```

`/diag/twist_tracking_error`로 발행. 힘축(`e_z`, `e_rx`, `e_ry`)이 임계를 넘으면 supervisor
이벤트.

| 임계 🟡 | 값 |
|---|---|
| `e_z` | 2 mm/s |
| `e_rx`, `e_ry` | 0.05 rad/s |

**비용이 거의 0입니다** (야코비안은 이미 계산됨). 그리고 **"QP가 실제로 필요한가"를 데이터로
판정**해 줍니다. 필요 없으면 DLS 확정, 필요하면 근거를 갖고 교체.

---

# 10. 상태 머신 (쟁점 9) 🟡

```
                        ┌────────┐
                        │  IDLE  │
                        └───┬────┘
                            │ operator engage
                            ▼
                  ┌───────────────────┐
        ┌────────►│  TELEOP_APPROACH  │◄──────────┐
        │         └─────────┬─────────┘           │
        │                   │ 접촉 감지 + 확인      │ 복구 실패
        │                   ▼                     │
        │         ┌───────────────────┐           │
        │         │  STAGE1A_CONTACT  │───────────┤
        │         │   Q_raw 힘탐색     │           │
        │         └─────────┬─────────┘           │
        │                   │ Q_raw ≥ θ           │
        │                   │ & valid 10프레임     │
        │                   ▼                     │
        │         ┌───────────────────┐           │
        │    ┌───►│  STAGE1B_FOCUS    │───────────┤
        │    │    │   Q_seg 미세탐색   │           │
        │    │    └─────────┬─────────┘           │
        │    │              │ F* 수렴              │
        │    │              ▼                     │
        │    │    ┌───────────────────┐           │
        │    └────┤   STAGE2_SCAN     │───────────┘
        │  Q 열화 │  + 배경 힘적응     │
        │         └─────────┬─────────┘
        │                   │ 안전 위반 / 센서 두절
        │                   ▼
        │            ┌─────────────┐
        └────────────┤   RETREAT   │
                     │ 법선 후퇴    │
                     │ 힘 → 0      │
                     └─────────────┘
```

## 10.1 전이 조건 🟡

| 전이 | 조건 |
|---|---|
| `IDLE → TELEOP_APPROACH` | 조작자 engage |
| `TELEOP_APPROACH → STAGE1A` | `F_n > 0.5 N`이 0.2 s 지속 **+ 조작자 확인** |
| `STAGE1A → STAGE1B` | `Q_raw ≥ 0.6` **및** `valid_for_control` 10프레임 연속 |
| `STAGE1B → STAGE2` | `F*` 탐색 완료 |
| `STAGE2 → STAGE1B` | `valid=False` 15프레임 연속 (≈0.5 s) **또는** `Q̄_raw` 25% 열화 3 s |
| `STAGE1B → TELEOP` | 복구 2회 연속 실패 |
| `any → RETREAT` | §12 안전 위반 |
| `RETREAT → TELEOP_APPROACH` | 후퇴 완료 + 조작자 확인 |

## 10.2 `rejection_reason` → 복구 동작 매핑 🟡

`validity.py:20`의 13종 코드를 활용합니다.

| 이유 | 해석 | 동작 |
|---|---|---|
| `empty_mask`, `mask_area_too_small` | 방광 소실 | STAGE1B 재진입 → 반복 시 TELEOP |
| `low_segmentation_confidence`, `high_boundary_entropy` | 접촉 불량 의심 | **STAGE1A 재진입** |
| `excessive_border_contact` | 면내 위치 문제 | 전이 없음 — policy 목표에 반영 |
| `low_temporal_warped_iou`, `excessive_centroid_jump` | 급격한 변화 | 면내 축 일시 홀드, 지속 시 STAGE1B |
| `mask_area_too_large` | 과압 가능성 | 힘 감소 방향 탐색 |
| `fragmented_mask` | 세그 품질 저하 | STAGE1B 재진입 |

## 10.3 신규 이유 코드 ✅ 구현됨

Stage 1a의 실패는 기존 13종 코드 체계 **밖**입니다. `Q_raw` 도입에 따른 신규 4종:

```
no_contact
poor_acoustic_coupling
excessive_shadowing
low_near_field_echo
```

`rus_perception.control.raw_quality.RAW_REJECTION_REASONS`로 구현되어 있습니다.

**두 집합은 의도적으로 분리되어 있고, 교집합이 없음을 테스트가 강제합니다**
(`test_raw_reason_codes_do_not_collide_with_segmentation_reason_codes`).
supervisor는 두 집합의 합집합을 소비합니다. 코드가 겹치면 하나의 관측이 서로 다른 두 복구
동작으로 매핑되기 때문입니다.

| 집합 | 의미 | 발행자 |
|---|---|---|
| `REJECTION_REASONS` (13) | **세그멘테이션**이 제어 측정치로 부적합 | `perception_node` |
| `RAW_REJECTION_REASONS` (4) | **접촉**이 부적합 (세그 무관) | `quality_raw_node` |

### 신규 4종 → 복구 동작 매핑 🟡

| 이유 | 해석 | 동작 |
|---|---|---|
| `no_contact` | 프로브가 공중 | **STAGE1A 이탈 → TELEOP_APPROACH** |
| `poor_acoustic_coupling` | 공기층 / 젤 부족 | STAGE1A 재진입 (힘 재탐색) |
| `excessive_shadowing` | 장가스·골 차폐 | 면내 이동 필요 — **힘으로 해결 안 됨**. 반복 시 TELEOP |
| `low_near_field_echo` | 접촉 약함 | 힘 상승 방향 탐색 |

⚠️ `excessive_shadowing`은 유일하게 **힘 축으로 해결되지 않는** 원시 실패입니다. 힘을 올리면
악화될 수 있으므로 Stage 1 격자 탐색을 계속 돌리면 안 됩니다.

## 10.5 모드 계약 — admittance 가 유일한 twist 발행자 ✅ (2026-08-18)

teleop 과 admittance 가 둘 다 `desired_twist` 를 발행하면 다툰다. 그리고
`TELEOP_APPROACH` 에서 조작자는 **z 를 포함한 6축**이 필요한데 z 는 힘 루프 소유다.

해결: 조작자 twist 를 `/control/teleop_twist` 로 우회시키고, **admittance 가 모드에 따라
동작을 바꾼다.** mux 경쟁이 사라지고 발행자가 하나로 유지된다.

| 상태 | 모드 | admittance 동작 |
|---|---|---|
| `IDLE` | `idle` | 영 twist |
| `TELEOP_APPROACH` | `teleop` | **조작자 6축 그대로 통과, 힘 루프 정지** |
| `STAGE1A` / `STAGE1B` / `STAGE2` | `contact` | 힘 3축 + 영상 3축 |
| `RETREAT` | `retreat` | 프로브 −z 등속 후퇴 |

**힘 루프를 끄는 것이 핵심이다.** 접근 단계에는 접촉이 없어 `F_n = 0` 인데, 힘 루프를
켜두면 `F_n*` 를 향해 스스로 내려가 조작자와 z 를 다툰다.

통과 모드에서도 §12.3 의 속도 클램프는 유지하고, 조작자 지령이 끊기면 발행을 멈춰
하위 워치독이 후퇴시킨다 — 프로브가 마지막 속도로 계속 가는 것을 막는다.

### 감독자가 죽으면 멈춘다 🟡

모드를 한 번이라도 받은 뒤 `mode_timeout_s`(0.5 s) 넘게 끊기면 admittance 는 발행을
멈춘다. 마지막 모드를 붙들고 계속 도는 것이 가장 위험하다.

단, 모드를 **한 번도** 받지 못했으면 `admittance.default_mode` 로 돈다 — supervisor
없이 단독 기동해 시험할 수 있어야 하기 때문이다.

## 10.4 TELEOP_APPROACH 원격조작 프로파일 ✅

Touch 햅틱 장치는 그대로 재사용하되, **입력 프로파일을 초음파용으로 분리**합니다
(`probe.yaml`의 `teleop:` 절, `touch_twist_node.cpp`).

기존 코드는 회전 데드존이 `0.1 rad/s`로 **하드코딩**되어 있었습니다. §12.3의 각속도 상한
`0.2 rad/s`와 결합하면 사용 가능 구간이 3:1밖에 안 나와, 접근 단계에서 프로브를 미세하게
기울이는 조작이 사실상 불가능했습니다.

| 프로파일 | 병진 스케일 | 회전 스케일 | 회전 데드존 | α | 회전 사용구간 |
|---|---|---|---|---|---|
| `laparoscopic` | 0.6 | 0.6 | 0.1 | 0.4 | **3.3 : 1** |
| `us_approach` 🟡 | 0.08 | 0.15 | **0.02** | 0.25 | **66.7 : 1** |

최소 지령이 `0.02 × 0.15 = 0.003 rad/s` (0.17 °/s)로 내려갑니다.

`teleop.profile`은 매 주기 읽으므로 조작 중에도 바꿀 수 있습니다:

```
ros2 param set /touch_teleop_node teleop.profile us_approach
```

알 수 없는 프로파일 이름은 **가장 보수적인 `us_approach`로 대체**하고 오류를 냅니다.
접근을 의도했는데 오타로 복강경 스케일(7.5배 빠름)이 도는 것이 가장 위험하기 때문입니다.

⏳ `angular_deadzone = 0.02`는 Touch 장치의 회전 잡음 바닥 위에 있어야 하는데 아직
실측 전입니다. 너무 낮으면 손을 멈춰도 프로브가 천천히 흐릅니다.

### 데드존은 자율 경로에 관여하지 않습니다

teleop 데드존은 **햅틱 입력 필터**이고, `policy_node`는 `/control/image_twist`로 직접
지령을 넣으므로 이 코드를 거치지 않습니다. 자율 회전의 분해능을 결정하는 것은 따로입니다:

| 축 | 분해능을 결정하는 것 |
|---|---|
| `rz` | policy 주기 5 Hz + ZOH → **200 ms 계단**. 대역 문제이지 데드존이 아님 |
| `rx`, `ry` | **`ft_sensor.moment_deadband_nm`** ← 구조적으로 동일한 문제가 힘 루프로 이사 |

F/T 잡음 바닥이 높아 모멘트 데드밴드를 크게 잡아야 한다면, 작은 오정렬이 영원히 보정되지
않습니다. ⏳ 잡음 실측 전에는 값을 정할 수 없습니다.

---

# 11. 타임스케일 (쟁점 10) 🟡

| 루프 | 주기 | 담당 |
|---|---|---|
| 서보 (관절) | **125 Hz** | ServoJ |
| 카테시안 + admittance | **100 Hz** | `F_n → F_n*`, `M → M*` 추종 |
| US 프레임 → 세그 → 특징 | **~30 Hz** | 영상 특징 |
| supervisor | **20 Hz** | 상태 전이, 재탐색 트리거 |
| 영상 policy | **~5 Hz** | 속도 3축 + setpoint 3개 |

힘 루프(100 Hz)와 영상 루프(5 Hz)가 **20× 벌어져** 깔끔하게 분리됩니다.

**0.2 Hz 루프는 없습니다.** 디더 기반 배경 적응을 폐기하고 절충안을 채택하면서(§8.4)
`F_n*` 적응이 policy 5 Hz에 흡수되었습니다. Stage 1의 명시적 탐색은 상태 전이 중 한 번만
도는 절차이지 상시 루프가 아닙니다.

## 11.1 위험 ⏳

**US 프레임그래버 지연을 반드시 실측해야 합니다.** 50 ms를 넘으면 5 Hz policy 루프의 위상
여유가 위험합니다.

현재 `diff_ik_node`(기원: `freespace_two_twist.py`)에는 **고정 주기 루프가 없습니다** —
twist 콜백 구동입니다. 100 Hz 고정 타이머를 세워야 합니다.

---

# 12. 안전

## 12.1 힘 한계 ✅ / 🟡

| 항목 | 값 | 상태 |
|---|---|---|
| `F_n` **하드 한계** | **7.0 N** | ✅ (기본값, 추후 정밀 설정) |
| `F_n` 소프트 한계 | 6.0 N → 경고 + 상승 금지 | 🟡 |
| 모멘트 한계 | 0.3 N·m | 🟡 |

## 12.2 워치독 — 2층 구조 ✅

`워치독 시간 × 접근 속도 = 최악 침투 깊이` (§1.1).

**후퇴는 카테시안 개념이므로 관절 공간 노드가 수행할 수 없습니다.** `servo_node`는 관절
적분기라 "법선 방향"을 모릅니다. 프로브 프레임을 아는 것은 `diff_ik_node`입니다. 따라서
워치독을 두 층으로 나눕니다.

| 층 | 감시 대상 | 만료 | 동작 |
|---|---|---|---|
| **상위** `diff_ik_node` | `/desired_twist` | 0.1 s 🟡 | **프로브 −z 후퇴 twist 자체 생성** |
| **하위** `servo_node` | `/joint_velocity_cmds` | 0.1 s 🟡 | **속도 0** — 최후 방어선 |

정상 경로에서는 상위 층이 후퇴를 수행합니다. `diff_ik_node`까지 죽으면 하위 층이 정지시킵니다.
"후퇴하다가 IK가 죽으면?"에 대한 답이 하위 층이고, 그래서 하위 층은 후퇴가 아니라 정지여야
합니다 — 방향을 모르는 채로 움직이는 것보다 서는 편이 안전합니다.

| 그 밖의 신선도 감시 | 만료 | 동작 |
|---|---|---|
| F/T | 0.05 s 🟡 | RETREAT |
| US | 0.5 s 🟡 | 영상축 0 (힘 루프는 유지) |
| joint_states | 0.2 s | 유지 (기존 값) |

핵심 변경: 상위 층 만료 시 동작이 "속도 0"이 아니라 **법선 방향 후퇴**입니다.
0.1 s × 10 mm/s = 최악 1 mm 침투.

후퇴 속도 🟡 5 mm/s, `F_n < 0.2 N`까지.

## 12.3 속도 한계 🟡

| 항목 | 현재 | 접촉 중 |
|---|---|---|
| 관절 속도 | 1.5 rad/s | **0.5 rad/s** |
| 카테시안 병진 | 없음 | **10 mm/s** |
| 카테시안 회전 | 없음 | **0.2 rad/s** |

## 12.4 종료 절차 ⚠️

현재 `perform_homing()` (`fr5_servo_joint_control_node.py:117-155`)은 `ServoMoveEnd` 후
`MoveJ(vel=15%)`로 초기 자세 복귀합니다. **접촉 중이면 프로브가 피부를 긁으며 이동합니다.**

✅ **강제 순서: 법선 후퇴 → `F_n < 0.2 N` 확인 → 그 다음 homing.**

## 12.5 무하드웨어 검증 — 로봇 백엔드 추상화 ✅

실로봇 앞에서만 검증할 수 있는 구조는 안전 로직을 시험하기 어렵습니다. 워치독 만료, F/T
드롭아웃, 힘 한계 초과 같은 **실패 경로는 실로봇에서 일부러 일으키기 곤란**합니다.

따라서 fairino RPC를 인터페이스 뒤로 감추고 두 구현을 둡니다.

```python
class RobotBackend(Protocol):
    def connect(self) -> None: ...
    def joint_positions_deg(self) -> list[float]: ...
    def joint_velocities_deg_s(self) -> list[float]: ...
    def joint_torques(self) -> list[float]: ...
    def tool_pose(self) -> list[float]:       # [x,y,z (mm), rx,ry,rz (deg)]
    def wrench_raw(self) -> list[float]:      # [Fx,Fy,Fz,Mx,My,Mz] 센서 프레임
    def servo_start(self) -> None: ...
    def servo_j(self, joint_pos_deg, cmd_t, cmd_id) -> int: ...
    def servo_end(self) -> None: ...
    def move_j(self, joint_pos_deg, vel) -> int: ...
```

| 구현 | 용도 |
|---|---|
| `FairinoBackend` | 실로봇. `Robot.RPC` 위임 |
| `MockBackend` | 무하드웨어. 지령을 적분해 관절 상태를 만들고, 합성 wrench를 낸다 |

`MockBackend`가 제공해야 하는 것:

- **지령 적분** — `servo_j`로 받은 관절 각을 그대로 상태로 반영 (완전 추종 가정)
- **합성 접촉** — 가상 평면을 두고 프로브가 파고든 깊이에 비례한 `F_z`를 낸다.
  Phase 1의 admittance 검증에 필요하다
- **주입 가능한 실패** — 파라미터로 F/T 드롭아웃, 상태 지연, 통신 끊김을 일으킬 수 있어야
  한다. 이것이 mock을 두는 주된 이유다

⚠️ **mock은 기구학만 맞습니다.** 조직의 점탄성, 실제 F/T 잡음, 컨트롤러 지연은 재현하지
않습니다. mock 통과는 "로직이 돈다"는 뜻이지 "제어가 맞다"는 뜻이 아닙니다.
Phase 1 이후의 검증 기준은 반드시 실로봇 + 팬텀입니다.

---

# 13. 전환 인벤토리

## 13.1 제거해야 하는 위험 잔재 ⚠️

| # | 위치 | 문제 |
|---|---|---|
| 1 | `fr5_servo_joint_control_node.py:158-173` | `gripper_callback`이 `ServoMoveEnd` → `MoveGripper` → `ServoMoveStart`. **접촉 중 서보 모드 중단** |
| 2 | `:106-114` | 기동 시 `ActGripper` + `MoveGripper` 2회 + `sleep(4)` |
| 3 | `:289-297` | `destroy_node`의 `ActGripper(1,0)` |
| 4 | `:117-155` | `perform_homing` — §12.4 |
| 5 | 5개 파일 | 툴 오프셋 하드코딩 불일치 — §4.2 |
| 6 | 전역 | 페이로드 설정 호출 **부재** — §2.3 |
| 7 | `:27` | `max_joint_vel = 1.5 rad/s` — 접촉 중 과도 |

## 13.2 패키지별 처리

### `fr5_control`

| 파일 | 처리 |
|---|---|
| `fairino/` SDK | **유지** — F/T API 전부 여기 |
| `fr5_servo_joint_control_node(_limit).py` | **개조(핵심 재사용)** — 125 Hz 적분기 + 워치독. 단일팔화, 그리퍼 제거, **wrench 발행 추가** |
| `fr5_status_node.py` | 개조 또는 서보노드 흡수 (RPC 연결 이중화 정리) |
| `fr5_dual_direct_control_node.py` / `_joint` | 폐기 |
| `fr5_dummy_test_node.py` | **유지** — 하드웨어 없이 루프 검증 |
| `fairino_driver.py` | 폐기 (스텁) |
| `reset2h5start.py` | 개조 — 단일팔 + 후퇴 선행 |

### `fr5_ik` — RCM 계열 전부 폐기

| 파일 | 처리 |
|---|---|
| `freespace_two_twist.py` | **개조 → `diff_ik_node`의 베이스.** 단일팔 + 100 Hz 고정 타이머 + `solve()` 분리 + 추종 모니터 |
| `rcm_two_twist` / `rcm_two_pos` / `rcm_two_delta` / `rcm_two_delta_new_rcm` / `rcm_two_absolute_new_rcm` / `rcm_control_node` | **폐기 (6개)** |
| `calc_gripper_wrt_new_rcm_node.py` | 폐기 |
| `free_control_node.py`, `freespace_two_pos.py` | 폐기 |

> RCM은 US에 대응물이 없습니다. "가상 구속"이 필요하다면 그것은 표면 법선 정렬이고,
> 합의대로 Mx/My로 처리합니다(§5). RCM 코드는 개조가 아니라 삭제입니다.

### `fr5_vision` — 거의 전부 폐기

| 파일 | 처리 |
|---|---|
| `camera_node.py` | 개조 → `us_frame_node`. **타이머 콜백 내 `cv2.imshow` + `waitKey(1)` (`:53-57`) 반드시 제거** — 캡처 경로에 GUI 지연 유입 |
| `calibrate_intrinsics` / `save_calibration_data` / `calculate_calibration` + `calibration_data_right/` | 폐기 — eye-to-hand 내시경 캘리브. US는 **probe-to-image 캘리브**(N-wire/cross-wire 팬텀)로 절차 자체가 다름 |
| `display_3d_to_2d_node.py`, `gt_sparse_depth_node.py` | 폐기 |

### 나머지

| 패키지 | 처리 |
|---|---|
| `fr5_inference` (suturing × 5) | 폐기 — **chunk delta 발행 구조는 policy 노드 템플릿으로 참조** |
| `dataset` | 개조 — wrench / US / ControlState 추가 + §14.3 동기 수정 |
| `keyboard_teleop` | 개조 — `/desired_pose_wrt_rcm` → twist |
| `touch_teleop` (C++) | **유지** — `TELEOP_APPROACH` 단계 |
| `laparo_umi` | 폐기 — 시리얼 UMI, 복강경 기구 전용 |
| `suturing_interfaces` | 폐기 → `us_interfaces` 신규 |

---

## 13.3 `Unet_seg` — 지각 리포의 경계 ✅

> **검증 계획:** `docs/VALIDATION_PLAN.md` — 데이터 도착 시 실행할 지표·그림·합격 기준.
> 신뢰도 지표는 `rus_perception/metrics/trust.py`, 그림은 `scripts/plot_report.py`.
>
> **도판:** `docs/architecture.html` 부록 A. 내보낸 이미지는 `docs/figures/`:
> `07-quality-definitions` (두 품질함수 전체 명세),
> `08-slim-unet` (망 구조 · 256² 기준 텐서 모양),
> `09-slim-vs-standard` (스테이지 차이).
> 각각 `.svg`(테마 대응) + `.png`(2배율, 라이트 고정) 쌍입니다.
> `python docs/export_figures.py`로 재생성합니다.

### 13.3.1 역할 정의

`Unet_seg`는 **ROS 스택의 일부가 아닙니다.** `perception_node`와 `quality_raw_node`가
호출하는 **지각 라이브러리 + 오프라인 학습/평가 워크벤치**입니다. 세 가지 역할을 가집니다:

| 역할 | 산출물 | 소비자 |
|---|---|---|
| 런타임 | `ControlState` (30 Hz), `Q_raw` (30 Hz) | `perception_node`, `quality_raw_node` |
| 어휘 정의 | `rejection_reason` 13종 + 신규 4종, `Q_seg`/`Q_raw` 가중치 | `supervisor` 상태 전이 (§10.2) |
| 오프라인 | 학습 / 평가 / ONNX / 지연 벤치마크 / `live_monitor` | Phase 3 이전 준비, §11.1 지연 실측 |

### 13.3.2 경계 규칙 ✅

이 경계는 리포 자체의 테스트로 강제됩니다
(`test_control_state_never_contains_robot_commands`, `test_raw_quality_never_emits_a_robot_command`).

| `Unet_seg`에 두는 것 | `fr5_control` 쪽에 두는 것 |
|---|---|
| 영상 → 스칼라/기하 특징 | **twist / 힘 / 관절 명령** |
| `Q_seg`, `Q_raw`, validity 게이트, 이유 코드 | **상태 전이 로직** (§10) — supervisor 소관 |
| 학습·평가·ONNX·벤치마크 | **ROS 의존 일체** — msg 변환은 노드 쪽 |
| US 영상 기하 (ROI 마스크, 깊이 밴드, A-line 규약) | 로봇 기하 (§4.2 `probe.yaml`) |

**`valid_for_control` / `usable_for_contact_search`는 "관측 판정"이지 "명령"이 아닙니다.**
그것을 받아 무엇을 할지는 전적으로 supervisor의 결정입니다.

### 13.3.3 왜 `Q_raw`도 `Unet_seg`에 두는가 ✅

§6.2의 `Q_raw`는 U-Net을 쓰지 않지만 계산 위치는 `Unet_seg`입니다. 세 가지 이유:

1. **US 영상 기하를 공유합니다.** 부채꼴 ROI 마스크, 깊이 밴드 비율, A-line 규약은
   *초음파 장비*의 성질이지 두 품질함수 각각의 성질이 아닙니다. 두 리포에 나눠 정의하면
   반드시 어긋납니다.
2. **§14.2-2 "`Q_raw` 4개 지표의 타당성 검증"은 오프라인 작업입니다.** 녹화 영상 + pytest
   하네스 + `live_monitor` 패널을 그대로 씁니다. ROS를 띄울 이유가 없습니다.
3. **`Q_raw`는 "제어"가 아니라 "영상 품질"입니다.** `Q_seg`와 같은 범주이고, 리포의 금지
   경계(로봇 명령 방출)를 침범하지 않습니다.

→ `quality_raw_node`는 계산 로직을 갖지 않습니다. `compute_raw_quality()` 호출 + 발행뿐입니다.

**구현 상태 (2026-08-14):** `rus_perception/control/raw_quality.py` — **골격 완료.**
구조 / 설정 스키마 / 이유 코드는 확정, **모든 수치 상수는 ⏳ `PROVISIONAL`**.
테스트 27개 통과. §14.1-4(US 영상 기하) 이전에는 어떤 상수도 확정할 수 없습니다.

### 13.3.4 설계 결정 두 가지 (검토 요망)

**(a) 스캔 기하.** `scan_geometry: linear`만 구현되어 있고, `sector`(convex/phased)는
**예외를 던집니다.** 스캔 변환 후 A-line은 이미지 컬럼이 아니라 apex에서 뻗는 광선이라,
컬럼을 A-line으로 취급하면 서로 다른 깊이를 조용히 평균냅니다. ⏳ §14.1-4에서 프로브
종류가 확정되면 구현합니다.

**(b) 음영 판정 기준.** A-line의 원거리 에너지 붕괴를 **프레임 자신의 원거리 90-percentile**
대비 상대값으로 판정합니다. 장비측 gain/TGC 변경이 "새로운 음영"으로 오검출되지 않게 하기
위함입니다. **중앙값(median)이 아닌 이유:** A-line의 절반 이상이 음영이면 중앙값 자체가
음영 레벨이 되어, 프레임이 가장 나쁠 때 정확히 눈이 멉니다. 균일 음영(참조가 아예 없는 경우)
대비로 낮은 절대 하한을 백스톱으로 둡니다.

### 13.3.5 패키지화 ✅

**문제였던 것:** 최상위 패키지 이름이 문자 그대로 `src`였고 `pyproject.toml`이 없었습니다.
`scripts/_common.py`의 `sys.path.insert` 해킹으로만 임포트가 됐습니다. ROS 2 워크스페이스에서
`import src`는 재현 가능한 방법이 없고, 그 해킹이 ROS 노드로 번집니다.

**조치 (2026-08-14 완료):**

```
src/  →  rus_perception/          (설치되는 유일한 최상위 이름)
pyproject.toml 추가                (dist name: rus-perception)
pip install -e Unet_seg
```

```python
from rus_perception.inference import Predictor
from rus_perception.control import compute_control_quality, compute_raw_quality
```

**의도적으로 설치하지 않는 것:** `unet/`, `utils/`, 루트 `train.py`/`predict.py`/
`evaluate.py`/`hubconf.py`. 전부 milesial/Pytorch-UNet 원본 무수정본이고, `unet`·`utils`
같은 범용 이름을 소비자 임포트 경로에 올리면 안 됩니다. 클론에서 실행할 때만 임포트되며
(baseline parity 테스트가 필요로 하는 전부), `scripts/`도 패키지에 포함하지 않습니다.

→ **Phase 0 항목입니다.** perception_node를 짜기 전에 끝나 있어야 합니다.

### 13.3.6 ROI — 두 품질함수가 공유하는 단일 정의 ✅ 배선 완료

부채꼴 ROI가 세그 경로 어디에도 없었습니다. 그래서 세 가지가 조용히 깨져 있었습니다:

| 값 | 깨진 방식 |
|---|---|
| `mask_area_ratio` | 분모가 **프레임 전체 직사각형**. 프레임그래버 크롭이 `mask_completeness`를 흔들고, 그건 힘에 민감한 항이므로 → **크롭이 최적 힘 F\*를 바꿨습니다** |
| `segmentation_confidence` | 프레임 전체 평균이라 **빔 바깥 검은 영역이 지배**. 경계에서 헤매도 점수가 높게 나오는데 `min_segmentation_confidence: 0.50`은 활성 게이트 |
| `border_contact_ratio` | 이미지 사각형 가장자리만 센다. 부채꼴이 프레임 안에 내접하면 **마스크가 그 가장자리에 절대 닿지 않아 이 기준이 영원히 발화하지 않습니다** |

**조치:** `rus_perception/control/roi.py` 신설. `full`(기본, 기존 동작과 완전 동일) /
`rect`(리니어) / `fan`(convex·phased) / `file` 4가지 모드. `extract_control_state()`와
`compute_raw_quality()`가 **같은 마스크를 받습니다** — Q_raw만 ROI를 알던 비대칭 해소.
`ControlState`에 `roi_mode` / `roi_area_px` 기록 (분모를 모르면 면적비는 의미가 없으므로).

⏳ 기하 파라미터는 전부 자리표시. §14.1-4 확정 시 `control.roi` YAML 한 곳만 바꿉니다.

---

# 14. 미결 / 실측 대기

## 14.1 하드웨어 ⏳

1. **프로브 마운트 CAD** — §4.2의 `j6_to_probe`, §4.3의 `r`. **곧 제공 예정**
2. **F/T 센서** — 모델 / 장착 위치 / 샘플링 레이트. §2.1 근거로 낙관적이나 미확인
3. **US 프레임그래버 지연** — §11.1. 50 ms 초과 시 policy 대역 재설계
4. **US 영상 기하** — `Q_raw`의 ROI/밴드 정의(§6.2). **`raw_quality.py`의 모든 상수를
   막고 있는 단일 항목**이고, `scan_geometry`가 `sector`면 A-line 샘플링 구현이 추가로
   필요합니다 (§13.3.4a)
5. **왼팔 물리적 존재 여부** — 정적 장애물 취급 필요성
6. **방광 US 데이터셋** ⚠️ — `Unet_seg/data/`가 **비어 있고 학습된 체크포인트가 없습니다.**
   방광 US에서의 정확도가 한 번도 측정된 적 없습니다. Phase 3의 크리티컬 패스는 하드웨어가
   아니라 **데이터 수집·라벨링**입니다. Phase 0–2와 **병렬로 즉시 시작해야** 합니다

## 14.2 설계 ❓

1. **policy 모델 형태** — world model / IBVS interaction matrix / behavior cloning. **미결**
   - 3-DoF로 축소되어(쟁점 7) 난이도가 크게 낮아진 상태에서 재검토
2. **`Q_raw` 4개 지표의 타당성** — §6.2. Phase 3에서 실증 필요.
   `metrics.trust.force_response()`가 단조·단봉 판정을 계산하고 `plot_report.py`가
   그림 B8로 그린다. 합격 기준은 `docs/VALIDATION_PLAN.md` 게이트 4
3. **`target_area_ratio = 0.15`** — §6.3. 실제 방광 영상으로 재설정.
   더 근본적으로, **`Q_seg`의 8개 가중치 전부가 임의값이다.**
   `metrics.trust.component_attribution()`이 각 항의 기여를 측정해 재도출 근거를 준다
   (그림 B5). 0 왼쪽 막대 = 그 항을 빼면 Q가 좋아진다는 뜻
4. **`accuracy_floor` / `target_bad_rate`** ❓ — "행동해도 되는 Dice 하한"과
   "신뢰한 프레임 중 틀려도 되는 비율". **측정이 아니라 제어 요구에서 오는 결정**이고,
   검증의 모든 합격/불합격이 여기 걸린다. `VALIDATION_PLAN.md` §1

## 14.3 알려진 결함 ⚠️

**시간 동기.** `fr5_h5_collector.py:103`이 "최신값 스냅샷"에 `time.time()`을 찍습니다.
`(pose, wrench, image)`의 정밀 페어링이 불가능합니다.

이것은 **학습 데이터의 근본 품질 문제**입니다. policy 학습 전에 반드시 해결해야 합니다
(`message_filters.ApproximateTimeSynchronizer` 또는 각 소스의 원 타임스탬프 보존).

### 옵티컬 플로우가 런타임 비용이라는 점 ⚠️ 지연 예산 미반영

`Unet_seg`의 `flow/`는 **학습 전용이 아닙니다.** §6.3에서 시간축 3항을 `valid_for_control`
게이트로 이관했는데, `rus_perception/control/features.py`의 `extract_control_state()`는
`flow=None`이면 이전 마스크를 **워핑 없이** 비교하고 `alignment="unwarped"`로 기록합니다.

→ 프로브가 **정상적으로 슬라이딩하는 동안** `temporal_warped_iou`가 떨어져
`low_temporal_warped_iou`가 뜹니다. §10.2 매핑상 "면내 축 홀드 → STAGE1B 재진입"이 상시
트리거됩니다. **정상 동작이 실패로 판정됩니다.**

**결론:** `perception_node`는 Farneback backward flow를 **30 Hz로 온라인 계산해야 합니다.**
그 비용(256×256 기준 수 ms~수십 ms)이 §11.1 지연 예산에 반영되어 있지 않습니다.
**Phase 3 지연 실측 항목에 "U-Net 추론 + 플로우 계산"을 분리 계측해 추가하십시오.**

대안(⏳ 미결): 시간축 게이트를 끄고(`min_temporal_warped_iou: null`) 플로우를 생략하는 선택.
그 경우 세그 튀는 것을 거를 수단이 사라지므로, 지연 실측 결과를 보고 결정합니다.

---

# 15. 구현 순서

각 Phase는 **검증 기준을 통과해야** 다음으로 넘어갑니다.

| Phase | 내용 | 검증 기준 |
|---|---|---|
| **0** | 단일팔 골격. servo_node 개조(그리퍼 제거, wrench 발행), 툴 YAML, diff_ik 개조 + `solve()` + 추종 모니터, **로봇 백엔드 추상화 + mock 구현**, **`pip install -e Unet_seg` (§13.3.5)** | mock 백엔드로 전 구간 무하드웨어 구동. twist 추종 오차 < 임계. ROS 워크스페이스에서 `import rus_perception` 성공 |
| **1** | F/T 통합. 페이로드 식별, wrench 기준점 이동, **z축 admittance만** | 팬텀에 일정 힘 유지. **부호 규약 검증**(§4.4) |
| **2** | rx/ry 정렬 추가 | 경사면(🟡 15°)에 프로브 자동 정렬 |
| **3** | US 파이프라인. `us_frame_node`, `perception_node`, `quality_raw_node`. US 영상 기하 확정 → `raw_quality.py` 상수 확정 (§13.3.3) | **지연 실측 — U-Net 추론 / 플로우 계산 분리 계측** (§14.3). `Q_raw` 4개 지표의 힘 응답 단조·단봉성 확인 |
| **4** | Stage 1 힘 탐색 (1a + 1b) | `F*` 재현성 — 동일 지점 반복 시 분산 |
| **5** | supervisor 상태머신 + 복구 | 강제 실패 주입 시 정상 복구 |
| **6** | policy (3-DoF). 데이터 수집 + 학습 | §14.2-1 모델 형태 결정 선행 |

Phase 0–2는 US 없이 진행 가능합니다. Phase 3의 하드웨어 대기와 병렬로 갈 수 있습니다.

## 15.1 Phase 0 검증 통과 후 정리할 것 ✅

아래는 **지금 지우지 않습니다.** `us_servo_node` / `us_diff_ik_node`가 실기에서 검증되기
전까지는 기존 복강경 경로가 유일하게 동작이 확인된 참조이기 때문입니다. Phase 0 검증
기준을 통과한 시점에 `legacy_laparoscopic/`으로 옮깁니다.

| 대상 | 남겨 둔 이유 | 옮길 때 함께 할 일 |
|---|---|---|
| `fr5_ik/freespace_two_twist.py` | `us_diff_ik_node`가 대체하지만 미검증. `calculate_initial_ref()`가 가짜 RCM 기준 프레임을 만들어 `gripper_wrt_rcm`을 발행하는 잔재 | `fr5_ik/setup.py` 진입점 제거, `teleop.launch.py` 갱신 |
| `fr5_control/fr5_servo_joint_control_node{,_limit}.py` | `us_servo_node`가 대체하지만 미검증. 듀얼암 + 그리퍼 | `fr5_control/setup.py` 진입점 제거 |
| `fr5_control/fr5_status_node.py` | `us_servo_node`에 흡수 예정 | RPC 연결 이중화 정리 |
| `keyboard_teleop` | **이미 동작 불능** — `/fr5_right/current_gripper_wrt_rcm`를 구독하는데 활성 트리에 발행자가 없음. 소비자였던 `*_two_pos`는 이미 legacy | 폐기하거나 twist 기반으로 재작성 |
| `touch_teleop/run_touch_teleop.sh` | **이미 깨짐** — legacy로 옮긴 `rcm_control`을 실행. 경로도 하드코딩 | 삭제 (런치파일로 대체됨) |
| `dataset/*` 3개의 `*_wrt_rcm` h5 필드명 | 기존 에피소드와의 호환 | 수집기 개조(Phase 6 데이터 수집) 때 함께 |

⚠️ 이 목록은 **Phase 0 검증 기준 통과가 조건**입니다. mock 통과만으로는 부족합니다 —
§12.5에 적은 대로 mock은 기구학만 맞습니다. 실로봇에서 twist 추종을 확인한 뒤에 옮깁니다.

---

## 부록: 결정 이력

| 날짜 | 쟁점 | 결정 |
|---|---|---|
| 2026-08-14 | 1 | 위치 탐색이 아닌 **힘 탐색**, 유계 지표 기준 |
| 2026-08-14 | 2 | **구간 평균 `Q̄`** 사용 |
| 2026-08-14 | 3 | **최대 품질을 내는 최소 힘**, 안전 구간(기본 7 N) 내 |
| 2026-08-14 | 4 | 동의 |
| 2026-08-14 | 5 | **Stage 1 / Stage 2 품질함수 분리** |
| 2026-08-14 | 6 | FORCE LOCK = **setpoint**, 힘 최적화는 **배경 상시** |
| 2026-08-14 | 7 | **rx/ry를 Mx/My로 정렬** → policy 5-DoF에서 **3-DoF**로 축소 |
| 2026-08-14 | 8 | **DLS로 상위 아키텍처 먼저 검증.** QP 보류, 인터페이스만 고정 |
| 2026-08-14 | 9 | 상태머신 — 🟡 제안값 채택 |
| 2026-08-14 | 10 | 타임스케일 — 🟡 제안값 채택 |
| 2026-08-14 | — | 제어 대상: **오른팔 단일** (`192.168.58.3`) |
| 2026-08-14 | — | 복강경 전용 코드 `legacy_laparoscopic/` 으로 분리 (196개 파일) |
| 2026-08-14 | — | 워치독 **2층 구조** — 후퇴는 상위(IK), 정지는 하위(servo). §12.2 |
| 2026-08-14 | — | `Unet_seg`는 **지각 라이브러리**, ROS 스택 아님. 경계 규칙 §13.3.2 |
| 2026-08-14 | — | **두 품질함수 모두 `Unet_seg`가 정의처.** `Q_raw`도 여기(§13.3.3). ROS 노드는 얇은 래퍼 |
| 2026-08-14 | — | 패키지 `src` → **`rus_perception`** 리네임 + `pyproject.toml`. Phase 0 항목 (§13.3.5) |
| 2026-08-14 | — | `Q_raw` 음영 판정은 **90-percentile 상대** 기준 (median 아님). §13.3.4b |
| 2026-08-14 | — | `scan_geometry: sector`는 **구현하지 않고 예외**. 컬럼≠A-line (§13.3.4a) |
| 2026-08-14 | — | **로봇 백엔드 추상화 + mock** 을 Phase 0에 포함. §12.5 |
| 2026-08-17 | — | **ROI를 두 품질함수가 공유**. `mask_area_ratio` 분모·`segmentation_confidence` 지지·`border_contact_ratio` 경계가 전부 ROI 기준 (§13.3.6) |
| 2026-08-17 | — | 신뢰도 검증을 **정확도와 분리**. `metrics/trust.py` + 그림 13종. 핵심 수치는 `trusted_bad_rate` (`docs/VALIDATION_PLAN.md`) |
| 2026-08-17 | — | `StandardUNet` `paper` 프리셋에 `in_channels/out_channels=1` 고정 — 없으면 8,636,418로 논문 대조가 성립하지 않음 |
| 2026-08-17 | — | **학습 : 검증 = 7 : 3, 시험셋 없음** (`ratios: [0.70, 0.30, 0.00]`). 비율은 프레임이 아니라 **환자 수**에 걸림. `strategy: patient_random`, `evaluation.split: val`. 대가는 임계를 고른 셋에서 성능을 보고하는 것 — 환자가 늘면 3분할로 복귀 |
| 2026-08-17 | — | Touch 원격조작 **프로파일 분리** — 회전 데드존 파라미터화. §10.4 |
| 2026-08-17 | — | 회전 중심 = 프로브 접촉면이 **RCM의 대체물**. §4.5 |
| 2026-08-17 | — | 복강경 잔재 정리는 **Phase 0 실기 검증 통과 후**. §15.1 |
| 2026-08-17 | — | **모멘트 setpoint 편향** — policy가 기울임을 쓴다. `M → M*`. §5.4 |
| 2026-08-17 | — | policy 출력 = **속도 3 + setpoint 3** `(v_x,v_y,ω_z,M*_x,M*_y,F_n*)`. §5 |
| 2026-08-17 | — | 디더 **폐기** → 절충안: Stage 1 명시 탐색 + Stage 2 policy. §8.4 |
| 2026-08-17 | — | 0.2 Hz 루프 소멸. 타임스케일 5단 → 5단(구성 변경). §11 |
| 2026-08-18 | — | **모드 계약** — admittance 가 유일한 twist 발행자, teleop 은 통과 모드. §10.5 |
| 2026-08-18 | — | 지각 노드 **하나로 통합** — 두 품질함수가 같은 ROI 를 봐야 한다. §3.1 |
| 2026-08-18 | — | setpoint 를 `WrenchStamped` 로 묶어 동기 확보. `us_interfaces` 연기. §3.2 |
| 2026-08-18 | — | `Q_raw` 측정 불가 = **NaN** (0 아님). 소비자 둘 다 명시 처리. §3.2 |
