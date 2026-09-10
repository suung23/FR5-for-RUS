# 팬텀 강성 측정 + 힘·초음파 모니터 (리눅스)

새 팬텀의 강성 `k = dF/dδ` 를 로봇의 **PX6D F/T 센서**로 재고, 그 과정을 **접촉력과
초음파 영상을 한 창에서** 보며 진행하기 위한 도구다.

```
stiffness_gui.py   창 하나 — 초음파 B-mode + 접촉력 시계열 + 강성 곡선. 점을 찍어 세션으로 저장
fit_stiffness.py   저장된 세션에서 강성을 적합하고 그림을 낸다 (힘 대역 선택, 이력, 접선 강성)
sources.py         힘·초음파·자세 세 입력을 각각 독립 스레드로 읽는다
stiffness.py       점 자료구조, 적합, 이력, 저장
korean_font.py     그림 폰트
```

---

## 이 셀의 배선 (2026-09-10 실측)

| 것 | 어디 | 확인 |
|---|---|---|
| PX6D F/T | `/dev/ttyACM0` (`usb-GigaDevice_GD32-CDC_ACM_…`) | fw **v1.0.2**, ~1 kHz, 잡음 sd ≈ 0.015 N |
| IMU (XIAO) | `/dev/ttyACM1` (`usb-SurgiTag_UMI_board-6-qc_…`) | 같이 꽂혀 있으면 **번호가 바뀔 수 있다** — `by-id` 로 확인할 것 |
| Wi-Fi 동글 | `wlx705dccf63db8` (Realtek RTL8814AU) | 프로브 AP 전용. 인터넷은 `wlp0s20f3` 이 계속 쥔다 |
| 프로브 | Konted **C10UR**, AP `US-1C …`, `192.168.1.1:5002/5003` | 160 라인 × 512 표본 극좌표, 10 fps |
| 로봇 | FR5 `192.168.58.3` (`eno1`) | 자세(FK)를 쓸 때만 필요 |

> ⚠️ **`--force-port` 를 확인하고 쓸 것.** PX6D 와 IMU 가 둘 다 CDC-ACM 이라
> 꽂는 순서에 따라 `ttyACM0/1` 이 뒤바뀐다. 고정하려면
> `--force-port /dev/serial/by-id/usb-GigaDevice_GD32-CDC_ACM_F9196A776C92-if00`.

### 파이썬 환경

셸 프로필이 `Unet_seg/.venv` 를 활성화한다. 그 venv 에 `rclpy`·`numpy`·`matplotlib`·
`cv2` 가 있고, **`pyserial` 은 2026-09-10 에 넣었다**. 시스템 `/usr/bin/python3` 에는
`pyserial` 은 있지만 `rclpy` 가 없다 — 그래서 GUI 는 venv 로 돈다.

`/dev/ttyACM*` 는 `dialout` 그룹이라 세션에 그룹이 안 붙어 있으면 열리지 않는다.
그럴 때 `sg dialout -c "…"` 로 감싸는데, **`sg` 가 여는 셸도 프로필을 읽어 venv 를
잡으므로** 안쪽에서 `python3` 를 그냥 부르면 venv 파이썬이 온다. 경로를 박아 두면 확실하다.

---

## 1. 프로브를 리눅스에 붙인다

프로브 배터리를 켜고 (USB 는 뽑은 상태), 뷰어가 붙어 있으면 그쪽 Wi-Fi 를 먼저 끊는다.

```bash
python3 imu_bench/host/probe_wifi_linux.py              # 스캔 → 접속 → ping
python3 imu_bench/host/probe_wifi_linux.py --scan-only  # AP 가 보이는지만
python3 imu_bench/host/probe_wifi_linux.py --disconnect # 다 쓰고 뗄 때
```

동글을 자동으로 고른다 (기본 경로가 걸린 인터페이스는 피한다). 프로필은
`ipv4.never-default` 로 만들어 **인터넷이 프로브 AP 로 넘어가지 않는다.**

> **프로브는 클라이언트를 하나만 받는다.** TCP 를 열었다 닫으면 한동안 다음 접속을
> 거부하고 Wi-Fi 재연결로만 풀린다. 그래서 기본 판정은 ping 까지만이다.
> `--frames` 로 프레임까지 시험했다면 GUI 를 열기 전에 `--disconnect` 후 다시 붙인다.

## 2. 창을 띄운다

```bash
# 화면 앞에서
python3 phantom_stiffness/stiffness_gui.py --label phantom_b

# SSH 라면 (창은 이 PC 의 모니터에 뜬다)
DISPLAY=:1 XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority \
  python3 phantom_stiffness/stiffness_gui.py --label phantom_b

# 자세(FK)까지 쓰려면 먼저
source ~/FR5-for-RUS/install/setup.bash
```

세 입력은 서로 독립이다 — **프로브가 꺼져 있어도 힘만으로 뜬다** (`--no-us`),
제어 스택이 없으면 깊이를 손으로 넣는다 (`--no-pose`, `[` / `]`).

| 키 | |
|---|---|
| `space` | 강성 점 기록 — 최근 `--settle` 초의 힘 평균을 지금 깊이와 짝지어 남긴다 |
| `u` | 직전 점 취소 |
| `c` | **접촉 기준** — 지금 자세를 깊이 0 으로 |
| `z` / `Z` | 힘 소프트 영점 / 해제 |
| `[` `]` | 수동 깊이 ∓`--manual-step` (자세 토픽이 없을 때) |
| `p` | 적재 ↔ 제하 전환 |
| `f` | 초음파 스캔 시작/정지 |
| `s` | 저장 (종료해도 자동 저장한다) |
| `q` | 종료 |

## 3. 측정 절차

**로봇을 움직이는 것은 이 도구가 아니다.** 압입은 기존 경로로 준다 — teleop 지령,
펜던트 조그, 또는 힘 목표 변경:

```bash
ros2 param set /us_diff_ik_node contact_control.target_force_n 1.0
```

힘 한계와 후퇴는 그대로 `us_diff_ik_node` 가 관리한다. 창의 빨간 선은 보기용
경고선일 뿐 **아무것도 멈추지 않는다.**

1. 프로브를 팬텀 **위 공중**에 두고 `z` — 힘 영점. 자중·오프셋이 빠진다.
2. 아주 천천히 내려 접촉이 막 잡히는 순간 (‖F‖ 가 잡음 위로 올라오는 곳, 약 0.1–0.2 N)
   에서 `c` — 깊이 0.
3. 한 단계 파고들고 (0.5–1 mm 권장), **힘이 앉을 때까지 기다린 뒤** `space`.
   기다리는 이유는 팬텀이 점탄성이라 같은 깊이에서도 힘이 내려앉기 때문이다 —
   멈추지 않고 훑으면 재는 것은 강성이 아니라 그 속도에서의 겉보기 임피던스다.
   점마다 `force_sd` 와 `relax_n`(응력완화량) 이 남으므로 나중에 걸러낼 수 있다.
4. 목표 상한까지 반복한다. 상한은 **제어가 실제로 도는 대역 위**로 잡는다
   (기존 힘 유지 검증이 0.5–4.0 N 이었다).
5. `p` 를 눌러 제하로 바꾸고 같은 단계로 되돌아 나온다 → 이력이 보인다.
6. `s` 저장.

권장: 접촉을 떼었다 다시 잡아 **3 회 이상 반복**한다. 접촉 위치·각도가 k 를 바꾸므로
한 번의 곡선은 그 자리의 값일 뿐이다.

## 4. 적합

```bash
python3 phantom_stiffness/fit_stiffness.py phantom_stiffness/runs/stiff_phantom_b_*
python3 phantom_stiffness/fit_stiffness.py .../stiff_b_* --force-band 0.5 4.0
python3 phantom_stiffness/fit_stiffness.py .../stiff_*_* --compare        # 회차 비교
```

`--force-band` 로 **제어가 실제로 도는 힘 대역**만 적합할 수 있다. 팬텀은 얕게
누를 때와 깊이 누를 때 기울기가 다르므로 (접촉 면적이 자라는 구간 + 재료가 굳는
구간), admittance 설계에 쓸 값은 그 대역에서 잰 k 다. 직선이 실제로 자료를
설명하는지는 R² 말고 **잔차 rms/max** 를 함께 본다.

내는 것: `stiffness_fit.json` · `stiffness_fit.csv` · `stiffness_fit.png`
(강성 곡선 + 접선 강성).

---

## 기존 값과의 관계

`force_hold_validation/outputs_pooled/phantom_stiffness.csv` 의 **0.3739 N/mm**
(정상상태 평형점 8 개, R² 0.997) 는 **다른 경로로 잰 같은 양**이다: 그쪽은 힘 목표를
바꿔 admittance 가 앉은 자리를 읽고, 이쪽은 깊이를 직접 주고 힘을 읽는다.

같은 팬텀에서 두 값이 어긋나면 admittance 의 정상상태 오차를 의심할 근거가 되고,
**새 팬텀이면 애초에 다른 값이 나오는 것이 정상이다** — 그 차이가 `admittance_b_z`
와 힘 유지 대역을 다시 잡아야 하는지의 근거가 된다
(`DESIGN_NOTES` §8.1, 한계주기 관련 `B_z` 1000→3000 논의).

## 알아둘 것

* **`CMD_SET_RATE` 가 듣지 않는다.** 펌웨어 v1.0.2 는 100 Hz 를 요청해도 1001.7 Hz,
  1000 Hz 를 요청해도 1002.7 Hz 로 보낸다 (2026-09-10 실측). `--force-rate` 는 물론
  `px6d_probe` · `px6d_monitor` 의 `--rate` 도 실효가 없다. 받는 쪽에서 시간으로 잘라 쓴다.
* **하드웨어 영점(`CMD_TARE`) 은 보내지 않는다.** 센서 내부 기준을 바꾸면 저장된
  교정 프로파일의 bias 가 무효가 되고 보상이 없는 오프셋을 계속 뺀다. `z` 는
  화면·기록에만 걸리는 소프트 영점이다.
* **축 배정은 잠정값이다.** `AXIS_ORDER` 는 매뉴얼 §5.3 과 §5.4 가 어긋나 아직
  실측 검증 전이다. 접촉력 축·부호는 `--force-axis` / `--force-sign` 으로 바꾼다
  (기본 `Fz`, 부호 −1 — `force_hold_validation` 과 같다). 한 축씩 눌러
  `px6d_monitor` 막대로 먼저 확인하는 편이 빠르다.
* **US 지연 ≈ 200 ms.** 영상은 힘보다 늦게 온다 (2026-09-10 사후 추정, 유효 지연,
  ±20 ms). 여기서 보정하지 않고 수신 시각 그대로 남긴다. 눌러서 변형이 화면에
  나타나기까지의 지체는 이 지연을 포함한다.
* **초음파 기하는 candidate 다.** 부채꼴 변환(R 59 mm / 반각 28° / 깊이 220 mm) 은
  뷰어 화면 실측이고 라인 좌우는 미검증이다 (`--fan-flip`). 저장은 언제나 극좌표 원본이다.
