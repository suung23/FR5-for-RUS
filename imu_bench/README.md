# imu_bench — BNO085 9축 IMU 스트리밍 + 호스트 퓨전

XIAO nRF52840 Sense 에 물린 **BNO085(9축)** 에서 가속·자이로·자력계 원시값을 받아
출력하고, 호스트에서 Madgwick 퓨전을 돌려 칩 자체 퓨전 결과와 비교한다.

## 출처

펌웨어와 바이너리 프로토콜 파서는 [SurgiTagV2](https://github.com/Rosota-Research/SurgiTagV2)
의 `MCU/` 에서 가져왔다.

* `firmware/umi_device_hardware/` — 원본 `umi_device_hardware/` + `UMI_IMU_ONLY` 옵션 추가
* `host/umi_protocol.py` — 원본 그대로

`mag_calib.py`, `imu_fusion_view.py`, `firmware/i2c_scan/`, `scripts/` 는 이 벤치용으로 새로 썼다.

## 이 벤치의 하드웨어 구성

I2C 스캔으로 확정한 실제 구성 (`./scripts/flash.sh i2c_scan` 으로 재확인 가능):

| 버스 | 주소 | 칩 |
|---|---|---|
| `Wire` (외부 D4/D5) | `0x4B` | **BNO085 — 9축 (가속 + 자이로 + 자력계)** |
| `Wire1` (내부) | — | 없음 (온보드 LSM6DS3TR-C 미응답) |

원본 SurgiTagV2 기기와 달리 **VL53L0X 거리센서(`0x29`)와 홀 센서가 없다.** 원본 펌웨어는
둘을 필수로 전제해서 VL53L0X 초기화 실패 시 `Error: VL53L0X Failed.` 를 찍고
`while(1)` 로 정지한다. 그래서 `UMI_IMU_ONLY` 빌드 옵션을 추가했다 — `LEGACY_CSV` /
`UMI_BOARD_ID` 와 같은 방식으로 `calibration.h` 에 선언되고, ToF·홀의 bring-up 과
샘플링 블록을 `#if !UMI_IMU_ONLY` 로 감싼다. 원본 동작은 기본값(`0`)에서 그대로다.

## 사용

```bash
./scripts/setup_toolchain.sh          # 최초 1회 (arduino-cli, 보드 코어, 라이브러리)
./scripts/flash.sh                    # IMU 전용 펌웨어 빌드 + 업로드
./scripts/run.sh --calibrate-mag 20   # 자력계 보정 — 8자 모션으로 돌려야 한다
./scripts/gui.sh --log                # 실시간 GUI + 로깅   ← 평소 쓰는 것
```

### GUI (`scripts/gui.sh` -> `host/imu_gui.py`)

한 화면에 6 패널: 3D 자세(호스트 실선 / 칩 점선), 오일러각, 가속도, 자이로, 자력계,
호스트 vs 칩 각도차. 상단에 모드·보드명·스트림별 실측 Hz·CRC 오류·시퀀스 누락·로깅 상태.

| 키 | 동작 |
|---|---|
| `R` | 로깅 시작 / 중지 |
| `Z` | **영점 재설정** (센서를 가만히 둔 채) |
| `C` | 자세 재설정 (TRIAD seed 와 프레임 정렬을 다시 잡는다) |
| `Q` | 종료 |

```bash
./scripts/gui.sh                 # 보기만
./scripts/gui.sh --log           # 시작과 동시에 로깅
./scripts/gui.sh --no-mag        # 6축 IMU-only 로 퓨전해 비교
./scripts/gui.sh --history 60    # 그래프에 60초 보기
```

수신·퓨전은 별도 스레드(`imu_stream.ImuStream`)가 돌리고 GUI 는 그리기만 한다. 그리기가
밀려도 로그에는 모든 퓨전 스텝이 남는다. 화면은 50 Hz 로 솎아 보관하고, **로깅은 솎지
않는다** — 화면은 픽셀 수만큼만 보여줄 수 있지만 데이터는 목적 그 자체이기 때문이다.

X 서버가 없는 셸(에이전트·서비스)에서는 `/tmp/.X11-unix` 를 뒤져 DISPLAY 를 자동으로 잡는다.
레이아웃만 확인하려면 X 없이도 되는 스냅샷 모드가 있다:

```bash
python3 host/imu_gui.py --snapshot /tmp/gui.png --snapshot-after 8
```

### 영점 캘리브레이션 (`host/zero_ref.py`)

**로깅을 시작할 때 자동으로 먼저 돈다** (기본 3 초, `--zero 0` 으로 끔, GUI 에서는 `[Z]` 로 재실행).
센서를 가만히 둔 상태에서 세 가지를 잡는다:

| 잡는 값 | 왜 필요한가 |
|---|---|
| **기준 자세 `q0`** | 이후 자세를 "처음 대비 얼마나 돌았는가"로 읽기 위한 원점. 로그에 `rel_roll/pitch/yaw` 로 남는다 |
| **지구프레임 가속도 바이어스 `b_E`** | 변위 계산의 핵심. 정지 시 지구프레임 가속도는 `(0,0,g)` 여야 하는데 스케일 오차로 어긋난다. 공칭 중력만 빼면 **1 초에 112 mm** 틀어진다 |
| **정지 여부 게이트** | 움직이는 동안 잡은 영점은 **조용히 쓰레기가 된다.** 자이로·가속도 산포로 판정하고 실패하면 재시도한다 |

출력 예:

```
=== 영점 캘리브레이션 3.0초 — 센서를 움직이지 마세요 ===
  기준 자세 q0      : [+0.8745 +0.0134 +0.0916 -0.4762]  rpy=(  -3.71   +9.95  -57.47)
  지구프레임 바이어스: (-0.0066 -0.0013 +10.0296) m/s^2
  측정 중력         : 10.0296 m/s^2   (표준 9.80665, 스케일 오차 +2.27 %)
  중력벡터 기울기   : 0.038 deg   (0 에 가까워야 자세 추정이 맞은 것)
  정지 판정         : 통과   (gyro σ 0.00021, mean 0.00002 / accel σ 0.0257, n=735, 3.0 s)
```

**중력벡터 기울기**가 큰 값으로 나오면 자세 추정이 틀린 것이다 — 센서 탓이 아니라 규약
문제일 가능성이 높으므로, 그 상태로 수집하면 안 된다.

영점은 로그의 `.meta.json` 에 통째로 저장되고, `displacement_check.py` 가 그대로 읽어 쓴다.

### 로깅 (`host/imu_log.py`)

퓨전 스텝마다(약 245 Hz) 한 행씩 남긴다. 기본 CSV, `--log-format hdf5` 로 HDF5.

```
logs/imu_<날짜>_<시각>.csv         29 개 컬럼 (아래)
logs/imu_<날짜>_<시각>.meta.json   포트·보드·beta·자력계 보정값·샘플수·구간 시각
```

컬럼: `pc_unix`(PC 절대시각, Unix epoch UTC), `dev_us`(디바이스 이벤트 시각, wrap 펼침),
`ax..az`, `gx..gz`, `mx_raw..mz_raw`, `mx..mz`(보정 후), `host_q*`/`host_roll,pitch,yaw`,
`chip_q*`/`chip_roll,pitch,yaw`, `diff_deg`, **`rel_roll/rel_pitch/rel_yaw`**(영점 대비 상대 자세).

`.meta.json` 에는 포트·보드·자력계 보정값과 함께 **영점 한 벌**(`zero_ref`)이 들어간다.

시각을 둘 다 남기는 이유: 다른 소스와 병합할 땐 절대시각이, 샘플 간격을 따질 땐 디바이스
시각이 기준이다. 둘 중 하나만 있으면 나중에 복구가 안 된다.

### 변위 오차 (`host/displacement_check.py`)

```bash
python3 host/displacement_check.py logs/imu_20260819_162310.csv
```

**"IMU 로 변위를 얼마나 정확히 잴 수 있는가"** 의 오차 바닥을 낸다. 안 움직인 구간에
변위 파이프라인을 그대로 돌려 "0 이 나와야 하는데 얼마가 나오는지" 를 재는 방식이라
로봇 GT 없이도 성능 상한을 알 수 있다. 세 보정 단계를 나란히 비교한다:

| 창 | A 무보정 | B 영점 바이어스 보정 | **C + ZUPT** |
|---|---|---|---|
| 0.5 s | 28 mm | 0.50 mm | **0.23 mm** |
| 1 s | 112 mm | 1.5 mm | **0.67 mm** |
| 2 s | 446 mm | 4.5 mm | **1.7 mm** |
| 5 s | 2.78 m | 19 mm | **7 mm** |

**C 가 실용 상한이다.** 구간 양 끝이 정지임을 알면(ZUPT) 상수 가속도 바이어스가 원리적으로
완전히 사라지기 때문이다. 로봇 point-to-point 이동이 이 조건을 공짜로 만족한다.
자세한 것은 [QC_PLAN.md](QC_PLAN.md) §2.6 · §3.6.

### 로봇 GT 대비 추적 QC (`qc_track/`)

**외부 GT 가 있는** 쪽이다. FR5 플랜지에 IMU 를 달고 teleop 으로 초음파 프로빙하듯
움직이면서, 회전과 병진을 로봇 FK 대비 얼마나 정확히 따라가는지 잰다. 결과 표에
[Surgilogger QC / Exp-Latency](https://github.com/Rosota-Research/QC) 의 실측값과
이 벤치의 변위 오차 바닥이 **기준 열로 같이** 찍힌다.

```bash
cd qc_track
python3 run_session.py --dry-run     # 계획과 조작 지시문
python3 simulate_session.py          # 로봇 없이 분석기 검증 (권장, 먼저)
python3 analyze_track.py --run raw_data_sim
```

자세한 것은 [qc_track/README.md](qc_track/README.md).

### 로그 분석 (`host/analyze_log.py`)

```bash
python3 host/analyze_log.py logs/imu_20260819_162310.csv
```

**외부 GT 가 필요 없는 QC 지표만** 낸다 (Rosota-Research/QC 의 `QC_PLAN.md` 를 이 벤치에
맞춰 좁힌 것):

1. **시간축 감사** — rate, 지터, 구멍, 디바이스시계 skew. **이게 먼저다.** 시간축이 틀어진
   데이터로 낸 노이즈는 노이즈가 아니라 정렬 오차다.
2. **정적 안정성** — 자세 SD 와 드리프트를 **대역폭별로**(샘플당 / 0.2 s / 1 s 평균) 낸다.
   대역폭 없는 "SD < X" 는 합격도 불합격도 마음대로 만들 수 있다.
3. **Allan 편차** — 각도 랜덤워크와 바이어스 안정도. 얼마나 오래 자유적분할 수 있는지.
4. **자기환경 감사** — |m| 과 자기복각의 안정도. 나쁘면 9축을 버리고 6축으로 가야 한다는
   뜻이므로 융합 설계 자체를 바꾼다.
5. **위치 사장추측 예산** — IMU 단독이 얼마나 가는지. 관측이 아니라 전파된 예산이다.

정지 구간은 자이로 크기로 자동 판정한다 (`--window T0 T1` 로 직접 지정 가능).

### 터미널 뷰어 (`scripts/run.sh` -> `host/imu_fusion_view.py`)

GUI 없이 확인할 때, 그리고 자력계 보정에 쓴다.

```bash
./scripts/run.sh                      # 값 + 퓨전
./scripts/run.sh --raw                # 프레임 단위 원시 레코드 덤프
./scripts/run.sh --no-mag             # 6축 IMU-only
./scripts/run.sh --log ../logs        # 로깅하며 보기
./scripts/run.sh --calibrate-mag 20   # 자력계 보정
```

기타:

```bash
./scripts/flash.sh i2c_scan  # I2C 버스에 뭐가 붙었는지 스캔
./scripts/flash.sh full      # ToF/홀 포함 원본 구성으로 빌드
```

## 자력계 보정이 왜 필수인가

보정 없이 받은 원시 자기장은 `|m| = 108~116 µT` 로 나온다. 지구 자기장은 25~65 µT 이므로
**측정값의 대부분이 지구 자기장이 아니라 하드아이언 오프셋**이다 (센서와 함께 회전하는
근처 자석·철심이 만드는 고정 편향). 이 상태에서는 heading 이 의미를 갖지 못한다.
칩도 같은 판단을 보고한다 — `MAG status=0` (unreliable), `RV acc_rad=π` (정확도 최악).

`--calibrate-mag` 는 센서를 모든 방향으로 돌리는 동안 자기장 샘플을 모아, 그 점들이
이루는 구의 중심(하드아이언)과 축별 반경 편차(대각 소프트아이언)를 최소자승으로 푼다.
보정 후에는 자세와 무관하게 `|m|` 이 일정해야 하고, 출력되는 **잔차 %** 가 그 척도다.
**커버리지 %** 는 얼마나 고르게 돌렸는지를 나타낸다 — 낮으면 다시 돌려야 한다.

보정 결과는 `mag_cal.json` 에 저장되고 이후 실행에서 자동으로 적용된다.

## 호스트 퓨전 vs 칩 퓨전 비교 읽는 법

두 필터는 **서로 다른 지구고정 프레임을 쓴다** — 호스트 Madgwick 은 NWU(X=북, Y=서, Z=위),
BNO085 는 ENU. 그래서 두 쿼터니언을 그냥 빼면 항상 큰 값이 나온다. 뷰어는 그 고정
오프셋을 한 번 재두고(`align`) 이후 **벌어짐만** 보여준다. 그 값이 시간에 따라 커지면
그게 실제 드리프트다.

또 하나: Madgwick 을 항등 쿼터니언에서 출발시키면 수렴에 수 초가 걸리는데, 그 과도상태는
드리프트와 구분되지 않는다. 그래서 첫 가속도+자력계 샘플에서 **TRIAD 로 절대 자세를 직접
풀어 필터를 seed** 한다 (`--no-seed` 로 끌 수 있다).

이 벤치에서 실측한 결과 (정지 상태):

```
host  rpy=(  -3.71   +9.94 -147.51)
chip  rpy=(  -3.71   +9.95  -57.47)
host(정렬 후) vs chip =   0.00 deg
```

roll/pitch 가 칩과 소수점까지 일치하고, yaw 차이는 정확히 90° — 위의 NWU/ENU 차이다.
정렬 후 0.00° 는 호스트 퓨전이 칩 내장 퓨전을 재현한다는 뜻이다.

## 파일

```
firmware/umi_device_hardware/   펌웨어 (SurgiTagV2 + UMI_IMU_ONLY 패치)
firmware/i2c_scan/              I2C 버스 스캐너
host/umi_protocol.py            바이너리 프레임 파서 (SurgiTagV2 원본)
host/fusion.py                  Madgwick AHRS + 쿼터니언 유틸 + TRIAD seeding
host/mag_calib.py               자력계 하드아이언/스케일 피팅
host/zero_ref.py                영점 캘리브레이션 (기준자세 + 가속도 바이어스)
host/imu_stream.py              시리얼->파싱->퓨전 백그라운드 스레드 (뷰어 공용)
host/imu_log.py                 세션 로거 (CSV / HDF5 + 메타 JSON)
host/imu_gui.py                 실시간 GUI 뷰어
host/imu_fusion_view.py         터미널 뷰어 + 자력계 보정
host/analyze_log.py             로그 -> QC 지표 (GT 불필요)
qc_track/                       로봇 GT 대비 추적 QC (회전 + 병진)
host/displacement_check.py      변위 오차 바닥 (A/B/C 보정 단계 비교)
scripts/setup_toolchain.sh      툴체인 설치 (최초 1회)
scripts/flash.sh                빌드 + 업로드
scripts/gui.sh                  GUI 실행
scripts/run.sh                  터미널 뷰어 실행
test/test_mag_calib.py          보정 피팅 + TRIAD seeding 검증 (센서 없이 실행 가능)
test/test_zero_ref.py           영점 캘리브레이션 검증 (정지 게이트 포함)
QC_PLAN.md                      QC 실행 계획
logs/                           세션 로그 (gitignore)
```

## 알아둘 것

* **시리얼 접근** — `/dev/ttyACM*` 는 `root:dialout 0660` 이다. `sudo usermod -aG dialout $USER`
  로 그룹에 넣고 재로그인한다. `chmod a+rw` 는 USB 재열거(업로드·리셋마다 발생)마다
  초기화되므로 영구 해결책이 아니다. 스크립트는 그룹을 상속 못 받은 셸에서도 되도록
  `sg dialout` 으로 감싼다.
* **부트로더 진입** — arduino-cli 는 이 코어에서 1200bps touch 를 스스로 걸지 못한다
  (`Touch disabled`). `flash.sh` 가 수동으로 건다. 성공하면 USB PID 가 `0x8045` → `0x0045`
  로 바뀐다.
* **보드 이름** — 펌웨어가 부팅 시 USB 디스크립터를 `CAL_NAME`(`board-6-qc`)로 바꾼다.
  `lsusb` 에 아직 `XIAO nRF52840 Sense` 로 보이면 이 펌웨어가 올라가 있지 않은 것이다.

## Windows 수집 — USB 프로브 (Konted C10UR) + IMU  `host/us_imu_gui_win.py`  (2026-09-09)

C10UR 은 USB 로 붙으면 Cypress FX3 벤더 전용 장치(VID 04B4 / PID BC0C) 이고 벤더 뷰어 WirelessUSG 가
CyUSB.dll 로 직접 말한다. 그 USB 프로토콜은 아직 모르고, 기존 Wi-Fi TCP 수집기(`us_imu_collect.py`) 는 다른
프로브(SL-2C) 용이다. 그래서 Windows 에서는 **뷰어 창을 `PrintWindow` 로 렌더해 오는 화면 캡처**가 초음파
소스다 (`host/us_screen_capture.py`). 다른 창에 가려져도 되고(이 GUI 가 위에 있어도 됨), 최소화만 안 된다.
세션 포맷은 `us_imu_collect.py` 와 같아 `policy_learning/scripts/inspect_session.py` 가 그대로 읽는다.

준비 (한 번): 드라이버 `USB_Probe_Driver\Win10x64\cyusb3.inf` (`pnputil /add-driver … /install`), 뷰어 설치,
뷰어 폴더 쓰기 권한 (`icacls "C:\Program Files (x86)\WirelessUSG" /grant <user>:(OI)(CI)M /T` — 뷰어가
`preference.json` 을 자기 폴더에 만들려 해서 없으면 시작 즉시 죽는다), `pip install pyserial mss`.

```
python host\us_imu_gui_win.py --select-roi          # 처음 한 번: 뷰어 스크린샷에서 영상 영역만 드래그 (Enter)
python host\us_imu_gui_win.py                       # COM 포트 자동 (VID 2886), 뷰어 창 'WirelessUSG' 캡처
```

키: **Z** 영점(정지 3 s) → **R** 녹화 시작 → 정지-이동-정지 스캔 → **R** 정지(= 저장). S ROI 재지정, W 창 다시 찾기, Q 종료.
상태줄 오른쪽 `cap=printwindow … mean=…` 이 캡처 백엔드와 프레임 평균이다. `mean` 이 250 근처면 뷰어가 아니라
다른 창을 찍고 있는 것이다 (`--capture-backend printwindow` 로 고정하거나 W).

알아둘 것:
* 프레임은 ROI → 그레이 → **정방형 레터박스** → 256×256. 레터박스·스케일은 `session.meta.json` 의 `us.capture` 에 남는다.
  비등방 축소를 피한 이유는 convex 부채꼴의 반경(hypot) 기하를 지키기 위해서다.
* 중복 제거: 직전 저장 프레임과 다를 때만 저장한다. 표시 fps 는 **새 프레임** 기준이라 뷰어가 FREEZE 면 0.
* 지연: 프로브 → 뷰어 → 화면 → 캡처 경로의 고정 지연이 붙는다. `inspect_session.py --latency` 로 실측해
  `timing.us_latency_s` 에 넣는다. 캡처 자체는 PrintWindow ~50 ms (2560×1600 창) 라 ~15 fps 상한 — 뷰어 창을
  줄이면 빨라진다.
* 자이로가 정지 중 정확히 0.0 으로 읽히고 `cal_gyr=0` 이었다 (2026-09-09 board-6-qc). 움직이면 값이 나온다.
  정지 판정 임계(`imu.still_gyro_sd`) 가 이 거동을 전제하지 않으므로 첫 세션에서 `inspect_session` 분위수 표를 볼 것.

## Windows Wi-Fi 수집 — C10UR 을 뷰어 없이 (`host/us_imu_gui.py --probe c10ur`)  (2026-09-09, 권장 경로)

동글(Realtek 8814AU, "Wi-Fi 2") 로 프로브 AP `US-1C GRCGBA010` 에 붙으면 (비밀번호 `usccgba010` — 뷰어가 만든
Windows 프로필에서 복구, `host/probe_wifi_win.py` 가 접속·판정) 프로브 192.168.1.1 의 TCP 5002/5003 이 열린다.
pktmon 으로 뷰어 세션을 캡처해 (`logs/pcap/`) 프로토콜을 확인했다: SL-2C 와 같은 골격, 상수만 다르고 **스캔 시작을
클라이언트가 명령**한다 (`fr5_vision/us_protocol.py` 의 `C10UR` 프로파일). 프레임은 160 블록 = 81,920 바이트 =
**160 라인 × 512 깊이 표본, scan conversion 이전 극좌표** (행 = A-line = 블록 1 개의 512 페이로드, 행 시작 = 근거리), 10 fps.
⚠️ 2026-09-09 저녁까지 코드·메타·이 문서에 320 × 256 으로 적혀 있었다 — 바이트 수가 같아 에러 없이 읽히지만 진짜 라인 하나가
두 행(얕은/깊은 절반)으로 쪼개져 부채꼴에 방사형 줄무늬가 났다. 16 세션 실측: 인접 라인 상관 320×256 은 0.1–0.3, 160×512 는
0.93–0.96. 프로토콜 프로파일·`bmode.py`·전 세션 메타(`us.frame_shape`, `fan_geometry.n_lines/n_samples`) 를 함께 고쳤고
원시 바이트는 그대로다. `host/session_to_images.py --shape` 로 두 배치를 나란히 확인할 수 있다.

```
python host\probe_wifi_win.py                                   # 동글 접속 + 포트 + 프레임 판정
python host\us_imu_gui.py --probe c10ur --host 192.168.1.1 --record-frames 1000    # IMU 포트 자동(VID 2886)
python host\us_imu_collect.py --probe c10ur --host 192.168.1.1 --port COM3 --max-frames 1000   # 헤드리스
```

**세션 시작마다 K → Z → R.** K 는 BNO085 칩 보정 안내(자이로: 탁자에 5 s 정지 → 가속도: 6 방향 각 2 s → 자력계: 8 자). Z 의 정지 판정은 `--zero-profile freehand`(기본) — 손으로 몸에 대고 정지한 떨림(자이로 sd ≈ 0.015 rad/s)이 통과한다. 예전 bench 임계(0.005)로는 첫날 15 세션 전부 Z 가 조용히 실패해 `zero_ref` 가 비었다 (파이프라인이 정지 표본으로 규약을 추정하므로 그 세션들도 쓸 수는 있다). 이제 결과(성공/실패 이유)가 체크리스트에 8 초간 뜬다. 초음파 패널 왼쪽 위 체크리스트가 보정 상태(a/g/m 0–3)와 영점을 표시하고, 상태줄에도 `cal a3/g2/m2` 로 보인다. 보정은 칩 안에서만 살아 있어 프로브 전원을 끄면 다시 한다. **프레임은 F 또는 R 을 눌러야 온다** (이 프로브는 클라이언트가 스캔 시작을 명령한다; GUI 창을 클릭해 포커스를 준 뒤). F 가 스캔 시작/정지, R 은 녹화(스캔이 꺼져 있으면 함께 시작). 표시는 `--display fan`(c10ur 기본) 으로 `host/us_scan_convert.py` 가 극좌표를 부채꼴로 바꾼다 (뷰어 화면 실측 R59 mm / 반각 28° / 깊이 220 mm — candidate, 라인 좌우 미검증, `--fan-flip`). 저장은 항상 극좌표 원본이고 기하는 `session.meta.json` 의 `us.fan_geometry` 에 남는다. 세션 포맷은 동일하고 `session.meta.json` 의
`us.frame_shape` 가 [160, 512] 이므로 `inspect_session.py` 가 그대로 읽는다. 뷰어 화면 캡처 경로(`us_imu_gui_win.py`)
보다 지연·CPU 모두 훨씬 낫다 (프레임당 수 ms).

**저장 위치와 형태.** 세션마다 `logs/us_imu_YYYYMMDD_HHMMSS/` 하나: `us_frames.bin` (uint8 160×512 극좌표 프레임을
그대로 이어 붙인 것, 1000 프레임 ≈ 82 MB) + `us_index.csv` (프레임별 `pc_unix` 시각) + `imu_*.csv` (가속도/자이로/자력계/
회전벡터, 같은 `pc_unix` 시계) + `session.meta.json` + 동기화 루프가 만드는 `sync.npz` / `sync_report.json` / `sync_check.png`.
극좌표 원본은 그대로는 U-Net 입력이 아니다 — 학습 파이프라인(`policy_learning/rus_policy/bmode.py` 의 `BmodeConverter`)
이 `us.fan_geometry` 로 부채꼴 B-mode 를 만들고 정방형 레터박스로 `perception.frame_size` 에 맞춘다 (SL-2C 세션은 항등).
눈으로 보려면 언제든:

```
python host\session_to_images.py logs\us_imu_20260909_163511               # images\sheet.png (프레임 12 장 + IMU 궤적)
python host\session_to_images.py logs\us_imu_20260909_163511 --every 10    # images\fan_000000.png ... (부채꼴)
python host\session_to_images.py logs\us_imu_20260909_163511 --every 1 --polar --out D:\dump   # 극좌표 원본 전부
```

미해결: 공기 중에서 스트림이 **약 18 s 뒤 프로브 쪽에서 강제 종료(RST)** 되고 AP 가 잠시 사라진다 (USB+뷰어에서는
42 프레임 뒤 정지). 접촉 자동 정지로 추정 — 젤/팬텀 접촉 상태에서 1000 프레임이 끊김 없이 오는지가 판정 기준.
끊기면 뷰어 세션을 60 s 이상 pktmon 으로 다시 캡처해 우리가 안 보내는 메시지(예: 프로브의 `5bb50000`) 를 본다.
WLAN 프로필은 자동 연결(auto) 이라 AP 가 돌아오면 Windows 가 다시 붙고, 수신기는 2 s 마다 TCP 재접속한다.
