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
