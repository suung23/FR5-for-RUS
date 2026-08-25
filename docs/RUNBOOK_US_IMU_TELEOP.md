# 런북 — 초음파 수신 · IMU · 통합 뷰어 · Teleop

터미널에 **무엇을 치면 무엇이 되는지**를 모아둔 실행 노트.
값·아키텍처 설명이 아니라 "이 명령 → 이 결과" 만 적는다.

- 로봇: FR5 오른팔 단일, 유선 `robot-net` (`eno1`, 192.168.58.x), 로봇 IP `192.168.58.3`
- 프로브 영상: Wi-Fi 동글 `wlx705dccf63db8` → 프로브 AP `SL-2C GMCEKC017` → TCP `192.168.1.1:5002/5003`, 256×256 candidate, ~8 fps
- IMU: BNO085(XIAO nRF52840 Sense), USB 시리얼 `/dev/ttyACM0`, 가속·자이로 ~200 Hz / 자력·RV ~100 Hz
- 시리얼은 `root:dialout` 이라 IMU 관련 명령은 **`sg dialout -c "..."`** 로 감싼다.
- 본체 화면에 창을 띄우려면 SSH 셸에서 `export DISPLAY=:1 XAUTHORITY=/run/user/1001/gdm/Xauthority`.

---

## 0. 한 번만 / 세션 시작 시

```bash
cd ~/FR5-for-RUS
source /opt/ros/jazzy/setup.bash
source install/setup.bash                 # ROS 노드(us_frame, us_servo, teleop 등) 실행 전 필수
```

빌드가 필요할 때 (소스 수정 후):
```bash
colcon build --packages-select fr5_vision fr5_control fr5_ik touch_teleop --symlink-install
```

---

## 1. 프로브 Wi-Fi 연결 (sudo 필요 — polkit)

```bash
sudo nmcli connection up "SL-2C GMCEKC017"          # 프로브 AP 접속 (동글 사용)
ip -4 addr show wlx705dccf63db8 | grep inet         # 192.168.1.x 받으면 성공
ip route | grep default                             # 여전히 wlp0s20f3 여야 인터넷 유지
```

연결이 자꾸 끊길 때 (절전 끄기 — 권장):
```bash
sudo iw dev wlx705dccf63db8 set power_save off
sudo nmcli connection modify "SL-2C GMCEKC017" wifi.powersave 2 connection.autoconnect yes
```

프로브 포트 도달 확인 (뷰어/수집기가 안 붙어 있을 때만 OK로 나온다 — TCP 1클라이언트):
```bash
for p in 5002 5003; do timeout 3 bash -c "echo >/dev/tcp/192.168.1.1/$p" 2>/dev/null \
  && echo "TCP $p OK" || echo "TCP $p FAIL"; done
```

> **주의:** 프로브는 Wi-Fi STA 도 TCP 클라이언트도 **하나만** 받는다.
> 뷰어·수집기·통합 GUI·`us_frame` 노드 중 **동시에 하나만** 프로브에 붙일 수 있다.
> 다른 PC(예: Windows WirelessUSG)가 붙어 있으면 우리가 `denied authentication` 으로 막힌다.

문제 진단:
| 증상 | 원인 / 확인 |
|---|---|
| `denied authentication (status 1)` | 다른 기기가 AP 슬롯 점유. 그 기기 Wi-Fi 끊기 / 프로브 재부팅 |
| `start auth` 반복, 응답 없음 | 드라이버 채널플랜. `cat /sys/module/8814au/parameters/rtw_country_code` = KR 확인 |
| 붙었는데 인터넷 끊김 | 프로브가 기본 경로 가로챔. profile 에 `ipv4.never-default yes` 확인 |
| `scanner_active=False`, 0 frames | 프로브가 스캔/스트리밍 상태가 아님 — 장비에서 스캔 시작(프리즈 해제) |

---

## 2. 초음파 영상만 보기 (뷰어)

### 2a. 실기 프로브
```bash
export DISPLAY=:1 XAUTHORITY=/run/user/1001/gdm/Xauthority
cd ~/FR5-for-RUS/tracer/ubuntu_22_04
python3 -m receiver.direct_tcp_receiver --host 192.168.1.1 --save-dir captures
```
창 안: 슬라이더 Zoom / Local gain / Orientation, `S` 원본 PNG 저장, `Q`/`Esc` 종료.
상단이 초록(STREAM ACTIVE)이면 스트리밍, 파랑이면 프로브 스캔 대기.

### 2b. 하드웨어 없이 (mock — 합성 영상)
```bash
# 터미널 A: 가짜 스캐너
cd ~/FR5-for-RUS/fr5_contorl/fr5_vision && python3 -m fr5_vision.us_mock_scanner --bind 127.0.0.1
# 터미널 B: 뷰어를 127.0.0.1 로
export DISPLAY=:1 XAUTHORITY=/run/user/1001/gdm/Xauthority
cd ~/FR5-for-RUS/tracer/ubuntu_22_04 && python3 -m receiver.direct_tcp_receiver --host 127.0.0.1
```

---

## 3. 초음파를 ROS 토픽으로 (`/us/image`)

```bash
source install/setup.bash
ros2 run fr5_vision us_frame --ros-args \
  --params-file fr5_contorl/fr5_control/config/probe.yaml     # us.host 기본 192.168.1.1
# mock 상대로: 뒤에  -p us.host:=127.0.0.1
ros2 topic hz /us/image                                        # ~8 Hz, mono8 256x256
ros2 run rqt_image_view rqt_image_view /us/image              # 영상 확인 (별도 창)
```
표시 방향 확정 시: `probe.yaml` 의 `us.orientation` (none/rot90_cw/rot90_ccw/rot180).

---

## 4. IMU 만 보기

```bash
cd ~/FR5-for-RUS/imu_bench
sg dialout -c "python3 host/imu_gui.py --log"     # 6패널 GUI. R 로깅 / Z 영점 / C 정렬 / Q 종료
sg dialout -c "python3 host/run.sh --calibrate-mag 20"   # 자력계 보정 (8자 모션)
python3 host/analyze_log.py logs/<파일>.csv       # 로그 시간축·안정성·Allan 감사
```

---

## 5. 초음파 + IMU 동기 수집 (같은 시간축 = pc_unix)

두 소스를 호스트 `time.time()` 로 찍어 `pc_unix` 로 조인한다. US 프레임당 IMU 가 1~2 ms 안에 있다.

### 5a. 통합 뷰어 — 보면서 키 하나로 동기 녹화  ← 평소 쓰는 것
```bash
export DISPLAY=:1 XAUTHORITY=/run/user/1001/gdm/Xauthority
cd ~/FR5-for-RUS/imu_bench
sg dialout -c "python3 host/us_imu_gui.py --host 192.168.1.1"
#   방향 바꿔 표시:  ... --host 192.168.1.1 --orientation rot90_cw
#   6축(자력계 미사용): ... --no-mag
```
한 창: **왼쪽 초음파 · 오른쪽 IMU 6패널**. 키:

| 키 | 동작 |
|---|---|
| **Z** | 영점 — IMU 를 현재 자세 기준으로 (몇 초 정지) |
| **R** | 녹화 시작/정지 — **US·IMU 를 한 세션 폴더에 동시에** |
| **C** | 자세 재설정 (호스트 퓨전 프레임 정렬) |
| **Q** | 종료 |

권장 순서: 프로브 스캔 시작 → 창에 영상 확인 → **Z**(영점) → **R**(녹화) … → **R**(정지).

### 5b. 헤드리스 수집 (GUI 없이, 장시간)
```bash
cd ~/FR5-for-RUS/imu_bench
sg dialout -c "python3 host/us_imu_collect.py --host 192.168.1.1"          # Ctrl+C 로 종료
sg dialout -c "python3 host/us_imu_collect.py --host 192.168.1.1 --duration 60 --show"
```

### 5c. 산출물 (`imu_bench/logs/us_imu_<시각>/`)
| 파일 | 내용 |
|---|---|
| `imu_<시각>.csv` | IMU 전 레이트 (~250 Hz). `analyze_log.py` 호환 |
| `us_frames.bin` | 256×256 uint8 candidate 원시 스트림 (무변환) |
| `us_index.csv` | `pc_unix, us_seq, frame_id, byte_offset` |
| `session.meta.json` | 두 로그를 묶는 메타 (조인 키 = `pc_unix`) |

되읽기:
```python
import numpy as np, csv, json
frames = np.fromfile("us_frames.bin", dtype=np.uint8).reshape(-1, 256, 256)  # us_seq 행 ↔ frames[us_seq]
# us_index.csv 의 pc_unix 로 imu_*.csv 의 pc_unix 와 최근접 조인
```

> **한계(정직):** pc_unix 는 **수신 시각** 정렬이다. IMU=USB(µs급), US=Wi-Fi TCP + 프로브 내부
> 획득 지연. 두 클럭 skew 는 `dev_us` 로 사후 보정 가능하나, **US 고정 엔드투엔드 지연은 미측정**
> (DESIGN_NOTES: US 프레임그래버 지연 ⏳). US 프레임은 방향·scan conversion 검증 전까지 candidate.

---

## 6. Teleop — 로봇 제어

Phase 0 골격 = **us_servo(서보) + us_diff_ik(미분 IK)** [+ teleop]. 모두 `probe.yaml` 을 읽는다.

### 6a. mock 로봇 (하드웨어 없음)
```bash
source install/setup.bash
ros2 launch fr5_launch us_phase0.launch.py                       # 서보 + IK, mock
```

### 6b. 실로봇
```bash
ros2 launch fr5_launch us_phase0.launch.py backend:=fairino      # 실로봇 (192.168.58.3)
```

### 6c. Touch(햅틱) 원격조작 함께 띄우기
```bash
ros2 launch fr5_launch us_phase0.launch.py backend:=fairino teleop:=true
#   ↑ freespace 스케일이 **기본**이다 (2026-08-25~). 초기 자세 접근이 주 용도라서다.
ros2 launch fr5_launch us_phase0.launch.py backend:=fairino teleop:=true freespace:=false
#   ↑ 접촉용 상한 (10 mm/s / 0.2 rad/s / 0.5 rad/s, teleop 프로파일 us_approach)
```
- Touch 노드: `touch_teleop touch_twist`. **버튼 1(회색) = 데드맨** — 누르는 동안만 동작, 놓으면 정지.
- 발행 토픽: `/fr5_right/desired_twist` (Twist), `/fr5_right/desired_gripper_pose`.
- 프로파일을 조작 중에 바꾸기 (매 주기 반영):
  ```bash
  ros2 param set /touch_teleop_node teleop.profile us_approach   # 정밀 접근 (probe.yaml 기본값)
  ros2 param set /touch_teleop_node teleop.profile freespace     # 빠름 (launch 기본값)
  ros2 param set /touch_teleop_node teleop.profile laparoscopic  # 복강경 잔재
  ```
  프로파일을 **내리는** 방향(freespace → us_approach)은 조작 중에도 바로 먹는다.
  올리는 방향은 `freespace:=false` 로 띄운 경우 클램프(10 mm/s)에 걸려 체감이 안 바뀐다.

> ⚠️ 기본값이 `freespace:=true` 다 — `probe.yaml` 의 **접촉용 속도 상한을 덮어써
> 15 배 빠르게** 돈다 (10 → 150 mm/s). **프로브가 조직/팬텀에 닿는 단계에서는
> `freespace:=false` 를 명시할 것.** 잊으면 접촉용 한계가 걸리지 않는다.
>
> Phase 0 에는 힘 되먹임이 없다 (admittance·supervisor·힘 배리어 모두 미구현,
> PX6D 도 이 launch 에 없다). 접촉 중 힘을 지키는 것은 조작자와 이 클램프뿐이다.

### 6d. 키보드 teleop (별도, 레거시 pose 경로)
```bash
ros2 run keyboard_teleop keyboard        # /fr5_right/desired_pose_wrt_rcm 발행
```

### 6e. 상태 확인
```bash
ros2 node list
ros2 topic echo /fr5_right/desired_twist          # teleop 출력 확인
ros2 param get /touch_teleop_node teleop.profile
```

---

## 7. 정리 / 종료

```bash
# 프로브에 붙은 US 프로세스 정리 (뷰어/수집기/노드)
pkill -f "[d]irect_tcp_receiver"; pkill -f "[u]s_mock_scanner"; pkill -f "us_imu_"
# Wi-Fi 프로브 연결 해제 (인터넷은 meta5G 유지)
sudo nmcli device disconnect wlx705dccf63db8
```

---

## 부록 A. Wi-Fi 동글 복구 (드라이버/모니터 흔적 원복)

동글이 안 보이거나 monitor 모드로 남았을 때:
```bash
sudo ip link set wlx705dccf63db8 down
sudo iw dev wlx705dccf63db8 set type managed
sudo ip link set wlx705dccf63db8 up
sudo nmcli device set wlx705dccf63db8 managed yes
sudo nmcli connection up "SL-2C GMCEKC017"
```
드라이버 재적재(모듈 옵션 변경 후):
```bash
sudo modprobe -r 8814au && sudo modprobe 8814au        # RTL8814AU DKMS (country_code=KR 고정)
```

## 부록 B. 파일 위치

| 무엇 | 경로 |
|---|---|
| US 프로토콜(벤더링) | `fr5_contorl/fr5_vision/fr5_vision/us_protocol.py` |
| US ROS 노드 | `fr5_contorl/fr5_vision/fr5_vision/us_frame_node.py` |
| US mock 스캐너 | `fr5_contorl/fr5_vision/fr5_vision/us_mock_scanner.py` |
| 통합 뷰어(US+IMU) | `imu_bench/host/us_imu_gui.py` |
| 헤드리스 수집기 | `imu_bench/host/us_imu_collect.py` |
| IMU 단독 GUI | `imu_bench/host/imu_gui.py` |
| 설정 단일 소스 | `fr5_contorl/fr5_control/config/probe.yaml` |
| Phase 0 런치 | `fr5_contorl/fr5_launch/launch/us_phase0.launch.py` |
| 프로브 Wi-Fi 설정 | `tracer/ubuntu_22_04/config/scanner.env` |
