#!/usr/bin/env python3
"""초음파 + BNO085 IMU 통합 뷰어 — 한 창에서 보고, 키 하나로 동기 녹화.

왼쪽에 초음파 candidate 영상, 오른쪽에 IMU 퓨전 6패널을 한 matplotlib 창에 함께
띄운다. IMU 뷰어(`imu_gui`)와 초음파 뷰어(tracer `direct_display`)를 합친 것이다.

두 소스는 물리 경로가 다르다 (IMU=USB 시리얼, US=Wi-Fi TCP). 같은 시간축에 올리는
방법은 `us_imu_collect` 와 같다 — **둘 다 호스트 수신 순간의 `time.time()`(pc_unix)로
찍는다.** IMU 스트림이 이미 그 시계로 로그를 남기므로, US 프레임도 도착 순간 같은
시계로 찍으면 두 로그가 pc_unix 로 직접 조인된다.

키:
    R  녹화 시작/정지 — US 와 IMU 를 **한 세션 폴더에 동시에**. `--record-frames N` 이면 N 프레임에서 자동 정지,
       0(기본) 이면 R 을 다시 눌러 정지 — 길이는 자유다.
    M  (녹화 중) 발견 표시 — "지금 목표(방광)가 보인다". 시각·US 프레임 번호가 메타 events 에 남는다.
    X  (녹화 중) 폐기 — 저장하지 않고 폴더를 discard_ 로 바꾼다 (실패한 에피소드). 에피소드 번호는 올라가지 않는다.
    F  스캔 시작/정지 — 클라이언트가 스캔을 명령하는 프로브(`--probe c10ur`)에서만. SL-2C 는 프로브 버튼.
    K  IMU 보정 안내 모드 — BNO085 의 칩 보정 상태(가속도/자이로/자력계/회전벡터 0~3) 를 보며 단계별 동작을 안내한다.
       전원 인가 뒤: K(자이로 정지 → 가속도 6방향 → 자력계 8자 워밍업, rv 3 확인) → S → Z → R.
    S  DCD 저장 — 칩의 동적 보정 데이터를 BNO085 플래시에 쓴다 (2026-09-10 펌웨어). 전원을 껐다 켜도 보정이 살아 있어
       첫 세션을 버리지 않는다. rv 3 이 아니면 미완 상태를 굳히지 않도록 보류한다. 응답(DCD,OK/FAIL)이 체크리스트에 뜬다.
       R 은 조건이 안 돼도 바로 시작한다 — 문제는 경고로 보여 주고 메타 imu_calibration.warnings / cal_override 에 남긴다.
    Z  영점(zero) — IMU 를 현재 자세 기준으로 잡는다 (몇 초간 정지). 정지 판정에 미달해도 채택한다 (quality.forced=True,
       still=False 로 메타에 남음). 표본이 아예 없을 때만 실패.
    C  자세 재설정 — 호스트 퓨전 프레임 정렬을 다시 잡는다.
    Q  종료.

에피소드 모드 (`--task find_bladder --episodes 100`, 2026-09-10): 세션 하나 = 에피소드 하나, 길이 자유.
    프로브를 방광이 안 보이는 곳에 대고 1 s 정지 → R → 정지-이동-정지로 탐색 (정지 ≥ 0.5 s) → 방광이 보이면 M →
    1~2 s 정지 → R (저장). 못 찾았거나 잘못 눌렀으면 X. 메타에 task / episode{index, outcome, found_us_seq} / events 가
    남고 번호는 out-dir 의 같은 task 세션 수에서 이어진다. 마지막 정지가 라벨 파이프라인의 "정지 B" 가 된다.

프로브 (`--probe`, 2026-09-09): sl2c (기본, 256×256) | c10ur (Konted, Wi-Fi AP "US-1C …", 160 라인 × 512 깊이 표본의
극좌표 candidate, 10 fps, 시작/정지 명령 있음). 상세는 fr5_vision/us_protocol.py 의 ProbeProfile.
IMU 포트를 비우면 VID 2886 의 포트를 자동으로 잡는다 (윈도우·리눅스 공통).

    sg dialout -c "python3 host/us_imu_gui.py --host 192.168.1.1"
    sg dialout -c "python3 host/us_imu_gui.py --host 192.168.1.1 --orientation rot90_cw"

US 프레임은 방향/scan conversion 이 검증되기 전까지 candidate 다. 화면의 회전은
표시용이며, 저장은 항상 원시 256x256 바이트를 무변환으로 한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
# 패키지 디렉터리 이름이 두 철자로 존재했다 (fr5_control / fr5_contorl). 있는 쪽을 쓴다.
_FR5_VISION = next((d for d in (
    os.path.abspath(os.path.join(_HERE, "..", "..", "fr5_control", "fr5_vision")),
    os.path.abspath(os.path.join(_HERE, "..", "..", "fr5_contorl", "fr5_vision")),
) if os.path.isdir(d)), os.path.abspath(os.path.join(_HERE, "..", "..", "fr5_control", "fr5_vision")))
for _p in (_HERE, _FR5_VISION):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# imu_gui 를 import 하면 DISPLAY 자동탐지 + matplotlib.use("TkAgg") 가 실행된다.
from imu_gui import OrientationView, TimeSeries, use_korean_font, AXIS_COLORS  # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.animation import FuncAnimation        # noqa: E402
import numpy as np                                    # noqa: E402

import mag_calib                                       # noqa: E402
from imu_stream import ImuStream                       # noqa: E402
from imu_log import SessionLogger                      # noqa: E402
from imu_fusion_view import run_zero_calibration       # noqa: E402
from umi_protocol import REC_ACCEL, REC_GYRO, REC_MAG, REC_RV  # noqa: E402
from fr5_vision.us_protocol import CANDIDATE_FRAME_SHAPE, UsScannerSession, PROFILES, SL2C  # noqa: E402
from us_scan_convert import FanGeometry, ScanConverter                  # noqa: E402
from us_imu_sync import SyncWorker                                      # noqa: E402

# np.rot90 k 값 — tracer 뷰어의 Orientation 과 같은 네 가지 (표시용).
ORIENTATIONS = {"none": 0, "rot90_ccw": 1, "rot180": 2, "rot90_cw": -1}

# BNO085 보정 상태 (SH-2 accuracy): 0 unreliable, 1 low, 2 medium, 3 high.  세션 시작 기준 🟡
# 2026-09-10: 회전벡터(rv) 3 을 시작 조건에 넣었다 — rv 는 자이로·가속도·자력계 보정을 합친 값이라, 자력계 워밍업(8 자)
# 을 빼먹으면 3 이 안 된다. 첫날 15 세션이 전부 cal_gyr 0 / rv 1 로 찍힌 것(구 펌웨어 + 워밍업 생략)을 막기 위한 게이트.
CAL_MIN = {"gyr": 2, "acc": 2, "mag": 1, "rv": 3}
FW_TAG_EXPECTED = "2026-09-10-dcd"       # firmware/umi_device_hardware FW_TAG — 'V' 에 답하지 않으면 구 펌웨어


def calib_ready(cal: dict) -> bool:
    return all((cal.get(k) or 0) >= v for k, v in CAL_MIN.items() if k != "mag")


def _display_units(s: str) -> int:
    """한글·CJK 는 2, 나머지는 1 — 패널 폭 계산용 (글꼴 폭의 대략값)."""
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def wrap_panel_lines(lines, max_units: int = 84):
    """안내 패널 폭을 넘는 줄을 공백에서 접는다 (이어지는 줄은 두 칸 들여쓰기). 영상 옆 IMU 그래프를 덮지 않기 위해."""
    out = []
    for line in lines:
        if _display_units(line) <= max_units:
            out.append(line)
            continue
        words, cur = line.split(" "), ""
        for w in words:
            cand = (cur + " " + w) if cur else w
            if cur and _display_units(cand) > max_units:
                out.append(cur)
                cur = "  " + w
            else:
                cur = cand
        if cur:
            out.append(cur)
    return out


def calib_step(cal: dict, dcd_saved: bool = False) -> str:
    """현재 칩 보정 상태에서 다음에 할 동작 (BNO085 동적 보정 절차)."""
    g, a, m, rv = (cal.get("gyr") or 0), (cal.get("acc") or 0), (cal.get("mag") or 0), (cal.get("rv") or 0)
    if g < CAL_MIN["gyr"]:
        return "1) 자이로 %d/3 — 프로브를 탁자에 내려놓고 5 초 이상 완전히 정지" % g
    if a < 3:
        return "2) 가속도 %d/3 — 프로브를 천천히 6 방향(위·아래·좌·우·앞·뒤)으로 돌려 각 자세에서 2 초 정지" % a
    if m < CAL_MIN["mag"] or rv < CAL_MIN["rv"]:
        return "3) 자력계 워밍업 — 8 자로 천천히 20~30 초 (m %d/3, rv %d/3 → rv 3 이 될 때까지)" % (m, rv)
    if not dcd_saved:
        return "4) 보정 완료 (a%d g%d m%d rv%d) — S 로 DCD 를 칩 플래시에 저장 (전원 재인가 후에도 유지) → Z" % (a, g, m, rv)
    return "보정 완료·DCD 저장됨 (a%d g%d m%d rv%d) — Z 로 영점을 잡으십시오" % (a, g, m, rv)


class UsReceiver(threading.Thread):
    """US 세션을 백그라운드로 물고, 최신 프레임을 보관하며, 녹화 중이면 파일에 쓴다.

    프로브는 TCP 클라이언트를 하나만 받으므로 이 스레드가 유일한 US 클라이언트다.
    """

    def __init__(self, host, profile=SL2C):
        super().__init__(daemon=True)
        self.host = host
        self.profile = profile
        self._session = UsScannerSession(host, profile=profile)
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()   # _stop 은 Thread._stop() 을 가려 join() 이 깨진다

        self.latest = None            # 최신 256x256 uint8 (표시용, 항상 갱신)
        self.frame_count = 0          # 세션 시작 이후 총 프레임
        self.scanner_active = False
        self.connected = False
        self.last_frame_wall = 0.0
        self._fps_stamps = []

        # 녹화 상태 (R 로 토글). 녹화는 pc_unix 로 US 를 IMU 와 같은 축에 남긴다.
        self._recording = False
        self._bin = None
        self._index = None
        self._index_writer = None
        self._byte_offset = 0
        self.rec_count = 0
        self.want_scan = False            # 사용자가 원하는 스캔 상태 — 재접속 뒤에도 복원한다
        self._scan_sent_at = 0.0

    # -- 녹화 제어 (GUI 스레드에서 호출) --
    def start_recording(self, session_dir):
        with self._lock:
            if self._recording:
                return
            self._bin = open(os.path.join(session_dir, "us_frames.bin"), "wb")
            self._index = open(os.path.join(session_dir, "us_index.csv"), "w", newline="")
            self._index_writer = csv.writer(self._index)
            self._index_writer.writerow(["pc_unix", "us_seq", "frame_id", "byte_offset"])
            self._byte_offset = 0
            self.rec_count = 0
            self._recording = True

    def stop_recording(self):
        with self._lock:
            if not self._recording:
                return 0
            n = self.rec_count
            self._recording = False
            for fh in (self._bin, self._index):
                if fh is not None:
                    fh.close()
            self._bin = self._index = self._index_writer = None
            return n

    @property
    def recording(self):
        return self._recording

    # -- 스캔 시작/정지 (프로브가 명령을 받는 경우만; 세션 스레드가 다음 keepalive 에 반영) --
    def toggle_scan(self):
        if not self._session.can_command_scan:
            return None
        if self.want_scan and (self._session.scanner_active or not self._session.is_open):
            self.want_scan = False
            if self._session.is_open:
                self._session.stop_scan()
            return False
        self.want_scan = True
        if self._session.is_open:
            self._session.start_scan(); self._scan_sent_at = time.monotonic()
        return True

    def snapshot(self):
        with self._lock:
            img = None if self.latest is None else self.latest
            fps = self._fps()
            return {
                "img": img, "fps": fps, "active": self.scanner_active,
                "connected": self.connected, "frames": self.frame_count,
                "recording": self._recording, "rec_count": self.rec_count,
                "stale": (time.time() - self.last_frame_wall) > 2.0 if self.last_frame_wall else True,
                "can_command": self._session.can_command_scan, "probe": self.profile.name,
            }

    def _fps(self):
        now = time.monotonic()
        self._fps_stamps = [t for t in self._fps_stamps if now - t < 2.0]
        return len(self._fps_stamps) / 2.0

    def stop(self):
        self._stop_evt.set()

    def run(self):
        last_connect = 0.0
        while not self._stop_evt.is_set():
            if not self._session.is_open:
                if time.monotonic() - last_connect < 2.0:
                    self._stop_evt.wait(0.05)
                    continue
                last_connect = time.monotonic()
                try:
                    self._session.open()
                    with self._lock:
                        self.connected = True
                    self._scan_sent_at = 0.0
                    print("US: 프로브 TCP 연결됨 (%s)%s" % (self.host, "  — 스캔 재요청 예정" if self.want_scan else ""), flush=True)
                except OSError as exc:
                    with self._lock:
                        if self.connected:
                            print("US: 연결 실패 %s: %s" % (self.host, exc), flush=True)
                        self.connected = False
                    continue

            # 재접속(프로브 자동 정지·AP 재기동) 뒤에도 원하는 스캔 상태를 복원한다: 세션이 열린 지 1.6 s 뒤 시작 요청
            if (self.want_scan and self._session.can_command_scan and not self._session.scanner_active
                    and self._scan_sent_at == 0.0 and time.monotonic() - self._session._started_at >= 1.6):
                self._session.start_scan(); self._scan_sent_at = time.monotonic()
                print("US: 스캔 시작 요청 전송", flush=True)
            was_active = self._session.scanner_active
            try:
                frames = self._session.poll(0.05)
            except (ConnectionError, OSError) as exc:
                print("US: 프로브가 연결을 끊음 (%s) — 재접속 시도" % exc, flush=True)
                self._session.close()
                with self._lock:
                    self.connected = False
                continue
            if self._session.scanner_active and not was_active:
                print("US: 스캔 활성 (프로브 응답)", flush=True)

            for frame in frames:
                pc_unix = time.time()
                with self._lock:
                    self.latest = frame.as_uint8_image()
                    self.frame_count += 1
                    self.scanner_active = self._session.scanner_active
                    self.last_frame_wall = pc_unix
                    self._fps_stamps.append(time.monotonic())
                    if self._recording and self._bin is not None:
                        raw = frame.raw_pixels
                        self._bin.write(raw)
                        self._index_writer.writerow(
                            [f"{pc_unix:.6f}", self.rec_count, frame.frame_id, self._byte_offset])
                        self._byte_offset += len(raw)
                        self.rec_count += 1
            with self._lock:
                self.scanner_active = self._session.scanner_active

        self._session.close()
        self.stop_recording()


class CombinedGui:
    def __init__(self, stream, us, args):
        self.stream = stream
        self.us = us
        self.args = args
        self.t0 = None
        self.rotate_k = ORIENTATIONS[args.orientation]
        self.session_dir = None
        self.frame_shape = tuple(getattr(us, "profile", SL2C).frame_shape)
        self.record_frames = int(getattr(args, "record_frames", 0) or 0)
        self.session_count = 0
        # 에피소드 모드 (--task/--episodes): 세션 = 에피소드 하나 (길이 자유, R 로 수동 정지). 번호는 out_dir 에 이미
        # 저장된 같은 task 의 세션 수에서 이어진다 — GUI 를 껐다 켜도 "몇 번째" 가 유지된다. 폐기(X) 는 세지 않는다.
        self.task = getattr(args, "task", None) or None
        self.episodes_target = int(getattr(args, "episodes", 0) or 0)
        self.episode_done = count_task_sessions(args.out_dir, self.task) if self.task else 0
        self.events = []                  # 녹화 중 M 으로 남기는 표시 [{name, t_pc, us_seq}] — 세션 메타에 들어간다
        self._stop_reason = "manual"
        self.event_msg = ""
        self.event_msg_until = 0.0
        # 수집 계획 (--plan JSON): 움직임 종류별 목표 수·규약·bin 순서. 패널이 "지금 무엇을, 몇 번째로" 를 보여 주고 저장된
        # 세션 메타(episode.motion) 를 세어 GUI 를 껐다 켜도 진행 수가 이어진다. N 다음 종류, 숫자 키로 종류 선택.
        self.plan = load_plan(getattr(args, "plan", None)) if getattr(args, "plan", None) else None
        self.plan_counts = {}
        self.plan_idx = None
        self.plan_manual = False          # 사용자가 숫자/N 으로 직접 고른 상태 — 목표를 채워도 자동으로 넘기지 않는다
        if self.plan:
            self.plan_counts = count_plan_sessions(args.out_dir, self.task, [it["key"] for it in self.plan["items"]])
            self.plan_idx = self._next_plan_item()
            if not self.episodes_target:
                # 전체 목표 = 계획 밖에서 이미 찍은 에피소드 (1차 100 개 등) + 계획 합계. 재시작해도 같은 값이 나온다.
                base = self.episode_done - sum(self.plan_counts.values())
                self.episodes_target = base + sum(int(it["target"]) for it in self.plan["items"])
        # 저장 완료 세션을 IMU·US 동기화 (sync.npz / sync_report.json) — 백그라운드 루프
        self.sync = SyncWorker(args.out_dir, period_s=5.0) if not getattr(args, "no_sync", False) else None
        if self.sync is not None:
            self.sync.start()
        # 표시용 scan conversion (극좌표 candidate → 부채꼴). 저장은 항상 원본이다.
        self.converter = None
        if getattr(args, "display", "polar") == "fan":
            geo = FanGeometry(radius_mm=args.fan_radius, half_angle_deg=args.fan_angle, depth_mm=args.fan_depth,
                              flip_lines=bool(getattr(args, "fan_flip", False)))
            self.converter = ScanConverter(geo, self.frame_shape[0], self.frame_shape[1], out_h=512)
        self.display_shape = (self.converter.out_h, self.converter.out_w) if self.converter else self.frame_shape

        # matplotlib 기본 단축키가 우리 키와 겹친다 — 특히 's' 는 "그림 저장" 대화상자를 띄워 GUI 를 막는다.
        # (f 전체화면, k 로그 x축, r 홈, c 뒤로, g 격자, l 로그 y축, o 줌, p 팬 도 마찬가지.)
        for km in ("save", "fullscreen", "xscale", "yscale", "home", "back", "forward", "grid", "grid_minor",
                   "zoom", "pan"):
            key = "keymap." + km
            if key in plt.rcParams:
                plt.rcParams[key] = []

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("US + IMU 통합 뷰어")
        gs = self.fig.add_gridspec(3, 3, width_ratios=(1.5, 1.0, 1.0),
                                   hspace=0.45, wspace=0.28,
                                   left=0.03, right=0.98, top=0.90, bottom=0.07)

        # 왼쪽: 초음파 영상 (위) + 안내 패널 (아래). 체크리스트·보정 안내·메시지는 전부 아래 패널에 그린다 —
        # 영상 위에 겹치면 방광을 찾는 동안 화면을 가린다 (2026-09-10).
        left = gs[:, 0].subgridspec(2, 1, height_ratios=(2.3, 1.0), hspace=0.05)
        self.us_ax = self.fig.add_subplot(left[0])
        self.us_ax.set_title("초음파 candidate [%s %s%s] — 표시 방향 %s"
                             % (getattr(us, "profile", SL2C).name, "x".join(map(str, self.frame_shape)),
                                " → 부채꼴 R%.0f/%.0f°/%.0fmm" % (args.fan_radius, args.fan_angle, args.fan_depth)
                                if self.converter else "", args.orientation),
                             fontsize=10)
        self.us_ax.set_xticks([]); self.us_ax.set_yticks([])
        blank = np.zeros(self.display_shape, dtype=np.uint8)
        # animated=True 를 주면 안 된다: matplotlib >= 3.8 은 blit 없는 애니메이션에서 animated 아티스트를
        # 일반 draw 에서 건너뛰어 (구버전의 AxesImage 예외가 사라짐) 영상 패널이 영영 흰색으로 남는다.
        self.us_im = self.us_ax.imshow(blank, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        self.us_text = self.us_ax.text(
            0.5, 0.5, "US 프레임 대기...", color="#8ab4ff", ha="center", va="center",
            transform=self.us_ax.transAxes, fontsize=12)
        # 세션 시작 체크리스트 (펌웨어 → 보정 → DCD → 영점 → 녹화) 와 안내·메시지 — 영상 아래 전용 패널
        self.panel_ax = self.fig.add_subplot(left[1])
        self.panel_ax.set_axis_off()
        self.checklist = self.panel_ax.text(
            0.0, 1.0, "", color="#ffd27a", ha="left", va="top", transform=self.panel_ax.transAxes, fontsize=9,
            linespacing=1.22, bbox=dict(boxstyle="round,pad=0.4", fc="#141414", ec="#ffd27a", alpha=0.95))
        self.calib_mode = False
        self.calib_started = None
        self.zero_msg = ""
        self.zero_msg_until = 0.0

        # 오른쪽: IMU 6패널 (2열 x 3행)
        self.orient = OrientationView(self.fig.add_subplot(gs[0, 1], projection="3d"))
        self.euler = TimeSeries(
            self.fig.add_subplot(gs[0, 2]),
            ["hr", "hp", "hy", "cr", "cp", "cy"],
            "오일러각 — 실선=호스트, 점선=칩", "deg",
            colors=AXIS_COLORS * 2, labels=("roll", "pitch", "yaw") * 2,
            styles=["-", "-", "-", "--", "--", "--"], ylim=(-185, 185))
        self.euler.ax.set_yticks([-180, -90, 0, 90, 180])
        self.accel = TimeSeries(self.fig.add_subplot(gs[1, 1]),
                                ["ax", "ay", "az"], "가속도", "m/s²", min_span=2.0)
        self.gyro = TimeSeries(self.fig.add_subplot(gs[1, 2]),
                               ["gx", "gy", "gz"], "자이로", "rad/s", min_span=0.2)
        self.mag = TimeSeries(self.fig.add_subplot(gs[2, 1]),
                              ["mx", "my", "mz"], "자력계", "µT", min_span=10.0)
        self.diff = TimeSeries(self.fig.add_subplot(gs[2, 2]), ["diff"],
                               "호스트 vs 칩 각도차", "deg",
                               colors=("#8a5cf6",), labels=("Δ",), min_span=2.0)

        self.status = self.fig.text(0.03, 0.965, "", fontsize=9, va="top")
        # 키 안내는 아래 여백에 — 위쪽 상태줄과 겹치지 않게 (2026-09-10)
        self.fig.text(0.98, 0.012,
                      "[K] 보정 안내  [S] DCD 저장  [Z] 영점  [R] 녹화 시작/정지  [M] 발견 표시  [X] 폐기  [F] 스캔  [C] 자세 재설정  [Q] 종료",
                      fontsize=9, ha="right", va="bottom", alpha=0.75)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    # -------------------------------------------------------------- 입력
    def on_key(self, event):
        k = (event.key or "").lower()
        if k == "q":
            plt.close(self.fig)
        elif k == "c":
            self.stream.reset_align()
        elif k == "z":
            self.zero_now()
        elif k == "k":
            self.calib_mode = not self.calib_mode
            self.calib_started = time.time() if self.calib_mode else None
            print("IMU 보정 안내 %s" % ("시작 — 안내를 따르십시오" if self.calib_mode else "종료"))
        elif k == "r":
            self._stop_reason = "manual"
            self.toggle_record()
        elif k == "s":
            self.save_dcd()
        elif k == "n":
            self.plan_next()
        elif k.isdigit() and k != "0":
            self.plan_select(int(k) - 1)
        elif k == "m":
            self.mark_event("found")
        elif k == "x":
            self.discard_recording()
        elif k == "f":
            res = self.us.toggle_scan() if hasattr(self.us, "toggle_scan") else None
            self.status.set_text("스캔 %s" % ("시작 요청" if res else ("정지" if res is False else "명령 불가 (프로브 버튼 사용)")))

    def _recording(self):
        return self.stream.logger is not None or self.us.recording

    # -------------------------------------------------------------- 수집 계획
    def _next_plan_item(self):
        """아직 목표를 못 채운 첫 종류의 인덱스 (계획 순서대로 블록 진행). 전부 채웠으면 None."""
        for i, it in enumerate(self.plan["items"]):
            if self.plan_counts.get(it["key"], 0) < it["target"]:
                return i
        return None

    def plan_current(self):
        if not self.plan or self.plan_idx is None:
            return None
        it = self.plan["items"][self.plan_idx]
        n = self.plan_counts.get(it["key"], 0)
        hint = it["bins"][n % len(it["bins"])] if it["bins"] else ""
        return {"i": self.plan_idx, "item": it, "count": n, "hint": hint}

    def plan_select(self, i):
        if not self.plan or not (0 <= i < len(self.plan["items"])):
            return
        self.plan_idx, self.plan_manual = i, True
        it = self.plan["items"][i]
        self.event_msg = "종류 선택: [%d] %s (%d/%d)" % (i + 1, it["label"], self.plan_counts.get(it["key"], 0), it["target"])
        self.event_msg_until = time.time() + 4.0
        print(self.event_msg)

    def plan_next(self):
        if not self.plan:
            return
        n = len(self.plan["items"])
        self.plan_select(((self.plan_idx if self.plan_idx is not None else -1) + 1) % n)

    def _plan_after_save(self, key):
        """저장 뒤: 종류 수 +1, 목표를 채웠고 사용자가 직접 고른 상태가 아니면 다음 미완 종류로."""
        if not self.plan or key is None:
            return
        self.plan_counts[key] = self.plan_counts.get(key, 0) + 1
        it = self.plan["items"][self.plan_idx] if self.plan_idx is not None else None
        if it is not None and self.plan_counts[key] >= it["target"]:
            self.plan_manual = False
            self.plan_idx = self._next_plan_item()
            print("종류 '%s' 목표 달성 → %s" % (it["label"], "다음: " + self.plan["items"][self.plan_idx]["label"]
                                             if self.plan_idx is not None else "계획 완료"))

    def save_dcd(self):
        """S: BNO085 의 동적 보정 데이터(DCD) 를 칩 플래시에 저장 — 전원을 껐다 켜도 보정이 유지된다.
        보정이 다 되지 않은 상태로 저장하면 나쁜 값이 고정되므로 rv 3 이 아닐 때는 막는다."""
        s = self.stream.snapshot()
        cal = s.get("cal_status") or {}
        if s.get("fw_tag") is None:
            self.event_msg = "S 불가 — 구 펌웨어 ('V' 응답 없음). host/flash_win.py 로 먼저 플래시"
        elif (cal.get("rv") or 0) < CAL_MIN["rv"]:
            self.event_msg = "S 보류 — rv %s/3. 8 자 워밍업으로 rv 3 을 만든 뒤 저장 (미완 상태를 굳히지 않기 위해)" % cal.get("rv")
        elif not self.stream.send(b"S"):
            self.event_msg = "S 실패 — IMU 포트가 열려 있지 않음"
        else:
            self._dcd_req_at = time.time()
            self.event_msg = "DCD 저장 요청 보냄 — 칩 응답 대기"
        self.event_msg_until = time.time() + 6.0
        print(self.event_msg)

    def mark_event(self, name):
        """녹화 중 M: '지금 목표(방광)를 찾았다' 표시. 시각(pc_unix)과 US 프레임 번호를 세션 메타 events 에 남긴다.
        표시 뒤 1–2 s 정지하고 R 로 정지하면 마지막 정지 구간이 라벨 파이프라인의 '정지 B' 가 된다."""
        if not self._recording():
            self.event_msg, self.event_msg_until = "녹화 중이 아닙니다 — M 은 녹화 중에만", time.time() + 4.0
            return
        ev = {"name": name, "t_pc": time.time(), "us_seq": int(self.us.snapshot().get("rec_count", 0))}
        self.events.append(ev)
        self.event_msg = "발견 표시 #%d (US seq %d) — 1~2 s 정지 후 R 로 저장" % (len(self.events), ev["us_seq"])
        self.event_msg_until = time.time() + 8.0
        print(self.event_msg)

    def discard_recording(self):
        """녹화 중 X: 이 세션을 저장하지 않고 버린다 (실패한 에피소드). 메타를 쓰지 않고 폴더를 discard_ 로 바꿔
        동기화 루프(us_imu_* 만 본다)와 리포(.gitignore) 둘 다에서 빠진다. 에피소드 번호는 올라가지 않는다."""
        if not self._recording():
            self.event_msg, self.event_msg_until = "녹화 중이 아닙니다 — X 는 녹화 중에만", time.time() + 4.0
            return
        if self.stream.logger is not None:
            self.stream.logger.close()
            self.stream.logger = None
        us_n = self.us.stop_recording()
        d = self.session_dir
        self.session_dir = None
        self.session_count = max(0, self.session_count - 1)
        target = os.path.join(os.path.dirname(d), "discard_" + os.path.basename(d))
        try:
            os.rename(d, target)
        except OSError as e:                       # Windows 에서 핸들이 아직 잡혀 있으면 표시 파일로 대신한다
            with open(os.path.join(d, "DISCARDED"), "w", encoding="utf-8") as fh:
                fh.write("discarded %s (%s)\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), e))
            target = d
        self.event_msg = "폐기: %s (US %d frames) — 저장 안 함" % (os.path.basename(target), us_n)
        self.event_msg_until = time.time() + 6.0
        print(self.event_msg)

    def zero_now(self):
        self.status.set_text("영점 캘리브레이션 %.1f초 — 센서를 움직이지 마세요..." % self.args.zero)
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
        ref = run_zero_calibration(self.stream, self.args.zero, retries=0, profile=self.args.zero_profile)
        if ref is not None:
            self.zero_msg = "영점 OK (%s: gyro sd %.3f, acc sd %.2f)" % (
                self.args.zero_profile, ref.quality["gyro_sd"], ref.quality["accel_sd"])
        else:
            # 정지 판정에 실패해도 영점은 **채택한다** (2026-09-10): Z 를 눌렀다는 것은 지금 자세를 기준으로 삼겠다는 뜻이고,
            # 파이프라인이 zero_ref 에서 주로 쓰는 것은 쿼터니언 규약과 기준 자세다. 판정 수치는 quality 에 그대로 남겨
            # (still=False, forced=True) 나중에 걸러낼 수 있게 한다. 샘플 자체가 부족했을 때만 실패.
            last = getattr(run_zero_calibration, "last", None)
            if last is not None:
                last.quality["forced"] = True
                self.stream.zero = last
                q, th = last.quality, last.quality.get("thresholds", {})
                self.zero_msg = "영점 채택 (정지 판정 미달 — gyro sd %.3f/%.3f, mean %.3f/%.3f, acc sd %.2f/%.2f). 메타에 forced 로 남김" % (
                    q.get("gyro_sd", float("nan")), th.get("gyro_sd", float("nan")),
                    q.get("gyro_mean_abs", float("nan")), th.get("gyro_mean_abs", float("nan")),
                    q.get("accel_sd", float("nan")), th.get("accel_sd", float("nan")))
            else:
                self.zero_msg = "영점 실패 — IMU 표본이 없음 (포트·스트림 확인). Z 다시"
        self.zero_msg_until = time.time() + 8.0
        print(self.zero_msg)

    def toggle_record(self):
        # 스캔 명령이 가능한 프로브인데 아직 스캔 중이 아니면 녹화 시작과 함께 스캔을 시작한다
        starting = not (self.stream.logger is not None or self.us.recording)
        self._cal_override = False
        self._start_warnings = []
        if starting:
            # R 은 막지 않는다 (2026-09-10 저녁 — 사용자 결정: 조건 미달이어도 바로 시작). 대신 시작 시점의 문제를
            # 경고로 보여 주고 메타 imu_calibration.warnings / cal_override 에 남겨 나중에 걸러낼 수 있게 한다.
            snap = self.stream.snapshot()
            cal = snap.get("cal_status") or {}
            problems = []
            if snap.get("fw_tag") is None:
                problems.append("구 펌웨어 (자이로 보정 꺼짐 — flash_win.py)")
            if not calib_ready(cal):
                problems.append("보정 미완 a%s g%s m%s rv%s (rv 3 권장 — 8 자 워밍업)" % (
                    cal.get("acc"), cal.get("gyr"), cal.get("mag"), cal.get("rv")))
            z = self.stream.zero
            if not z:
                problems.append("영점 없음 (Z)")
            elif not z.quality.get("still"):
                problems.append("영점 정지 판정 미달 (forced)")
            if problems:
                self._cal_override = True
                self._start_warnings = problems
                self.event_msg = "⚠ 녹화 시작 (경고: " + " / ".join(problems) + ")"
                self.event_msg_until = time.time() + 6.0
                print(self.event_msg)
        if starting and hasattr(self.us, "toggle_scan") and self.us.snapshot().get("can_command") \
                and not self.us.snapshot().get("active"):
            self.us.toggle_scan()
        if self.stream.logger is not None or self.us.recording:
            # 정지 — 둘 다 닫고 세션 메타 기록
            imu_path = imu_n = None
            if self.stream.logger is not None:
                imu_path = self.stream.logger.close()
                imu_n = self.stream.logger.n
                self.stream.logger = None
            us_n = self.us.stop_recording()
            self._write_session_meta(imu_path, imu_n, us_n)
            ep = ""
            if self.task:
                self.episode_done += 1
                ep = "  [%s 에피소드 %d%s, %s]" % (self.task, self.episode_done,
                                                 "/%d" % self.episodes_target if self.episodes_target else "",
                                                 "발견 표시 있음" if self.events else "발견 표시 없음 (M 안 누름)")
            print("녹화 정지: %s  (US %s frames, IMU %s rows)%s — 저장 완료. 동기화는 백그라운드에서 몇 초 뒤. "
                  "R 로 다음 세션을 바로 시작할 수 있습니다." % (self.session_dir, us_n, imu_n, ep))
            self._plan_after_save((getattr(self, "_plan_at_start", None) or {}).get("key"))
            self.session_dir = None
            self.events = []
        else:
            # 시작 — 한 세션 폴더에 US 와 IMU 를 동시에
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self.session_dir = os.path.join(self.args.out_dir, "us_imu_" + stamp)
            os.makedirs(self.session_dir, exist_ok=True)
            self.events = []
            self._stop_reason = "manual"
            snap0 = self.stream.snapshot()
            self._start_state = {"cal_status_at_start": dict(snap0.get("cal_status") or {}),
                                 "cal_override": bool(self._cal_override),
                                 "warnings": list(getattr(self, "_start_warnings", []) or []),
                                 "firmware": snap0.get("fw_tag"),
                                 "dcd_saved_at": snap0.get("dcd_saved_at")}
            cur = self.plan_current()
            self._plan_at_start = ({"key": cur["item"]["key"], "label": cur["item"]["label"], "bin_hint": cur["hint"],
                                    "index_in_kind": cur["count"] + 1, "target": cur["item"]["target"],
                                    "plan_file": os.path.basename(self.plan["path"])} if cur else None)
            self.stream.logger = self._new_imu_logger()
            self.us.start_recording(self.session_dir)
            self.session_count += 1
            cal = snap0.get("cal_status") or {}
            warn = ""
            if not self.stream.zero:
                warn += "   ⚠ 영점(zero_ref) 없음 — Z 를 먼저"
            if not calib_ready(cal):
                warn += "   ⚠ IMU 보정 미완 (a%s g%s m%s) — K 로 보정" % (cal.get("acc"), cal.get("gyr"), cal.get("mag"))
            ep = "  (%s 에피소드 #%d%s — 방광 밖에서 시작, 찾으면 M, 1~2 s 정지 후 R)" % (
                self.task, self.episode_done + 1, "/%d" % self.episodes_target if self.episodes_target else "") \
                if self.task else ""
            print("녹화 시작 #%d: %s%s%s" % (self.session_count, self.session_dir, ep, warn))

    def _new_imu_logger(self):
        z = self.stream.zero
        return SessionLogger(
            self.session_dir, fmt="csv", prefix="imu",
            meta={"source": "BNO085 via XIAO nRF52840 Sense",
                  "port": self.args.port, "use_mag": not self.args.no_mag,
                  "zero_ref": z.to_dict() if z else None,
                  "cal_status_at_start": dict(self.stream.snapshot().get("cal_status") or {})})

    def _write_session_meta(self, imu_path, imu_n, us_n):
        if self.session_dir is None:
            return
        meta = {
            "created": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
            "time_axis": "pc_unix (host time.time() at reception, shared by both streams)",
            "join_key": "pc_unix",
            "caveat": ("수신 시각 정렬. US 고정 엔드투엔드 지연은 미측정."),
            "us": {"host": self.us.host, "frames": us_n,
                   "frame_shape": list(self.frame_shape), "dtype": "uint8",
                   "bin": "us_frames.bin", "index": "us_index.csv",
                   "display_orientation": self.args.orientation,
                   "probe": getattr(self.us, "profile", SL2C).name,
                   "probe_note": getattr(self.us, "profile", SL2C).note,
                   "fan_geometry": self.converter.meta() if self.converter else None,
                   "note": ("candidate — 방향/scan conversion 미검증, 원시 바이트 무변환 저장"
                            if self.frame_shape == tuple(CANDIDATE_FRAME_SHAPE) else
                            "candidate — 극좌표 (행 = A-line, 열 = 깊이 표본, 행 시작 = 근거리). scan conversion 전. 원시 바이트 무변환 저장")},
            "imu": {"port": self.args.port, "rows": imu_n,
                    "csv": None if imu_path is None else os.path.basename(imu_path)},
            "record_stop": self._stop_reason if not self.record_frames else
                           ("auto %d frames" % self.record_frames if self._stop_reason == "auto" else "manual"),
            "events": list(self.events),
            # 시작 시점의 칩 보정 상태·펌웨어·DCD 저장 여부 — 세션을 나중에 걸러낼 때의 근거 (2026-09-10)
            "imu_calibration": dict(getattr(self, "_start_state", {}) or {}),
        }
        if self.task:
            found = next((e for e in self.events if e["name"] == "found"), None)
            meta["task"] = self.task
            meta["episode"] = {
                "index": self.episode_done + 1, "target": self.episodes_target or None,
                "outcome": "found" if found else "not_marked",
                "found_t_pc": found["t_pc"] if found else None,
                "found_us_seq": found["us_seq"] if found else None,
                "found_marker": "found" if found else "implicit_end",
                "protocol": "방광 밖에서 시작 → 정지-이동-정지로 탐색 → 방광이 보이면 (M 또는) 1~2 s 정지 → R 로 저장. 실패는 X 로 폐기",
            }
            ps = getattr(self, "_plan_at_start", None)
            if ps:
                # 수집 계획의 종류 (episode.motion) — 매니페스트·라벨 분류의 근거. bin_hint 는 GUI 가 제안한 부호/크기이고
                # 실제 움직임은 라벨러가 IMU 로 잰다.
                meta["episode"].update({"motion": ps["key"], "motion_label": ps["label"], "bin_hint": ps["bin_hint"],
                                        "index_in_kind": ps["index_in_kind"], "kind_target": ps["target"],
                                        "plan_file": ps["plan_file"]})
        with open(os.path.join(self.session_dir, "session.meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------- 그리기
    def artists(self):
        return ([self.us_im] + self.orient.artists() + self.euler.artists()
                + self.accel.artists() + self.gyro.artists()
                + self.mag.artists() + self.diff.artists())

    def _update_us(self):
        snap = self.us.snapshot()
        img = snap["img"]
        if img is not None:
            if self.converter is not None:
                try:
                    img = self.converter.convert(img)
                except ValueError:
                    pass
            shown = img if self.rotate_k == 0 else np.rot90(img, self.rotate_k)
            self.us_im.set_data(shown)
            self.us_text.set_text("" if not snap["stale"] else "US 정지 — 프로브 스캔 상태 확인")
        else:
            self.us_text.set_text("US 프레임 대기...  (연결=%s active=%s)"
                                  % (snap["connected"], snap["active"]))
        return snap

    def update(self, _frame):
        us_snap = self._update_us()
        if self.us.recording and self.record_frames > 0 and us_snap.get("rec_count", 0) >= self.record_frames:
            print("US %d 프레임 도달 — 자동 정지" % self.record_frames)
            self._stop_reason = "auto"
            self.toggle_record()

        s = self.stream.snapshot()
        if s["error"]:
            self.status.set_text("IMU 오류: %s   |   US frames=%d fps=%.1f"
                                 % (s["error"], us_snap["frames"], us_snap["fps"]))
            return self.artists()
        hist = s["hist"]
        if len(hist["t"]) >= 2:
            if self.t0 is None:
                self.t0 = hist["t"][0]
            self.orient.update(s["host_q"], s["aligned_q"])
            for panel in (self.euler, self.accel, self.gyro, self.mag, self.diff):
                panel.update(hist, self.t0)

        r = s["rates"]
        cal = s.get("cal_status") or {}
        z = s.get("zero")
        zero_ok = bool(z)
        zero_forced = bool(z and not z.quality.get("still"))
        cal_ok = calib_ready(cal)
        fw = s.get("fw_tag")
        dcd_at = s.get("dcd_saved_at")
        dcd_ok = dcd_at is not None
        if fw is None:
            fw_line = "[ ] 펌웨어: 구버전 ('V' 응답 없음 — 자이로 보정 꺼짐) → host\\flash_win.py 로 플래시"
        elif fw != FW_TAG_EXPECTED:
            fw_line = "[?] 펌웨어 %s (기대 %s)" % (fw, FW_TAG_EXPECTED)
        else:
            fw_line = "[v] 펌웨어 %s" % fw
        lines = ["세션 시작 체크리스트",
                 fw_line,
                 "%s [K] IMU 보정  a%s g%s m%s rv%s  (rv 3 = 자력계 8 자 워밍업 완료)" % (
                     "[v]" if cal_ok else "[ ]", cal.get("acc", "-"), cal.get("gyr", "-"), cal.get("mag", "-"), cal.get("rv", "-")),
                 "%s [S] DCD 저장%s" % ("[v]" if dcd_ok else "[ ]",
                                        " (%s)" % time.strftime("%H:%M:%S", time.localtime(dcd_at)) if dcd_ok else " — 보정 뒤 한 번, 전원 재인가 후에도 유지"),
                 "%s [Z] 영점 (정지 %.0f s)%s" % ("[v]" if zero_ok else "[ ]", self.args.zero, " — 정지 판정 미달, 강제 채택" if zero_forced else ""),
                 "%s [R] 녹화 (%s)" % ("[v]" if self.session_count else "[ ]",
                                      "%d 프레임 자동 정지" % self.record_frames if self.record_frames else "수동 정지: R 다시")]
        if self.task:
            nxt = self.episode_done + 1
            tgt = "/%d" % self.episodes_target if self.episodes_target else ""
            if self._recording():
                lines.append("▶ %s 에피소드 #%d%s 녹화 중 — 방광 중앙에서 1~2 s 정지 후 R 저장 · X 폐기%s"
                             % (self.task, nxt, tgt, "  [M %d]" % len(self.events) if self.events else ""))
            else:
                lines.append("%s: 에피소드 %d%s 저장됨 · 다음 #%d" % (self.task, self.episode_done, tgt, nxt))
        if self.plan:
            items = self.plan["items"]
            done_total = sum(min(self.plan_counts.get(it["key"], 0), it["target"]) for it in items)
            total = sum(it["target"] for it in items)
            prog = " · ".join("%s %d/%d" % (it["label"], self.plan_counts.get(it["key"], 0), it["target"]) for it in items)
            lines.append("수집 계획 %d/%d — %s" % (done_total, total, prog))
            cur = (self._plan_at_start if self._recording() else None) or None
            pc = self.plan_current()
            if self._recording() and cur:
                lines.append("▶ 지금: [%d] %s %d/%d회%s" % (
                    next(i for i, it in enumerate(items) if it["key"] == cur["key"]) + 1, cur["label"], cur["index_in_kind"],
                    cur["target"], " — " + cur["bin_hint"] if cur["bin_hint"] else ""))
            elif pc:
                it = pc["item"]
                lines.append("▶ 다음: [%d] %s — %d/%d 완료, 이번은 %d번째%s" % (
                    pc["i"] + 1, it["label"], pc["count"], it["target"], pc["count"] + 1,
                    " · " + pc["hint"] if pc["hint"] else ""))
                lines.append("  규약: " + it["protocol"])
                lines.append("  (N 다음 종류 · 숫자 키로 종류 선택)")
            else:
                lines.append("✔ 수집 계획 완료 — 이후 세션은 종류 없이 저장됨 (숫자 키로 종류를 고르면 계속 셈)")
        if self.event_msg and time.time() < self.event_msg_until:
            lines.append(self.event_msg)
        if self.zero_msg and time.time() < self.zero_msg_until:
            lines.append(self.zero_msg)
        info = s.get("last_info")
        if info and time.time() - info[1] < 6.0 and info[0].startswith(("DCD,", "CAL,")):
            lines.append({"DCD,OK": "✔ DCD 저장 완료 — 칩 플래시에 기록됨 (전원 재인가 후에도 유지)",
                          "DCD,FAIL": "✘ DCD 저장 실패 — 칩이 거부. 보정을 다시 하고 S",
                          "CAL,OK": "동적 보정 재적용 OK", "CAL,FAIL": "동적 보정 재적용 실패"}.get(info[0], info[0]))
        if self.calib_mode:
            lines.append("보정 안내: " + calib_step(cal, dcd_ok))
            if cal_ok and (cal.get("acc") or 0) >= 3 and dcd_ok:
                lines.append("(K 로 안내 닫기)")
        elif not cal_ok:
            lines.append("K 를 눌러 보정 안내를 여십시오")
        self.checklist.set_text("\n".join(wrap_panel_lines(lines)))
        rec = "● 녹화중 #%d" % self.session_count if (self.stream.logger or self.us.recording) else \
            ("○ 대기 (세션 %d 저장됨)" % self.session_count if self.session_count else "○ 대기")
        rec_detail = ""
        if self.stream.logger or self.us.recording:
            rec_detail = "  US %d%s / IMU %d" % (us_snap["rec_count"],
                                                ("/%d" % self.record_frames) if self.record_frames else "", s["log_n"])
        if self.sync is not None and self.sync.last_message:
            rec_detail += "   |   " + self.sync.last_message[:90]
        z = s.get("zero")
        zero_state = "영점 OK" if (z and z.quality.get("still")) else ("영점 (강제)" if z else "영점 없음")
        us_state = ("active" if us_snap["active"] else "idle")
        if not us_snap["connected"]:
            us_state = "연결 안 됨"
        self.status.set_text(
            "IMU  A%5.1f G%5.1f M%5.1f RV%5.1f Hz  cal a%s/g%s/m%s  crc=%d gaps=%d  %s   |   "
            "US  %s  %.1f fps  frames=%d   |   %s%s"
            % (r[REC_ACCEL], r[REC_GYRO], r[REC_MAG], r[REC_RV],
               cal.get("acc", "-"), cal.get("gyr", "-"), cal.get("mag", "-"),
               s["crc_errors"], s["gaps"], zero_state,
               us_state, us_snap["fps"], us_snap["frames"], rec, rec_detail))
        return self.artists()


def count_task_sessions(out_dir, task):
    """out_dir 의 저장된 세션 중 session.meta.json 의 task 가 같은 것의 수 — 에피소드 번호를 이어 붙이기 위해.
    폐기 폴더(discard_*) 는 us_imu_* 가 아니므로 세지 않는다."""
    n = 0
    try:
        names = sorted(os.listdir(out_dir))
    except OSError:
        return 0
    for name in names:
        if not name.startswith("us_imu_"):
            continue
        p = os.path.join(out_dir, name, "session.meta.json")
        try:
            with open(p, encoding="utf-8") as fh:
                if json.load(fh).get("task") == task:
                    n += 1
        except (OSError, ValueError):
            continue
    return n


def load_plan(path):
    """수집 계획 JSON (imu_bench/collect_plan_<task>.json):
    {"task", "note", "common", "items": [{"key", "label", "target", "protocol", "bins": [...]}, ...]}"""
    with open(path, encoding="utf-8") as fh:
        plan = json.load(fh)
    items = plan.get("items") or []
    if not items:
        raise ValueError("계획에 items 가 없습니다: %s" % path)
    for it in items:
        it.setdefault("bins", [])
        it["target"] = int(it.get("target", 0))
    plan["path"] = os.path.abspath(path)
    return plan


def count_plan_sessions(out_dir, task, keys):
    """out_dir 의 저장된 세션 중 task 가 같고 episode.motion 이 각 key 인 것의 수 — 계획 진행률의 근거."""
    counts = {k: 0 for k in keys}
    try:
        names = sorted(os.listdir(out_dir))
    except OSError:
        return counts
    for name in names:
        if not name.startswith("us_imu_"):
            continue
        p = os.path.join(out_dir, name, "session.meta.json")
        try:
            with open(p, encoding="utf-8") as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        if m.get("task") != task:
            continue
        k = (m.get("episode") or {}).get("motion")
        if k in counts:
            counts[k] += 1
    return counts


def _wait_probe(host, timeout=5.0):
    """프로브 TCP 가 잠깐 사이에 열려 있는지만 확인 (경고용, 차단하지 않음)."""
    try:
        with socket.create_connection((host, 5002), timeout):
            return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="192.168.1.1", help="프로브 AP 주소")
    ap.add_argument("--probe", choices=sorted(PROFILES), default="sl2c", help="프로브 프로파일 (us_protocol.PROFILES)")
    ap.add_argument("--record-frames", type=int, default=0, help="이 수의 US 프레임이 저장되면 녹화 자동 정지 (0 = 수동)")
    ap.add_argument("--task", default=None, metavar="NAME",
                    help="에피소드 모드 — 세션 = 에피소드 하나 (예: find_bladder). 메타에 task/episode/events 를 남기고 "
                         "번호를 out-dir 의 같은 task 세션 수에서 잇는다. 녹화 중 M = 발견 표시, X = 폐기")
    ap.add_argument("--episodes", type=int, default=0, metavar="N", help="에피소드 목표 수 (표시용, 예: 100; --plan 이 있으면 계획 합계)")
    ap.add_argument("--plan", default=None, metavar="JSON",
                    help="수집 계획 (imu_bench/collect_plan_<task>.json): 움직임 종류별 목표·규약·bin. 패널이 '지금 무엇을 몇 번째로' "
                         "를 안내하고 메타 episode.motion 에 종류를 남긴다. N 다음 종류, 숫자 키 선택")
    ap.add_argument("--display", choices=["auto", "polar", "fan"], default="auto",
                    help="표시: polar(원본) | fan(scan conversion, 저장은 원본). auto = c10ur 이면 fan")
    ap.add_argument("--fan-radius", type=float, default=59.0, help="부채꼴 반경 mm (뷰어 실측 59, R60)")
    ap.add_argument("--fan-angle", type=float, default=28.0, help="섹터 반각 deg (뷰어 실측 ≈28)")
    ap.add_argument("--fan-depth", type=float, default=220.0, help="깊이 mm (뷰어 D:220mm)")
    ap.add_argument("--fan-flip", action="store_true", help="라인 순서 반전 (좌우 검증용)")
    ap.add_argument("--no-sync", action="store_true", help="저장 후 자동 동기화(sync.npz) 를 끈다")
    # 리눅스도 비워 둔다 — VID 2886 자동 탐지는 `list_ports` 가 플랫폼과 무관하게 해 준다.
    # 예전 기본값 `/dev/ttyACM0` 은 이 셀에서 **PX6D F/T 센서**다 (IMU 는 ttyACM1, 게다가
    # 꽂는 순서로 뒤바뀐다). 그대로 두면 IMU 대신 힘 센서를 열고, 921600 bps 스트림을
    # 115200 으로 읽어 조용히 쓰레기를 낸다. 2026-09-10.
    ap.add_argument("--port", default=os.environ.get("IMU_PORT"),
                    help="IMU 시리얼 포트 (기본: VID 2886 자동 탐지)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--no-mag", action="store_true", help="6축 IMU-only 퓨전")
    ap.add_argument("--mag-cal", default=os.path.join(_HERE, os.pardir, "mag_cal.json"))
    ap.add_argument("--orientation", choices=list(ORIENTATIONS), default="none",
                    help="US 표시 방향 (저장은 항상 원시)")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "..", "logs"))
    ap.add_argument("--zero", type=float, default=3.0, metavar="SEC")
    ap.add_argument("--zero-profile", default="freehand", choices=("freehand", "bench"),
                    help="영점 정지 판정 임계: freehand(손으로 든 프로브, 기본) / bench(탁자·로봇 마운트)")
    ap.add_argument("--fps", type=float, default=20.0, help="화면 갱신 fps")
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    if args.port is None:
        try:
            from serial.tools import list_ports
            args.port = next((p.device for p in list_ports.comports() if p.vid == 0x2886), None)
        except ImportError:
            args.port = None
        if args.port is None:
            print("IMU 포트를 찾지 못했습니다 (VID 2886) — 보드가 꽂혀 있는지 확인하거나 "
                  "--port 로 지정하십시오.\n"
                  "  ⚠ 리눅스에서 /dev/ttyACM0 은 PX6D F/T 센서일 수 있습니다. "
                  "/dev/serial/by-id/ 로 확인하십시오."); return 1
    profile = PROFILES[args.probe]
    if args.display == "auto":
        args.display = "fan" if profile.name == "c10ur" else "polar"

    use_korean_font()
    cal = mag_calib.load(args.mag_cal) if os.path.exists(args.mag_cal) else None

    stream = ImuStream(port=args.port, baud=args.baud, cal=cal, beta=args.beta,
                       use_mag=not args.no_mag, logger=None)
    stream.start()

    # ⚠ 프로브 포트를 미리 열어 보지 않는다 (_wait_probe). C10UR 은 클라이언트를 하나만 받고, 열었다 닫은 뒤 한동안
    # 다음 접속을 거부해 정작 수신기가 못 붙는다 (2026-09-09). 연결 상태는 상태줄의 "US 연결 안 됨" 으로 본다.

    us = UsReceiver(args.host, profile=profile)
    us.start()
    print("프로브 %s (%s), IMU %s, 저장 %s" % (profile.name, "x".join(map(str, profile.frame_shape)), args.port, args.out_dir))

    gui = CombinedGui(stream, us, args)
    anim = FuncAnimation(gui.fig, gui.update, interval=int(1000 / args.fps),
                         blit=False, cache_frame_data=False)
    try:
        plt.show()
    finally:
        if stream.logger is not None or us.recording:
            gui.toggle_record()   # 창을 그냥 닫아도 열린 로그를 안전하게 닫는다
        if gui.sync is not None:
            gui.sync.stop()
        stream.stop()
        us.stop()
        del anim


if __name__ == "__main__":
    main()
