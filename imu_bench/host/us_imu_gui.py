#!/usr/bin/env python3
"""초음파 + BNO085 IMU 통합 뷰어 — 한 창에서 보고, 키 하나로 동기 녹화.

왼쪽에 초음파 candidate 영상, 오른쪽에 IMU 퓨전 6패널을 한 matplotlib 창에 함께
띄운다. IMU 뷰어(`imu_gui`)와 초음파 뷰어(tracer `direct_display`)를 합친 것이다.

두 소스는 물리 경로가 다르다 (IMU=USB 시리얼, US=Wi-Fi TCP). 같은 시간축에 올리는
방법은 `us_imu_collect` 와 같다 — **둘 다 호스트 수신 순간의 `time.time()`(pc_unix)로
찍는다.** IMU 스트림이 이미 그 시계로 로그를 남기므로, US 프레임도 도착 순간 같은
시계로 찍으면 두 로그가 pc_unix 로 직접 조인된다.

키:
    R  녹화 시작/정지 — US 와 IMU 를 **한 세션 폴더에 동시에**. `--record-frames N` 이면 N 프레임에서 자동 정지.
    F  스캔 시작/정지 — 클라이언트가 스캔을 명령하는 프로브(`--probe c10ur`)에서만. SL-2C 는 프로브 버튼.
    Z  영점(zero) — IMU 를 현재 자세 기준으로 잡는다 (몇 초간 정지 필요).
    C  자세 재설정 — 호스트 퓨전 프레임 정렬을 다시 잡는다.
    Q  종료.

프로브 (`--probe`, 2026-09-09): sl2c (기본, 256×256) | c10ur (Konted, Wi-Fi AP "US-1C …", 320 라인 × 256 깊이 표본의
극좌표 candidate, 10 fps, 시작/정지 명령 있음). 상세는 fr5_vision/us_protocol.py 의 ProbeProfile.
Windows 에서는 IMU 포트를 비우면 VID 2886 의 COM 포트를 자동으로 잡는다.

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
        self._stop = threading.Event()

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
        self._stop.set()

    def run(self):
        last_connect = 0.0
        while not self._stop.is_set():
            if not self._session.is_open:
                if time.monotonic() - last_connect < 2.0:
                    self._stop.wait(0.05)
                    continue
                last_connect = time.monotonic()
                try:
                    self._session.open()
                    with self._lock:
                        self.connected = True
                    self._scan_sent_at = 0.0
                except OSError:
                    with self._lock:
                        self.connected = False
                    continue

            # 재접속(프로브 자동 정지·AP 재기동) 뒤에도 원하는 스캔 상태를 복원한다: 세션이 열린 지 1.6 s 뒤 시작 요청
            if (self.want_scan and self._session.can_command_scan and not self._session.scanner_active
                    and self._scan_sent_at == 0.0 and time.monotonic() - self._session._started_at >= 1.6):
                self._session.start_scan(); self._scan_sent_at = time.monotonic()
            try:
                frames = self._session.poll(0.05)
            except (ConnectionError, OSError):
                self._session.close()
                with self._lock:
                    self.connected = False
                continue

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

        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.manager.set_window_title("US + IMU 통합 뷰어")
        gs = self.fig.add_gridspec(3, 3, width_ratios=(1.5, 1.0, 1.0),
                                   hspace=0.45, wspace=0.28,
                                   left=0.03, right=0.98, top=0.90, bottom=0.07)

        # 왼쪽: 초음파 영상 (세로 전체)
        self.us_ax = self.fig.add_subplot(gs[:, 0])
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
        self.fig.text(0.98, 0.965,
                      "[R] 녹화 시작/정지 (US+IMU 동시)   [F] 스캔 시작/정지   [Z] 영점   [C] 자세 재설정   [Q] 종료",
                      fontsize=9, ha="right", va="top", alpha=0.75)
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
        elif k == "r":
            self.toggle_record()
        elif k == "f":
            res = self.us.toggle_scan() if hasattr(self.us, "toggle_scan") else None
            self.status.set_text("스캔 %s" % ("시작 요청" if res else ("정지" if res is False else "명령 불가 (프로브 버튼 사용)")))

    def zero_now(self):
        self.status.set_text("영점 캘리브레이션 %.1f초 — 센서를 움직이지 마세요..." % self.args.zero)
        self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
        run_zero_calibration(self.stream, self.args.zero, retries=0)

    def toggle_record(self):
        # 스캔 명령이 가능한 프로브인데 아직 스캔 중이 아니면 녹화 시작과 함께 스캔을 시작한다
        starting = not (self.stream.logger is not None or self.us.recording)
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
            print("녹화 정지: %s  (US %s frames, IMU %s rows) — 저장 완료. 동기화는 백그라운드에서 몇 초 뒤. "
                  "R 로 다음 세션을 바로 시작할 수 있습니다." % (self.session_dir, us_n, imu_n))
            self.session_dir = None
        else:
            # 시작 — 한 세션 폴더에 US 와 IMU 를 동시에
            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self.session_dir = os.path.join(self.args.out_dir, "us_imu_" + stamp)
            os.makedirs(self.session_dir, exist_ok=True)
            self.stream.logger = self._new_imu_logger()
            self.us.start_recording(self.session_dir)
            self.session_count += 1
            print("녹화 시작 #%d: %s%s" % (self.session_count, self.session_dir,
                                          "" if self.stream.zero else "   ⚠ 영점(zero_ref) 없음 — Z 를 먼저 누르는 것을 권장"))

    def _new_imu_logger(self):
        z = self.stream.zero
        return SessionLogger(
            self.session_dir, fmt="csv", prefix="imu",
            meta={"source": "BNO085 via XIAO nRF52840 Sense",
                  "port": self.args.port, "use_mag": not self.args.no_mag,
                  "zero_ref": z.to_dict() if z else None})

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
        }
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
        rec = "● 녹화중 #%d" % self.session_count if (self.stream.logger or self.us.recording) else \
            ("○ 대기 (세션 %d 저장됨)" % self.session_count if self.session_count else "○ 대기")
        rec_detail = ""
        if self.stream.logger or self.us.recording:
            rec_detail = "  US %d%s / IMU %d" % (us_snap["rec_count"],
                                                ("/%d" % self.record_frames) if self.record_frames else "", s["log_n"])
        if self.sync is not None and self.sync.last_message:
            rec_detail += "   |   " + self.sync.last_message[:90]
        z = s.get("zero")
        zero_state = "영점 OK" if (z and z.quality.get("still")) else ("영점 불량" if z else "영점 없음")
        us_state = ("active" if us_snap["active"] else "idle")
        if not us_snap["connected"]:
            us_state = "연결 안 됨"
        self.status.set_text(
            "IMU  A%5.1f G%5.1f M%5.1f RV%5.1f Hz  crc=%d gaps=%d  %s   |   "
            "US  %s  %.1f fps  frames=%d   |   %s%s"
            % (r[REC_ACCEL], r[REC_GYRO], r[REC_MAG], r[REC_RV],
               s["crc_errors"], s["gaps"], zero_state,
               us_state, us_snap["fps"], us_snap["frames"], rec, rec_detail))
        return self.artists()


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
    ap.add_argument("--display", choices=["auto", "polar", "fan"], default="auto",
                    help="표시: polar(원본) | fan(scan conversion, 저장은 원본). auto = c10ur 이면 fan")
    ap.add_argument("--fan-radius", type=float, default=59.0, help="부채꼴 반경 mm (뷰어 실측 59, R60)")
    ap.add_argument("--fan-angle", type=float, default=28.0, help="섹터 반각 deg (뷰어 실측 ≈28)")
    ap.add_argument("--fan-depth", type=float, default=220.0, help="깊이 mm (뷰어 D:220mm)")
    ap.add_argument("--fan-flip", action="store_true", help="라인 순서 반전 (좌우 검증용)")
    ap.add_argument("--no-sync", action="store_true", help="저장 후 자동 동기화(sync.npz) 를 끈다")
    ap.add_argument("--port", default=os.environ.get("IMU_PORT", None if sys.platform == "win32" else "/dev/ttyACM0"),
                    help="IMU 시리얼 포트 (Windows 기본: VID 2886 자동 탐지)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--no-mag", action="store_true", help="6축 IMU-only 퓨전")
    ap.add_argument("--mag-cal", default=os.path.join(_HERE, os.pardir, "mag_cal.json"))
    ap.add_argument("--orientation", choices=list(ORIENTATIONS), default="none",
                    help="US 표시 방향 (저장은 항상 원시)")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "..", "logs"))
    ap.add_argument("--zero", type=float, default=3.0, metavar="SEC")
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
            print("IMU 포트를 찾지 못했습니다 (VID 2886) — --port 로 지정하십시오"); return 1
    profile = PROFILES[args.probe]
    if args.display == "auto":
        args.display = "fan" if profile.name == "c10ur" else "polar"

    use_korean_font()
    cal = mag_calib.load(args.mag_cal) if os.path.exists(args.mag_cal) else None

    stream = ImuStream(port=args.port, baud=args.baud, cal=cal, beta=args.beta,
                       use_mag=not args.no_mag, logger=None)
    stream.start()

    if not _wait_probe(args.host):
        print("주의: 프로브 TCP %s:5002 에 지금 닿지 않음 — Wi-Fi/스캔 상태 확인. "
              "뷰어는 계속 재연결을 시도한다." % args.host)

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
