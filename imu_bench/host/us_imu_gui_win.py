#!/usr/bin/env python3
"""Windows: WirelessUSG 뷰어 화면 캡처 + BNO085 IMU 를 한 창에서 보고, 영점 → 녹화 → 저장.

`us_imu_gui.py`(Wi-Fi 프로브용) 의 화면·키·세션 포맷을 그대로 쓰고, 초음파 소스만
`us_screen_capture.ScreenUsReceiver`(벤더 뷰어 창 화면 캡처) 로 바꿨다. 왜 화면 캡처인지는
그 모듈 docstring 에 있다 (USB 프로토콜 미지 — 지연은 `inspect_session.py --latency` 로 실측).

화면: 왼쪽 초음파 candidate 영상, 오른쪽 IMU 6 패널 (자세 3D, 오일러, 가속도, 자이로, 자력계, 호스트-칩 각도차).

키:
    Z  영점(zero) 캘리브레이션 — 프로브+IMU 를 몇 초 정지 (기본 3 s). 녹화 **전에** 한 번.
    R  녹화 시작/정지 — US 와 IMU 를 한 세션 폴더에 동시 저장. 시작 시 뷰어가 FREEZE 면 LIVE 로 바꾼다.
       `--record-frames N` (기본 1000) 개의 US 프레임이 저장되면 **자동 정지** = session.meta.json 기록 = 저장 완료.
    F  뷰어 Freeze/Live 토글 (UI Automation 으로 뷰어 버튼을 누른다 — 뷰어 창이 숨겨져 있어도 됨).
    V  뷰어 창 보이기/숨기기 (기본 숨김 = 화면 밖. 최소화가 아니므로 캡처는 계속된다).
    S  캡처 ROI 재지정 — 뷰어 창 스크린샷 위에서 영상 영역을 드래그 (Enter 확정). roi 파일에 저장된다.
    W  뷰어 창 다시 찾기 (창을 옮겼거나 크기를 바꿨을 때).
    C  자세 재설정 (호스트 퓨전 프레임 정렬).
    Q  종료 (녹화 중이면 안전하게 닫는다).

뷰어(WirelessUSG) 는 프로브와 말하는 유일한 프로세스라 **없앨 수는 없지만**, 이 GUI 가 실행·숨김·조작을 대신한다:
시작 시 뷰어가 없으면 실행하고, 창을 화면 밖으로 보내고, 상태줄에 뷰어 상태(LIVE/FREEZE, 프레임 카운터, 깊이, 게인,
프로브 이름) 를 보인다 (`us_viewer_control.py`).

    python imu_bench\\host\\us_imu_gui_win.py                      # COM 포트 자동 (VID 2886), ROI 파일 사용
    python imu_bench\\host\\us_imu_gui_win.py --select-roi         # ROI 만 지정하고 종료
    python imu_bench\\host\\us_imu_gui_win.py --port COM3 --no-mag --out-dir D:\\us_sessions

절차 (수집 첫날):
    1. 뷰어(WirelessUSG) 를 실행해 B-mode 가 나오게 한다. 창은 최소화하지 않는다.
    2. 이 스크립트 실행. 처음이면 --select-roi 로 영상 영역만 잡는다 (시계·카운터 텍스트 제외).
    3. 프로브+IMU 를 고정한 채 정지 → Z (영점). 상태줄이 "영점 OK" 가 되면
    4. R 로 녹화 시작 → 정지-이동-정지 프로토콜로 스캔 → R 로 정지. 폴더 하나가 세션 하나.
    5. python policy_learning\\scripts\\inspect_session.py <세션폴더> --latency

산출물 (out_dir/us_imu_<stamp>/): us_frames.bin, us_index.csv, imu_<stamp>.csv(+meta.json), session.meta.json
— `us_imu_collect.py` 와 같은 포맷. `session.meta.json` 의 us.capture 에 캡처 기하(ROI, 레터박스, 스케일) 가 남는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# us_imu_gui 를 import 하면 imu_gui 가 matplotlib.use("TkAgg") 를 건다.
from us_imu_gui import CombinedGui, ORIENTATIONS            # noqa: E402
from imu_gui import use_korean_font                           # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402

import mag_calib                                              # noqa: E402
from imu_stream import ImuStream                              # noqa: E402
from imu_log import SessionLogger                             # noqa: E402
from us_screen_capture import (                               # noqa: E402
    CaptureRoi, ScreenUsReceiver, grab_client_gray, select_roi_interactive,
    DEFAULT_WINDOW_TITLE, DEFAULT_FRAME_SIZE,
)
from us_viewer_control import ViewerControl, ViewerPoller, DEFAULT_VIEWER_EXE   # noqa: E402

IMU_USB_VID = 0x2886      # Seeed XIAO nRF52840 Sense (board-6-qc 펌웨어)
# 뷰어 기본 레이아웃(최대화, 2560×1600 실측)에서 영상 영역의 분율. 하단 "LIVE"/프레임 카운터 텍스트와
# 우측 스크롤바를 제외한다 (카운터가 ROI 에 들어가면 중복 제거가 무력해진다). 창 크기가 다르면 S 로 다시 잡는다.
DEFAULT_ROI = CaptureRoi(0.105, 0.060, 0.875, 0.880)


def autodetect_imu_port(vid: int = IMU_USB_VID):
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    for p in list_ports.comports():
        if p.vid == vid:
            return p.device
    return None


class WinGui(CombinedGui):
    """CombinedGui + 화면 캡처 소스에 맞는 상태 표시·메타·ROI 키."""

    def __init__(self, stream, us, args, roi_path, viewer=None, poller=None):
        super().__init__(stream, us, args)
        self.roi_path = roi_path
        self.viewer = viewer
        self.poller = poller
        self.record_frames = int(args.record_frames) if args.record_frames else 0
        self.us_ax.set_title("초음파 candidate — 뷰어 화면 캡처 (표시 방향 %s)" % args.orientation, fontsize=10)
        # 키 안내 덧붙이기
        self.fig.text(0.98, 0.945, "[F] 뷰어 Freeze/Live   [V] 뷰어 창 보이기/숨기기   [S] 캡처 ROI 지정   [W] 뷰어 창 다시 찾기",
                      fontsize=9, ha="right", va="top", alpha=0.75)
        self.viewer_status = self.fig.text(0.03, 0.945, "", fontsize=9, va="top", color="#335599")
        # 그리기 비용 분리: 6 패널 전체 재그리기는 ~240 ms/프레임 (3D 자세 포함) 라 매 tick 하면 GUI 가 포화된다.
        # tick 마다는 초음파 패널만 blit 으로 갱신하고, 전체 재그리기는 full_every tick 에 한 번.
        self.full_every = max(1, int(args.plot_every))
        self._tick = 0
        self._us_bg = None
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)

    def _on_draw(self, _event):
        # 전체 그리기 직후의 초음파 축 배경을 저장한다 (blit 용)
        try:
            self._us_bg = self.fig.canvas.copy_from_bbox(self.us_ax.bbox)
        except Exception:  # noqa: BLE001
            self._us_bg = None

    def _blit_us(self):
        """초음파 이미지와 상태 텍스트만 다시 그린다 (수 ms)."""
        snap = self._update_us()
        if self._us_bg is None:
            return snap
        try:
            self.fig.canvas.restore_region(self._us_bg)
            self.us_ax.draw_artist(self.us_im)
            self.us_ax.draw_artist(self.us_text)
            self.fig.canvas.blit(self.us_ax.bbox)
        except Exception:  # noqa: BLE001
            self._us_bg = None
        return snap

    def on_key(self, event):
        k = (event.key or "").lower()
        if k == "s":
            self.select_roi()
        elif k == "w":
            self.us.refind_window()
            if self.viewer is not None:
                self.viewer.attach()
        elif k == "f":
            self.toggle_viewer_freeze()
        elif k == "v":
            self.toggle_viewer_window()
        else:
            super().on_key(event)

    # -------------------------------------------------------------- 뷰어 제어
    def toggle_viewer_freeze(self):
        if self.viewer is None:
            self.status.set_text("뷰어 제어 없음 (--no-viewer-control)"); return
        if self.viewer.freeze_enabled() is False:
            self.status.set_text("뷰어 Freeze 버튼이 비활성 — 프로브가 스트림을 주지 않는 상태 (프로브 전원/USB 재연결)")
            return
        ok = self.viewer.toggle_freeze()
        self.status.set_text("뷰어 Freeze/Live 토글 %s" % ("전송" if ok else "실패 — 창을 못 찾음"))

    def toggle_viewer_window(self):
        if self.viewer is None:
            return
        if self.viewer.hidden:
            self.viewer.show(); print("뷰어 창 보임")
        else:
            self.viewer.hide(); print("뷰어 창 숨김 (화면 밖)")

    def toggle_record(self):
        starting = not (self.stream.logger is not None or self.us.recording)
        if starting and self.viewer is not None:
            snap = self.poller.snapshot() if self.poller else {}
            if snap.get("state", "").upper() == "FREEZE":
                res = self.viewer.set_live(True)
                print("녹화 시작 전 뷰어 LIVE 전환 → %s" % ("LIVE" if res else "실패 (Freeze 버튼 비활성?)"))
        super().toggle_record()

    def select_roi(self):
        if self.us.recording:
            self.status.set_text("녹화 중에는 ROI 를 바꾸지 않는다 — R 로 먼저 정지")
            return
        try:
            gray = grab_client_gray(self.us.title)
        except RuntimeError as exc:
            self.status.set_text(str(exc)); return
        roi = select_roi_interactive(gray, current=self.us.roi)
        if roi is not None:
            self.us.set_roi(roi)
            roi.save(self.roi_path)
            print("ROI 저장: %s  %s" % (self.roi_path, roi))

    def _update_us(self):
        snap = self.us.snapshot()
        img = snap["img"]
        if img is not None:
            shown = img if self.rotate_k == 0 else __import__("numpy").rot90(img, self.rotate_k)
            self.us_im.set_data(shown)
            if snap["stale"]:
                self.us_text.set_text("US 정지 — 뷰어가 FREEZE 이거나 프로브가 스캔 중이 아님")
            else:
                self.us_text.set_text("")
        else:
            self.us_text.set_text(snap["error"] or "뷰어 창 캡처 대기...")
        return snap

    def tick(self):
        """Tk 타이머 콜백. FuncAnimation 은 blit=False 면 매 tick 전체를 다시 그리므로 쓰지 않는다."""
        self._tick += 1
        if self._tick % self.full_every != 0 and self._us_bg is not None:
            # 빠른 tick: 초음파 패널만 blit (수 ms). 녹화 자동 정지 판정은 여기서도 한다.
            snap = self._blit_us()
            if self.us.recording and self.record_frames > 0 and snap.get("rec_count", 0) >= self.record_frames:
                print("US %d 프레임 도달 — 자동 정지" % self.record_frames)
                self.toggle_record()
            return
        self.update(self._tick)
        self.fig.canvas.draw_idle()

    def update(self, _frame):
        arts = super().update(_frame)
        snap = self.us.snapshot()
        lm = snap.get("latest_mean")
        rec_txt = ""
        if self.us.recording and self.record_frames > 0:
            rec_txt = "   |   US 저장 %d / %d" % (snap.get("rec_count", 0), self.record_frames)
            if snap.get("rec_count", 0) >= self.record_frames:
                print("US %d 프레임 도달 — 자동 정지" % self.record_frames)
                self.toggle_record()
                rec_txt = "   |   %d 프레임 저장 완료" % self.record_frames
        self.status.set_text(self.status.get_text() + "   |   cap=%s mean=%s%s" % (
            snap.get("backend"), "-" if lm is None else "%.1f" % lm, rec_txt))
        if self.poller is not None:
            v = self.poller.snapshot()
            if not v.get("attached"):
                self.viewer_status.set_text("뷰어: 창 없음 (V 로 실행/찾기)")
            else:
                fe = v.get("freeze_enabled")
                self.viewer_status.set_text(
                    "뷰어: %s  %s  %s  %s  프로브 %s  창 %s%s" % (
                        v.get("state") or "?", v.get("count") or "", v.get("depth") or "", v.get("gain") or "",
                        v.get("probe") or "없음", "숨김" if v.get("hidden") else "보임",
                        "" if fe is not False else "  ⚠ Freeze 비활성(스트림 없음)"))
        return arts

    def _write_session_meta(self, imu_path, imu_n, us_n):
        if self.session_dir is None:
            return
        meta = {
            "created": time.strftime("%Y%m%d_%H%M%S", time.localtime()),
            "time_axis": "pc_unix (host time.time() at reception, shared by both streams)",
            "join_key": "pc_unix",
            "caveat": ("수신 시각 정렬. US 는 뷰어 화면 캡처라 프로브→뷰어→화면→캡처 고정 지연이 붙는다 — "
                       "inspect_session.py --latency 로 실측해 timing.us_latency_s 에 넣을 것."),
            "us": {"host": self.us.host, "frames": us_n,
                   "frame_shape": [self.us.frame_size, self.us.frame_size], "dtype": "uint8",
                   "bin": "us_frames.bin", "index": "us_index.csv",
                   "display_orientation": self.args.orientation,
                   "probe": "Konted C10UR (USB, Cypress 04B4:BC0C) via WirelessUSG 2.1.4",
                   "capture": self.us.capture_meta(),
                   "note": "candidate — 뷰어 렌더 영상의 그레이스케일 레터박스 축소. scan conversion 은 뷰어가 이미 함."},
            "imu": {"port": self.args.port, "rows": imu_n,
                    "csv": None if imu_path is None else os.path.basename(imu_path),
                    "meta": None if imu_path is None else os.path.basename(imu_path).replace(".csv", ".meta.json")},
        }
        with open(os.path.join(self.session_dir, "session.meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=os.environ.get("IMU_PORT"), help="IMU COM 포트 (기본: VID 2886 자동)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--no-mag", action="store_true", help="6축 IMU-only 퓨전 (자력계 미사용)")
    ap.add_argument("--mag-cal", default=os.path.join(_HERE, os.pardir, "mag_cal.json"))
    ap.add_argument("--orientation", choices=list(ORIENTATIONS), default="none", help="US 표시 방향 (저장은 항상 원시)")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "..", "logs"))
    ap.add_argument("--zero", type=float, default=3.0, metavar="SEC", help="영점 캘리브레이션 정지 시간")
    ap.add_argument("--fps", type=float, default=10.0, help="tick 주기 (초음파 패널 blit 갱신). 기본 10")
    ap.add_argument("--plot-every", type=int, default=10,
                    help="몇 tick 마다 IMU 6 패널을 전체 재그리기할지 (기본 10 → 10 fps 에서 1 Hz). 전체 재그리기는 ~240 ms")
    ap.add_argument("--window-title", default=DEFAULT_WINDOW_TITLE, help="캡처할 뷰어 창 제목 (부분 일치)")
    ap.add_argument("--capture-hz", type=float, default=10.0,
                    help="화면 캡처 폴링 주기. PrintWindow 는 뷰어에 렌더를 시키므로 프로브 fps(~13) 보다 크게 잡을 이유가 없다")
    ap.add_argument("--frame-size", type=int, default=DEFAULT_FRAME_SIZE, help="저장 프레임 한 변 (px)")
    ap.add_argument("--roi-file", default=None, help="ROI(분율) JSON. 기본 <out-dir>/capture_roi.json")
    ap.add_argument("--no-dedupe", action="store_true", help="중복 프레임도 저장")
    ap.add_argument("--capture-backend", choices=["auto", "printwindow", "mss", "pil"], default="auto",
                    help="auto: PrintWindow(가려져도 됨) 우선, 검은 화면이면 화면 영역 캡처로 물러남")
    ap.add_argument("--record-frames", type=int, default=1000,
                    help="이 수의 US 프레임이 저장되면 녹화를 자동 정지 (0 = 수동). 기본 1000")
    ap.add_argument("--viewer-exe", default=DEFAULT_VIEWER_EXE, help="뷰어 실행 파일 (없으면 실행한다)")
    ap.add_argument("--viewer-visible", action="store_true", help="뷰어 창을 화면에 둔다 (기본: 화면 밖으로 숨김)")
    ap.add_argument("--no-viewer-control", action="store_true", help="뷰어 실행/숨김/조작을 하지 않는다")
    ap.add_argument("--select-roi", action="store_true", help="ROI 만 지정해 저장하고 종료")
    args = ap.parse_args()
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    roi_path = args.roi_file or os.path.join(args.out_dir, "capture_roi.json")

    use_korean_font()

    # ROI
    if os.path.isfile(roi_path):
        roi = CaptureRoi.load(roi_path)
        print("ROI 파일 사용: %s  %s" % (roi_path, roi))
    else:
        roi = DEFAULT_ROI
        print("ROI 기본값 (뷰어 최대화 레이아웃 기준) — 창 크기가 다르면 S 로 다시 잡으십시오: %s" % roi)
    if args.select_roi:
        gray = grab_client_gray(args.window_title)
        new = select_roi_interactive(gray, current=roi)
        if new is None:
            print("취소됨"); return 1
        new.save(roi_path)
        print("ROI 저장: %s  %s" % (roi_path, new))
        return 0

    # IMU
    port = args.port or autodetect_imu_port()
    if port is None:
        print("IMU COM 포트를 찾지 못했습니다 (VID 2886). --port COM3 처럼 지정하십시오.")
        return 1
    args.port = port
    cal = mag_calib.load(args.mag_cal) if os.path.exists(args.mag_cal) else None
    stream = ImuStream(port=port, baud=args.baud, cal=cal, beta=args.beta, use_mag=not args.no_mag, logger=None)
    stream.start()

    # 뷰어: 없으면 실행, 기본으로 화면 밖에 숨김, 상태 폴링
    viewer = poller = None
    if not args.no_viewer_control:
        try:
            hwnd = ViewerControl.launch(args.viewer_exe, title=args.window_title)
            if hwnd is None:
                print("뷰어 창이 뜨지 않았습니다 — 수동으로 실행하십시오 (%s)" % args.viewer_exe)
            viewer = ViewerControl(args.window_title)
            if viewer.attached:
                viewer.close_dialogs()
                if not args.viewer_visible:
                    viewer.hide()
                st = viewer.state()
                print("뷰어: %s %s %s 프로브 %s (창 %s)" % (st.get("state"), st.get("count"), st.get("depth"),
                                                       st.get("probe") or "없음", "숨김" if viewer.hidden else "보임"))
            poller = ViewerPoller(viewer, period_s=2.0, slow_period_s=10.0)
            poller.start()
        except Exception as exc:  # noqa: BLE001
            print("뷰어 제어 초기화 실패 (%s) — 캡처만 계속" % exc)
            viewer = poller = None

    # US (화면 캡처)
    us = ScreenUsReceiver(title=args.window_title, roi=roi, capture_hz=args.capture_hz,
                          frame_size=args.frame_size, dedupe=not args.no_dedupe, backend=args.capture_backend)
    us.start()

    print("IMU %s @ %d   US 화면 캡처 '%s'   저장 %s" % (port, args.baud, args.window_title, args.out_dir))
    print("키: [Z] 영점  [R] 녹화 시작/정지(%s)  [F] 뷰어 Freeze/Live  [V] 뷰어 창  [S] ROI  [W] 창 다시 찾기  [C] 자세 재설정  [Q] 종료"
          % ("%d 프레임 자동 정지" % args.record_frames if args.record_frames else "수동"))

    gui = WinGui(stream, us, args, roi_path, viewer=viewer, poller=poller)
    timer = gui.fig.canvas.new_timer(interval=int(1000 / args.fps))
    timer.add_callback(gui.tick)
    timer.start()
    try:
        plt.show()
    finally:
        timer.stop()
        if stream.logger is not None or us.recording:
            gui.toggle_record()          # 창을 그냥 닫아도 열린 로그를 안전하게 닫는다
        stream.stop()
        us.stop()
        if poller is not None:
            poller.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
