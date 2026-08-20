#!/usr/bin/env python3
"""실시간 GUI 뷰어 — 9축 원시값, 호스트 퓨전, 칩 퓨전을 한 화면에서 본다.

    python3 imu_gui.py                    # 보기만
    python3 imu_gui.py --log ../logs      # 시작과 동시에 로깅
    python3 imu_gui.py --no-mag           # 자력계 없이 6축 퓨전으로 비교

단축키
    [R] 로깅 시작/중지     [C] 자세 재설정(seed+정렬 다시 잡기)     [Q] 종료

시리얼 수신·퓨전은 별도 스레드(ImuStream)가 돌리고 여기서는 그리기만 한다.
그려야 할 게 밀려도 데이터는 놓치지 않는다.
"""

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import matplotlib                                    # noqa: E402
if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    # 이 PC 의 X 소켓을 찾아 붙는다 — 에이전트/서비스 셸에는 DISPLAY 가 없다.
    for sock in sorted(os.listdir("/tmp/.X11-unix")) if os.path.isdir("/tmp/.X11-unix") else []:
        os.environ["DISPLAY"] = ":" + sock.lstrip("X")
        break
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt                      # noqa: E402


def use_korean_font():
    """한글 글리프가 있는 폰트를 고른다. 없으면 라벨이 전부 두부(_)로 나온다.

    Noto Sans CJK 는 지역 변종(JP/KR/SC/TC)이 모두 같은 글리프 집합을 담고 있어
    JP 판만 깔려 있어도 한글이 정상으로 나온다.
    """
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic",
                 "Noto Sans KR", "Malgun Gothic", "UnDotum"):
        if name in have:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False   # 이 폰트들엔 U+2212 가 없다
            return name
    return None

import numpy as np                                   # noqa: E402
from matplotlib.animation import FuncAnimation       # noqa: E402

import mag_calib                                     # noqa: E402
from fusion import quat_to_matrix                    # noqa: E402
from imu_stream import ImuStream                     # noqa: E402
from imu_log import SessionLogger                    # noqa: E402
from imu_fusion_view import run_zero_calibration     # noqa: E402
from umi_protocol import REC_ACCEL, REC_GYRO, REC_MAG, REC_RV  # noqa: E402

# 축 색: X/Y/Z 를 화면 어디서나 같은 색으로 읽도록 통일한다.
AXIS_COLORS = ("#d1495b", "#3f8f5b", "#2a6f97")
AXIS_LABELS = ("x", "y", "z")


class OrientationView:
    """센서 축 3개를 3D 로 그린다. 호스트는 실선, 칩은 점선."""

    def __init__(self, ax):
        self.ax = ax
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("N"); ax.set_ylabel("W"); ax.set_zlabel("U")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title("자세 (지구 NWU 프레임)", fontsize=10)

        self.host = [ax.plot([0, 0], [0, 0], [0, 0], color=c, lw=3)[0] for c in AXIS_COLORS]
        self.chip = [ax.plot([0, 0], [0, 0], [0, 0], color=c, lw=1.5, ls="--", alpha=0.7)[0]
                     for c in AXIS_COLORS]

    def artists(self):
        return self.host + self.chip

    def update(self, host_q, chip_aligned_q):
        for lines, q in ((self.host, host_q), (self.chip, chip_aligned_q)):
            if q is None:
                for ln in lines:
                    ln.set_data([], []); ln.set_3d_properties([])
                continue
            R = quat_to_matrix(q)      # 센서 -> 지구: 열이 센서 축의 지구 좌표
            for i, ln in enumerate(lines):
                v = R[:, i]
                ln.set_data([0, v[0]], [0, v[1]])
                ln.set_3d_properties([0, v[2]])


class TimeSeries:
    """공통 롤링 시계열 패널."""

    def __init__(self, ax, keys, title, ylabel, colors=AXIS_COLORS, labels=AXIS_LABELS,
                 styles=None, min_span=None, ylim=None):
        self.ax = ax
        self.keys = keys
        self.min_span = min_span      # 데이터가 평평할 때 축이 노이즈까지 확대되는 걸 막는다
        self.fixed_ylim = ylim
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        styles = styles or ["-"] * len(keys)
        self.lines = [ax.plot([], [], color=colors[i % len(colors)], lw=1.2,
                              ls=styles[i], label=labels[i % len(labels)])[0]
                      for i in range(len(keys))]
        ax.legend(loc="upper left", fontsize=7, ncol=len(keys), framealpha=0.6)

    def artists(self):
        return self.lines

    def update(self, hist, t0):
        t = hist["t"]
        if len(t) < 2:
            return
        x = t - t0
        lo, hi = np.inf, -np.inf
        for ln, k in zip(self.lines, self.keys):
            y = hist[k]
            ln.set_data(x, y)
            if len(y) and np.isfinite(y).any():
                lo = min(lo, np.nanmin(y)); hi = max(hi, np.nanmax(y))
        self.ax.set_xlim(x[0], max(x[-1], x[0] + 1e-3))
        if self.fixed_ylim is not None:
            return
        if np.isfinite(lo) and np.isfinite(hi):
            if self.min_span and (hi - lo) < self.min_span:
                mid = 0.5 * (lo + hi)
                lo, hi = mid - self.min_span / 2, mid + self.min_span / 2
            pad = max(0.05 * (hi - lo), 1e-3)
            self.ax.set_ylim(lo - pad, hi + pad)


class Gui:
    def __init__(self, stream, args):
        self.stream = stream
        self.args = args
        self.t0 = None

        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title("imu_bench — BNO085 9축 + 퓨전")
        gs = self.fig.add_gridspec(3, 2, width_ratios=(1.0, 1.35),
                                   hspace=0.45, wspace=0.2,
                                   left=0.06, right=0.98, top=0.90, bottom=0.06)

        self.orient = OrientationView(self.fig.add_subplot(gs[0, 0], projection="3d"))
        self.euler = TimeSeries(
            self.fig.add_subplot(gs[0, 1]),
            ["hr", "hp", "hy", "cr", "cp", "cy"],
            "오일러각 — 실선=호스트 Madgwick, 점선=칩 BNO085", "deg",
            colors=AXIS_COLORS * 2, labels=("roll", "pitch", "yaw") * 2,
            styles=["-", "-", "-", "--", "--", "--"], ylim=(-185, 185))
        self.euler.ax.set_yticks([-180, -90, 0, 90, 180])
        self.accel = TimeSeries(self.fig.add_subplot(gs[1, 0]),
                                ["ax", "ay", "az"], "가속도", "m/s²", min_span=2.0)
        self.gyro = TimeSeries(self.fig.add_subplot(gs[1, 1]),
                               ["gx", "gy", "gz"], "자이로", "rad/s", min_span=0.2)
        self.mag = TimeSeries(self.fig.add_subplot(gs[2, 0]),
                              ["mx", "my", "mz"], "자력계", "µT", min_span=10.0)
        self.diff = TimeSeries(self.fig.add_subplot(gs[2, 1]), ["diff"],
                               "호스트 vs 칩 각도차 (프레임 정렬 후)", "deg",
                               colors=("#8a5cf6",), labels=("Δ",), min_span=2.0)
        self.diff.ax.set_xlabel("경과 시간 [s]", fontsize=9)
        self.mag.ax.set_xlabel("경과 시간 [s]", fontsize=9)

        self.status = self.fig.text(0.06, 0.955, "", fontsize=10, va="top")
        self.hint = self.fig.text(0.98, 0.955,
                                  "[R] 로깅  [Z] 영점  [C] 자세 재설정  [Q] 종료",
                                  fontsize=9, ha="right", va="top", alpha=0.7)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    # ------------------------------------------------------------------- 입력
    def on_key(self, event):
        k = (event.key or "").lower()
        if k == "q":
            plt.close(self.fig)
        elif k == "c":
            self.stream.reset_align()
        elif k == "r":
            self.toggle_log()
        elif k == "z":
            self.zero_now()

    def toggle_log(self):
        if self.stream.logger is not None:
            path = self.stream.logger.close()
            n = self.stream.logger.n
            self.stream.logger = None
            print("로깅 중지: %s (%d 샘플)" % (path, n))
        else:
            self.stream.logger = self.new_logger()
            print("로깅 시작: %s" % self.stream.logger.path)

    def zero_now(self):
        """[Z] — 영점을 다시 잡는다. 수집 동안 화면이 멈추므로 상태줄로 알린다."""
        self.status.set_text("영점 캘리브레이션 %.1f초 — 센서를 움직이지 마세요..."
                             % self.args.zero)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        run_zero_calibration(self.stream, self.args.zero, retries=0)

    def new_logger(self):
        z = self.stream.zero
        return SessionLogger(self.args.log_dir, fmt=self.args.log_format,
                             meta={"port": self.args.port, "board": self.stream.board_name,
                                   "beta": self.args.beta, "use_mag": not self.args.no_mag,
                                   "mag_cal": self.stream.cal,
                                   "zero_ref": z.to_dict() if z else None})

    # -------------------------------------------------------------------- 그리기
    def artists(self):
        return (self.orient.artists() + self.euler.artists() + self.accel.artists()
                + self.gyro.artists() + self.mag.artists() + self.diff.artists())

    def update(self, _frame):
        s = self.stream.snapshot()
        if s["error"]:
            self.status.set_text("오류: %s" % s["error"])
            return self.artists()
        hist = s["hist"]
        if len(hist["t"]) < 2:
            self.status.set_text("데이터 대기 중...  board=%s" % (s["board"] or "?"))
            return self.artists()
        if self.t0 is None:
            self.t0 = hist["t"][0]

        self.orient.update(s["host_q"], s["aligned_q"])
        for panel in (self.euler, self.accel, self.gyro, self.mag, self.diff):
            panel.update(hist, self.t0)

        r = s["rates"]
        mode = ("9축 MARG" if (s["saw_mag"] and not self.args.no_mag) else "6축 IMU-only")
        cal_state = "보정됨" if self.stream.cal else "미보정"
        log_state = ("● 기록 중 %d" % s["log_n"]) if self.stream.logger else "○ 미기록"
        z = s.get("zero")
        zero_state = "영점 OK" if (z and z.quality.get("still")) else (
            "영점 불량" if z else "영점 없음")
        diff_txt = "—" if s["diff_deg"] is None else "%.2f°" % s["diff_deg"]
        self.status.set_text(
            "%s [%s]  board=%s   A%5.1f G%5.1f M%5.1f RV%5.1f Hz   "
            "crc_err=%d gaps=%d   Δ=%s   %s   %s%s"
            % (mode, cal_state, s["board"] or "?",
               r[REC_ACCEL], r[REC_GYRO], r[REC_MAG], r[REC_RV],
               s["crc_errors"], s["gaps"], diff_txt, zero_state, log_state,
               "" if s.get("rel_euler") is None else
               "\n영점 대비 상대회전  roll %+7.2f°  pitch %+7.2f°  yaw %+7.2f°"
               % s["rel_euler"]))
        return self.artists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--beta", type=float, default=0.05, help="Madgwick gain")
    ap.add_argument("--no-mag", action="store_true", help="자력계 무시(6축 IMU-only)")
    ap.add_argument("--mag-cal", default=os.path.join(_HERE, os.pardir, "mag_cal.json"))
    ap.add_argument("--history", type=float, default=30.0, help="그래프에 보여줄 시간 [s]")
    ap.add_argument("--fps", type=float, default=20.0)
    ap.add_argument("--log", dest="log_dir", nargs="?",
                    const=os.path.join(_HERE, os.pardir, "logs"),
                    help="시작과 동시에 로깅한다 (경로 생략 시 imu_bench/logs)")
    ap.add_argument("--log-format", choices=("csv", "hdf5"), default="csv")
    ap.add_argument("--zero", type=float, default=3.0, metavar="SEC",
                    help="시작 시 영점 캘리브레이션 시간 (0 = 건너뜀). [Z] 로 다시 잡는다")
    ap.add_argument("--snapshot", metavar="PNG",
                    help="창을 띄우는 대신 데이터를 모아 한 프레임을 PNG 로 저장한다 "
                         "(레이아웃/폰트 확인용, X 서버 불필요)")
    ap.add_argument("--snapshot-after", type=float, default=6.0, metavar="SEC")
    args = ap.parse_args()

    if args.snapshot:
        matplotlib.use("Agg", force=True)
    if args.log_dir is None:
        args.log_dir = os.path.join(_HERE, os.pardir, "logs")

    font = use_korean_font()
    if font is None:
        print("경고: 한글 폰트를 찾지 못했습니다 - 라벨이 깨져 보일 수 있습니다.")

    cal = None if args.no_mag else mag_calib.load(args.mag_cal)
    stream = ImuStream(port=args.port, baud=args.baud, cal=cal, beta=args.beta,
                       use_mag=not args.no_mag, history_s=args.history)
    stream.start()

    gui = Gui(stream, args)

    if args.zero > 0:
        # 스트림이 살아나기를 기다린 뒤 영점을 잡는다 (데이터가 있어야 잡힌다)
        t_wait = time.time() + 5.0
        while time.time() < t_wait and stream.n_fuse < 50:
            if stream.error:
                print(stream.error)
                return
            time.sleep(0.1)
        run_zero_calibration(stream, args.zero)

    if "--log" in sys.argv:
        stream.logger = gui.new_logger()
        print("로깅 시작: %s" % stream.logger.path)

    if args.snapshot:
        deadline = time.time() + args.snapshot_after
        while time.time() < deadline:
            time.sleep(0.2)
        gui.update(0)
        gui.fig.savefig(args.snapshot, dpi=110)
        stream.stop()
        print("스냅샷 저장: %s (퓨전 %d 스텝)" % (args.snapshot, stream.n_fuse))
        return

    anim = FuncAnimation(gui.fig, gui.update, interval=1000.0 / args.fps,
                         blit=False, cache_frame_data=False)
    gui._anim = anim          # GC 방지
    try:
        plt.show()
    finally:
        stream.stop()
        if stream.logger is not None:
            print("로깅 저장: %s (%d 샘플)" % (stream.logger.close(), stream.logger.n))


if __name__ == "__main__":
    main()
