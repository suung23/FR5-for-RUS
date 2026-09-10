#!/usr/bin/env python3
"""팬텀 강성 측정 + 힘·초음파 동시 모니터 (리눅스).

한 창에서 **PX6D 접촉력**과 **초음파 B-mode** 를 같이 보며, 단계별로 압입 깊이와
정착 힘을 찍어 강성 k = dF/dδ 를 그 자리에서 적합한다.

    source ~/FR5-for-RUS/install/setup.bash          # 자세(FK)를 쓸 때만 필요
    python3 phantom_stiffness/stiffness_gui.py --label phantom_b
    python3 phantom_stiffness/stiffness_gui.py --label phantom_b --no-us      # 힘만
    python3 phantom_stiffness/stiffness_gui.py --us-host 192.168.1.1 --probe c10ur

세 입력은 서로 독립이다 — **하나가 없어도 나머지는 돈다.** 프로브 전원이 꺼져 있으면
초음파 패널만 비고, 제어 스택이 없으면 깊이를 손으로 넣는다 (`[` / `]`).

키
--
    space  강성 점 기록 — 최근 `--settle` 초의 힘을 평균 내 지금 깊이와 짝지어 남긴다
    u      직전 점 취소
    c      **접촉 기준** — 지금 자세를 깊이 0 으로 잡는다. 프로브가 팬텀에 막 닿았을 때
    z      힘 소프트 영점 (센서 기준은 건드리지 않는다) · Z 영점 해제
    [ ]    수동 깊이를 `--manual-step` 만큼 빼기 / 더하기 (자세 토픽이 없을 때)
    p      적재 ↔ 제하 전환. 왕복하면 이력이 보인다
    f      초음파 스캔 시작/정지 (C10UR 은 클라이언트가 명령한다)
    s      지금까지를 저장 (종료해도 자동 저장한다)
    q      종료

⚠️ **이것은 로봇을 움직이지 않는다.** 압입은 조작자가 기존 경로로 준다 — teleop 지령,
펜던트 조그, 또는 `ros2 param set /us_diff_ik_node contact_control.target_force_n <N>`.
힘 한계와 후퇴는 그대로 제어 스택(`us_diff_ik_node`) 이 관리한다. 여기서 문턱을 새로
만들지 않는다. 화면의 빨간 선은 보기용 경고선일 뿐 아무것도 멈추지 않는다.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from sources import Px6dReader, UsReader, PoseReader          # noqa: E402
from stiffness import StiffnessPoint, StiffnessRun            # noqa: E402
from korean_font import use_korean_font                        # noqa: E402

import matplotlib                                              # noqa: E402


def _pick_backend() -> str:
    """실제로 임포트되는 대화형 백엔드를 고른다.

    ``matplotlib.use`` 는 지연 임포트라 바인딩이 없어도 그 자리에서는 통과한다 —
    창을 띄우는 순간에야 죽는다. 그래서 여기서 한 번 진짜로 불러 본다.
    이 PC 의 venv 에는 Qt 바인딩이 없고 tkinter 만 있다 (2026-09-10).
    """
    forced = os.environ.get("MPLBACKEND")
    if forced:
        return forced
    for backend, module in (("QtAgg", "PyQt6"), ("TkAgg", "tkinter")):
        try:
            __import__(module)
            return backend
        except ImportError:
            continue
    return "Agg"          # 화면이 없으면 그림만 그린다 — 창은 안 뜬다


matplotlib.use(_pick_backend())
import matplotlib.pyplot as plt                                # noqa: E402
from matplotlib.animation import FuncAnimation                 # noqa: E402
from matplotlib.gridspec import GridSpec                       # noqa: E402

AXIS_NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


class StiffnessApp:
    """창 하나 — 초음파, 힘 시계열, 강성 곡선, 상태줄."""

    def __init__(self, args) -> None:
        self.args = args
        self.force = Px6dReader(args.force_port, rate_hz=args.force_rate,
                                history_s=max(30.0, args.settle * 4))
        self.us = None
        self.pose = None
        self.manual_depth_mm = 0.0
        self.phase = "load"
        self.saved_path: str | None = None
        self.message = ""
        self.message_until = 0.0

        # US 프레임은 극좌표 원본으로 모은다 (수집기와 같은 레이아웃).
        self.us_frames: list[np.ndarray] = []
        self.us_frame_t: list[float] = []
        self._converter = None

        self.run = StiffnessRun(args.label, args.out_dir, meta={
            "purpose": "phantom stiffness by stepwise indentation (PX6D)",
            "force": {"port": args.force_port, "rate_hz": args.force_rate,
                      "axis": AXIS_NAMES[args.force_axis], "sign": args.force_sign,
                      "settle_s": args.settle,
                      "axis_caveat": "PX6D AXIS_ORDER 는 매뉴얼 §5.3/§5.4 불일치로 잠정값이다"},
            "depth": {"source": "ros_fk" if not args.no_pose else "manual",
                      "convention": "probe z axis projection, positive = into tissue",
                      "manual_step_mm": args.manual_step},
        })

    # -- 파생값 -------------------------------------------------------------
    def contact_force(self, w: np.ndarray | None) -> float:
        """부호를 맞춘 접촉력 [N]. 양수 = 압축."""
        if w is None or (hasattr(w, "size") and w.size == 0):
            return float("nan")
        return float(self.args.force_sign * np.asarray(w)[..., self.args.force_axis])

    def depth_mm(self) -> float | None:
        if self.pose is not None:
            d = self.pose.depth_mm()
            if d is not None:
                return d
        return self.manual_depth_mm

    def depth_source(self) -> str:
        if self.pose is not None and self.pose.depth_mm() is not None:
            return "FK"
        return "수동"

    def notify(self, text: str, seconds: float = 4.0) -> None:
        self.message = text
        self.message_until = time.time() + seconds

    # -- 동작 ---------------------------------------------------------------
    def record_point(self) -> None:
        """정착 창의 힘을 평균 내 점 하나를 남긴다."""
        w = self.force.window(self.args.settle)
        if w.shape[0] < 3:
            self.notify("힘 표본이 모자란다 — 센서가 붙어 있는지 확인하십시오")
            return
        f = self.args.force_sign * w[:, self.args.force_axis]
        half = max(1, len(f) // 2)
        depth = self.depth_mm()
        if depth is None:
            self.notify("깊이가 없다 — c 로 접촉 기준을 잡거나 [ ] 로 수동 입력하십시오")
            return

        us_seq = -1
        if self.us is not None:
            frame, t_frame = self.us.current()
            if frame is not None:
                us_seq = len(self.us_frames)
                self.us_frames.append(frame)
                self.us_frame_t.append(t_frame)

        snap = self.pose.snapshot() if self.pose is not None else {}
        p = self.run.add(StiffnessPoint(
            index=-1, t_pc=time.time(), depth_mm=float(depth),
            force_n=float(np.mean(f)), force_sd=float(np.std(f)), samples=int(f.size),
            relax_n=float(np.mean(f[:half]) - np.mean(f[half:])),
            phase=self.phase, wrench=np.mean(w, axis=0).tolist(),
            pose_xyz=snap.get("pos") or [], pose_quat=snap.get("quat") or [],
            us_seq=us_seq,
        ))
        fit = self.run.fit(self.phase)
        k = f"k={fit['k_n_per_mm']:.4f} N/mm  R²={fit['r_squared']:.4f}" if fit.get("ok") else "적합에 점이 더 필요하다"
        self.notify(f"점 {p.index}: δ={p.depth_mm:+.3f} mm  F={p.force_n:.3f}±{p.force_sd:.3f} N  ({k})")

    def undo_point(self) -> None:
        p = self.run.undo()
        if p is None:
            self.notify("취소할 점이 없다")
            return
        # US 프레임은 마지막 점의 것일 때만 되돌린다 (인덱스가 어긋나지 않도록).
        if p.us_seq >= 0 and p.us_seq == len(self.us_frames) - 1:
            self.us_frames.pop()
            self.us_frame_t.pop()
        self.notify(f"점 {p.index} 취소")

    def save(self) -> None:
        if not self.run.points:
            self.notify("저장할 점이 없다")
            return
        extra = {"us": self._save_us_frames()}
        self.saved_path = self.run.save(extra)
        fit = self.run.fit("load")
        k = f"{fit['k_n_per_mm']:.4f} N/mm" if fit.get("ok") else "적합 불가"
        self.notify(f"저장: {self.saved_path}  (k={k})", 8.0)
        print(f"\n저장 → {self.saved_path}\n  적재 구간 적합: {fit}")

    def _save_us_frames(self) -> dict:
        """점마다 한 장씩 모은 극좌표 원본을 수집기와 같은 레이아웃으로 쓴다."""
        if not self.us_frames:
            return {"frames": 0}
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.run.started))
        path = os.path.join(self.args.out_dir, f"stiff_{self.args.label}_{stamp}")
        os.makedirs(path, exist_ok=True)
        shape = self.us_frames[0].shape
        with open(os.path.join(path, "us_frames.bin"), "wb") as fh:
            for fr in self.us_frames:
                fh.write(fr.tobytes())
        with open(os.path.join(path, "us_index.csv"), "w") as fh:
            fh.write("pc_unix,us_seq\n")
            for i, t in enumerate(self.us_frame_t):
                fh.write(f"{t:.6f},{i}\n")
        return {
            "frames": len(self.us_frames), "frame_shape": list(shape), "dtype": "uint8",
            "bin": "us_frames.bin", "index": "us_index.csv",
            "host": self.args.us_host, "probe": self.args.probe,
            "note": "점마다 한 장. candidate 극좌표 원본, scan conversion 이전. 무변환 저장",
            "latency_s": 0.2,
            "latency_note": "2026-09-10 사후 추정 유효 지연. 여기서 보정하지 않았다",
        }

    # -- 키 ---------------------------------------------------------------
    def on_key(self, event) -> None:
        k = event.key
        if k == " ":
            self.record_point()
        elif k == "u":
            self.undo_point()
        elif k == "c":
            if self.pose is not None and self.pose.set_reference():
                self.notify("접촉 기준을 잡았다 — 깊이 0 (FK)")
            else:
                self.manual_depth_mm = 0.0
                self.notify("자세 토픽이 없어 수동 깊이를 0 으로 잡았다")
        elif k == "z":
            self.notify("힘 영점" if self.force.soft_zero(0.5) else "영점에 쓸 표본이 없다")
        elif k == "Z":
            self.force.clear_zero()
            self.notify("힘 영점 해제")
        elif k == "]":
            self.manual_depth_mm += self.args.manual_step
            self.notify(f"수동 깊이 {self.manual_depth_mm:+.3f} mm")
        elif k == "[":
            self.manual_depth_mm -= self.args.manual_step
            self.notify(f"수동 깊이 {self.manual_depth_mm:+.3f} mm")
        elif k == "p":
            self.phase = "unload" if self.phase == "load" else "load"
            self.notify(f"구간: {'제하 (unload)' if self.phase == 'unload' else '적재 (load)'}")
        elif k == "f":
            if self.us is None:
                self.notify("초음파가 꺼져 있다")
            elif not self.us.profile.control_start:
                self.notify(f"{self.args.probe}: 스캔은 프로브 버튼으로 시작한다")
            else:
                self.notify("스캔 시작 요청" if self.us.toggle_scan() else "스캔 정지 요청")
        elif k == "s":
            self.save()
        elif k in ("q", "escape"):
            plt.close(self.fig)

    # -- 화면 ---------------------------------------------------------------
    def build(self) -> None:
        use_korean_font()
        self.fig = plt.figure(figsize=(15.5, 8.6))
        self.fig.canvas.manager.set_window_title(f"팬텀 강성 · 힘/초음파 모니터 — {self.args.label}")
        gs = GridSpec(2, 2, figure=self.fig, width_ratios=[1.15, 1.0],
                      left=0.045, right=0.985, top=0.94, bottom=0.10, wspace=0.20, hspace=0.30)

        self.ax_us = self.fig.add_subplot(gs[:, 0])
        self.ax_us.set_title("초음파 B-mode (candidate 부채꼴)")
        self.ax_us.set_xticks([]); self.ax_us.set_yticks([])
        self.im = self.ax_us.imshow(np.zeros((512, 591), np.uint8), cmap="gray", vmin=0, vmax=255)
        self.us_note = self.ax_us.text(0.5, 0.5, "초음파 없음", transform=self.ax_us.transAxes,
                                       ha="center", va="center", color="0.6", fontsize=13)

        self.ax_force = self.fig.add_subplot(gs[0, 1])
        self.ax_force.set_title("접촉력 (정착 창 회색)")
        self.ax_force.set_xlabel("시간 [s]"); self.ax_force.set_ylabel("F [N]")
        self.ax_force.grid(alpha=0.3)
        (self.ln_force,) = self.ax_force.plot([], [], lw=1.4, color="#1f77b4")
        self.settle_band = self.ax_force.axvspan(-self.args.settle, 0, color="0.85", zorder=0)
        self.warn_line = self.ax_force.axhline(self.args.warn_force, color="#d62728", ls="--", lw=1.0)

        self.ax_k = self.fig.add_subplot(gs[1, 1])
        self.ax_k.set_title("강성 곡선  F(δ)")
        self.ax_k.set_xlabel("압입 깊이 δ [mm]"); self.ax_k.set_ylabel("F [N]")
        self.ax_k.grid(alpha=0.3)
        (self.pt_load,) = self.ax_k.plot([], [], "o", ms=6, color="#1f77b4", label="적재")
        (self.pt_unload,) = self.ax_k.plot([], [], "s", ms=6, mfc="none", color="#ff7f0e", label="제하")
        (self.ln_fit,) = self.ax_k.plot([], [], "-", lw=1.5, color="#2ca02c", label="적합")
        self.ax_k.legend(loc="upper left", fontsize=8, framealpha=0.9)
        self.k_text = self.ax_k.text(0.98, 0.04, "", transform=self.ax_k.transAxes,
                                     ha="right", va="bottom", fontsize=10, family="monospace")

        self.status = self.fig.text(0.045, 0.035, "", fontsize=9.5, family="monospace", va="bottom")
        self.msg = self.fig.text(0.045, 0.005, "", fontsize=9.5, color="#b8860b", va="bottom")
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def _fan(self, polar: np.ndarray) -> np.ndarray:
        """극좌표를 화면용 부채꼴로. 변환 테이블은 한 번만 만든다."""
        if self._converter is None:
            from us_scan_convert import FanGeometry, ScanConverter
            geo = FanGeometry(radius_mm=self.args.fan_radius, half_angle_deg=self.args.fan_half_angle,
                              depth_mm=self.args.fan_depth, flip_lines=self.args.fan_flip)
            self._converter = ScanConverter(geo, polar.shape[0], polar.shape[1])
        return self._converter.convert(polar)

    def update(self, _frame) -> tuple:
        # -- 초음파
        if self.us is not None:
            polar, _t = self.us.current()
            if polar is not None:
                img = self._fan(polar) if self.args.display == "fan" else polar
                self.im.set_data(img)
                self.im.set_extent((-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5))
                self.ax_us.set_xlim(-0.5, img.shape[1] - 0.5)
                self.ax_us.set_ylim(img.shape[0] - 0.5, -0.5)
                self.us_note.set_text("")
            elif self.us.error:
                self.us_note.set_text(f"초음파 오류\n{self.us.error}")
            elif not self.us.scanning:
                self.us_note.set_text("스캔 대기 — 창을 클릭하고 f")

        # -- 힘 시계열
        t, w = self.force.history()
        if t.size:
            rel = t - t[-1]
            keep = rel >= -self.args.plot_seconds
            f = self.args.force_sign * w[keep, self.args.force_axis]
            self.ln_force.set_data(rel[keep], f)
            self.ax_force.set_xlim(-self.args.plot_seconds, 0.5)
            lo, hi = float(np.min(f)), float(np.max(f))
            pad = max(0.25, 0.15 * (hi - lo))
            self.ax_force.set_ylim(min(lo - pad, -0.3), max(hi + pad, self.args.warn_force * 1.1))

        # -- 강성 곡선
        dl, fl = self.run.arrays("load")
        du, fu = self.run.arrays("unload")
        self.pt_load.set_data(dl, fl)
        self.pt_unload.set_data(du, fu)
        fit = self.run.fit("load")
        if fit.get("ok"):
            xs = np.linspace(float(np.min(dl)), float(np.max(dl)), 2)
            self.ln_fit.set_data(xs, fit["k_n_per_mm"] * xs + fit["intercept_n"])
            self.k_text.set_text(f"k = {fit['k_n_per_mm']:.4f} N/mm\n"
                                 f"R² = {fit['r_squared']:.4f}   n = {fit['points']}")
        else:
            self.ln_fit.set_data([], [])
            self.k_text.set_text(f"점 {len(self.run.points)} 개 — 2 개 이상 필요")
        if dl.size or du.size:
            self.ax_k.relim(); self.ax_k.autoscale_view()

        # -- 상태줄
        cur = self.force.current()
        fn = self.contact_force(cur)
        depth = self.depth_mm()
        parts = [
            f"F={fn:+7.3f} N" if np.isfinite(fn) else "F=   —   ",
            f"δ={depth:+7.3f} mm [{self.depth_source()}]" if depth is not None else "δ=   —   ",
            f"힘 {self.force.rate:5.1f} Hz" + (f" fw {self.force.version}" if self.force.version else ""),
        ]
        if self.us is not None:
            parts.append(f"US {self.us.rate:4.1f} fps {'스캔중' if self.us.scanning else '대기'}")
        if self.pose is not None and self.pose.available:
            parts.append(f"자세 {self.pose.messages}")
        parts.append(f"구간 {self.phase}")
        parts.append(f"점 {len(self.run.points)}")
        if self.force.error:
            parts.append(f"⚠ 힘: {self.force.error}")
        self.status.set_text("   ".join(parts))
        self.msg.set_text(self.message if time.time() < self.message_until else
                          "space 기록 · u 취소 · c 접촉기준 · z 영점 · [ ] 수동깊이 · p 적재/제하 · f 스캔 · s 저장 · q 종료")
        return ()

    # -- 실행 ---------------------------------------------------------------
    def start(self) -> int:
        a = self.args
        self.force.start()
        if not a.no_us:
            try:
                self.us = UsReader(a.us_host, a.probe)
                self.us.start()
            except Exception as exc:  # noqa: BLE001
                print(f"초음파를 열 수 없다: {exc} — 힘만으로 계속한다", file=sys.stderr)
                self.us = None
        if not a.no_pose:
            self.pose = PoseReader(a.namespace, a.wrench_topic)
            self.pose.start()

        time.sleep(1.2)     # 센서 핸드셰이크가 끝난 뒤 상태를 보여 준다
        if self.force.error:
            print(f"⚠ 힘 센서: {self.force.error}", file=sys.stderr)
            if "dialout" not in (self.force.error or "") and "Permission" in (self.force.error or ""):
                print('   sg dialout -c "…" 로 감싸십시오.', file=sys.stderr)
        if self.pose is not None and self.pose.error:
            print(f"· 자세: {self.pose.error}", file=sys.stderr)

        self.build()
        self.anim = FuncAnimation(self.fig, self.update, interval=a.refresh_ms,
                                  blit=False, cache_frame_data=False)
        try:
            plt.show()
        finally:
            if self.run.points and self.saved_path is None:
                self.save()             # 종료해도 찍은 점을 잃지 않는다
            self.force.stop()
            if self.us is not None:
                self.us.stop()
            if self.pose is not None:
                self.pose.stop()
        return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="phantom", help="세션 이름. 폴더 이름이 된다")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "runs"))

    g = ap.add_argument_group("힘 (PX6D)")
    g.add_argument("--force-port", default="/dev/ttyACM0",
                   help="PX6D 시리얼. 이 셀에서는 %(default)s (IMU 는 ttyACM1)")
    g.add_argument("--force-rate", type=int, default=1000,
                   help="회신 주기 Hz 요청값, 4 의 배수. ⚠ 펌웨어 v1.0.2 는 이 명령을 무시하고 늘 ~1 kHz 로 보낸다")
    g.add_argument("--force-axis", type=int, default=2, choices=range(6),
                   help="접촉력으로 쓸 축 인덱스 (기본 2 = Fz). AXIS_ORDER 는 잠정값이다")
    g.add_argument("--force-sign", type=float, default=-1.0,
                   help="접촉력 부호. force_hold_validation 과 같은 기본 -1 (압축이 양수)")
    g.add_argument("--settle", type=float, default=1.0, help="점 하나의 정착 평균 창 [s]")
    g.add_argument("--warn-force", type=float, default=5.0, help="경고선 [N] — 보기용, 아무것도 멈추지 않는다")

    g = ap.add_argument_group("초음파")
    g.add_argument("--no-us", action="store_true", help="초음파 없이 힘만")
    g.add_argument("--us-host", default="192.168.1.1")
    g.add_argument("--probe", default="c10ur", help="프로브 프로파일 (기본 %(default)s)")
    g.add_argument("--display", default="fan", choices=("fan", "polar"))
    g.add_argument("--fan-radius", type=float, default=59.0)
    g.add_argument("--fan-half-angle", type=float, default=28.0)
    g.add_argument("--fan-depth", type=float, default=220.0)
    g.add_argument("--fan-flip", action="store_true", help="라인 좌우 반전 (검증 전 candidate)")

    g = ap.add_argument_group("깊이 (자세)")
    g.add_argument("--no-pose", action="store_true", help="ROS 자세를 쓰지 않고 수동 깊이만")
    g.add_argument("--namespace", default="/fr5_right")
    g.add_argument("--wrench-topic", default=None,
                   help="제어가 쓰는 렌치도 함께 볼 때 (예: /fr5_right/wrench_px6d)")
    g.add_argument("--manual-step", type=float, default=0.5, help="[ ] 한 번의 수동 깊이 [mm]")

    ap.add_argument("--plot-seconds", type=float, default=20.0, help="힘 그래프 창 [s]")
    ap.add_argument("--refresh-ms", type=int, default=100, help="화면 갱신 주기 [ms]")
    return ap.parse_args(argv if argv is not None else sys.argv[1:])


def main(argv=None) -> int:
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    return StiffnessApp(args).start()


if __name__ == "__main__":
    raise SystemExit(main())
