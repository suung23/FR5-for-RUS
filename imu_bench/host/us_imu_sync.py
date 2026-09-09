#!/usr/bin/env python3
"""저장이 끝난 세션의 IMU 와 초음파 시계열을 **한 시간축으로 동기화**한다 — 세션마다 sync.npz + sync_report.json.

    python imu_bench\\host\\us_imu_sync.py <session_dir>            # 한 세션
    python imu_bench\\host\\us_imu_sync.py --watch imu_bench\\logs   # 루프: 새로 저장된 세션을 찾아 처리 (GUI 가 내부에서도 돈다)

두 스트림은 수신 시각 `pc_unix` 로 이미 같은 시계에 찍혀 있다 (`us_imu_collect.py` docstring). 동기화란 그 위에서
**프레임마다 IMU 상태를 붙이는 것**이다:

    frame_t_pc[i]  −  us_latency_s                 → 프레임의 획득 시각 추정 (지연 보정, 설정값. 미측정이면 0)
    IMU 를 그 시각에 보간:  acc·gyr 선형, 쿼터니언 nlerp(최근접 두 행), 보정 상태·정지 판정은 최근접
    프레임 구간 [t_{i-1}, t_i] 에 속하는 IMU 행 범위 (imu_start[i], imu_stop[i])  → 원시 행을 직접 자를 때

산출 (세션 폴더):
    sync.npz          frame_t_pc, frame_t_acq, frame_id, frame_index, imu_row_nearest, imu_dt_ms, imu_start, imu_stop,
                      acc (N,3), gyr (N,3), quat_chip (N,4), quat_host (N,4), gravity_probe (N,3), still (N,), cal (N,4)
    sync_report.json  fps, IMU rate, 시계 적합, 프레임 간격 통계, 커버리지, 결손, 지연값, 판정
    sync_check.png    프레임 시각 위에 IMU 커버리지·정지 마스크 (있으면 matplotlib)

판정 규칙 (report["ok"]):  IMU 범위 밖 프레임 ≤ 1 %,  프레임 간격 > 3/f_us 인 결손 ≤ 2 %,  IMU 행 간격 > 50 ms 인 결손 ≤ 0.5 %
(IMU 행은 레코드 종류가 인터리브돼 4 ms 중앙값에 12 ms 꼬리가 정상이다 — 상대 규칙은 쓰지 않는다).

의존: policy_learning/rus_policy/session.py 의 로더 (같은 세션 포맷) 와 imu_labels.still_mask.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_POLICY = os.path.abspath(os.path.join(_HERE, "..", "..", "policy_learning"))
for _p in (_HERE, _POLICY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SYNC_NPZ = "sync.npz"
SYNC_REPORT = "sync_report.json"
SYNC_PNG = "sync_check.png"


def _nlerp(q0: np.ndarray, q1: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(n,4)×(n,4)×(n,) → (n,4). 부호 정렬 후 선형보간·정규화 (짧은 구간엔 slerp 와 사실상 같다)."""
    dot = np.sum(q0 * q1, axis=1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    q = (1.0 - w)[:, None] * q0 + w[:, None] * q1
    n = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(n > 0, n, 1.0)


def sync_session(session_dir: str | Path, us_latency_s: Optional[float] = None, write_png: bool = True,
                 still_cfg=None) -> dict:
    """세션 하나를 동기화해 sync.npz / sync_report.json 을 쓰고 report 를 돌려준다."""
    from rus_policy.config import ImuConfig
    from rus_policy.imu_labels import gravity_in_frame, sensor_to_earth_rotations, earth_convention, still_mask
    from rus_policy.session import load_session

    root = Path(session_dir)
    s = load_session(root)
    imu = s.imu
    cfg = still_cfg or ImuConfig()
    if us_latency_s is None:
        us_latency_s = float((s.meta.get("us") or {}).get("latency_s", 0.0) or 0.0)

    t_acq = s.frame_t_pc - us_latency_s
    t_imu = imu.t_pc
    n_f, n_i = t_acq.size, t_imu.size
    if n_f == 0 or n_i < 2:
        raise ValueError(f"{root.name}: 프레임 {n_f} / IMU {n_i} — 동기화할 것이 없습니다")

    # --- 프레임마다 IMU 보간 ---
    j = np.clip(np.searchsorted(t_imu, t_acq), 1, n_i - 1)       # t_imu[j-1] <= t < t_imu[j]
    t0, t1 = t_imu[j - 1], t_imu[j]
    w = np.clip((t_acq - t0) / np.maximum(t1 - t0, 1e-9), 0.0, 1.0)
    inside = (t_acq >= t_imu[0]) & (t_acq <= t_imu[-1])
    lin = lambda a: (1.0 - w)[:, None] * a[j - 1] + w[:, None] * a[j]          # noqa: E731
    acc = lin(imu.acc)
    gyr = lin(imu.gyr)
    qc = imu.quaternion("chip")
    quat_chip = _nlerp(qc[j - 1], qc[j], w)
    quat_host = _nlerp(imu.quat_host[j - 1], imu.quat_host[j], w)
    nearest = np.where(w < 0.5, j - 1, j)
    dt_ms = (t_acq - t_imu[nearest]) * 1e3
    cal = imu.cal[nearest] if imu.cal is not None and len(imu.cal) == n_i else np.full((n_f, 4), np.nan)

    # 정지 마스크 (라벨과 같은 판정) 와 중력 방향 (프로브 프레임 — R_SP 는 설정값, 기본 단위행렬)
    still = still_mask(imu.t_dev, imu.gyr, imu.acc, cfg)
    still_f = still[nearest]
    conv = (s.zero_ref or {}).get("quat_convention") if s.zero_ref else None
    if conv not in ("R", "R.T"):
        sel = still if still.any() else np.ones(n_i, bool)
        conv, _amb = earth_convention(qc[sel], imu.acc[sel])
    R_SE = sensor_to_earth_rotations(quat_chip, conv)                  # (N,3,3) 센서→지구
    R_sp = cfg.R_sp()
    gravity_probe = np.stack([gravity_in_frame(R_sp @ R.T) for R in R_SE])   # 지구 +z 를 프로브 프레임에서

    # 프레임 구간의 IMU 행 범위
    edges = np.concatenate([[t_acq[0] - np.median(np.diff(t_acq)) if n_f > 1 else t_acq[0] - 0.1], t_acq])
    imu_start = np.searchsorted(t_imu, edges[:-1], side="left")
    imu_stop = np.searchsorted(t_imu, edges[1:], side="left")

    # --- 리포트 ---
    d_f = np.diff(t_acq) if n_f > 1 else np.zeros(0)
    d_i = np.diff(t_imu)
    fps = 1.0 / float(np.median(d_f)) if d_f.size else float("nan")
    f_imu = 1.0 / float(np.median(d_i)) if d_i.size else float("nan")
    frame_gaps = int((d_f > 3.0 / fps).sum()) if d_f.size and np.isfinite(fps) else 0
    imu_gaps = int((d_i > 0.05).sum()) if d_i.size else 0          # 50 ms = 표본 ~12 개 결손
    outside = int((~inside).sum())
    a, b = imu.clock
    resid = imu.t_pc - imu.dev_to_pc(imu.t_dev)
    report = {
        "session": root.name, "n_frames": int(n_f), "n_imu": int(n_i),
        "duration_s": float(t_imu[-1] - t_imu[0]),
        "us_fps": fps, "imu_hz": f_imu,
        "us_latency_s_applied": float(us_latency_s),
        "frame_dt_ms": {"median": float(np.median(d_f) * 1e3) if d_f.size else None,
                        "p95": float(np.percentile(d_f, 95) * 1e3) if d_f.size else None,
                        "max": float(d_f.max() * 1e3) if d_f.size else None},
        "frame_gaps_gt_3x": frame_gaps, "frame_gap_fraction": frame_gaps / max(1, d_f.size),
        "imu_gaps_gt_50ms": imu_gaps, "imu_gap_fraction": imu_gaps / max(1, d_i.size),
        "imu_dt_ms": {"median": float(np.median(d_i) * 1e3), "p99": float(np.percentile(d_i, 99) * 1e3),
                      "max": float(d_i.max() * 1e3)},
        "frames_outside_imu": outside, "frames_outside_fraction": outside / n_f,
        "imu_nearest_dt_ms": {"median_abs": float(np.median(np.abs(dt_ms))), "max_abs": float(np.abs(dt_ms).max())},
        "clock": {"a": a, "b": b, "skew_ppm": (b - 1.0) * 1e6, "residual_p95_ms": float(np.percentile(np.abs(resid), 95) * 1e3)},
        "still_fraction_frames": float(still_f.mean()),
        "quat_convention": conv, "zero_ref": bool(s.zero_ref),
        "cal_status_mode": [int(np.nanmedian(cal[:, k])) if np.isfinite(cal[:, k]).any() else None for k in range(4)],
        "frame_shape": list(s.frames.shape[1:]) if s.frames.ndim == 3 else None,
    }
    report["ok"] = bool(report["frames_outside_fraction"] <= 0.01 and report["frame_gap_fraction"] <= 0.02
                        and report["imu_gap_fraction"] <= 0.005)
    report["warnings"] = []
    if not s.zero_ref:
        report["warnings"].append("영점(zero_ref) 없음 — 다음 세션은 Z 를 R 보다 먼저")
    if report["cal_status_mode"][1] is not None and report["cal_status_mode"][1] < 2:
        report["warnings"].append("BNO085 자이로 보정 상태 낮음 (cal_gyr < 2) — 회전 라벨 신뢰도 주의")
    if us_latency_s == 0.0:
        report["warnings"].append("us_latency_s 미측정 (0 적용) — inspect_session.py --latency")

    np.savez_compressed(
        root / SYNC_NPZ,
        frame_t_pc=s.frame_t_pc, frame_t_acq=t_acq, frame_id=s.frame_id, frame_index=np.arange(n_f),
        imu_row_nearest=nearest, imu_dt_ms=dt_ms, imu_start=imu_start, imu_stop=imu_stop, inside=inside,
        acc=acc.astype(np.float32), gyr=gyr.astype(np.float32),
        quat_chip=quat_chip.astype(np.float32), quat_host=quat_host.astype(np.float32),
        gravity_probe=gravity_probe.astype(np.float32), still=still_f, cal=cal.astype(np.float32),
        imu_t_pc=t_imu, imu_still=still,
    )
    with open(root / SYNC_REPORT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    if write_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            tt = t_imu - t_imu[0]
            fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
            ax[0].plot(tt, imu.gyr, lw=0.5); ax[0].fill_between(tt, *ax[0].get_ylim(), where=still, color="g", alpha=0.12)
            ax[0].set_ylabel("gyro rad/s"); ax[0].set_title("%s — sync check (green = still, ticks = frames)" % root.name)
            ax[1].plot(tt, imu.acc, lw=0.5); ax[1].set_ylabel("accel m/s²")
            ax[2].vlines(t_acq - t_imu[0], 0, 1, lw=0.3, color="k"); ax[2].set_ylabel("US frames"); ax[2].set_yticks([])
            ax[2].plot(t_acq[1:] - t_imu[0], np.clip(d_f / (1.0 / fps), 0, 4) / 4.0, ".", ms=2, color="r", label="frame dt / nominal (÷4)")
            ax[2].legend(loc="upper right", fontsize=8); ax[2].set_xlabel("t [s]")
            fig.tight_layout(); fig.savefig(root / SYNC_PNG, dpi=80); plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            report["png_error"] = str(exc)
    return report


def session_ready(d: Path) -> bool:
    """저장이 끝난 세션 = session.meta.json 이 있고 (녹화 정지 시 마지막에 쓴다) 아직 sync 산출물이 없다."""
    return (d / "session.meta.json").is_file() and (d / "us_index.csv").is_file() and not (d / SYNC_REPORT).is_file()


def watch(logs_dir: str | Path, period_s: float = 5.0, once: bool = False, stop_event: Optional[threading.Event] = None,
          on_done=None) -> None:
    logs = Path(logs_dir)
    while True:
        for d in sorted(logs.glob("us_imu_*")):
            if d.is_dir() and session_ready(d):
                try:
                    rep = sync_session(d)
                    msg = "동기화 완료 %s: %d 프레임 %.1f fps, IMU %.0f Hz, %s%s" % (
                        d.name, rep["n_frames"], rep["us_fps"], rep["imu_hz"], "OK" if rep["ok"] else "⚠ 판정 실패",
                        ("  경고: " + "; ".join(rep["warnings"])) if rep["warnings"] else "")
                except Exception as exc:  # noqa: BLE001
                    msg = "동기화 실패 %s: %s" % (d.name, exc)
                    with open(d / SYNC_REPORT, "w", encoding="utf-8") as fh:
                        json.dump({"session": d.name, "ok": False, "error": str(exc)}, fh, ensure_ascii=False, indent=2)
                print(msg, flush=True)
                if on_done is not None:
                    on_done(msg)
        if once:
            return
        if stop_event is not None:
            if stop_event.wait(period_s):
                return
        else:
            time.sleep(period_s)


class SyncWorker(threading.Thread):
    """GUI 안에서 도는 감시 루프. 녹화 정지(= 저장 완료) 뒤 몇 초 안에 그 세션을 동기화한다."""

    def __init__(self, logs_dir: str | Path, period_s: float = 5.0):
        super().__init__(daemon=True)
        self.logs_dir = Path(logs_dir)
        self.period = period_s
        self._stop = threading.Event()
        self.last_message: Optional[str] = None

    def run(self) -> None:
        watch(self.logs_dir, self.period, stop_event=self._stop, on_done=self._note)

    def _note(self, msg: str) -> None:
        self.last_message = msg

    def stop(self) -> None:
        self._stop.set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", help="세션 디렉터리 (하나만 처리)")
    ap.add_argument("--watch", metavar="LOGS_DIR", help="이 디렉터리의 새 세션을 계속 처리")
    ap.add_argument("--once", action="store_true", help="--watch 를 한 바퀴만")
    ap.add_argument("--latency", type=float, default=None, help="us_latency_s 강제 (기본: 세션 메타 또는 0)")
    ap.add_argument("--force", action="store_true", help="이미 sync 산출물이 있어도 다시")
    args = ap.parse_args()
    if args.session:
        if args.force:
            for f in (SYNC_NPZ, SYNC_REPORT, SYNC_PNG):
                p = Path(args.session) / f
                if p.exists():
                    p.unlink()
        rep = sync_session(args.session, us_latency_s=args.latency)
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0 if rep["ok"] else 2
    if args.watch:
        print("감시 시작: %s (5 s 주기, Ctrl+C 로 종료)" % args.watch, flush=True)
        try:
            watch(args.watch, once=args.once)
        except KeyboardInterrupt:
            pass
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
