"""``us_imu_collect.py`` 세션 디렉터리를 읽는다.

세션 레이아웃 (imu_bench/host/us_imu_collect.py docstring):

    us_imu_<stamp>/
      imu_<stamp>.csv         IMU 전 레이트 로그 (imu_log.COLUMNS, 33열)
      imu_<stamp>.meta.json   IMU 메타 (zero_ref 가 있을 수 있음)
      us_frames.bin           256x256 uint8 프레임 원시 스트림
      us_index.csv            pc_unix, us_seq, frame_id, byte_offset
      session.meta.json       조인 키 = pc_unix

시간축 원칙:
  * 두 스트림의 **조인**은 ``pc_unix`` (호스트 수신 시각) 로 한다.
  * IMU **적분**의 시간축은 ``dev_us`` (디바이스 이벤트 시각) 로 한다 — 수신 지터가
    적분에 들어가지 않게. dev_us → pc_unix 는 세션마다 1차식으로 맞춘다
    (qc_common.fit_clock 과 같은 취지).
  * US 고정 지연(⏳)은 ``timing.us_latency_s`` 로 프레임 시각에서 뺀다.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

FRAME_SHAPE = (256, 256)   # fr5_vision.us_protocol.CANDIDATE_FRAME_SHAPE

# imu_log.COLUMNS 와 같은 순서. 이름으로 읽되, 위치 의존 소비자를 위해 순서를 고정한다.
IMU_COLUMNS = [
    "pc_unix", "dev_us",
    "ax", "ay", "az", "gx", "gy", "gz",
    "mx_raw", "my_raw", "mz_raw", "mx", "my", "mz",
    "host_qw", "host_qx", "host_qy", "host_qz", "host_roll", "host_pitch", "host_yaw",
    "chip_qw", "chip_qx", "chip_qy", "chip_qz", "chip_roll", "chip_pitch", "chip_yaw",
    "diff_deg", "rel_roll", "rel_pitch", "rel_yaw",
    "cal_acc", "cal_gyr", "cal_mag", "cal_rv",
]


class SessionError(RuntimeError):
    pass


@dataclass
class ImuTable:
    """IMU 로그. 모든 배열은 같은 길이 n."""

    t_pc: np.ndarray          # (n,) pc_unix [s]
    t_dev: np.ndarray         # (n,) dev_us*1e-6 [s], wrap 펼침 완료
    acc: np.ndarray           # (n,3) [m/s^2]  센서 프레임
    gyr: np.ndarray           # (n,3) [rad/s]  센서 프레임
    quat_chip: np.ndarray     # (n,4) wxyz  (NaN 가능)
    quat_host: np.ndarray     # (n,4) wxyz
    cal: np.ndarray           # (n,4) cal_acc/gyr/mag/rv (NaN 가능)
    columns: dict[str, np.ndarray] = field(default_factory=dict)   # 나머지 열 (이름→배열)
    clock: tuple[float, float] = (0.0, 1.0)   # pc ≈ clock[0] + clock[1]*t_dev

    @property
    def n(self) -> int:
        return int(self.t_pc.size)

    def dev_to_pc(self, t_dev: np.ndarray | float) -> np.ndarray | float:
        a, b = self.clock
        return a + b * np.asarray(t_dev, float)

    def pc_to_dev(self, t_pc: np.ndarray | float) -> np.ndarray | float:
        a, b = self.clock
        return (np.asarray(t_pc, float) - a) / b

    @property
    def rate_hz(self) -> float:
        d = np.diff(self.t_dev)
        med = float(np.median(d)) if d.size else float("nan")
        return 1.0 / med if med > 0 else float("nan")

    def quaternion(self, which: str = "chip") -> np.ndarray:
        q = self.quat_chip if which == "chip" else self.quat_host
        if which == "chip" and not np.all(np.isfinite(q)):
            # 칩 RV 가 빠진 행은 호스트 퓨전으로 메운다 (100 Hz vs 250 Hz 레이트 차이)
            bad = ~np.all(np.isfinite(q), axis=1)
            q = q.copy()
            q[bad] = _ffill_rows(self.quat_chip, self.quat_host)[bad]
        return q


def _ffill_rows(primary: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    """primary 의 NaN 행을 직전 유효 행으로 채우고, 그래도 없으면 fallback."""
    out = primary.copy()
    ok = np.all(np.isfinite(out), axis=1)
    if not ok.any():
        return fallback.copy()
    idx = np.where(ok, np.arange(len(out)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = out[idx]
    first_ok = int(np.argmax(ok))
    filled[:first_ok] = fallback[:first_ok] if np.all(np.isfinite(fallback[:first_ok])) else out[first_ok]
    return filled


@dataclass
class Session:
    path: Path
    frames: np.ndarray           # (N,H,W) uint8 (memmap)
    frame_t_pc: np.ndarray       # (N,) pc_unix
    frame_id: np.ndarray         # (N,) 프로브 frame_id
    imu: ImuTable
    meta: dict[str, Any]
    imu_meta: dict[str, Any]
    zero_ref: Optional[dict[str, Any]] = None
    name: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.frame_t_pc.size)

    @property
    def us_fps(self) -> float:
        d = np.diff(self.frame_t_pc)
        med = float(np.median(d)) if d.size else float("nan")
        return 1.0 / med if med > 0 else float("nan")

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "frames": self.n_frames,
            "us_fps": round(self.us_fps, 2),
            "imu_rows": self.imu.n,
            "imu_hz": round(self.imu.rate_hz, 1),
            "duration_s": round(float(self.imu.t_pc[-1] - self.imu.t_pc[0]), 1) if self.imu.n else 0.0,
            "zero_ref": self.zero_ref is not None,
        }


# --------------------------------------------------------------------------- 읽기
def _read_csv_float(path: Path, expected: list[str]) -> dict[str, np.ndarray]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    if not rows:
        raise SessionError(f"빈 CSV: {path}")
    width = len(header)
    arr = np.full((len(rows), width), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j, v in enumerate(row[:width]):
            if v != "":
                try:
                    arr[i, j] = float(v)
                except ValueError:
                    pass
    out = {name: arr[:, j] for j, name in enumerate(header)}
    missing = [c for c in expected if c not in out]
    if missing:
        raise SessionError(f"{path}: 열이 빠져 있습니다: {missing}")
    return out


def _read_imu_h5(path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as fh:
        ds = fh["samples"]
        cols = [c.decode() if isinstance(c, bytes) else str(c) for c in ds.attrs["columns"]]
        data = np.asarray(ds[...], dtype=np.float64)
    return {name: data[:, j] for j, name in enumerate(cols)}


def _fit_clock(t_dev: np.ndarray, t_pc: np.ndarray) -> tuple[float, float]:
    """pc ≈ a + b·dev. 수신 지터에 강하도록 잔차가 큰 점을 두 번 걸러낸다."""
    ok = np.isfinite(t_dev) & np.isfinite(t_pc)
    x, y = t_dev[ok], t_pc[ok]
    if x.size < 2:
        return (float(y[0] - x[0]) if x.size else 0.0, 1.0)
    keep = np.ones(x.size, bool)
    a, b = 0.0, 1.0
    for _ in range(3):
        A = np.stack([np.ones(keep.sum()), x[keep]], axis=1)
        sol, *_ = np.linalg.lstsq(A, y[keep], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        resid = y - (a + b * x)
        scale = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-6
        keep = np.abs(resid - np.median(resid)) < 4.0 * scale
        if keep.sum() < 2:
            break
    return a, b


def _unwrap_dev_us(dev_us: np.ndarray) -> np.ndarray:
    """로거가 이미 펼쳤지만, 혹시 남은 32-bit wrap 을 한 번 더 편다."""
    d = np.diff(dev_us)
    wraps = np.where(d < -2**31)[0]
    if wraps.size == 0:
        return dev_us
    out = dev_us.astype(np.float64).copy()
    for w in wraps:
        out[w + 1:] += 2**32
    return out


def load_imu_table(path: Path) -> ImuTable:
    if path.suffix == ".h5":
        cols = _read_imu_h5(path)
    else:
        cols = _read_csv_float(path, ["pc_unix", "dev_us", "ax", "ay", "az", "gx", "gy", "gz"])
    t_pc = cols["pc_unix"]
    dev = cols["dev_us"]
    ok = np.isfinite(t_pc) & np.all(np.isfinite(np.stack([cols[c] for c in ("ax", "ay", "az", "gx", "gy", "gz")], 1)), 1)
    if not ok.all():
        cols = {k: v[ok] for k, v in cols.items()}
        t_pc, dev = cols["pc_unix"], cols["dev_us"]
    if np.isfinite(dev).all():
        t_dev = _unwrap_dev_us(dev) * 1e-6
        # 단조 증가 보장 (디바이스 시계가 되돌아간 행은 버린다)
        mono = np.concatenate([[True], np.diff(t_dev) > 0])
        if not mono.all():
            cols = {k: v[mono] for k, v in cols.items()}
            t_pc, t_dev = cols["pc_unix"], t_dev[mono]
        clock = _fit_clock(t_dev, t_pc)
    else:
        t_dev = t_pc - t_pc[0]
        clock = (float(t_pc[0]), 1.0)

    def _stack(names):
        return np.stack([cols[n] if n in cols else np.full_like(t_pc, np.nan) for n in names], axis=1)

    rest = {k: v for k, v in cols.items()
            if k not in ("pc_unix", "dev_us", "ax", "ay", "az", "gx", "gy", "gz")}
    return ImuTable(
        t_pc=t_pc, t_dev=t_dev,
        acc=_stack(["ax", "ay", "az"]), gyr=_stack(["gx", "gy", "gz"]),
        quat_chip=_stack(["chip_qw", "chip_qx", "chip_qy", "chip_qz"]),
        quat_host=_stack(["host_qw", "host_qx", "host_qy", "host_qz"]),
        cal=_stack(["cal_acc", "cal_gyr", "cal_mag", "cal_rv"]),
        columns=rest, clock=clock,
    )


def load_session(session_dir: str | Path, frame_shape: tuple[int, int] = FRAME_SHAPE) -> Session:
    """세션 디렉터리 → Session. 프레임은 memmap 으로 열어 메모리를 아낀다."""
    root = Path(session_dir)
    if not root.is_dir():
        raise SessionError(f"세션 디렉터리가 없습니다: {root}")
    meta_path = root / "session.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    us_bin = root / (meta.get("us", {}).get("bin") or "us_frames.bin")
    us_index = root / (meta.get("us", {}).get("index") or "us_index.csv")
    if not us_bin.is_file() or not us_index.is_file():
        raise SessionError(f"US 파일이 없습니다: {us_bin} / {us_index}")
    shape = tuple(meta.get("us", {}).get("frame_shape") or frame_shape)
    n_bytes = us_bin.stat().st_size
    per = shape[0] * shape[1]
    n_frames = n_bytes // per
    frames = np.memmap(us_bin, dtype=np.uint8, mode="r", shape=(n_frames, shape[0], shape[1])) \
        if n_frames > 0 else np.zeros((0, shape[0], shape[1]), np.uint8)

    idx = _read_csv_float(us_index, ["pc_unix", "us_seq"]) if n_frames > 0 else \
        {"pc_unix": np.zeros(0), "us_seq": np.zeros(0), "frame_id": np.zeros(0)}
    seq = idx["us_seq"].astype(int)
    if seq.size != n_frames:
        # 마지막 프레임이 잘렸을 수 있다 — 인덱스와 바이너리 중 짧은 쪽에 맞춘다
        n = min(seq.size, n_frames)
        frames = frames[:n]
        seq = seq[:n]
        idx = {k: v[:n] for k, v in idx.items()}
    order = np.argsort(seq)
    frame_t = idx["pc_unix"][order]
    frame_id = idx.get("frame_id", seq)[order].astype(int)
    if not np.all(seq[order] == np.arange(seq.size)):
        raise SessionError(f"{us_index}: us_seq 가 0..N-1 연속이 아닙니다")

    imu_files = sorted(root.glob("imu_*.csv")) + sorted(root.glob("imu_*.h5"))
    if not imu_files:
        raise SessionError(f"IMU 로그가 없습니다: {root}")
    imu_path = imu_files[0]
    imu = load_imu_table(imu_path)
    imu_meta_path = imu_path.with_suffix("").with_suffix(".meta.json") \
        if imu_path.suffix == ".csv" else imu_path.with_suffix(".meta.json")
    imu_meta = json.loads(imu_meta_path.read_text(encoding="utf-8")) if imu_meta_path.is_file() else {}
    zero_ref = imu_meta.get("zero_ref")

    return Session(path=root, frames=frames, frame_t_pc=frame_t, frame_id=frame_id,
                   imu=imu, meta=meta, imu_meta=imu_meta, zero_ref=zero_ref, name=root.name)


# --------------------------------------------------------------------------- 세션 목록
@dataclass
class SessionRecord:
    session_dir: str
    subject: str = ""
    source: str = "freehand"      # freehand | teleop
    split: str = ""               # train | val | test | "" (미지정)
    note: str = ""


def read_sessions_manifest(path: str | Path, root: Optional[Path] = None) -> list[SessionRecord]:
    """sessions.csv: session_dir, subject, source, split[, note]. 상대경로는 root 기준."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"세션 목록이 없습니다: {path}")
    out: list[SessionRecord] = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("session_dir") or row["session_dir"].lstrip().startswith("#"):
                continue
            d = Path(row["session_dir"].strip())
            if not d.is_absolute() and root is not None:
                d = root / d
            out.append(SessionRecord(
                session_dir=str(d), subject=(row.get("subject") or "").strip(),
                source=(row.get("source") or "freehand").strip().lower(),
                split=(row.get("split") or "").strip().lower(), note=(row.get("note") or "").strip(),
            ))
    if not out:
        raise ValueError(f"{path}: 세션이 한 줄도 없습니다")
    for r in out:
        if r.source not in ("freehand", "teleop"):
            raise ValueError(f"{r.session_dir}: source 는 freehand|teleop 이어야 합니다 (got {r.source!r})")
    return out


def write_sessions_manifest(path: str | Path, records: list[SessionRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session_dir", "subject", "source", "split", "note"])
        for r in records:
            w.writerow([r.session_dir, r.subject, r.source, r.split, r.note])
