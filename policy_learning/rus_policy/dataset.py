"""세션 묶음 → 학습 샘플 HDF5, 그리고 torch Dataset.

샘플 하나 = 분절 시연 구간 하나 (정지→이동→정지) = chunk 하나.

    obs/frames       (N, m, H, W) uint8     앵커 이전 마지막 m 장 (오래된 순)
    obs/frame_dt     (N, m) float32         프레임 시각 − 앵커 [s] (≤ 0)
    obs/frame_valid  (N, m) bool            패딩/너무 오래된 프레임은 False
    obs/state        (N, m, STATE_DIM)      perception.STATE_FEATURE_NAMES
    obs/vec          (N, OBS_VEC_DIM)       OBS_VEC_NAMES
    label/P          (N, k+1, 3) float32    policy 축 궤적 [x mm, y mm, θz deg], P[0]=0
    label/P6         (N, k+1, 6)            6축 (진단·힘축 마스킹 확인용)
    label/sigma_net  (N, 3)                 순변위 σ (소스별, §5.3a)
    label/sigma_shape(N, 3)
    label/Q          (N, k) float32         chunk 격자에서의 Q_seg (NaN 허용)
    label/Q_valid    (N, k) bool
    label/F          (N, k) float32         법선력 (텔레오퍼레이션 전용, 지금은 NaN)
    label/F_valid    (N, k) bool
    label/Fn_star    (N,) float32
    meta/…           세션·피험자·소스·분할·시각·진단
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np

from .config import PolicyConfig
from .imu_labels import SegmentLabel, SessionLabels, label_session, previous_motion
from .perception import STATE_DIM, STATE_FEATURE_NAMES, PerceptionResult, apply_frame_transform, \
    build_backend, perceive_session
from .session import Session, SessionRecord, load_session, read_sessions_manifest

logger = logging.getLogger(__name__)

OBS_VEC_NAMES: tuple[str, ...] = (
    "prev_dx_mm", "prev_dy_mm", "prev_dtheta_deg",   # Ã_{t−1} 실현치 (§1.2.1)
    "prev_valid", "prev_gap_s",
    "gravity_px", "gravity_py", "gravity_pz",         # 지구 +z 를 프로브 프레임에서 (§1.7 roll/pitch)
    "F_n", "M_x", "M_y", "F_n_star",                  # wrench (프리핸드: 0)
    "barrier_b", "delay_w",                           # arbiter 포화 (§1.5, 프리핸드: 0)
    "has_force",
)
OBS_VEC_DIM = len(OBS_VEC_NAMES)
SOURCE_IDS = {"freehand": 0, "teleop": 1}
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- 샘플 조립
@dataclass
class Sample:
    frames: np.ndarray       # (m,H,W) uint8
    frame_dt: np.ndarray     # (m,)
    frame_valid: np.ndarray  # (m,)
    state: np.ndarray        # (m, STATE_DIM)
    vec: np.ndarray          # (OBS_VEC_DIM,)
    P: np.ndarray            # (k+1,3)
    P6: np.ndarray           # (k+1,6)
    sigma_net: np.ndarray
    sigma_shape: np.ndarray
    Q: np.ndarray            # (k,)
    Q_valid: np.ndarray
    meta: dict[str, Any]


def _resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if frames.shape[-2:] == tuple(size):
        return np.ascontiguousarray(frames)
    from PIL import Image

    out = np.empty((len(frames), size[0], size[1]), np.uint8)
    for i, f in enumerate(frames):
        out[i] = np.asarray(Image.fromarray(np.asarray(f, np.uint8)).resize((size[1], size[0]), Image.BILINEAR))
    return out


def assemble_sample(cfg: PolicyConfig, session: Session, frames_t: np.ndarray, frames_all: np.ndarray,
                    perc: PerceptionResult, labels: SessionLabels, i: int) -> Optional[Sample]:
    lab: SegmentLabel = labels.labels[i]
    m, k = cfg.timing.obs_frames, cfg.timing.chunk_steps
    t_a = lab.t_anchor_pc
    n_before = int(np.searchsorted(frames_t, t_a, side="right"))
    if n_before == 0:
        return None
    idx = np.arange(max(0, n_before - m), n_before)
    pad = m - idx.size
    idx_padded = np.concatenate([np.full(pad, idx[0]), idx]) if pad > 0 else idx
    frame_dt = (frames_t[idx_padded] - t_a).astype(np.float32)
    valid = np.ones(m, bool)
    valid[:pad] = False
    valid &= (-frame_dt) <= cfg.timing.obs_frame_max_age_s
    if not valid.any():
        return None

    frames = _resize_frames(frames_all[idx_padded], tuple(cfg.perception.frame_size))
    state = perc.state[idx_padded].astype(np.float32)

    # Ã_{t−1}, 중력, 힘 채널
    prev_vec, gap, prev_valid = previous_motion(labels.labels, i, cfg.imu)
    vec = np.zeros(OBS_VEC_DIM, np.float32)
    vec[0:3] = prev_vec
    vec[3] = prev_valid
    vec[4] = min(gap, 10.0) if np.isfinite(gap) else 10.0
    vec[5:8] = lab.gravity_P
    # wrench/arbiter/has_force: 프리핸드 0 (텔레오퍼레이션 로더가 채운다)

    # Q̃: chunk 격자 (i = 1..k) 시각의 가장 가까운 프레임
    grid_pc = session.imu.dev_to_pc(lab.grid_t_dev[1:])
    Q = np.full(k, np.nan, np.float32)
    Q_valid = np.zeros(k, bool)
    if frames_t.size > 0:
        fps_dt = float(np.median(np.diff(frames_t))) if frames_t.size > 1 else 0.125
        j = np.clip(np.searchsorted(frames_t, grid_pc), 1, frames_t.size - 1)
        left, right = j - 1, j
        pick = np.where(np.abs(frames_t[left] - grid_pc) <= np.abs(frames_t[right] - grid_pc), left, right)
        near = np.abs(frames_t[pick] - grid_pc) <= 0.6 * fps_dt
        q = perc.quality[pick]
        ok = near & np.isfinite(q)
        Q[ok] = q[ok]
        Q_valid = ok

    meta = {
        "t_anchor_pc": float(t_a), "t_anchor_dev": lab.t_anchor_dev, "move_s": lab.move_s,
        "still_before_s": lab.still_before_s, "still_after_s": lab.still_after_s,
        "net_mm": lab.net_mm, "segment_index": i, "n_obs_valid": int(valid.sum()),
        "token_at_anchor": int(perc.token[idx[-1]]), "Q_at_anchor": float(perc.quality[idx[-1]]),
        **{f"diag_{k_}": float(v) for k_, v in lab.diagnostics.items()},
    }
    return Sample(frames=frames, frame_dt=frame_dt, frame_valid=valid, state=state, vec=vec,
                  P=lab.P.astype(np.float32), P6=lab.P6.astype(np.float32),
                  sigma_net=lab.sigma_net.astype(np.float32), sigma_shape=lab.sigma_shape.astype(np.float32),
                  Q=Q, Q_valid=Q_valid, meta=meta)


# --------------------------------------------------------------------------- 분할
def assign_splits(records: list[SessionRecord], cfg: PolicyConfig) -> dict[str, str]:
    """세션 디렉터리 → split. 피험자 단위로 나눠 누수를 막는다."""
    strategy = cfg.split.strategy
    if strategy == "manifest":
        missing = [r.session_dir for r in records if r.split not in SPLITS]
        if missing:
            raise ValueError(f"split.strategy=manifest 인데 split 이 비었거나 잘못된 세션: {missing[:3]}…"
                             " (train|val|test 를 채우거나 split.strategy: subject_random)")
        out = {r.session_dir: r.split for r in records}
    elif strategy == "subject_random":
        subjects = sorted({r.subject or r.session_dir for r in records})
        rng = np.random.RandomState(cfg.split.seed)
        rng.shuffle(subjects)
        ratios = np.asarray(cfg.split.ratios, float)
        ratios = ratios / ratios.sum()
        n = len(subjects)
        n_train = int(round(ratios[0] * n))
        n_val = int(round(ratios[1] * n))
        if n >= 3:
            n_train = max(1, min(n_train, n - 2))
            n_val = max(1, min(n_val, n - n_train - 1))
        elif n == 2:
            n_train, n_val = 1, 1
        else:
            n_train, n_val = 1, 0
        split_of = {}
        for j, s in enumerate(subjects):
            split_of[s] = "train" if j < n_train else ("val" if j < n_train + n_val else "test")
        out = {r.session_dir: split_of[r.subject or r.session_dir] for r in records}
    else:
        raise ValueError(f"split.strategy 는 manifest|subject_random: {strategy!r}")
    # 누수 검사
    subj_splits: dict[str, set[str]] = {}
    for r in records:
        subj_splits.setdefault(r.subject or r.session_dir, set()).add(out[r.session_dir])
    leak = {s: v for s, v in subj_splits.items() if len(v) > 1}
    if leak:
        raise ValueError(f"피험자가 여러 split 에 걸쳐 있습니다: {leak}")
    return out


# --------------------------------------------------------------------------- 빌드
class _H5Writer:
    def __init__(self, path: Path, cfg: PolicyConfig, compress: bool):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = h5py.File(path, "w")
        m, k = cfg.timing.obs_frames, cfg.timing.chunk_steps
        H, W = cfg.perception.frame_size
        self.n = 0
        comp = {"compression": "gzip", "compression_opts": 3} if compress else {}

        def mk(name, shape, dtype, chunks=None, **kw):
            return self.fh.create_dataset(name, shape=(0,) + shape, maxshape=(None,) + shape, dtype=dtype,
                                          chunks=(1,) + shape if chunks is None else chunks, **kw)

        self.d = {
            "obs/frames": mk("obs/frames", (m, H, W), np.uint8, **comp),
            "obs/frame_dt": mk("obs/frame_dt", (m,), np.float32),
            "obs/frame_valid": mk("obs/frame_valid", (m,), bool),
            "obs/state": mk("obs/state", (m, STATE_DIM), np.float32),
            "obs/vec": mk("obs/vec", (OBS_VEC_DIM,), np.float32),
            "label/P": mk("label/P", (k + 1, 3), np.float32),
            "label/P6": mk("label/P6", (k + 1, 6), np.float32),
            "label/sigma_net": mk("label/sigma_net", (3,), np.float32),
            "label/sigma_shape": mk("label/sigma_shape", (3,), np.float32),
            "label/Q": mk("label/Q", (k,), np.float32),
            "label/Q_valid": mk("label/Q_valid", (k,), bool),
            "label/F": mk("label/F", (k,), np.float32),
            "label/F_valid": mk("label/F_valid", (k,), bool),
            "label/Fn_star": mk("label/Fn_star", (), np.float32),
            "meta/session": mk("meta/session", (), np.int32),
            "meta/source": mk("meta/source", (), np.int8),
            "meta/split": mk("meta/split", (), h5py.string_dtype()),
            "meta/subject": mk("meta/subject", (), h5py.string_dtype()),
            "meta/json": mk("meta/json", (), h5py.string_dtype()),
        }
        g = self.fh.create_group("schema")
        g.attrs["state_features"] = json.dumps(STATE_FEATURE_NAMES)
        g.attrs["obs_vec"] = json.dumps(OBS_VEC_NAMES)
        g.attrs["units"] = "P: x,y mm; theta deg. axes: x lateral, y elevational, z beam"
        g.attrs["config"] = json.dumps(cfg.to_dict(), ensure_ascii=False)

    def append(self, s: Sample, session_idx: int, source: str, split: str, subject: str) -> None:
        i = self.n
        for ds in self.d.values():
            ds.resize(i + 1, axis=0)
        d = self.d
        d["obs/frames"][i] = s.frames
        d["obs/frame_dt"][i] = s.frame_dt
        d["obs/frame_valid"][i] = s.frame_valid
        d["obs/state"][i] = s.state
        d["obs/vec"][i] = s.vec
        d["label/P"][i] = s.P
        d["label/P6"][i] = s.P6
        d["label/sigma_net"][i] = s.sigma_net
        d["label/sigma_shape"][i] = s.sigma_shape
        d["label/Q"][i] = s.Q
        d["label/Q_valid"][i] = s.Q_valid
        d["label/F"][i] = np.full(s.Q.shape, np.nan, np.float32)
        d["label/F_valid"][i] = np.zeros(s.Q.shape, bool)
        d["label/Fn_star"][i] = np.nan
        d["meta/session"][i] = session_idx
        d["meta/source"][i] = SOURCE_IDS[source]
        d["meta/split"][i] = split
        d["meta/subject"][i] = subject
        d["meta/json"][i] = json.dumps(s.meta)
        self.n += 1

    def finish(self, sessions_info: list[dict[str, Any]]) -> None:
        self.fh["schema"].attrs["sessions"] = json.dumps(sessions_info, ensure_ascii=False)
        self.fh["schema"].attrs["n_samples"] = self.n
        self.fh.close()


def build_dataset(cfg: PolicyConfig, out_path: Optional[Path] = None, compress: bool = False,
                  records: Optional[list[SessionRecord]] = None) -> dict[str, Any]:
    """sessions.csv 의 모든 세션 → HDF5. 세션별 요약을 돌려준다."""
    out_path = cfg.resolve(cfg.paths.dataset) if out_path is None else Path(out_path)
    if records is None:
        manifest = cfg.resolve(cfg.paths.sessions_manifest)
        records = read_sessions_manifest(manifest, root=manifest.parent)
    splits = assign_splits(records, cfg)
    backend = build_backend(cfg)
    writer = _H5Writer(out_path, cfg, compress)
    info: list[dict[str, Any]] = []
    try:
        for si, rec in enumerate(records):
            if rec.source == "teleop":
                raise NotImplementedError(
                    f"{rec.session_dir}: source=teleop 는 로봇 FK 라벨 로더가 필요한데 아직 US+FK 동기 수집기가 "
                    "없습니다 (fr5_h5_collector.py 는 구 복강경용). 프리핸드(IMU) 세션만 지원합니다.")
            session = load_session(rec.session_dir)
            summary = session.summary()
            logger.info("세션 %s: %s", session.name, summary)
            conv = (session.zero_ref or {}).get("quat_convention")
            labels = label_session(session.imu, cfg.timing, cfg.imu, cfg.labels, source=rec.source,
                                   convention=conv)
            perc = perceive_session(cfg, session, backend)
            frames_all = apply_frame_transform(np.asarray(session.frames), cfg.perception.frame_transform)
            frames_t = session.frame_t_pc - cfg.timing.us_latency_s
            n_written = 0
            for i in range(len(labels.labels)):
                s = assemble_sample(cfg, session, frames_t, frames_all, perc, labels, i)
                if s is None:
                    continue
                writer.append(s, si, rec.source, splits[rec.session_dir], rec.subject)
                n_written += 1
            entry = {**summary, "session_dir": rec.session_dir, "subject": rec.subject, "source": rec.source,
                     "split": splits[rec.session_dir], "samples": n_written,
                     "perception": perc.backend, "checkpoint_id": perc.checkpoint_id, **labels.diagnostics}
            info.append(entry)
            logger.info("  → %d 샘플 (정지 %d 구간, 이동 %d 구간, 규약 %s)", n_written,
                        labels.diagnostics["n_still_segments"], labels.diagnostics["n_move_segments"],
                        labels.convention)
    finally:
        writer.finish(info)
    return {"path": str(out_path), "n_samples": writer.n, "sessions": info}


# --------------------------------------------------------------------------- torch Dataset
class PolicyH5Dataset:
    """HDF5 → 샘플 dict (torch.Tensor). 워커별로 파일을 늦게 연다."""

    def __init__(self, path: str | Path, split: Optional[str] = None, augment: bool = False,
                 indices: Optional[np.ndarray] = None, max_samples: Optional[int] = None, seed: int = 0):
        import torch  # noqa: F401  (torch 없이도 모듈 import 는 되게)

        self.path = str(path)
        self.augment = augment
        self._fh: Optional[h5py.File] = None
        with h5py.File(self.path, "r") as fh:
            n = int(fh["schema"].attrs["n_samples"])
            all_splits = np.asarray(fh["meta/split"][...]).astype(str) if n else np.zeros(0, str)
            self.config = json.loads(fh["schema"].attrs["config"])
            self.state_features = json.loads(fh["schema"].attrs["state_features"])
        if indices is None:
            indices = np.arange(n) if split is None else np.where(all_splits == split)[0]
        if max_samples is not None and indices.size > max_samples:
            indices = np.random.RandomState(seed).choice(indices, max_samples, replace=False)
            indices.sort()
        self.indices = np.asarray(indices, int)
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return int(self.indices.size)

    def _open(self) -> h5py.File:
        if self._fh is None:
            self._fh = h5py.File(self.path, "r")
            # 워커마다 다른 증강 난수열 (같은 seed 가 복제되지 않게)
            try:
                import torch.utils.data as tud
                info = tud.get_worker_info()
                if info is not None:
                    self.rng = np.random.RandomState((int(self.rng.randint(1 << 30)) + info.id) % (1 << 31))
            except Exception:  # noqa: BLE001
                pass
        return self._fh

    def __getitem__(self, i: int) -> dict[str, Any]:
        import torch

        fh = self._open()
        j = int(self.indices[i])
        frames = np.asarray(fh["obs/frames"][j], np.float32) / 255.0
        if self.augment:
            frames = self._photometric(frames)
        item = {
            "frames": torch.from_numpy(np.ascontiguousarray(frames)),
            "frame_dt": torch.from_numpy(np.asarray(fh["obs/frame_dt"][j], np.float32)),
            "frame_valid": torch.from_numpy(np.asarray(fh["obs/frame_valid"][j], bool)),
            "state": torch.from_numpy(np.nan_to_num(np.asarray(fh["obs/state"][j], np.float32))),
            "vec": torch.from_numpy(np.nan_to_num(np.asarray(fh["obs/vec"][j], np.float32))),
            "P": torch.from_numpy(np.asarray(fh["label/P"][j], np.float32)),
            "sigma_net": torch.from_numpy(np.asarray(fh["label/sigma_net"][j], np.float32)),
            "sigma_shape": torch.from_numpy(np.asarray(fh["label/sigma_shape"][j], np.float32)),
            "Q": torch.from_numpy(np.nan_to_num(np.asarray(fh["label/Q"][j], np.float32))),
            "Q_valid": torch.from_numpy(np.asarray(fh["label/Q_valid"][j], bool)),
            "F": torch.from_numpy(np.nan_to_num(np.asarray(fh["label/F"][j], np.float32))),
            "F_valid": torch.from_numpy(np.asarray(fh["label/F_valid"][j], bool)),
            "Fn_star": torch.tensor(float(np.nan_to_num(fh["label/Fn_star"][j]))),
            "source": torch.tensor(int(fh["meta/source"][j])),
            "index": torch.tensor(j),
        }
        return item

    def _photometric(self, frames: np.ndarray) -> np.ndarray:
        """밝기·대비·감마·잡음. 기하 변환은 하지 않는다 (부호 단서 보존, §3.8)."""
        r = self.rng
        gain = r.uniform(0.8, 1.2)
        bias = r.uniform(-0.08, 0.08)
        gamma = r.uniform(0.8, 1.25)
        out = np.clip(frames, 0, 1) ** gamma * gain + bias
        out += r.normal(0.0, 0.01, size=out.shape).astype(np.float32)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def summary(self) -> dict[str, Any]:
        with h5py.File(self.path, "r") as fh:
            src = np.asarray(fh["meta/source"][...])[self.indices] if len(self) else np.zeros(0)
            qv = np.asarray(fh["label/Q_valid"][...])[self.indices] if len(self) else np.zeros((0, 1))
        return {"n": len(self), "freehand": int((src == 0).sum()), "teleop": int((src == 1).sum()),
                "Q_valid_fraction": float(qv.mean()) if qv.size else 0.0}
