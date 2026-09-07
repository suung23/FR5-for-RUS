"""설정 — YAML 한 장 + ``--set key.path=value`` 덮어쓰기.

모든 수치의 근거는 ``docs/POLICY_LEARNING_MATH.md`` 의 절 번호로 주석에 남긴다.
🟡 표시는 제안값(튜닝 대상), ⏳ 는 실측 후 갱신해야 하는 값이다.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
UNET_ROOT = REPO_ROOT / "Unet_seg"
IMU_BENCH_ROOT = REPO_ROOT / "imu_bench"


# --------------------------------------------------------------------------- 섹션
@dataclass
class PathsConfig:
    sessions_manifest: str = "policy_learning/data/sessions.csv"
    dataset: str = "policy_learning/data/policy_dataset.h5"
    output_dir: str = "policy_learning/runs/default"
    unet_checkpoint: str = "Unet_seg/checkpoints/exp_seed43/best.pt"
    unet_config: str = "Unet_seg/configs/exp_seed43_retro.yaml"
    perception_cache_dir: str = "policy_learning/data/perception_cache"


@dataclass
class TimingConfig:
    policy_hz: float = 5.0          # f_p  (§1.2)
    chunk_steps: int = 8            # k    (§2.5 🟡)
    obs_frames: int = 16            # m    (§1.3 🟡, f_us 확정 후 재산정)
    us_latency_s: float = 0.0       # ⏳ US 고정 엔드투엔드 지연 (§7.1). 실측 후 갱신
    obs_frame_max_age_s: float = 4.0  # 관측 프레임이 이보다 오래되면 무효 마스크

    @property
    def chunk_dt(self) -> float:
        return 1.0 / self.policy_hz

    @property
    def chunk_horizon_s(self) -> float:
        return self.chunk_steps / self.policy_hz


@dataclass
class ImuConfig:
    quaternion: str = "chip"        # chip (BNO085 RV) | host (Madgwick)
    still_win_s: float = 0.30       # qc_common.still_mask 와 동일
    # ⏳ 정지 판정 임계. qc_common 값(0.005/0.010/0.15)은 "손으로 들면 통과하지 않을 만큼" 빡빡해
    # 로봇/거치대용이다. 프리핸드 정지(피부에 댄 손)는 그보다 떨리므로 느슨하게 시작하고,
    # scripts/inspect_session.py 로 실제 세션의 산포 분포를 보고 조정한다. 느슨할수록 ZUPT 앵커의
    # 잔류 속도가 커져 라벨 σ 가 실제보다 낙관적일 수 있다 (진단 zupt_v_end_raw 를 볼 것).
    still_gyro_sd: float = 0.02     # rad/s
    still_gyro_mean: float = 0.03   # rad/s
    still_accel_sd: float = 0.25    # m/s^2
    min_still_s: float = 0.30       # §2.3 분절 프로토콜의 정지 길이 🟡 (QC 는 0.8)
    max_move_s: float = 8.0         # protocol.ZUPT_MAX_MOVE_S
    min_move_s: float = 0.15        # 이보다 짧은 "이동" 은 잡음으로 본다
    anchor_fraction: float = 0.25   # 정지 구간 끝쪽 25 % 안에 앵커 (analyze_track.move_segments)
    # ⏳ 센서→프로브 회전 R_SP. 프로브 프레임: +x lateral, +y elevational, +z beam
    # (docs/FRAMES_AND_SE2.md §1). IMU 마운트 방향을 측정해 채워야 한다.
    sensor_to_probe: list = field(default_factory=lambda: [[1.0, 0.0, 0.0],
                                                           [0.0, 1.0, 0.0],
                                                           [0.0, 0.0, 1.0]])
    prev_motion_max_gap_s: float = 5.0   # Ã_{t−1} 을 유효로 보는 최대 공백

    def R_sp(self) -> np.ndarray:
        R = np.asarray(self.sensor_to_probe, float)
        if R.shape != (3, 3):
            raise ValueError("imu.sensor_to_probe 는 3x3 행렬이어야 합니다")
        if (not np.allclose(R @ R.T, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(R), 1.0, atol=1e-6)):
            raise ValueError("imu.sensor_to_probe 는 회전행렬(직교, det=+1)이어야 합니다")
        return R


@dataclass
class LabelConfig:
    # ε(τ) = c · τ^1.5  (§2.2, QC_PLAN 9번 실측 적합)
    sigma_translation_coeff_mm: float = 0.7
    sigma_rotation_deg: float = 0.1          # AHRS 직접 (§5.3a 🟡)
    sigma_shape_fraction: float = 0.3        # σ_shape = fraction · σ_net 🟡 (§5.2 "≪")
    teleop_sigma_translation_mm: float = 0.1  # FK (§5.3a 🟡)
    teleop_sigma_rotation_deg: float = 0.05
    sigma_floor_mm: float = 0.05             # 0 으로 나누기 방지
    sigma_floor_deg: float = 0.01


@dataclass
class PerceptionConfig:
    backend: str = "unet"               # unet | none
    frame_size: list = field(default_factory=lambda: [256, 256])  # 저장 관측 해상도
    frame_transform: str = "none"       # ⏳ candidate 프레임 방향: none|rot90_cw|rot90_ccw|rot180|flip_h|flip_v
    device: str = "auto"
    cache: bool = True
    hold_deadband_px: float = 8.0916    # lateral_instruction.json thresholds["5%"]["px"]


@dataclass
class SplitConfig:
    strategy: str = "manifest"          # manifest | subject_random
    ratios: list = field(default_factory=lambda: [0.7, 0.15, 0.15])
    seed: int = 42


@dataclass
class ModelConfig:
    head: str = "cvae"                  # cvae | discrete  (§9-1 미결 — 설정으로 선택)
    d_model: int = 256
    n_heads: int = 8
    encoder_layers: int = 4
    decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    z_dim: int = 32
    frame_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    frame_input_size: list = field(default_factory=lambda: [128, 128])  # 인코더 입력 (저장본을 리사이즈)
    # 이산 헤드: 축당 bins, ±range. 🟡 §5.3(g) 축당 21빈(±20 mm, 2 mm)
    discrete_bins: int = 21
    discrete_range_mm: float = 20.0
    discrete_range_deg: float = 10.0
    q_head_hidden: int = 256


@dataclass
class LossConfig:
    # §5.4 계수 요약 (전부 🟡)
    lambda_force: float = 0.3
    lambda_quality: float = 1.0
    lambda_smooth: float = 0.05
    lambda_feas: float = 0.5
    lambda_risk: float = 0.2
    w_minus_over_plus: float = 5.0
    beta_kl: float = 0.02
    w_shape: float = 0.3
    huber_delta_sigma: float = 2.0      # δ = 2σ
    B_z: float = 1000.0                 # N·s/m  (§8.1 🟡)
    v_z_max_mm_s: float = 10.0          # mm/s
    net_weight_cap: float = 400.0       # (1/σ²) 상한 — FK 라벨이 배치를 독점하지 않게


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-4
    warmup_epochs: int = 2
    amp: bool = False
    device: str = "auto"
    num_workers: int = 0
    seed: int = 0
    grad_clip: float = 1.0
    z_samples_eval: int = 32            # M (§3.3 🟡)
    gamma_mode_consistency: float = 0.2  # γ (3.3) — Q̂ 스케일 대비 🟡
    log_every: int = 20
    max_train_samples: Optional[int] = None   # 디버그용 서브샘플
    augment: bool = True


@dataclass
class PolicyConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    SECTIONS = ("paths", "timing", "imu", "labels", "perception", "split", "model", "loss", "train")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyConfig":
        sections = {
            "paths": PathsConfig, "timing": TimingConfig, "imu": ImuConfig,
            "labels": LabelConfig, "perception": PerceptionConfig, "split": SplitConfig,
            "model": ModelConfig, "loss": LossConfig, "train": TrainConfig,
        }
        unknown = set(data) - set(sections)
        if unknown:
            raise ValueError(f"알 수 없는 설정 섹션: {sorted(unknown)}")
        kwargs = {}
        for name, klass in sections.items():
            section = dict(data.get(name) or {})
            bad = set(section) - set(klass.__dataclass_fields__)
            if bad:
                raise ValueError(f"{name}: 알 수 없는 키 {sorted(bad)}")
            section = {k: _coerce(klass.__dataclass_fields__[k].type, v, f"{name}.{k}") for k, v in section.items()}
            kwargs[name] = klass(**section)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.SECTIONS:
            section = getattr(self, name)
            out[name] = {k: copy.deepcopy(getattr(section, k)) for k in section.__dataclass_fields__}
        return out

    def resolve(self, path: str | Path) -> Path:
        """저장소 루트 기준 상대경로를 절대경로로."""
        p = Path(path)
        return p if p.is_absolute() else (REPO_ROOT / p)


def _coerce(annotation: Any, value: Any, where: str) -> Any:
    """dataclass 필드 타입에 맞게 스칼라를 변환한다 (YAML 1.1 은 '3e-4' 를 문자열로 읽는다)."""
    ann = str(annotation)
    if value is None:
        return None
    try:
        if ann.startswith("float"):
            return float(value)
        if ann.startswith("int"):
            return int(value)
        if ann.startswith("bool"):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if ann.startswith("Optional[int]"):
            return int(value)
        if ann.startswith("Optional[float]"):
            return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {value!r} 를 {ann} 로 바꿀 수 없습니다") from exc
    return value


def _apply_overrides(data: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    for pair in overrides:
        if "=" not in pair:
            raise ValueError(f"--set 은 key.path=value 형식이어야 합니다: {pair!r}")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        node = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return data


def _deep_merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in top.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], dict(v))
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_chain(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("defaults", None)
    if parent:
        parent_path = Path(parent) if Path(parent).is_absolute() else (path.parent / parent)
        data = _deep_merge(_load_yaml_chain(parent_path), data)
    return data


def load_config(path: Optional[str | Path] = None, overrides: Sequence[str] = ()) -> PolicyConfig:
    """YAML → PolicyConfig. ``defaults:`` 키로 다른 YAML 을 상속할 수 있다."""
    data: dict[str, Any] = {}
    if path is not None:
        data = _load_yaml_chain(Path(path))
    data = _apply_overrides(data, list(overrides))
    return PolicyConfig.from_dict(data)


def save_config(config: PolicyConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config.to_dict(), fh, allow_unicode=True, sort_keys=False)
