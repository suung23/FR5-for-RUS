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
    # ⏳ m = ceil(f_us / f_dither). f_us 는 가정하지 않는다 — inspect_session.py 가 세션에서 잰 값으로
    # 재산정한다 (§1.3). 16 은 "8 fps × 2 s" 의 자리표시자일 뿐이다.
    obs_frames: int = 16
    # US 고정 엔드투엔드 지연 (§7.1). 200ms 는 사후 추정 확정값 — 정책 스텝(1/5s)과 정확히 같아서
    # 0 으로 두면 관측과 행동이 한 스텝 어긋난다. dataset.py 가 빌드 때 프레임 시각에서 빼므로
    # 이 값을 바꾸면 **데이터셋을 다시 빌드해야** 반영된다 (학습만 다시 돌려선 안 바뀐다).
    us_latency_s: float = 0.200
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
    # 센서→프로브 회전 R_SP (p = R_SP · s). 프로브 프레임: +x lateral, +y elevational, +z beam (docs/FRAMES_AND_SE2.md §1).
    # 2026-09-10 C10UR 2차 수집 데이터로 확정 (board-6-qc 마운트):
    #   * 정지 시 가속도가 센서 −x 에 −9.1 m/s² → 센서 +x 가 아래(빔) 방향  ⇒ z_p = +x_s
    #   * 횡이동 세션 69 개: 가속 에너지가 센서 y (1.11 vs 0.17/0.18), 출발 가속 부호와 영상 속 구조물의 라인 이동 부호가
    #     146/146 일치  ⇒ x_p = +y_s  (+x_p = 구조물이 높은 A-line 번호 쪽(부채꼴 표시의 오른쪽)으로 이동하는 방향)
    #   * 면외 스윕 20 개: 가속 에너지가 센서 z (0.69 vs 0.09/0.14)  ⇒ y_p = +z_s  (= z_p × x_p, 오른손 좌표계 ✔)
    # 행 = 프로브 축을 센서 좌표로 쓴 것. 로봇 배치 때 부호(좌우)는 flip_lines 와 함께 한 번 더 확인할 것.
    sensor_to_probe: list = field(default_factory=lambda: [[0.0, 1.0, 0.0],
                                                           [0.0, 0.0, 1.0],
                                                           [1.0, 0.0, 0.0]])
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
class DatasetConfig:
    # 관측 창 증강 (§8-2 2026-09-08): 정지 A 안의 어느 시각을 "지금" 으로 잡아도 같은 라벨의 유효한 관측이다
    # (프로브가 정지해 있으므로 이동은 동일). 앵커 외에 (obs_anchor_samples − 1) 개를 정지 A 안에서 추가로 뽑는다.
    obs_anchor_samples: int = 1
    obs_anchor_min_still_s: float = 0.15   # 정지 시작 후 이 시간이 지난 뒤부터 뽑는다 (정지 판정 창의 반)
    obs_anchor_seed: int = 0
    require_chunk_covers_move: bool = False  # True 면 이동이 chunk 창(1.6 s) 을 넘는 구간을 버린다


@dataclass
class OffsetFilterConfig:
    """§3.9 오프셋 필터 (2026-09-08 채택). 값의 근거는 offset_filter.py 참조. 전부 🟡."""
    d_range_mm: float = 30.0
    d_step_mm: float = 0.5
    R_grid_mm: list = field(default_factory=lambda: [30.0, 40.0, 50.0, 60.0, 75.0, 90.0])
    A0_grid: list = field(default_factory=lambda: [0.85, 1.0, 1.15])
    sigma_q: float = 0.03            # 정규화 면적의 관측 잡음 (호흡·변형 포함) ⏳ 실측
    process_coeff_mm: float = 0.7    # 이동 후 위치 불확실성 c·τ^1.5 (라벨 σ 와 같은 식)
    process_tau_s: float = 1.6
    process_floor_mm: float = 0.5
    converge_std_mm: float = 2.0     # 사후 표준편차가 이보다 작으면 평균으로 간다
    probe_snr: float = 2.0           # 탐침은 두 가설의 |Δh| ≥ probe_snr·σ_q 가 되게
    min_probe_mm: float = 1.0
    max_probe_mm: float = 8.0
    bimodal_min_mass: float = 0.2    # 양쪽 부호에 이 이상 질량이 있으면 이봉으로 본다


@dataclass
class SplitConfig:
    strategy: str = "manifest"          # manifest | subject_random
    ratios: list = field(default_factory=lambda: [0.7, 0.15, 0.15])
    seed: int = 42
    allow_subject_overlap: bool = False  # True: same subject may span splits (session-level split; single-subject pilot data). Warns.


@dataclass
class ModelConfig:
    head: str = "cvae"                  # cvae | discrete  (§9-1 미결 — 설정으로 선택)
    # 2026-09-08: 첫 실데이터 런용 소형 기본값. 샘플이 구간당 하나라 표본이 적다 (§8-2).
    # 큰 모델은 --set model.d_model=256 model.encoder_layers=4 model.decoder_layers=4 model.dim_feedforward=1024
    d_model: int = 128
    n_heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    # z 용량이 곧 누설 상한이다 (§5.3g 2026-09-08). 모드 부호 + 스타일 몇 비트면 충분하다.
    z_dim: int = 4
    frame_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    frame_input_size: list = field(default_factory=lambda: [128, 128])  # 인코더 입력 (저장본을 리사이즈)
    # 이산 헤드: 축당 bins, ±range. 🟡 §5.3(g) 축당 21빈(±20 mm, 2 mm)
    # 범위를 ±20→±80mm 로 넓히면서 빈 폭이 2→8mm 가 되므로 개수도 같이 올린다 (폭 4mm).
    # 실행시 경로의 오차 하한이 곧 빈 폭이라, 도달 목표(수 mm)보다 굵으면 의미가 없다.
    discrete_bins: int = 41
    discrete_range_mm: float = 80.0    # 라벨 σ 가 27mm — ±20mm 로는 상당수 라벨이 표현 범위 밖이었다 (2026-09-10)
    discrete_range_deg: float = 10.0
    q_head_hidden: int = 256
    # 잔차 분해 Q̂ = b(o) + [g(o,A) − g(o,0)] (2026-09-10). 단일 MLP 로는 E[Q|o] 만 맞혀도 MSE 가
    # 거의 최소라 행동을 무시하는 해에 갇힌다 (행동을 0 으로 지워도 출력 변화가 Q std 의 2~7%).
    # 관측 경로 b 와 행동 경로 g 를 나누고, g 의 목표를 b 가 설명 못 한 잔차로 고정한다.
    quality_residual: bool = True


@dataclass
class LossConfig:
    # §5.4 계수 요약 (전부 🟡)
    lambda_force: float = 0.3
    lambda_quality: float = 1.0
    # 품질 헤드가 맞힐 라벨. "Q" = 옛 Q_seg (면적 15 %, 방광이 없어도 0.89 · 있어도 0.95 로
    # 포화). "Q_area" = chunk 격자의 세션 정규화 면적 — 데이터셋이 그 포화를 우회하려고
    # 만들어 두었지만 2026-09-11 까지 로드만 되고 손실에 쓰이지 않았다.
    quality_target: str = "Q"           # "Q" | "Q_area"
    lambda_smooth: float = 0.05
    lambda_feas: float = 0.5
    lambda_risk: float = 0.2
    w_minus_over_plus: float = 5.0
    # β (§5.3g 2026-09-08): 0.02 는 누설 쪽으로 치우친다. 0.5 에서 시작해 {0.1, 0.5, 1, 5} 스윕,
    # 선택 규칙은 train.py 의 leak_gap_mm. 워밍업은 train.beta_warmup_epochs.
    # 2026-09-10 exp1: β=0.5 고정은 ep19 에 leak_gap 14.25mm (σ_net,y≈1.42mm 의 10배) 로 누설했다.
    # train.beta_adapt 가 켜져 있으면 이 값은 시작점일 뿐이고 실제 β 는 epoch 마다 조정된다.
    beta_kl: float = 0.5
    w_shape: float = 0.3
    huber_delta_sigma: float = 2.0      # δ = 2σ
    # δ 하한 (2026-09-10). 실측 라벨 |Δ| 는 평균 (14.5, 13.4, 0.96), σ 는 (27.2, 25.8, 1.5) 인데
    # σ_net 은 1mm 수준이라 δ=2σ_net≈2mm 로는 잔차 전체가 L1 영역이었다. 64 샘플 암기 시험:
    # δ≈2mm 에서 train mae 12mm 로 80 epoch 정체 → δ≈20mm 에서 4.7mm. σ_net 이 작은 FK 라벨
    # (σ=0.1mm)도 이 바닥 덕에 같이 구제된다 (huber_delta_sigma 만 올려서는 안 되던 부분).
    huber_delta_min_mm: float = 20.0
    huber_delta_min_deg: float = 2.0
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
    beta_warmup_epochs: int = 5         # KL 워밍업 (0 → beta_kl 선형)
    amp: bool = False
    device: str = "auto"
    num_workers: int = 0
    seed: int = 0
    grad_clip: float = 1.0
    z_samples_eval: int = 32            # M (§3.3 🟡)
    gamma_mode_consistency: float = 0.2  # γ (3.3) — Q̂ 스케일 대비 🟡
    log_every: int = 20
    max_train_samples: Optional[int] = None   # 디버그용 서브샘플
    # 가중치만 가져올 체크포인트. --resume 과 달리 옵티마이저·epoch·best 는 새로 시작한다 —
    # 다른 목표로 다시 학습하는 것이지 같은 학습을 잇는 것이 아니다.
    init_from: str = ""
    # "quality" = 품질 헤드(quality_head · quality_base)만 학습한다. 나머지는 가중치를 얼리고
    # **eval 모드로 묶는다** — frame_encoder 의 BatchNorm 은 requires_grad 를 꺼도 train 모드면
    # running 통계가 바뀌어, 얼린 인코더가 조용히 변하고 그 위의 행동 디코더도 따라 변한다.
    # 행동 후보를 만드는 부분이 그대로여야 "후보는 같고 줄 세우는 기준만 바뀐" 비교가 된다.
    train_only: str = ""                # "" | "quality"
    augment: bool = True
    # 체크포인트 기준 (§5.3g 2026-09-10). "select_nmae" = 실행시 경로(사전분포 z + Q̂)의 σ 정규화 오차.
    # "total" 은 사후분포 경로(라벨이 z 로 들어감)라 누설이 심해질수록 좋아진다 — exp1 은 그 기준으로
    # ep19 의 가장 많이 누설된 체크포인트를 best 로 골랐다.
    checkpoint_metric: str = "select_nmae"    # "select_nmae" | "total"
    # β 자동 조정 — train.py 헤더의 선택 규칙(gap ≤ σ_net 이면서 vy_std > 0 인 최소 β)을 epoch 마다 적용.
    # exp1(β=0.5 고정)은 ep5~7 붕괴 → ep19 누설로 넘어갔다. 고정 β 하나로는 양쪽을 다 못 피한다.
    # 고정 β 스윕을 하려면 beta_adapt=false.
    beta_adapt: bool = True
    beta_adapt_target_sigma: float = 1.0      # 누설 상한: leak_gap ≤ 이 값 × σ_net,y
    beta_adapt_collapse_sigma: float = 0.25   # 붕괴 하한: vy_std < 이 값 × σ_net,y
    beta_adapt_rate: float = 1.5              # epoch 당 최대 배율
    # z 다양성은 Q̂ 가 그중에서 옳은 걸 고를 수 있을 때만 값어치가 있다. 선택기가 우연 수준이면
    # β 를 낮춰 봐야 후보만 흩어지고 실행시 오차가 커진다 (exp2 ep5→6: β 0.5→0.333 에서
    # sel_nmae 9.70→10.86). select_vy_sign_acc 가 이 값을 넘을 때만 하향을 허용한다.
    beta_adapt_selector_acc: float = 0.55
    beta_min: float = 0.05
    beta_max: float = 32.0


@dataclass
class PolicyConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    imu: ImuConfig = field(default_factory=ImuConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    offset_filter: OffsetFilterConfig = field(default_factory=OffsetFilterConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    SECTIONS = ("paths", "timing", "imu", "labels", "perception", "dataset", "offset_filter", "split",
                "model", "loss", "train")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PolicyConfig":
        sections = {
            "paths": PathsConfig, "timing": TimingConfig, "imu": ImuConfig,
            "labels": LabelConfig, "perception": PerceptionConfig, "dataset": DatasetConfig,
            "offset_filter": OffsetFilterConfig, "split": SplitConfig,
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
