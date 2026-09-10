"""프레임별 지각 특징 s_t, Q_seg, 액션 토큰 τ=(a, ê, Q).

U-Net (``Unet_seg/rus_perception``) 을 세션의 모든 프레임에 돌려 ``ControlState`` 를
얻고, 그중 policy 관측에 들어갈 스칼라만 고정 순서의 벡터로 뽑는다. 결과는 세션·
체크포인트별로 캐시한다 (npz) — U-Net 추론이 데이터셋 빌드에서 가장 비싼 단계다.

주의 (§3.8): ControlState 특징만으로는 ±d 부호 정보가 지각 단계에서 사라진다.
원시 프레임 m 장이 관측의 하중 부분이고, 이 벡터는 보조다. ROI 를 방광으로 crop 하지
않는다.

액션 토큰 규칙 (Paper/build_lateral_instruction_manuscript.py 표 1):
    contrast ≤ 0 (또는 마스크 없음)      → check-filling  (게이트)
    contrast > 0, |ê| < deadband       → hold           (데드밴드)
    contrast > 0, ê > 0                → move-left      (ê = A − ĉ: 루멘이 빔축 왼쪽)
    contrast > 0, ê < 0                → move-right
"""

from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import PerceptionConfig, PolicyConfig, UNET_ROOT

logger = logging.getLogger(__name__)

STATE_FEATURE_NAMES: tuple[str, ...] = (
    "has_mask", "valid_for_control",
    "centroid_dx", "centroid_dy",          # 정규화 중심 − 0.5 (state.py 규약: +x 오른쪽, +y 아래)
    "area_ratio", "major_axis_norm", "minor_axis_norm", "orient_cos2", "orient_sin2",
    "segmentation_confidence", "lumen_contrast", "contrast_valid",
    "border_contact_ratio", "largest_component_ratio", "boundary_entropy",
    "quality",                              # Q_seg (control_quality_score)
    "e_hat_norm",                           # (A − ĉ)/W
    "token_check_filling", "token_hold", "token_move_left", "token_move_right",
)
STATE_DIM = len(STATE_FEATURE_NAMES)
TOKEN_NAMES = ("check-filling", "hold", "move-left", "move-right")
TOKEN_CHECK_FILLING, TOKEN_HOLD, TOKEN_MOVE_LEFT, TOKEN_MOVE_RIGHT = range(4)


def action_token(e_hat_px: Optional[float], contrast: Optional[float], deadband_px: float) -> int:
    """τ 의 이산 상태 a. 게이트가 선결이고, 데드밴드가 그 다음."""
    if contrast is None or not np.isfinite(contrast) or contrast <= 0.0 or e_hat_px is None:
        return TOKEN_CHECK_FILLING
    if abs(e_hat_px) < deadband_px:
        return TOKEN_HOLD
    return TOKEN_MOVE_LEFT if e_hat_px > 0 else TOKEN_MOVE_RIGHT


def apply_frame_transform(frames: np.ndarray, transform: str) -> np.ndarray:
    """candidate 프레임 방향 보정 (⏳ scan conversion 검증 전). (…,H,W) 배열."""
    if transform in ("none", "", None):
        return frames
    if transform == "rot90_cw":
        return np.rot90(frames, k=-1, axes=(-2, -1))
    if transform == "rot90_ccw":
        return np.rot90(frames, k=1, axes=(-2, -1))
    if transform == "rot180":
        return np.rot90(frames, k=2, axes=(-2, -1))
    if transform == "flip_h":
        return frames[..., :, ::-1]
    if transform == "flip_v":
        return frames[..., ::-1, :]
    raise ValueError(f"알 수 없는 frame_transform: {transform!r}")


@dataclass
class PerceptionResult:
    state: np.ndarray        # (N, STATE_DIM) float32
    quality: np.ndarray      # (N,) float32, NaN = 측정 안 됨
    token: np.ndarray        # (N,) int8, −1 = 없음
    e_hat_px: np.ndarray     # (N,) float32, NaN 가능
    backend: str
    checkpoint_id: str
    beam_axis_px: float
    image_size: tuple[int, int]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, state=self.state, quality=self.quality, token=self.token,
                            e_hat_px=self.e_hat_px, backend=self.backend, checkpoint_id=self.checkpoint_id,
                            beam_axis_px=self.beam_axis_px, image_size=np.asarray(self.image_size),
                            feature_names=np.asarray(STATE_FEATURE_NAMES))

    @classmethod
    def load(cls, path: Path) -> "PerceptionResult":
        z = np.load(path, allow_pickle=False)
        names = tuple(str(n) for n in z["feature_names"])
        if names != STATE_FEATURE_NAMES:
            raise ValueError(f"캐시 {path} 의 특징 순서가 현재 코드와 다릅니다 — 캐시를 지우고 다시 빌드하십시오")
        return cls(state=z["state"], quality=z["quality"], token=z["token"], e_hat_px=z["e_hat_px"],
                   backend=str(z["backend"]), checkpoint_id=str(z["checkpoint_id"]),
                   beam_axis_px=float(z["beam_axis_px"]), image_size=tuple(int(v) for v in z["image_size"]))


def empty_result(n: int, image_size: tuple[int, int], backend: str = "none") -> PerceptionResult:
    """backend=none: 특징 0, Q NaN, 토큰 −1. Q̂ loss 는 마스크된다."""
    return PerceptionResult(
        state=np.zeros((n, STATE_DIM), np.float32), quality=np.full(n, np.nan, np.float32),
        token=np.full(n, -1, np.int8), e_hat_px=np.full(n, np.nan, np.float32),
        backend=backend, checkpoint_id="none", beam_axis_px=image_size[1] / 2.0, image_size=image_size,
    )


# --------------------------------------------------------------------------- U-Net 백엔드
def _bootstrap_unet_path() -> None:
    if str(UNET_ROOT) not in sys.path:
        sys.path.insert(0, str(UNET_ROOT))


class UnetPerception:
    """Unet_seg 의 Predictor + extract_control_state 를 세션 프레임에 돌린다."""

    def __init__(self, checkpoint: Path, unet_config: Path, device: str = "auto",
                 hold_deadband_px: float = 8.0916):
        _bootstrap_unet_path()
        import torch
        from rus_perception.control.features import FeatureExtractionConfig, extract_control_state
        from rus_perception.control.roi import RoiConfig, build_roi_mask
        from rus_perception.inference.predictor import Predictor, PredictorConfig
        from rus_perception.utils.config import load_config as load_unet_config

        self._extract = extract_control_state
        cfg = load_unet_config(str(unet_config))
        size = tuple(int(v) for v in cfg.section("data")["image_size"])
        roi_dict = dict(cfg.section("control").get("roi") or {})
        # roi.path 는 Unet_seg 스크립트처럼 CWD 기준 상대경로다 — 여기서는 Unet_seg 루트 기준으로 푼다
        if roi_dict.get("path") and not Path(str(roi_dict["path"])).is_absolute():
            cand = UNET_ROOT / str(roi_dict["path"])
            if cand.is_file():
                roi_dict["path"] = str(cand)
        roi_cfg = RoiConfig.from_dict(roi_dict or None)
        roi = build_roi_mask(size, roi_cfg)
        self.roi = None if roi is None else (np.asarray(roi) > 0)
        self.image_size = size
        self.beam_axis_px = float(np.nonzero(self.roi)[1].mean()) if self.roi is not None else size[1] / 2.0
        self.feature_config = FeatureExtractionConfig.from_dict(
            {"postprocess": cfg.section("postprocess"), **cfg.section("control")})
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = Predictor.from_checkpoint(
            str(checkpoint), model_config=cfg.section("model"),
            predictor_config=PredictorConfig(
                input_size=size,
                intensity_normalization=str(cfg.get("data.intensity_normalization", "zero_one")),
                normalization_stats=cfg.get("data.normalization_stats"),
                device=device, restore_original_size=False, roi=roi_cfg),
            feature_config=self.feature_config,
        )
        self.checkpoint_id = str(getattr(self.predictor, "checkpoint_identifier", "unknown"))
        self.hold_deadband_px = hold_deadband_px

    def reset(self) -> None:
        """스트리밍 상태를 버린다. 새 세션이나 영상이 끊겼다 돌아왔을 때 부른다."""
        self._prev = None

    def raw_quality(self, frame: np.ndarray) -> float:
        """``Q_raw`` — 세그멘테이션에 기대지 않는 접촉·에코 품질 (DESIGN_NOTES §6.2).

        ``Q_seg`` 와 **다른 신호**다. Q_seg 는 "방광을 제대로 보이게" (영상 축), Q_raw 는
        "일단 제대로 닿게" (힘 축) 이고, 힘 탐색은 Q_raw 를 최대화하는 최소 F_n* 를 찾는다
        (§333). 분할이 아무것도 못 내놓는 구간에서도 돌아야 하므로 신경망도 이전 프레임도
        쓰지 않는다.

        Returns:
            [0, 1] 의 점수. **측정 불가면 NaN** 이다 — 0.0 이 아니다. A-line 이 부족하면
            "나쁘다" 가 아니라 "재지 못했다" 이고, 그것을 0 으로 적으면 힘 탐색이 없는
            열화를 쫓는다.
        """
        from rus_perception.control.raw_quality import compute_raw_quality

        img = np.asarray(frame, np.float32) / 255.0
        r = compute_raw_quality(img, roi_mask=self.roi)
        return float("nan") if r.score is None else float(r.score)

    def step(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, float]:
        """프레임 한 장 — **직전 프레임 상태를 이어받는다**. (state, quality, token, e_hat)

        ``extract_control_state`` 는 ``previous_state`` 를 받아 시간 의존 특징을 만든다
        (§10.2 의 temporal warped IoU · centroid jump, 그리고 ``quality`` 자체). 실시간 경로가
        매 프레임 ``run(frame[None])`` 을 부르면 그 인자가 매번 None 이 되어 **모든 프레임이
        "첫 프레임" 으로 처리된다** — 학습 때 세션을 순차로 돌린 것과 분포가 달라지고,
        state 안의 quality 가 직접 틀어진다. 실시간에서는 이 메서드를 쓴다.
        """
        return self.step_detailed(frame)[:4]

    def step_detailed(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, float, Any]:
        """``step`` 과 같되 ``ControlState`` 를 함께 돌려준다 — 마스크가 필요한 곳용.

        화면에 마스크를 겹쳐 보이려면 특징 벡터가 아니라 ``cs.binary_mask`` 와, 그것이
        어느 그림 위의 마스크인지(= 여기서 실제로 넣은 256² B-mode)가 있어야 한다. 그
        둘을 따로 계산하는 경로를 하나 더 만들면 화면과 정책이 **다른 전처리** 를 보게
        되므로, 지각은 이 메서드 하나로만 돈다. ``step`` 은 앞 넷을 자른 것이다.

        스트리밍 상태(``_prev``)도 같은 것을 쓴다 — 시각화 때문에 시간 의존 특징이
        갈라지지 않는다.
        """
        from rus_perception.data.io import resize_image

        image = np.asarray(frame, np.float32) / 255.0
        image = resize_image(image, self.image_size)
        prob, _ = self.predictor.predict_probability(image)
        cs = self._extract(prob, image=image, previous_state=getattr(self, "_prev", None),
                           config=self.feature_config, roi_mask=self.roi)
        self._prev = cs
        vec, q, tok, e_px = control_state_to_vector(
            cs, self.beam_axis_px, self.image_size, self.hold_deadband_px)
        return vec, q, tok, e_px, cs

    def run(self, frames: np.ndarray, progress: bool = True) -> PerceptionResult:
        """세션 프레임 전체를 순차로. ``step`` 을 처음부터 다시 돌리는 것과 같다."""
        n = len(frames)
        state = np.zeros((n, STATE_DIM), np.float32)
        quality = np.full(n, np.nan, np.float32)
        token = np.full(n, -1, np.int8)
        e_hat = np.full(n, np.nan, np.float32)
        self.reset()
        it = range(n)
        if progress and n > 0:
            try:
                from tqdm import tqdm
                it = tqdm(it, desc="U-Net", unit="frame")
            except ImportError:
                pass
        for i in it:
            state[i], quality[i], token[i], e_hat[i] = self.step(frames[i])
        return PerceptionResult(state=state, quality=quality, token=token, e_hat_px=e_hat, backend="unet",
                                checkpoint_id=self.checkpoint_id, beam_axis_px=self.beam_axis_px,
                                image_size=self.image_size)


def control_state_to_vector(cs: Any, beam_axis_px: float, image_size: tuple[int, int],
                            deadband_px: float) -> tuple[np.ndarray, float, int, float]:
    """ControlState → (STATE_DIM 벡터, Q, 토큰, ê px)."""
    H, W = image_size
    v = np.zeros(STATE_DIM, np.float32)
    has_mask = cs.centroid_x_px is not None and cs.mask_area_px > 0
    v[0] = float(has_mask)
    v[1] = float(bool(cs.valid_for_control))
    e_px = float("nan")
    if has_mask:
        v[2] = float(cs.centroid_x_normalized) - 0.5
        v[3] = float(cs.centroid_y_normalized) - 0.5
        v[4] = float(cs.mask_area_ratio)
        v[5] = float(cs.major_axis_length or 0.0) / W
        v[6] = float(cs.minor_axis_length or 0.0) / W
        th = np.radians(float(cs.orientation_degrees or 0.0))
        v[7], v[8] = np.cos(2 * th), np.sin(2 * th)
        e_px = beam_axis_px - float(cs.centroid_x_px)
    v[9] = float(cs.segmentation_confidence)
    contrast = cs.lumen_surrounding_contrast
    if contrast is not None and np.isfinite(contrast):
        v[10], v[11] = float(contrast), 1.0
    v[12] = float(cs.border_contact_ratio)
    v[13] = float(cs.largest_component_ratio)
    v[14] = float(cs.mean_boundary_entropy)
    q = float(cs.control_quality_score)
    v[15] = q
    v[16] = (e_px / W) if np.isfinite(e_px) else 0.0
    tok = action_token(None if not np.isfinite(e_px) else e_px, contrast, deadband_px)
    v[17 + tok] = 1.0
    return v, q, tok, e_px


# --------------------------------------------------------------------------- 진입점
def perception_cache_path(cfg: PolicyConfig, session_name: str, checkpoint_id: str, bmode_tag: str = "") -> Path:
    key = hashlib.sha1(f"{session_name}|{checkpoint_id}|{cfg.perception.frame_transform}|{bmode_tag}".encode()).hexdigest()[:10]
    return cfg.resolve(cfg.paths.perception_cache_dir) / f"{session_name}__{key}.npz"


def session_bmode_frames(cfg: PolicyConfig, session, progress: bool = False) -> np.ndarray:
    """세션 프레임 → 학습용 B-mode (필요하면 극좌표 → 부채꼴 → 레터박스) → frame_transform. (N, H, W) uint8."""
    from .bmode import BmodeConverter

    raw = np.asarray(session.frames)
    conv = BmodeConverter(session.meta, tuple(raw.shape[1:]) if raw.ndim == 3 else (256, 256),
                          out_size=max(cfg.perception.frame_size))
    frames = conv.convert_all(raw, progress=progress)
    return apply_frame_transform(frames, cfg.perception.frame_transform)


def session_bmode_tag(session) -> str:
    from .bmode import BmodeConverter

    raw = session.frames
    conv = BmodeConverter(session.meta, tuple(raw.shape[1:]) if raw.ndim == 3 else (256, 256))
    d = conv.describe()
    return d["mode"] + ("|ss%d" % d["supersample"] if "supersample" in d else "")


def build_backend(cfg: PolicyConfig) -> Optional[UnetPerception]:
    if cfg.perception.backend == "none":
        return None
    if cfg.perception.backend != "unet":
        raise ValueError(f"perception.backend 는 unet|none: {cfg.perception.backend!r}")
    ckpt = cfg.resolve(cfg.paths.unet_checkpoint)
    ucfg = cfg.resolve(cfg.paths.unet_config)
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"U-Net 체크포인트가 없습니다: {ckpt}\n"
            "  학습 머신의 checkpoints/exp_seed43/best.pt 를 복사하거나 paths.unet_checkpoint 를 바꾸십시오.\n"
            "  지각 없이 파이프라인만 검증하려면 --set perception.backend=none")
    if not ucfg.is_file():
        raise FileNotFoundError(f"U-Net 설정이 없습니다: {ucfg}")
    return UnetPerception(ckpt, ucfg, device=cfg.perception.device,
                          hold_deadband_px=cfg.perception.hold_deadband_px)


def perceive_session(cfg: PolicyConfig, session, backend: Optional[UnetPerception]) -> PerceptionResult:
    """세션 프레임 전체의 지각 결과 (캐시 사용)."""
    frames = session_bmode_frames(cfg, session, progress=True)
    if backend is None:
        return empty_result(len(frames), tuple(frames.shape[1:]) if frames.ndim == 3 else (256, 256))
    path = perception_cache_path(cfg, session.name, backend.checkpoint_id, session_bmode_tag(session))
    if cfg.perception.cache and path.is_file():
        try:
            res = PerceptionResult.load(path)
            if res.state.shape[0] == len(frames):
                logger.info("지각 캐시 사용: %s", path)
                return res
        except Exception as exc:  # noqa: BLE001
            logger.warning("캐시 %s 를 읽지 못해 다시 계산합니다: %s", path, exc)
    res = backend.run(frames)
    if cfg.perception.cache:
        res.save(path)
    return res
