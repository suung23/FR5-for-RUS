"""Single-frame predictor: preprocessing, inference and control-state extraction.

The runtime path is deliberately **stateless with respect to the network**: one
ultrasound frame goes in, one probability map comes out. No recurrence, no
hidden state, no multi-frame input. Temporal information is used only to compare
the new prediction against the previous one *after* the network has run, which
keeps latency low, ONNX export trivial and deployment simple.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from ..control.features import FeatureExtractionConfig, extract_control_state
from ..control.roi import RoiConfig, build_roi_mask
from ..control.state import ControlState
from ..data.io import normalize_intensity, resize_image
from ..models.registry import build_model
from ..utils.checkpoint import load_checkpoint
from ..metrics.latency import synchronize

logger = logging.getLogger(__name__)

__all__ = ["PredictorConfig", "Predictor"]


@dataclass
class PredictorConfig:
    """Runtime configuration of a :class:`Predictor`.

    Attributes:
        input_size: ``(height, width)`` the model is fed.
        intensity_normalization: Preprocessing mode, see
            :func:`rus_perception.data.io.normalize_intensity`.
        normalization_stats: ``{"mean": .., "std": ..}`` for ``mean_std``.
        device: Torch device string.
        amp: Use autocast mixed precision (CUDA only; ignored on CPU).
        channels_last: Use the channels-last memory format.
        restore_original_size: Resize the probability map back to the input
            frame's resolution before postprocessing, so control coordinates
            refer to the original image geometry.
        roi: Imaged-sector description. Built once per frame shape and reused, so
            it costs nothing on the real-time path. ``None`` (or ``mode="full"``)
            means the whole frame, which is the previous behaviour exactly.
    """

    input_size: tuple[int, int] = (128, 128)
    intensity_normalization: str = "zero_one"
    normalization_stats: Optional[dict[str, float]] = None
    device: str = "cpu"
    amp: bool = False
    channels_last: bool = False
    restore_original_size: bool = True
    roi: Optional[RoiConfig] = None


class Predictor:
    """Runs a trained segmentation model on individual ultrasound frames.

    Args:
        model: A model returning raw logits.
        config: Runtime configuration.
        feature_config: Control-feature extraction configuration.
        model_version: Written into every :class:`ControlState`.
        checkpoint_identifier: Written into every :class:`ControlState`.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[PredictorConfig] = None,
        feature_config: Optional[FeatureExtractionConfig] = None,
        model_version: str = "unknown",
        checkpoint_identifier: str = "unknown",
    ) -> None:
        self.config = config or PredictorConfig()
        self.feature_config = feature_config or FeatureExtractionConfig()
        self.model_version = model_version
        self.checkpoint_identifier = checkpoint_identifier

        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; falling back to CPU.")
            self.device = torch.device("cpu")
        if self.config.amp and self.device.type != "cuda":
            logger.info("Mixed precision requested on %s; ignoring (CUDA only).", self.device.type)

        self.model = model.to(self.device).eval()
        if self.config.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # The ROI depends only on the frame shape, and a live source holds one
        # shape for its whole run -- so build it once and keep it.
        self._roi_cache: dict[tuple[int, int], Optional[np.ndarray]] = {}

    # -- construction ------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_config: Optional[dict[str, Any]] = None,
        predictor_config: Optional[PredictorConfig] = None,
        feature_config: Optional[FeatureExtractionConfig] = None,
    ) -> "Predictor":
        """Build a predictor from a checkpoint, reusing its embedded config.

        The checkpoint's own ``model`` section is authoritative: it is the only
        description guaranteed to match the stored weights. ``model_config`` is
        used as a **fallback** for checkpoints that carry no config, not as an
        override, so pointing a config file at a checkpoint trained with a
        different architecture cannot silently produce a shape-mismatch wall.

        Args:
            checkpoint_path: Checkpoint written by
                :func:`rus_perception.utils.checkpoint.save_checkpoint`.
            model_config: Fallback ``model`` section, used only when the
                checkpoint does not carry one.
            predictor_config: Runtime configuration; ``input_size`` defaults to
                the value stored in the checkpoint's config.
            feature_config: Control-feature configuration; defaults to the
                checkpoint's ``postprocess`` and ``control`` sections.

        Returns:
            A ready :class:`Predictor`.

        Raises:
            ValueError: If no model configuration can be determined, or if the
                weights do not fit the resolved architecture.
        """
        payload = load_checkpoint(checkpoint_path, map_location="cpu")
        stored_config = payload.get("config") or {}
        resolved_model_config = stored_config.get("model") or model_config
        if not resolved_model_config:
            raise ValueError(
                f"Checkpoint {checkpoint_path} has no 'model' config section. "
                "Pass model_config explicitly."
            )
        if model_config and stored_config.get("model") and model_config != stored_config["model"]:
            logger.info(
                "Using the model architecture stored in %s; the supplied model config "
                "differs and was ignored (it only applies to checkpoints without one).",
                checkpoint_path,
            )

        model = build_model(resolved_model_config)
        try:
            model.load_state_dict(payload["model_state"])
        except RuntimeError as exc:
            raise ValueError(
                f"The weights in {checkpoint_path} do not fit the architecture described "
                f"by its own config ({resolved_model_config}). The checkpoint is "
                f"inconsistent and cannot be loaded: {exc}"
            ) from exc

        if predictor_config is None:
            input_size = resolved_model_config.get("input_size") or stored_config.get(
                "data", {}
            ).get("image_size", [128, 128])
            predictor_config = PredictorConfig(
                input_size=(int(input_size[0]), int(input_size[1])),
                intensity_normalization=stored_config.get("data", {}).get(
                    "intensity_normalization", "zero_one"
                ),
                normalization_stats=stored_config.get("data", {}).get("normalization_stats"),
            )
        if feature_config is None:
            feature_config = FeatureExtractionConfig.from_dict(
                {
                    "postprocess": stored_config.get("postprocess"),
                    **(stored_config.get("control") or {}),
                }
            )

        return cls(
            model=model,
            config=predictor_config,
            feature_config=feature_config,
            model_version=payload.get("model_version", resolved_model_config.get("name", "unknown")),
            checkpoint_identifier=payload.get("checkpoint_id", "unknown"),
        )

    # -- inference ---------------------------------------------------------
    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Convert a raw frame into a ``1 x 1 x H x W`` model input tensor.

        Args:
            frame: ``H x W`` grayscale or ``H x W x 3`` colour frame. Values may
                be ``uint8`` in ``[0, 255]`` or float in ``[0, 1]``.

        Returns:
            A batched tensor on the predictor's device.

        Raises:
            ValueError: If the frame is not 2-D or 3-D.
        """
        array = np.asarray(frame)
        if array.ndim == 3:
            array = array.mean(axis=2)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2-D or 3-D frame, got shape {array.shape}.")

        array = array.astype(np.float32)
        if array.max() > 1.0 + 1e-6:
            array = array / 255.0
        array = np.clip(array, 0.0, 1.0)

        array = resize_image(array, self.config.input_size)
        stats = self.config.normalization_stats or {}
        array = normalize_intensity(
            array, self.config.intensity_normalization, stats.get("mean"), stats.get("std")
        )

        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
        tensor = tensor.to(self.device)
        if self.config.channels_last:
            tensor = tensor.to(memory_format=torch.channels_last)
        return tensor

    @torch.inference_mode()
    def infer_logits(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run the model, returning raw logits."""
        use_amp = self.config.amp and self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            return self.model(tensor).float()

    @torch.inference_mode()
    def predict_probability(
        self, frame: np.ndarray, output_size: Optional[tuple[int, int]] = None
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Predict a probability map for one frame.

        Args:
            frame: Raw input frame.
            output_size: ``(height, width)`` of the returned map. Defaults to the
                input frame's own size when ``restore_original_size`` is set,
                otherwise to the model's input size.

        Returns:
            ``(probability_map, latencies_ms)``.
        """
        array = np.asarray(frame)
        native = (int(array.shape[0]), int(array.shape[1]))

        start = time.perf_counter()
        tensor = self.preprocess(frame)
        synchronize(self.device)
        preprocess_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        logits = self.infer_logits(tensor)
        synchronize(self.device)
        inference_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        target = output_size or (native if self.config.restore_original_size else self.config.input_size)
        if tuple(logits.shape[-2:]) != tuple(target):
            logits = torch.nn.functional.interpolate(
                logits, size=tuple(target), mode="bilinear", align_corners=False
            )
        # Sigmoid is applied here -- never inside the model's forward pass.
        probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
        synchronize(self.device)
        resize_ms = (time.perf_counter() - start) * 1000.0

        return probability, {
            "preprocessing_latency_ms": preprocess_ms,
            "inference_latency_ms": inference_ms,
            "postprocessing_latency_ms": resize_ms,
        }

    def roi_for(self, shape: tuple[int, int]) -> Optional[np.ndarray]:
        """Return the cached imaged-sector mask for a frame shape, building it once.

        Returns ``None`` for a whole-frame ROI, which every consumer treats as
        "no mask" without allocating.
        """
        key = (int(shape[0]), int(shape[1]))
        if key not in self._roi_cache:
            self._roi_cache[key] = build_roi_mask(key, self.config.roi)
        return self._roi_cache[key]

    def predict_control_state(
        self,
        frame: np.ndarray,
        previous_state: Optional[ControlState] = None,
        flow: Optional[np.ndarray] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ControlState:
        """Predict and extract a full :class:`ControlState` for one frame.

        Args:
            frame: Raw input frame.
            previous_state: Previous frame's state, for temporal features.
            flow: ``H x W x 2`` backward flow from this frame into the previous
                one, at the probability map's resolution.
            metadata: Provenance merged into the state.

        Returns:
            The populated :class:`ControlState`.
        """
        started = time.perf_counter()
        probability, latencies = self.predict_probability(frame)

        image = np.asarray(frame)
        if image.ndim == 3:
            image = image.mean(axis=2)
        image = image.astype(np.float32)
        if image.max() > 1.0 + 1e-6:
            image = image / 255.0
        if image.shape != probability.shape:
            image = resize_image(np.clip(image, 0.0, 1.0), probability.shape)

        roi_mask = self.roi_for(probability.shape)
        state = extract_control_state(
            probability_map=probability,
            image=image,
            previous_state=previous_state,
            flow=flow,
            metadata={
                "model_version": self.model_version,
                "checkpoint_id": self.checkpoint_identifier,
                "roi_mode": (self.config.roi.mode if self.config.roi else "full"),
                **(metadata or {}),
            },
            config=self.feature_config,
            latencies=latencies,
            roi_mask=roi_mask,
        )
        state.end_to_end_latency_ms = (time.perf_counter() - started) * 1000.0
        return state
