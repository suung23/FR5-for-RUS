"""Config-driven training loop for the Standard U-Net and Slim U-Net models.

The same loop trains all three experimental conditions:

* Standard U-Net baseline (spatial loss only),
* paper-based Slim U-Net (spatial loss only),
* Slim U-Net with temporal-consistency regularisation.

Temporal training feeds *pairs* of adjacent frames through the **shared**
network, warps the previous probability map into the current frame using the
detached backward flow, and adds the reliability-gated temporal terms to the
spatial loss. The network itself never sees more than one frame at a time, so
the trained weights deploy as a stateless single-frame model.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.augment import AugmentationConfig
from ..data.image_dataset import UltrasoundFrameDataset
from ..data.manifest import Manifest
from ..data.video_dataset import SequentialUltrasoundDataset
from ..flow.reliability import ReliabilityConfig, compute_reliability
from ..flow.warp import warp_backward
from ..losses.segmentation import SpatialSegmentationLoss
from ..losses.temporal import TemporalConsistencyLoss
from ..utils.checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from ..utils.config import Config, ConfigError
from ..utils.seeding import make_generator, worker_init_fn

logger = logging.getLogger(__name__)

__all__ = ["TrainerState", "Trainer", "build_optimizer", "build_scheduler", "resolve_device"]


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve a device string, falling back to CPU when CUDA is unavailable."""
    if name in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("Device %r requested but CUDA is unavailable; using CPU.", name)
        return torch.device("cpu")
    return device


def build_optimizer(parameters: Any, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Build an optimizer from a ``train.optimizer`` config section.

    Raises:
        ConfigError: If the optimizer name is unknown.
    """
    name = str(config.get("name", "adam")).lower()
    lr = float(config.get("lr", 1e-3))
    weight_decay = float(config.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(config.get("momentum", 0.9)),
            nesterov=bool(config.get("nesterov", False)),
        )
    if name == "rmsprop":
        return torch.optim.RMSprop(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(config.get("momentum", 0.0)),
        )
    raise ConfigError(
        f"Unknown optimizer {name!r}. Available: adam, adamw, sgd, rmsprop."
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any], epochs: int
) -> tuple[Optional[Any], bool]:
    """Build an LR scheduler.

    Returns:
        ``(scheduler, steps_on_metric)``; ``steps_on_metric`` is ``True`` for
        ``ReduceLROnPlateau``, which must be stepped with the validation score.

    Raises:
        ConfigError: If the scheduler name is unknown.
    """
    name = str(config.get("name", "none")).lower()
    if name in ("none", "", "null"):
        return None, False
    if name == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(config.get("t_max", epochs)),
                eta_min=float(config.get("eta_min", 0.0)),
            ),
            False,
        )
    if name == "step":
        return (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(config.get("step_size", max(1, epochs // 3))),
                gamma=float(config.get("gamma", 0.1)),
            ),
            False,
        )
    if name == "plateau":
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=float(config.get("factor", 0.5)),
                patience=int(config.get("patience", 5)),
            ),
            True,
        )
    raise ConfigError(f"Unknown scheduler {name!r}. Available: none, cosine, step, plateau.")


@dataclass
class TrainerState:
    """Mutable training progress, persisted in checkpoints."""

    epoch: int = 0
    global_step: int = 0
    best_score: float = -math.inf
    best_epoch: int = -1
    history: list[dict[str, Any]] = field(default_factory=list)


class Trainer:
    """Trains a segmentation model according to a validated :class:`Config`.

    Args:
        config: The full experiment configuration.
        model: Model to train.
        train_manifest: Manifest restricted to the training split.
        val_manifest: Manifest restricted to the validation split.
        output_dir: Destination for checkpoints and logs. Defaults to
            ``train.checkpoint_dir``.
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_manifest: Manifest,
        val_manifest: Optional[Manifest] = None,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest

        train_cfg = config.section("train")
        data_cfg = config.section("data")
        loss_cfg = config.section("loss")
        temporal_cfg = dict(loss_cfg.get("temporal") or {})

        self.device = resolve_device(str(train_cfg.get("device", "auto")))
        self.epochs = int(train_cfg.get("epochs", 10))
        self.batch_size = int(train_cfg.get("batch_size", 8))
        self.grad_clip = train_cfg.get("grad_clip")
        self.use_temporal = bool(temporal_cfg.get("enabled", False))
        self.output_dir = Path(output_dir or train_cfg.get("checkpoint_dir", "checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_size = self._resolve_image_size()
        self.model = self.model.to(self.device)

        # -- losses ---------------------------------------------------------
        self.spatial_loss = SpatialSegmentationLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
            jaccard_weight=float(loss_cfg.get("jaccard_weight", 1.0)),
            smooth=float(loss_cfg.get("smooth", 1.0)),
            eps=float(loss_cfg.get("eps", 1e-7)),
            pos_weight=loss_cfg.get("pos_weight"),
        ).to(self.device)

        self.temporal_loss = TemporalConsistencyLoss(
            lambda_pixel=float(temporal_cfg.get("lambda_temp_pixel", 0.2)),
            lambda_control=float(temporal_cfg.get("lambda_temp_control", 0.05)),
            warmup_epochs=int(temporal_cfg.get("temporal_warmup_epochs", temporal_cfg.get("warmup_epochs", 5))),
            ramp_epochs=int(temporal_cfg.get("ramp_epochs", 0)),
            enabled=self.use_temporal,
            pixel_kwargs={
                key: value
                for key, value in {
                    "robust_kind": temporal_cfg.get("robust_kind"),
                    "charbonnier_eps": temporal_cfg.get("charbonnier_eps"),
                    "min_reliable_ratio": temporal_cfg.get("min_reliable_ratio"),
                    "reliable_pixel_threshold": temporal_cfg.get("reliable_pixel_threshold"),
                    "detach_target": temporal_cfg.get("detach_target"),
                }.items()
                if value is not None
            },
            control_kwargs={
                key: value
                for key, value in {
                    "centroid_weight": temporal_cfg.get("centroid_weight"),
                    "area_weight": temporal_cfg.get("area_weight"),
                    "min_mass": temporal_cfg.get("min_mass"),
                    "detach_target": temporal_cfg.get("detach_target"),
                }.items()
                if value is not None
            },
        ).to(self.device)

        self.reliability_config = ReliabilityConfig.from_dict(
            config.get("flow.reliability", {})
        )

        # -- optimisation ---------------------------------------------------
        self.optimizer = build_optimizer(self.model.parameters(), train_cfg.get("optimizer", {}))
        self.scheduler, self.scheduler_on_metric = build_scheduler(
            self.optimizer, train_cfg.get("scheduler", {}) or {}, self.epochs
        )
        self.amp = bool(train_cfg.get("amp", False)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        if bool(train_cfg.get("amp", False)) and not self.amp:
            logger.info("Mixed precision requested but the device is %s; disabled.", self.device.type)

        selection = dict(train_cfg.get("model_selection") or {})
        self.selection_spatial_weight = float(selection.get("spatial_weight", 0.8))
        self.selection_temporal_weight = float(selection.get("temporal_weight", 0.2))
        if not self.use_temporal:
            self.selection_temporal_weight = 0.0
        if self.selection_spatial_weight <= 0:
            raise ConfigError(
                "train.model_selection.spatial_weight must be > 0: model selection must "
                "primarily reward spatial accuracy, since temporal stability alone does "
                "not imply anatomical correctness."
            )

        self.seed = int(config.get("experiment.seed", 42))
        self.state = TrainerState()
        self.data_cfg = data_cfg
        self.train_cfg = train_cfg

        self.train_loader = self._build_loader(train_manifest, training=True)
        self.val_loader = (
            self._build_loader(val_manifest, training=False) if val_manifest is not None else None
        )

    # -- setup helpers -----------------------------------------------------
    def _resolve_image_size(self) -> tuple[int, int]:
        size = self.config.get("model.input_size") or self.config.get("data.image_size") or [128, 128]
        return (int(size[0]), int(size[1]))

    def _build_loader(self, manifest: Manifest, training: bool) -> DataLoader:
        """Build a DataLoader for one split."""
        augmentation = (
            AugmentationConfig.from_dict(self.config.section("augmentation"))
            if training
            else AugmentationConfig(enabled=False)
        )
        common = dict(
            image_size=self.image_size,
            intensity_normalization=str(self.data_cfg.get("intensity_normalization", "zero_one")),
            normalization_stats=self.data_cfg.get("normalization_stats"),
            seed=self.seed,
        )
        if self.use_temporal:
            dataset: Any = SequentialUltrasoundDataset(
                manifest,
                interval=int(self.data_cfg.get("temporal_interval", 1)),
                augmentation=augmentation,
                require_labeled_current=bool(self.data_cfg.get("require_labeled_current", True)),
                require_flow=bool(self.config.get("flow.require_flow", False)),
                flow_backend=self._maybe_flow_backend(),
                **common,
            )
        else:
            dataset = UltrasoundFrameDataset(
                manifest, augmentation=augmentation, labeled_only=True, **common
            )

        num_workers = int(self.train_cfg.get("num_workers", 0))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=training,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            generator=make_generator(self.seed) if training else None,
            persistent_workers=num_workers > 0,
        )

    def _maybe_flow_backend(self) -> Optional[Any]:
        """Build an on-the-fly flow backend when the config asks for one."""
        flow_cfg = self.config.section("flow")
        backend = str(flow_cfg.get("backend", "precomputed"))
        if backend in ("precomputed", "none", ""):
            return None
        from ..flow.backends import build_flow_backend

        logger.warning(
            "flow.backend=%r computes optical flow on the fly for every epoch. "
            "Precomputing it with scripts/precompute_flow.py is much faster.",
            backend,
        )
        return build_flow_backend(flow_cfg)

    # -- training ----------------------------------------------------------
    def _forward_pair(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run the shared model on both frames of a pair and warp the previous map."""
        image_previous = batch["image_previous"].to(self.device, non_blocking=True)
        image_current = batch["image_current"].to(self.device, non_blocking=True)

        logits_current = self.model(image_current)
        logits_previous = self.model(image_previous)

        prob_current = torch.sigmoid(logits_current)
        prob_previous = torch.sigmoid(logits_previous)

        flow_backward = batch["flow_backward"].to(self.device, non_blocking=True)
        flow_forward = batch["flow_forward"].to(self.device, non_blocking=True)

        # The flow is auxiliary evidence: it is detached and no gradient reaches
        # any flow estimator. Gradients do flow through the warped probability
        # map into the shared network, which is what makes this a regulariser.
        prob_previous_warped, _ = warp_backward(prob_previous, flow_backward.detach())

        reliability = compute_reliability(
            flow_backward=flow_backward,
            flow_forward=flow_forward,
            image_current=image_current,
            image_previous=image_previous,
            config=self.reliability_config,
        ).reliability
        reliability = (
            reliability
            * batch["precomputed_reliability"].to(self.device, non_blocking=True)
            * batch["geometry_valid"].to(self.device, non_blocking=True)
        ).clamp(0.0, 1.0)

        return {
            "logits_current": logits_current,
            "logits_previous": logits_previous,
            "prob_current": prob_current,
            "prob_previous_warped": prob_previous_warped,
            "reliability": reliability,
            "mask_current": batch["mask_current"].to(self.device, non_blocking=True),
            "mask_previous": batch["mask_previous"].to(self.device, non_blocking=True),
            "labeled_current": batch["labeled_current"].to(self.device, non_blocking=True),
            "labeled_previous": batch["labeled_previous"].to(self.device, non_blocking=True),
            "pair_valid": batch["flow_valid"].to(self.device, non_blocking=True),
        }

    def _compute_losses(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the total loss and a flat dictionary of logged components."""
        logs: dict[str, float] = {}

        if not self.use_temporal:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            weights = batch["labeled"].to(self.device, non_blocking=True)
            logits = self.model(images)
            spatial = self.spatial_loss(logits, masks, weights)
            logs.update(spatial.as_log_dict())
            logs["loss/total"] = float(spatial.total.detach())
            return spatial.total, logs

        forward = self._forward_pair(batch)

        spatial_current = self.spatial_loss(
            forward["logits_current"], forward["mask_current"], forward["labeled_current"]
        )
        spatial_previous = self.spatial_loss(
            forward["logits_previous"], forward["mask_previous"], forward["labeled_previous"]
        )
        # Average the two frames' supervision so a pair contributes the same
        # spatial gradient magnitude as a single supervised frame would.
        spatial_total = 0.5 * (spatial_current.total + spatial_previous.total)

        temporal = self.temporal_loss(
            prob_current=forward["prob_current"],
            prob_previous_warped=forward["prob_previous_warped"],
            reliability=forward["reliability"],
            epoch=self.state.epoch,
            pair_valid=forward["pair_valid"],
        )

        total = spatial_total + temporal.total
        logs.update(spatial_current.as_log_dict())
        logs.update(temporal.as_log_dict())
        logs["loss/spatial_previous"] = float(spatial_previous.total.detach())
        logs["loss/total"] = float(total.detach())
        return total, logs

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch and return the mean of every logged term."""
        self.model.train()
        accumulator: dict[str, float] = {}
        batches = 0
        nonfinite_batches = 0
        started = time.perf_counter()

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp):
                loss, logs = self._compute_losses(batch)

            if not torch.isfinite(loss):
                nonfinite_batches += 1
                logger.warning(
                    "Non-finite loss at step %d; skipping this batch.", self.state.global_step
                )
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.grad_clip)
                )
                logs["train/grad_norm"] = float(grad_norm)
                if not torch.isfinite(grad_norm):
                    nonfinite_batches += 1
                    logger.warning(
                        "Non-finite gradient norm at step %d; skipping the update.",
                        self.state.global_step,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    # The gradients of this step have already been unscaled, and
                    # under AMP a non-finite gradient is precisely the signal the
                    # scaler exists to react to. Skipping the update() would leave
                    # the scaler in its unscaled state, so the next iteration's
                    # unscale_() would raise instead of training.
                    self.scaler.update()
                    continue
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key, value in logs.items():
                accumulator[key] = accumulator.get(key, 0.0) + float(value)
            batches += 1
            self.state.global_step += 1

        if batches == 0:
            raise RuntimeError(
                "No training batches produced a finite loss. Check the data, the "
                "learning rate and the loss weights."
            )
        summary = {key: value / batches for key, value in accumulator.items()}
        summary["train/epoch_seconds"] = time.perf_counter() - started
        summary["train/batches"] = float(batches)
        summary["train/nonfinite_batches"] = float(nonfinite_batches)
        summary["train/lr"] = float(self.optimizer.param_groups[0]["lr"])
        return summary

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluate on the validation split, returning Dice, IoU and temporal IoU."""
        if self.val_loader is None:
            return {}
        self.model.eval()

        threshold = float(self.config.get("postprocess.threshold", 0.5))
        dice_values: list[float] = []
        iou_values: list[float] = []
        temporal_iou_values: list[float] = []
        losses: list[float] = []

        for batch in self.val_loader:
            if self.use_temporal:
                images = batch["image_current"].to(self.device)
                masks = batch["mask_current"].to(self.device)
                weights = batch["labeled_current"].to(self.device)
            else:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                weights = batch["labeled"].to(self.device)

            logits = self.model(images)
            spatial = self.spatial_loss(logits, masks, weights)
            losses.append(float(spatial.total))

            predicted = (torch.sigmoid(logits) >= threshold).float()
            selected = weights.reshape(-1) > 0
            if selected.any():
                p = predicted[selected].flatten(1)
                t = masks[selected].flatten(1)
                intersection = (p * t).sum(dim=1)
                union = p.sum(dim=1) + t.sum(dim=1) - intersection
                total = p.sum(dim=1) + t.sum(dim=1)
                dice_values.extend(
                    ((2 * intersection + 1e-7) / (total + 1e-7)).detach().cpu().tolist()
                )
                iou_values.extend(
                    ((intersection + 1e-7) / (union + 1e-7)).detach().cpu().tolist()
                )

            if self.use_temporal:
                previous_logits = self.model(batch["image_previous"].to(self.device))
                flow_backward = batch["flow_backward"].to(self.device)
                warped, _ = warp_backward(torch.sigmoid(previous_logits), flow_backward)
                warped_mask = (warped >= threshold).float()
                valid = batch["flow_valid"].to(self.device).reshape(-1) > 0
                if valid.any():
                    p = predicted[valid].flatten(1)
                    w = warped_mask[valid].flatten(1)
                    intersection = (p * w).sum(dim=1)
                    union = p.sum(dim=1) + w.sum(dim=1) - intersection
                    temporal_iou_values.extend(
                        ((intersection + 1e-7) / (union + 1e-7)).detach().cpu().tolist()
                    )

        metrics = {
            "val/loss": float(np.mean(losses)) if losses else float("nan"),
            "val/dice": float(np.mean(dice_values)) if dice_values else 0.0,
            "val/iou": float(np.mean(iou_values)) if iou_values else 0.0,
        }
        if temporal_iou_values:
            metrics["val/temporal_iou"] = float(np.mean(temporal_iou_values))
        metrics["val/selection_score"] = self.selection_score(metrics)
        return metrics

    def selection_score(self, metrics: dict[str, float]) -> float:
        """Combine validation metrics into the model-selection score.

        The score deliberately weights **spatial accuracy** most heavily and adds
        only a smaller temporal-stability term, because a perfectly stable but
        anatomically wrong prediction must never win model selection.
        """
        spatial = float(metrics.get("val/dice", 0.0))
        temporal = float(metrics.get("val/temporal_iou", 0.0))
        total_weight = self.selection_spatial_weight + self.selection_temporal_weight
        if total_weight <= 0:
            return spatial
        return (
            self.selection_spatial_weight * spatial + self.selection_temporal_weight * temporal
        ) / total_weight

    # -- checkpointing -----------------------------------------------------
    def _metadata(self, best_score: float) -> CheckpointMetadata:
        return CheckpointMetadata(
            epoch=self.state.epoch,
            best_metric=float(best_score),
            best_metric_name="val/selection_score",
            model_version=str(
                self.config.get(
                    "model.version",
                    f"{self.config.get('model.name')}"
                    f"{'-' + str(self.config.get('model.preset')) if self.config.get('model.preset') else ''}",
                )
            ),
            seed=self.seed,
            config=self.config.to_dict(),
        )

    def save(self, name: str, best_score: Optional[float] = None) -> Path:
        """Write a checkpoint into the output directory."""
        return save_checkpoint(
            self.output_dir / name,
            model=self.model,
            metadata=self._metadata(
                self.state.best_score if best_score is None else best_score
            ),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler if self.amp else None,
            extra={"global_step": self.state.global_step, "history": self.state.history},
        )

    def resume(self, path: str | Path) -> None:
        """Restore model, optimizer, scheduler, scaler and progress from a checkpoint.

        Raises:
            RuntimeError: If the checkpoint's weights do not fit the model.
        """
        payload = load_checkpoint(path, map_location=self.device)
        try:
            self.model.load_state_dict(payload["model_state"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint {path} does not match the configured model architecture: {exc}"
            ) from exc

        if payload.get("optimizer_state"):
            self.optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scheduler_state") and self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler_state"])
        if payload.get("scaler_state") and self.amp:
            self.scaler.load_state_dict(payload["scaler_state"])

        self.state.epoch = int(payload.get("epoch", 0)) + 1
        self.state.global_step = int(payload.get("global_step", 0))
        self.state.best_score = float(payload.get("best_metric", -math.inf))
        self.state.history = list(payload.get("history", []))
        logger.info(
            "Resumed from %s at epoch %d (best %s=%.4f)",
            path,
            self.state.epoch,
            payload.get("best_metric_name", "score"),
            self.state.best_score,
        )

    # -- driver ------------------------------------------------------------
    def fit(self) -> TrainerState:
        """Run the full training schedule, saving ``last.pt`` and ``best.pt``.

        Returns:
            The final :class:`TrainerState`.
        """
        logger.info(
            "Training %s for %d epoch(s) on %s | temporal=%s | train batches=%d",
            type(self.model).__name__,
            self.epochs,
            self.device,
            self.use_temporal,
            len(self.train_loader),
        )
        parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("Trainable parameters: %s", f"{parameters:,}")

        start_epoch = self.state.epoch
        for epoch in range(start_epoch, self.epochs):
            self.state.epoch = epoch
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            record = {"epoch": epoch, **train_metrics, **val_metrics}
            self.state.history.append(record)

            if self.scheduler is not None:
                if self.scheduler_on_metric:
                    self.scheduler.step(val_metrics.get("val/selection_score", 0.0))
                else:
                    self.scheduler.step()

            logger.info(
                "epoch %d/%d | loss=%.4f | val_dice=%.4f | val_temporal_iou=%s | "
                "score=%.4f | temporal_weight=%.2f | %.1fs",
                epoch + 1,
                self.epochs,
                train_metrics.get("loss/total", float("nan")),
                val_metrics.get("val/dice", float("nan")),
                f"{val_metrics['val/temporal_iou']:.4f}" if "val/temporal_iou" in val_metrics else "n/a",
                val_metrics.get("val/selection_score", float("nan")),
                self.temporal_loss.weight_at(epoch),
                train_metrics.get("train/epoch_seconds", 0.0),
            )

            score = val_metrics.get("val/selection_score")
            improved = score is not None and score > self.state.best_score
            if improved:
                self.state.best_score = float(score)
                self.state.best_epoch = epoch
            # last.pt is written after the best score is updated, so both
            # checkpoints agree on what the best score so far was.
            self.save("last.pt")
            if improved:
                self.save("best.pt", best_score=self.state.best_score)
                logger.info("New best model at epoch %d (score=%.4f)", epoch + 1, score)

        if self.state.best_epoch < 0 and self.val_loader is not None:
            logger.warning(
                "No epoch improved the validation score; best.pt may not exist. "
                "Model selection never falls back to training loss."
            )
        return self.state
