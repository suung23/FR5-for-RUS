"""학습 루프 + 진단.

진단 (§5.3g · §6 "조용한 실패"):
  * mode_collapse_vy_std  검증 관측에서 z 를 M 개 뽑아 v_y 순변위의 표준편차. 0 으로 가면 붕괴.
  * leak_gap_mm           select_mae_y (사전분포 z) − mae_y (사후분포 z). σ_net 을 크게 넘으면 **누설** —
                          인코더가 라벨을 z 로 흘려보내고 디코더가 관측을 무시한다 (§5.3g 2026-09-08).
                          β 선택 규칙: gap ≤ σ_net 이면서 vy_std > 0 인 최소 β.
  * KL 워밍업             β 를 train.beta_warmup_epochs 동안 0 → loss.beta_kl 로 올린다.
  * mode_margin           Q̂ 로 고른 모드와 반대 부호 모드의 점수 차. 잡음 수준이면 대칭 붕괴 (L11).
  * vy_sign_acc           |Δy| > σ 인 샘플에서 v_y 부호 정확도.
  * mae_*                 축별 순변위 절대오차 (mm, mm, deg).

체크포인트: <output_dir>/last.pt, best.pt (검증 total 기준). 이력: metrics.jsonl.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PolicyConfig, save_config
from .dataset import PolicyH5Dataset
from .losses import compute_loss
from .model import ActPolicy, build_policy, count_parameters

logger = logging.getLogger(__name__)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_loaders(cfg: PolicyConfig, dataset_path: Optional[Path] = None
                 ) -> tuple[DataLoader, Optional[DataLoader], PolicyH5Dataset, Optional[PolicyH5Dataset]]:
    path = cfg.resolve(cfg.paths.dataset) if dataset_path is None else Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"데이터셋이 없습니다: {path} — 먼저 scripts/build_dataset.py 를 실행하십시오")
    tr = PolicyH5Dataset(path, split="train", augment=cfg.train.augment,
                         max_samples=cfg.train.max_train_samples, seed=cfg.train.seed)
    va = PolicyH5Dataset(path, split="val", augment=False, seed=cfg.train.seed)
    if len(tr) == 0:
        raise ValueError(f"{path}: train 샘플이 0 개입니다 (split 배정을 확인하십시오)")
    kw = dict(num_workers=cfg.train.num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=cfg.train.num_workers > 0)
    tl = DataLoader(tr, batch_size=cfg.train.batch_size, shuffle=True, drop_last=len(tr) > cfg.train.batch_size, **kw)
    vl = DataLoader(va, batch_size=cfg.train.batch_size, shuffle=False, **kw) if len(va) else None
    return tl, vl, tr, va if len(va) else None


class Trainer:
    def __init__(self, cfg: PolicyConfig, dataset_path: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.cfg = cfg
        self.device = pick_device(cfg.train.device)
        seed_everything(cfg.train.seed)
        self.out = cfg.resolve(cfg.paths.output_dir) if output_dir is None else Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.out / "config.yaml")
        self.train_loader, self.val_loader, self.train_ds, self.val_ds = make_loaders(cfg, dataset_path)
        self.model: ActPolicy = build_policy(cfg.model, cfg.timing).to(self.device)
        logger.info("모델 파라미터 %.2fM, device=%s, train=%d val=%d", count_parameters(self.model) / 1e6,
                    self.device, len(self.train_ds), 0 if self.val_ds is None else len(self.val_ds))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        steps_per_epoch = max(1, len(self.train_loader))
        total = cfg.train.epochs * steps_per_epoch
        warm = cfg.train.warmup_epochs * steps_per_epoch

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / max(1, warm)
            prog = (step - warm) / max(1, total - warm)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.use_amp = bool(cfg.train.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.epoch = 0
        self.best = float("inf")
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ 체크포인트
    def save(self, name: str) -> Path:
        path = self.out / name
        torch.save({
            "model_state": self.model.state_dict(), "optimizer_state": self.opt.state_dict(),
            "scheduler_state": self.sched.state_dict(), "scaler_state": self.scaler.state_dict(),
            "epoch": self.epoch, "best": self.best, "config": self.cfg.to_dict(),
            "state_features": self.train_ds.state_features,
        }, path)
        return path

    def resume(self, path: Path) -> None:
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state"])
        self.opt.load_state_dict(ck["optimizer_state"])
        self.sched.load_state_dict(ck["scheduler_state"])
        self.scaler.load_state_dict(ck["scaler_state"])
        self.epoch = int(ck["epoch"])
        self.best = float(ck["best"])
        logger.info("재개: %s (epoch %d, best %.4f)", path, self.epoch, self.best)

    # ------------------------------------------------------------------ 한 epoch
    def beta_scale(self) -> float:
        """KL 워밍업: 첫 epoch 에서 1/warm, beta_warmup_epochs 번째 epoch 부터 1 (선형)."""
        warm = int(self.cfg.train.beta_warmup_epochs)
        if warm <= 0:
            return 1.0
        return float(min(1.0, (self.epoch + 1) / warm))

    def train_epoch(self) -> dict[str, float]:
        self.model.train()
        agg: dict[str, list[float]] = {}
        t0 = time.time()
        bscale = self.beta_scale()
        for it, batch in enumerate(self.train_loader):
            batch = to_device(batch, self.device)
            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                out = self.model(batch, use_posterior=True)
                loss, logs = compute_loss(self.model, out, batch, self.cfg.loss, beta_scale=bscale)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.cfg.train.grad_clip > 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()
            self.sched.step()
            for k, v in logs.items():
                if np.isfinite(v):
                    agg.setdefault(k, []).append(v)
            if self.cfg.train.log_every and (it + 1) % self.cfg.train.log_every == 0:
                logger.info("  ep %d it %d/%d loss %.4f", self.epoch + 1, it + 1, len(self.train_loader), logs["total"])
        res = {k: float(np.mean(v)) for k, v in agg.items()}
        res["time_s"] = time.time() - t0
        res["lr"] = float(self.opt.param_groups[0]["lr"])
        return res

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader]) -> dict[str, float]:
        if loader is None:
            return {}
        self.model.eval()
        agg: dict[str, list[float]] = {}
        spreads, margins, sel_mae_y, sel_sign = [], [], [], []
        for batch in loader:
            batch = to_device(batch, self.device)
            out = self.model(batch, use_posterior=True)
            _, logs = compute_loss(self.model, out, batch, self.cfg.loss)
            for k, v in logs.items():
                if np.isfinite(v):
                    agg.setdefault(k, []).append(v)
            # 실행시 선택 경로의 진단 (§6): z 표본 → Q̂ 선택
            sel = self.model.select_action(batch, n_samples=self.cfg.train.z_samples_eval,
                                           gamma=self.cfg.train.gamma_mode_consistency,
                                           prev_dy=batch["vec"][:, 1] * batch["vec"][:, 3])
            P_lab = batch["P"]
            if "vy_spread" in sel:
                spreads.append(sel["vy_spread"].mean().item())
            m = sel["mode_margin"]
            m = m[torch.isfinite(m)]
            if m.numel():
                margins.append(m.mean().item())
            sel_mae_y.append((sel["net"][:, 1] - P_lab[:, -1, 1]).abs().mean().item())
            big = P_lab[:, -1, 1].abs() > batch["sigma_net"][:, 1]
            if big.any():
                sel_sign.append((torch.sign(sel["net"][big, 1]) == torch.sign(P_lab[big, -1, 1])).float().mean().item())
        res = {k: float(np.mean(v)) for k, v in agg.items()}
        if spreads:
            res["mode_collapse_vy_std"] = float(np.mean(spreads))
        if margins:
            res["mode_margin"] = float(np.mean(margins))
        if sel_mae_y:
            res["select_mae_y_mm"] = float(np.mean(sel_mae_y))
        if sel_sign:
            res["select_vy_sign_acc"] = float(np.mean(sel_sign))
        if sel_mae_y and "mae_y_mm" in res:
            res["leak_gap_mm"] = res["select_mae_y_mm"] - res["mae_y_mm"]
        return res

    # ------------------------------------------------------------------ 전체
    def fit(self) -> list[dict[str, Any]]:
        log_path = self.out / "metrics.jsonl"
        while self.epoch < self.cfg.train.epochs:
            tr = self.train_epoch()
            va = self.evaluate(self.val_loader)
            self.epoch += 1
            row = {"epoch": self.epoch, **{f"train/{k}": v for k, v in tr.items()},
                   **{f"val/{k}": v for k, v in va.items()}}
            self.history.append(row)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            score = va.get("total", tr["total"])
            msg = (f"epoch {self.epoch}/{self.cfg.train.epochs}  train {tr['total']:.4f}  "
                   f"val {va.get('total', float('nan')):.4f}  "
                   f"mae(x,y,θ)=({va.get('mae_x_mm', float('nan')):.2f},{va.get('mae_y_mm', float('nan')):.2f},"
                   f"{va.get('mae_th_deg', float('nan')):.2f})")
            if "mode_collapse_vy_std" in va:
                msg += f"  vy_std {va['mode_collapse_vy_std']:.2f}"
            if "mode_margin" in va:
                msg += f"  margin {va['mode_margin']:.3f}"
            if "leak_gap_mm" in va:
                msg += f"  leak_gap {va['leak_gap_mm']:.2f}mm"
            msg += f"  beta {tr.get('beta', float('nan')):.3f}"
            logger.info(msg)
            self.save("last.pt")
            if score < self.best:
                self.best = score
                self.save("best.pt")
        return self.history


def load_policy(path: str | Path, device: str = "auto") -> tuple[ActPolicy, PolicyConfig]:
    """학습된 체크포인트 → (모델, 설정)."""
    dev = pick_device(device)
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = PolicyConfig.from_dict(ck["config"])
    model = build_policy(cfg.model, cfg.timing).to(dev)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, cfg
